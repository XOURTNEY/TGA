import torch
import time
from scene import Scene
import os
from os import makedirs
from gaussian_renderer import render, integrate
import random
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, FlameGaussianModel
import numpy as np
import trimesh
from tetranerf.utils.extension import cpp
from utils.tetmesh import marching_tetrahedra
from scipy.spatial import KDTree
from bvh import RayTracer # xxadd0109

def merge_mesh(coarse_mesh, fine_mesh, threshold = 0.020):
    # 提取差异部分（使用 Trimesh 内建的布尔操作）
    # result = coarse_mesh.difference(fine_mesh)

    # 1. 计算 fine mesh 每个面片的质心
    fine_faces = fine_mesh.faces
    fine_vertices = fine_mesh.vertices
    fine_face_centroids = fine_vertices[fine_faces].mean(axis=1)  # 面片质心

    # 2. 建立 coarse mesh 的 KDTree
    coarse_vertices = coarse_mesh.vertices
    tree = KDTree(coarse_vertices)

    # 3. 计算 fine mesh 的面片质心到 coarse mesh 的最小距离
    distances, _ = tree.query(fine_face_centroids)

    # 4. 筛选出距离小于阈值的面片
    valid_faces_mask = distances < threshold
    filtered_faces = fine_faces[valid_faces_mask]

    # 5. 创建过滤后的 fine mesh
    filtered_fine_mesh = trimesh.Trimesh(
        vertices=fine_vertices, faces=filtered_faces, process=False
    )

    # 6. 合并 coarse mesh 和过滤后的 fine mesh
    merged_vertices = np.vstack([coarse_mesh.vertices, filtered_fine_mesh.vertices])
    offset = len(coarse_mesh.vertices)
    merged_faces = np.vstack([
        coarse_mesh.faces,
        filtered_fine_mesh.faces + offset  # 调整 fine mesh 面片索引
    ])

    merged_mesh = trimesh.Trimesh(vertices=merged_vertices, faces=merged_faces, process=False)

    # return merged_mesh
    return filtered_fine_mesh

@torch.no_grad()
def evaluage_alpha(points, views, gaussians, pipeline, background, kernel_size, return_color=False):
    final_alpha = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
    if return_color:
        final_color = torch.ones((points.shape[0], 3), dtype=torch.float32, device="cuda")

    # integrate是算每个points的，render是算gaussian的渲染
    views_sparse = [views[0],views[7],views[14]] # xxchange0304 第八个视角被抽出去了
    with torch.no_grad():
        for _, view in enumerate(tqdm(views_sparse, desc="Rendering progress")):
            ret = integrate(points, view, gaussians, pipeline, background, kernel_size=kernel_size)
            alpha_integrated = ret["alpha_integrated"]
            # 选择alpha最小的视角，来收集color
            if return_color:
                color_integrated = ret["color_integrated"]
                final_color = torch.where((alpha_integrated < final_alpha).reshape(-1, 1), color_integrated,
                                          final_color)
            final_alpha = torch.min(final_alpha, alpha_integrated)

        alpha = 1 - final_alpha
    if return_color:
        return alpha, final_color
    return alpha

@torch.no_grad()
def marching_tetrahedra_with_binary_search(start_from, render_path, views, gaussians, pipeline, background,
                                           kernel_size, filter_mesh: bool, texture_mesh: bool,
                                           alpha_last, xyz_last, scales_last, cells_last, frame_idx, wrapper):

    global ray_tracer
    makedirs(render_path, exist_ok=True)

    # generate tetra points here
    points, points_scale = gaussians.get_tetra_points() # gs+8个顶点的xyz/scale
    alpha = evaluage_alpha(points, views, gaussians, pipeline, background, kernel_size)
    alpha_now = alpha.clone()
    xyz_now = points.clone() # 本身即为xyz
    scales_now = points_scale.clone().expand(-1, 3)

    points_rotations = torch.zeros(points.shape[0], 4) # xxadd0109
    points_rotations[:, 0] = 1

    timestamp0 = time.time()
    # if frame_idx==0:
    if frame_idx>=0:
        # xxadd0109
        ray_tracer = RayTracer(xyz_now, scales_now, points_rotations)

        # create cell and save cells
        print("create cells and save")
        cells = wrapper.triangulate(points)
        # we should filter the cell if it is larger than the gaussians
        # torch.save(cells, os.path.join(render_path, "cells.pt")) # 暂时不保存
    # load cell if exists
    # if os.path.exists(os.path.join(render_path, "cells.pt")):
    #     print("load existing cells")
    #     cells = torch.load(os.path.join(render_path, "cells.pt"))
    # else:
    #     # create cell and save cells
    #     print("create cells and save")
    #     cells = cpp.triangulate(points)
    #     # we should filter the cell if it is larger than the gaussians
    #     torch.save(cells, os.path.join(render_path, "cells.pt"))

    # evaluate alpha
    if frame_idx<0: # 20251010 hide for camera ready.
        delta_alpha = alpha_now - alpha_last
        alpha_exceed_idx = torch.where(torch.abs(delta_alpha) > 10)[0]
        delta_xyz = xyz_now - xyz_last
        delta_xyz_magnitude = torch.norm(delta_xyz, dim=1)
        xyz_exceed_idx = torch.where(delta_xyz_magnitude > 0.0005)[0]


        xyz_noexceed_idx = torch.where(delta_xyz_magnitude < 0.5)[0]
        xyz_now[xyz_noexceed_idx] = xyz_last[xyz_noexceed_idx]
        scales_now[xyz_noexceed_idx] = scales_last[xyz_noexceed_idx]
        nochanges_flags = torch.zeros(xyz_now.shape[0], dtype=torch.int32)
        nochanges_flags[xyz_noexceed_idx] = 1
        # ray_tracer.update_bvh(xyz_now, scales_now, points_rotations, nochanges_flags) # xxadd0109

        # dynamic_scores = ray_tracer.scores[points.shape[0]-1: 2*points.shape[0]-1]
        # dynamic_idx = torch.where(dynamic_scores > 0.0) [0]
        # print(str(frame_idx) + ":" + str(alpha_exceed_idx.shape[0]) + " " + str(xyz_exceed_idx.shape[0]) + " " + str(dynamic_idx.shape[0]) + " points to move\n")

        # 改一下这个idx的求法 有点倒反天罡了 不亚于说直接>0 xxchange0109 done
        # 考虑增量式
        cat_exceed_idx = torch.unique(torch.cat([alpha_exceed_idx, xyz_exceed_idx])) # xxadd1128 防止有的点被重复删除而报错
        points_to_move_xyz = xyz_now[cat_exceed_idx]

        # xxadd0109
        # if((alpha_exceed_idx.shape[0] + xyz_exceed_idx.shape[0]) > 0):
        #     cells = wrapper.batch_move(cat_exceed_idx, points_to_move_xyz) # int64 float32
        # if dynamic_idx.shape[0] > 0:
        #     cells = wrapper.batch_move(dynamic_idx, xyz_now[dynamic_idx])  # int64 float32
        # else: cells = cells_last
        cells = cells_last
# 移动的点太多可能会导致Process finished with exit code 139 (interrupted by signal 11:SIGSEGV) 来自于gaussians的重复xyz


    vertices = points.cuda()[None]
    tets = cells.cuda().long()

    print(vertices.shape, tets.shape, alpha.shape)

    def alpha_to_sdf(alpha):
        sdf = alpha - 0.5
        sdf = sdf[None]
        return sdf

    sdf = alpha_to_sdf(alpha)

    torch.cuda.empty_cache()
    verts_list, scale_list, faces_list, _ = marching_tetrahedra(vertices, tets, sdf, points_scale[None])
    torch.cuda.empty_cache()

    end_points, end_sdf = verts_list[0]
    end_scales = scale_list[0]

    faces = faces_list[0].cpu().numpy()
    points = (end_points[:, 0, :] + end_points[:, 1, :]) / 2.

    left_points = end_points[:, 0, :]
    right_points = end_points[:, 1, :]
    left_sdf = end_sdf[:, 0, :]
    right_sdf = end_sdf[:, 1, :]
    left_scale = end_scales[:, 0, 0]
    right_scale = end_scales[:, 1, 0]
    distance = torch.norm(left_points - right_points, dim=-1)
    scale = left_scale + right_scale

    n_binary_steps = 8
    for step in range(n_binary_steps):
        # print("binary search in step {}".format(step))
        mid_points = (left_points + right_points) / 2
        alpha = evaluage_alpha(mid_points, views, gaussians, pipeline, background, kernel_size)
        mid_sdf = alpha_to_sdf(alpha).squeeze().unsqueeze(-1)

        ind_low = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))

        left_sdf[ind_low] = mid_sdf[ind_low]
        right_sdf[~ind_low] = mid_sdf[~ind_low]
        left_points[ind_low.flatten()] = mid_points[ind_low.flatten()]
        right_points[~ind_low.flatten()] = mid_points[~ind_low.flatten()]

        points = (left_points + right_points) / 2
        if step not in [7]:
            continue

        if texture_mesh:
            _, color = evaluage_alpha(points, views, gaussians, pipeline, background, kernel_size, return_color=True)
            vertex_colors = (color.cpu().numpy() * 255).astype(np.uint8)
        else:
            vertex_colors = None
        mesh = trimesh.Trimesh(vertices=points.cpu().numpy(), faces=faces, vertex_colors=vertex_colors, process=False)

        # filter
        if filter_mesh:
            mask = (distance <= 2.0 * scale).cpu().numpy() # xxchange1209: 2.0* ->1.0*
            face_mask = mask[faces].all(axis=1)
            mesh.update_vertices(mask)
            mesh.update_faces(face_mask)


        # xxadd 1204: hole-filling
        # import pymeshlab
        # ms = pymeshlab.MeshSet()# 将 trimesh 转换为 MeshLab 对象
        # mesh_pymeshlab = pymeshlab.Mesh(mesh.vertices.tolist(), mesh.faces.tolist())
        #mesh_pymeshlab.vertex_colors = vertex_colors
        # ms.add_mesh(mesh_pymeshlab)
        # ms.apply_filter('meshing_close_holes', maxholesize=30,
        #               selected=False,
        #               newfaceselected=True,
        #               selfintersection=True,
        #               refinehole=False,
        #               refineholeedgelen=pymeshlab.PercentageValue(3))  # 10表示填充最大孔洞尺寸（可根据实际情况调整）
        # output_path = os.path.join(render_path, f"mesh_binary_search_{frame_idx}.ply")
        # ms.save_current_mesh(output_path)


        # xxadd 1128
        # normals = mesh.vertex_normals
        # # 翻转法线，如果需要的话
        # inverted_normals = -normals
        # mesh.vertex_normals = inverted_normals
        # success = trimesh.repair.repair(mesh)  # 修复孔洞, 没有用。只能修复单三角形空洞

        # xxadd1209 stitch创建三角形扇，插入新的顶点. fan包含了每个三角形的三个顶点索引，vertices插入的新顶点的坐标
        # boundary_edges = mesh.boundary # 获取网格的所有边界（非闭合边）
        # boundary_loops = [boundary_edges]  # 这里假设只有一个边界环，实际情况需要根据拓扑结构处理
        # boundary_faces = [] # 提取面片与边界边的关系
        # for loop in boundary_loops:# 假设每个边界边都有一个相应的面片索引
        #     loop_faces = []
        #     for edge in loop:
        #         # 通过边索引找到与该边相连的面片索引
        #         edge_faces = mesh.face_neighbors[edge[0], edge[1]]  # 获取与该边相邻的面片
        #         loop_faces.extend(edge_faces)
        #     boundary_faces.append(loop_faces)
        # for boundary_loop in boundary_loops:
        #     fan, new_vertices = trimesh.repair.stitch(mesh, faces=boundary_loop, insert_vertices=True)
        #     if new_vertices is not None: # 更新网格的顶点列表
        #         mesh.vertices = np.vstack([mesh.vertices, new_vertices])
        #     mesh.faces = np.vstack([mesh.faces, fan]) # 将新生成的三角形面片添加到网格的面片列表


        # areas = mesh.area_faces
        # valid_faces = areas > 1e-7  # 根据需要设置阈值
        # mesh.update_faces(valid_faces)


        # mesh.fix_normals()  # 修复法线

        timestamp1 = time.time()
        print("用时" + str(timestamp1 - timestamp0) + "s\n")

        mesh.export(os.path.join(render_path, f"mesh_binary_search_frame{start_from}_{frame_idx}.ply"))


    # linear interpolation
    # right_sdf *= -1
    # points = (left_points * left_sdf + right_points * right_sdf) / (left_sdf + right_sdf)
    # mesh = trimesh.Trimesh(vertices=points.cpu().numpy(), faces=faces)
    # mesh.export(os.path.join(render_path, f"mesh_binary_search_interp.ply"))
    return alpha_now, xyz_now, scales_now, cells, mesh

def extract_mesh(dataset: ModelParams, iteration: int, start_from: int, pipeline: PipelineParams, filter_mesh: bool, texture_mesh: bool):
    render_path = os.path.join(dataset.model_path, "meshh", "ours_{}".format(iteration), "fusion")
    with torch.no_grad():
        # xxchange 0926
        if dataset.bind_to_mesh:
            # gaussians = FlameGaussianModel(dataset.sh_degree, dataset.disable_flame_static_offset)
            gaussians = FlameGaussianModel(dataset.sh_degree)
        else:
            gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # xxhide 0926
        # gaussians.load_ply(os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = dataset.kernel_size

        cams = scene.getTrainCameras()
        # xxchange 0926
        # 操作一下
        # 想一下现在得到的结果是neutral吗？？还是都有分布？todo
        # 反光怎么搞？改最值？ 先不管 先加入depthnormalloss看看（再加入dnloss后解决了）
        # 后脑勺补齐？？(再加入dnloss后解决了)

        # xxadd 1009
        # 现在的情况变成会漏洞了
        # 但是neutral还是都有我还是不确定

        # xxadd 1203
        # 高斯点云基于xyz去重
        points = gaussians.get_xyz
        unique_points, unique_idx = torch.unique(points, dim=0, return_inverse=True)
        counts = torch.bincount(unique_idx)
        duplicates_mask = counts > 1
        duplicate_idx = torch.nonzero(duplicates_mask).squeeze()
        print("\nIndices of Duplicate Points:")
        print(duplicate_idx)
        mask = torch.zeros(points.size(0), dtype=torch.bool, device=points.device)
        mask[torch.isin(unique_idx, duplicate_idx)] = True
        gaussians.prune_points_really(mask)
        gaussians.compute_3D_filter(cameras=cams[0:15])  # .filter_3D为过滤器

        # xxadd 1203
        # 验证重复点
        points = gaussians.get_xyz
        unique_points, counts = torch.unique(points, dim=0, return_counts=True)
        duplicates = unique_points[counts > 1]
        if duplicates.size(0) > 0:
            print("\nDuplicate Points Found:")
            print(duplicates)
        else:
            print("\nNo Duplicate Points Found.")

        # cams = scene.getValCameras() # 这样获得的都是同一个视角
        cams15 = cams.cameras[0:15] # 看看用同一个时间点的高斯位置。也可以用cam.timestamp来筛选

        start = 15*start_from # xxchange0304 第八个视角被抽出去了
        n = 1000 #1 # 20251010:for camera ready
        cameras = cams.cameras[start:start+15*n]
        alpha_now = torch.tensor([1,1,1])
        xyz_now = torch.tensor([1,1,1])
        scales_now = torch.tensor([1, 1, 1])
        cells_now = torch.tensor([1,1,1])

        wrapper = cpp.TriangulationWrapper()
        ray_tracer = None
        for frame_idx in range(n):
            gaussians.select_mesh_by_timestep(cameras[frame_idx*15].timestep) # xxdelete0304 start+
            a=gaussians.flame_model.mask.f

            if not os.path.exists(render_path):
                os.makedirs(render_path)

            gaussians.save_ply_with_color(os.path.join(render_path + f"/pcd_{frame_idx}.ply"))  # xxadd0120

            # 打印flame mesh用于补齐牙齿、后脑勺
            flame_mesh = trimesh.Trimesh(gaussians.verts.squeeze().cpu().numpy(), gaussians.faces.cpu().numpy())
            flame_mesh.export(os.path.join(render_path, f"flame_{frame_idx}.ply"))

            alpha_now, xyz_now, scales_now, cells_now, our_mesh = marching_tetrahedra_with_binary_search(start_from, render_path, cameras[15*frame_idx:15*(frame_idx+1)],
                                                                        gaussians, pipeline, background, kernel_size, filter_mesh, texture_mesh,
                                                                                             alpha_now, xyz_now, scales_now, cells_now, frame_idx,wrapper)

            fused_mesh = merge_mesh(flame_mesh, our_mesh)
            fused_mesh.export(os.path.join(render_path, f"fused_{frame_idx}.ply"))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--filter_mesh", action="store_true")
    parser.add_argument("--texture_mesh", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

    extract_mesh(model.extract(args), args.iteration, args.start, pipeline.extract(args), args.filter_mesh, args.texture_mesh)