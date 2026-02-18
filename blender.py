import bpy, math, mathutils
import os

# 找到第一个颜色属性的名字（Blender 4.x 用 color_attributes，3.x 还可能有 vertex_colors）
def first_color_attr_name(mesh):
    # Blender 4.x 推荐 color_attributes
    if hasattr(mesh, "color_attributes") and len(mesh.color_attributes) > 0:
        return mesh.color_attributes[0].name
    # 兼容 3.x 的 vertex_colors
    if hasattr(mesh, "vertex_colors") and len(mesh.vertex_colors) > 0:
        return mesh.vertex_colors[0].name
    return None


# ===== 你的相机参数 =====
fx = 2046.625
fy = 2046.643798828125
cx = 274.8968200683594
cy = 400.91552734375
H, W = 802, 550

import numpy as np
transform_matrix = np.array([
    [ 0.9954220652580261,   0.02225305698812008,  -0.0929495245218277,   -0.10245557501912117],
    [ 0.00161041808314621,  0.9684742093086243,    0.24910885095596313,   0.2677168846130371],
    [ 0.0955626368522644,  -0.24811816215515137,   0.964004635810852,     1.1484033912420273],
    [ 0.0,                  0.0,                   0.0,                   1.0]
], dtype=np.float64)

M = transform_matrix.copy()
IS_C2W = True  # 如果你确定这是 world-to-camera，请改为 False

# ===== 清场景 =====
bpy.ops.wm.read_factory_settings(use_empty=True)

# 分辨率 & 帧率
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.render.resolution_x = W
scene.render.resolution_y = H
scene.render.fps = 30
scene.cycles.samples = 60

# ===== 建相机并设置内参 =====
cam_data = bpy.data.cameras.new("Cam")
cam_obj = bpy.data.objects.new("Cam", cam_data)
bpy.context.scene.collection.objects.link(cam_obj)
scene.camera = cam_obj

# 用水平视场角匹配 fx（让传感器以宽度为准）
# FOVx = 2 * atan(W / (2 * fx))
fov_x = 2.0 * math.atan(W / (2.0 * fx))
cam_data.lens_unit = 'FOV'
cam_data.angle = fov_x
cam_data.sensor_fit = 'HORIZONTAL'

# principal point 偏移：Blender 4.x 可用 camera.shift_x/y（以相机宽/高的半宽高为单位）
# shift_x = (cx - W/2) / (W/2), shift_y = -(cy - H/2) / (H/2)
cam_data.shift_x = (cx - W * 0.5) / (W * 0.5)
cam_data.shift_y = - (cy - H * 0.5) / (H * 0.5)

# ===== 外参到 Blender 的相机位姿 =====
# 若 M 是 c2w（OpenGL 约定：相机朝 -Z，Y 向上），可直接用作相机的世界矩阵；
# 若 M 是 w2c：用其逆矩阵作为相机的 world matrix。
if not IS_C2W:
    M = np.linalg.inv(M)

# 注意：Blender 世界是 Z 向上、相机局部看向 -Z，和 NeRF/OpenGL 一致；
# 如果你的矩阵来自 OpenCV（相机看 +Z，Y 向下），需要做一次坐标系转换：
#   Cv2Gl = diag([1, -1, -1])，然后：
#   R_gl = R_cv @ Cv2Gl.T,  t_gl = t_cv
# 本最小例假定已是 OpenGL/NeRF 风格。

# 把 numpy 矩阵塞进 Blender
mw = mathutils.Matrix([[M[0,0], M[0,1], M[0,2], M[0,3]],
                       [M[1,0], M[1,1], M[1,2], M[1,3]],
                       [M[2,0], M[2,1], M[2,2], M[2,3]],
                       [M[3,0], M[3,1], M[3,2], M[3,3]]])
cam_obj.matrix_world = mw

# ===== 简单灯光 =====
light = bpy.data.lights.new("Key", type='SUN')
light.energy = 1.5
light_obj = bpy.data.objects.new("Key", light)
light_obj.rotation_euler = (math.radians(40), math.radians(0), math.radians(20))
bpy.context.scene.collection.objects.link(light_obj)

base_dir = "/home/gb/GaussianAvatars/output/460_r2/meshh/ours_200000/fusion"
out_dir  = os.path.join(base_dir, "blender")
os.makedirs(out_dir, exist_ok=True)

last_obj = None




# 👉 新增：在 for 循环之前放置（只建一次）
clay_mat = bpy.data.materials.new("ClayWhite")
clay_mat.use_nodes = True
n = clay_mat.node_tree.nodes
l = clay_mat.node_tree.links
for x in list(n):
    n.remove(x)
out_node = n.new("ShaderNodeOutputMaterial")
bsdf = n.new("ShaderNodeBsdfPrincipled")
bsdf.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 0.5)  # 纯白
bsdf.inputs["Roughness"].default_value = 0.8                     # 偏哑光               # 少许高光，避免“塑料感”
l.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])




for idx in range(0, 501):  # 渲染 0~100
    mesh_path = os.path.join(base_dir, f"mesh_binary_search_frame1_{idx}.ply")
    if not os.path.exists(mesh_path):
        print(f"[WARN] 缺失: {mesh_path}，跳过")
        continue

    # 删除上一个 mesh（保留相机/灯光/世界设置）
    if last_obj and last_obj.name in bpy.data.objects:
        bpy.ops.object.select_all(action='DESELECT')
        last_obj.select_set(True)
        bpy.ops.object.delete(use_global=False)
        # 清理孤立 mesh 数据块
        for b in list(bpy.data.meshes):
            if b.users == 0:
                bpy.data.meshes.remove(b)

    # 导入当前 mesh（沿用你原来的导入逻辑）
    ext = os.path.splitext(mesh_path)[1].lower()
    if ext == ".obj":
        try:
            bpy.ops.wm.obj_import(files=[{"name": os.path.basename(mesh_path)}], directory=os.path.dirname(mesh_path))
        except Exception:
            bpy.ops.import_scene.obj(filepath=mesh_path)
    elif ext == ".ply":
        try:
            bpy.ops.wm.ply_import(files=[{"name": os.path.basename(mesh_path)}], directory=os.path.dirname(mesh_path))
        except Exception:
            bpy.ops.import_mesh.ply(filepath=mesh_path)
    else:
        print(f"[WARN] 不支持的格式: {mesh_path}，跳过")
        continue

    # 选中导入的 mesh
    obj = [o for o in bpy.context.selected_objects if o.type == 'MESH'][0]
    last_obj = obj

    # === 环境光 / HDRI ===
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # 清空旧节点
    for n in list(nodes):
        nodes.remove(n)

    bg = nodes.new("ShaderNodeBackground")
    bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)  # 👉 修改处：背景设为纯白
    bg.inputs[1].default_value = 0.4 # 强度(Exposure)，1.0~3.0 之间调
    bg.inputs["Color"].default_value = (0.3, 0.3, 0.3, 0.3)
    out = nodes.new("ShaderNodeOutputWorld")
    links.new(bg.outputs["Background"], out.inputs["Surface"])

    # scene.render.film_transparent = True

    # —— 顶点颜色材质（复用你前面定义的 first_color_attr_name）——
    col_name = first_color_attr_name(obj.data)
    # if col_name is None:
    #     print(f"[WARN] {mesh_path} 没有顶点颜色，使用灰材质")
    #     mat = bpy.data.materials.new("NoVCol")
    #     mat.use_nodes = True
    #     bsdf = mat.node_tree.nodes.get("Principled BSDF")
    #     bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
    #     bsdf.inputs["Roughness"].default_value = 0.4
    #     obj.data.materials.clear()
    #     obj.data.materials.append(mat)
    # else:
    #     mat = bpy.data.materials.new("VertexColorMat")
    #     mat.use_nodes = True
    #     nodes = mat.node_tree.nodes
    #     links = mat.node_tree.links
    #     for n in list(nodes):
    #         nodes.remove(n)
    #     out_node = nodes.new("ShaderNodeOutputMaterial")
    #     bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    #     attr = nodes.new("ShaderNodeAttribute")
    #     attr.attribute_name = col_name
    #     links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    #     links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])
    #     bsdf.inputs["Roughness"].default_value = 0.8
    #     obj.data.materials.clear()
    #     obj.data.materials.append(mat)

    bpy.ops.object.shade_smooth()

    # 渲染到指定目录
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = os.path.join(out_dir, f"frame_{idx:03d}.png")
    bpy.ops.render.render(write_still=True)
    print(f"[OK] Saved: {scene.render.filepath}")


# BLENDER_DIR=/home/gb/blender-git/build_linux/bin/blender
