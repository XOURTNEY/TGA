import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from utils.general_utils import build_rotation

from bvh_tracing import _C


# AABB 是一种与坐标轴对齐的包围盒，即盒子的每一条边都平行于坐标轴。
# 这种盒子用两个对角点的坐标来定义——最小点坐标和最大点坐标.
# 物体位置或大小发生变化时，我们可以仅更新其父级及少数相关节点，而无需重新构建整个结构
class RayTracer:
    def __init__(self, means3D, scales, rotations):#(P, 3)
        P = means3D.shape[0] # 3D点数量。AABB结构中有P个是叶节点，P-1个二叉树的父节点。
        rot = build_rotation(rotations)
        nodes = torch.full((2 * P - 1, 5), -1, device="cuda").int() #（2P-1，5）存储 BVH。每个元素都是fill_value、形状为size、数据类型为dtype的Tensor
        nodes[:P - 1, 4] = 0
        nodes[P - 1:, 4] = 1 # 最后一列前半部是0，后半部是1
        aabbs = torch.zeros(2 * P - 1, 6, device="cuda").float() #（2P-1，6）存储 AABB，每个对象的最小和最大边界值初始化为极大值和极小值。
        aabbs[:, :3] = 100000
        aabbs[:, 3:] = -100000 # 前三列（最小边界值）是10000，后三列（最大边界值）是-100000

        a, b, c = rot[:, :, 0], rot[:, :, 1], rot[:, :, 2] # 旋转列向量
        m = 3
        sa, sb, sc = m * scales[:, 0], m * scales[:, 1], m * scales[:, 2] # 三倍sigma长

        # AABB 的 8 个顶点坐标
        x111 = means3D + a * sa[:, None] + b * sb[:, None] + c * sc[:, None] #(P, 3)
        x110 = means3D + a * sa[:, None] + b * sb[:, None] - c * sc[:, None]
        x101 = means3D + a * sa[:, None] - b * sb[:, None] + c * sc[:, None]
        x100 = means3D + a * sa[:, None] - b * sb[:, None] - c * sc[:, None]
        x011 = means3D - a * sa[:, None] + b * sb[:, None] + c * sc[:, None]
        x010 = means3D - a * sa[:, None] + b * sb[:, None] - c * sc[:, None]
        x001 = means3D - a * sa[:, None] - b * sb[:, None] + c * sc[:, None]
        x000 = means3D - a * sa[:, None] - b * sb[:, None] - c * sc[:, None]
        aabb_min = torch.minimum(torch.minimum(
            torch.minimum(torch.minimum(torch.minimum(torch.minimum(torch.minimum(x111, x110), x101), x100), x011),
                          x010), x001), x000)#(P, 3)
        aabb_max = torch.maximum(torch.maximum(torch.maximum(torch.maximum(
            torch.maximum(torch.maximum(torch.maximum(x111, x110), x101), x100), x011), x010), x001), x000)

        aabbs[P - 1:] = torch.cat([aabb_min, aabb_max], dim=-1) #(P, 6)

        self.tree, self.aabb, self.morton = _C.create_bvh(means3D, scales, rotations, nodes, aabbs)

    @torch.no_grad()
    def trace_visibility(self, rays_o, rays_d, means3D, symm_inv, opacity, normals):
        cotrib, opa = _C.trace_bvh_opacity(self.tree, self.aabb,
                                                 rays_o, rays_d,
                                                 means3D, symm_inv,
                                                 opacity, normals)
        return {
            "visibility": opa.unsqueeze(-1),
            "contribute": cotrib.unsqueeze(-1),
        }
