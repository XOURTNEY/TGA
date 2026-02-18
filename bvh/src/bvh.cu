#ifndef BVH_BVH_CU
#define BVH_BVH_CU
#include <tuple>
#include "bvh.h"
#include "construct.cuh"
#include "trace.cuh"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
create_bvh(const torch::Tensor& means3D, const torch::Tensor& scales, const torch::Tensor& rotations, const torch::Tensor& nodes, const torch::Tensor& aabbs){
    //                          (P, 3)                        (P, 3)                       (P, 3)                 （2P-1，5）                  （2P-1，6）
    const uint32_t P = means3D.size(0);

    auto int_opts = means3D.options().dtype(torch::kInt32);
    auto float_opts = means3D.options().dtype(torch::kFloat32); // 存储位置一致

    torch::Tensor mortons = torch::zeros({P}, means3D.options().dtype(torch::kLong));

    construct_bvh(
            P,
            means3D.contiguous().data_ptr<float>(),
            scales.contiguous().data_ptr<float>(),
            rotations.contiguous().data_ptr<float>(),
            (int32_t*)nodes.contiguous().data_ptr<int>(),
            aabbs.contiguous().data_ptr<float>(),
            (uint64_t*)mortons.contiguous().data_ptr<int64_t>()
    );
    return std::make_tuple(nodes, aabbs, mortons); // 摩顿码：空间填充曲线，帮助将三维空间索引映射为一维。
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
trace_bvh(const torch::Tensor& nodes, const torch::Tensor& aabbs,
          const torch::Tensor& rays_o, const torch::Tensor& rays_d,
          const torch::Tensor& means3D, const torch::Tensor& covs3D,
          const torch::Tensor& opacities){
    int32_t num_rays = rays_o.size(0);

    auto int_opts = rays_o.options().dtype(torch::kInt32); // 张量的配置（类型和设备）
    auto float_opts = rays_o.options();
    torch::Tensor num_contributes = torch::zeros({num_rays, 1}, int_opts); // 记录每条射线与物体交叉的次数

    auto result = trace_bvh_cuda(num_rays,
                   nodes.contiguous().data_ptr<int32_t>(),
                   aabbs.contiguous().data_ptr<float>(),
                   (float3*)rays_o.contiguous().data_ptr<float>(),
                   (float3*)rays_d.contiguous().data_ptr<float>(),
                   (float3*)means3D.contiguous().data_ptr<float>(),
                   covs3D.contiguous().data_ptr<float>(),
                   opacities.contiguous().data_ptr<float>(),
                   num_contributes.contiguous().data_ptr<int32_t>());

//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start);
//     float milliseconds = 0;

    int32_t num_rendered = std::get<0>(result); // 光线与物体相交的数量
    //  CUDA的在GPU上存储操作的数据结构，类似于std::vector，是动态数组
    thrust::device_vector<int32_t>& point_list_vec = std::get<1>(result); // 相交点
    thrust::device_vector<float3>& position_list_vec = std::get<2>(result); // 相交位置
    thrust::device_vector<int32_t>& ray_id_list_vec = std::get<3>(result); // 光线 ID
    if (num_rendered == 0){ // 没有光线与任何物体相交
        torch::Tensor point_list_tensor = torch::zeros({0, 1}, int_opts);
        torch::Tensor position_list_tensor = torch::zeros({0, 3}, float_opts);
        torch::Tensor ray_id_list_tensor = torch::zeros({0, 3}, float_opts);
        return std::make_tuple(num_contributes, point_list_tensor, position_list_tensor, ray_id_list_tensor);
    }

    // 将 thrust 向量转换为 PyTorch 张量
    int32_t* point_list = thrust::raw_pointer_cast(point_list_vec.data()); // 获取point_list_vec存储数据的指针，转换为普通的 C++ 原始指针（int32_t*）
    int32_t size = point_list_vec.size();
    torch::Tensor point_list_tensor = torch::from_blob(point_list, {size, 1}, int_opts);// (size, 1)，从现有的数据指针创建一个张量
    // from_blob不复制数据，直接使用传入的指针 point_list，即张量的数据与指针共享同一片内存。
    point_list_tensor = point_list_tensor.clone(); // 创建一个与原张量相同的副本，同时分配新的内存

    float* position_list = (float*)thrust::raw_pointer_cast(position_list_vec.data());
    torch::Tensor position_list_tensor = torch::from_blob(position_list, {size, 3}, float_opts);
    position_list_tensor = position_list_tensor.clone();

    int32_t* ray_id_list = thrust::raw_pointer_cast(ray_id_list_vec.data());
    torch::Tensor ray_id_list_tensor = torch::from_blob(ray_id_list, {size, 1}, int_opts);
    ray_id_list_tensor = ray_id_list_tensor.clone();

//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     std::cout << "after time: " << milliseconds << " ms" << std::endl;
//     cudaEventRecord(start);
    return std::make_tuple(num_contributes, point_list_tensor, position_list_tensor, ray_id_list_tensor);
    //                     光线的贡献数量          碰撞点索引            碰撞位置               光线 ID
}


std::tuple<torch::Tensor, torch::Tensor>
trace_bvh_opacity(const torch::Tensor& nodes, const torch::Tensor& aabbs,
          const torch::Tensor& rays_o, const torch::Tensor& rays_d,
          const torch::Tensor& means3D, const torch::Tensor& covs3D,
          const torch::Tensor& opacities, const torch::Tensor& normals){
    int32_t num_rays = rays_o.numel() / rays_o.size(-1);
    auto rays_o_shape = rays_o.sizes().slice(0, rays_o.dim() - 1);
//     auto rays_o_shape = rays_o.sizes().vec();
//     rays_o_shape.pop_back();
//     rays_o_shape.push_back(1);

    auto int_opts = rays_o.options().dtype(torch::kInt32);
    auto float_opts = rays_o.options();
    torch::Tensor num_contributes = torch::zeros(rays_o_shape, int_opts);
    torch::Tensor rendered_opacity = torch::ones(rays_o_shape, float_opts);

    trace_bvh_opacity_cuda(num_rays,
                   nodes.contiguous().data_ptr<int32_t>(),
                   aabbs.contiguous().data_ptr<float>(),
                   (float3*)rays_o.contiguous().data_ptr<float>(),
                   (float3*)rays_d.contiguous().data_ptr<float>(),
                   (float3*)means3D.contiguous().data_ptr<float>(),
                   covs3D.contiguous().data_ptr<float>(),
                   opacities.contiguous().data_ptr<float>(),
                   (float3*)normals.contiguous().data_ptr<float>(),
                   num_contributes.contiguous().data_ptr<int32_t>(),
                   rendered_opacity.contiguous().data_ptr<float>());
    return std::make_tuple(num_contributes, rendered_opacity);
}

#endif //BVH_BVH_CU