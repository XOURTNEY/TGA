#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <string>

#include "triangulation.h"

namespace py = pybind11;
using namespace pybind11::literals;

// 检查函数宏
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_DEVICE(x) TORCH_CHECK(x.device() == this->device, #x " must be on the same device")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must have float32 type")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT_DIM3(x) \
    CHECK_INPUT(x);         \
    CHECK_DEVICE(x);        \
    CHECK_FLOAT(x);         \
    TORCH_CHECK(x.size(-1) == 3, #x " must have last dimension with size 3")

// 绑定的三角剖分函数
torch::Tensor TriangulationWrapper::py_triangulate(const torch::Tensor &points) {
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, "points must have shape [num_points, 3]");

    const auto points_ = points.cpu().contiguous();
    auto cells = this->triangulate_wrapper(points_.size(0), reinterpret_cast<float3*>(points_.data_ptr()));

    auto cells_out = torch::empty({(long)cells.size(), 4}, torch::dtype(torch::kInt32).device(torch::kCPU));
    memcpy(cells_out.data_ptr(), reinterpret_cast<void*>(cells.data()), cells.size() * sizeof(uint4));
    return cells_out.to(points.device());
}


// 绑定的批量移动函数
torch::Tensor TriangulationWrapper::py_batch_move( const torch::Tensor &vertices_to_move, const torch::Tensor &new_points) {

    TORCH_CHECK(vertices_to_move.dim() == 1, "vertices_to_move must be a 1D tensor");
    TORCH_CHECK(new_points.dim() == 2 && new_points.size(1) == 3, "new_points must have shape [num_points, 3]");
    TORCH_CHECK(vertices_to_move.size(0) == new_points.size(0), "vertices_to_move and new_points must have the same size");

    const auto vertices_to_move_ = vertices_to_move.cpu().contiguous();
    const auto new_points_ = new_points.cpu().contiguous(); 

    const long* v_data_ptr = vertices_to_move_.data_ptr<long>();// 获取指向数据的指针

    // 使用正确的类型创建 std::vector
    std::vector<unsigned int> vertices_to_move_vec(vertices_to_move_.size(0));
    for (size_t i = 0; i < vertices_to_move_.size(0); ++i) {
        vertices_to_move_vec[i] = static_cast<long>(v_data_ptr[i]);
    }

    // 创建 std::vector 用于存储 new_points
    std::vector<Point> new_points_vec;
    for (int i = 0; i < new_points_.size(0); ++i) {
        float x = new_points_.index({i, 0}).item<float>(); // 获取x坐标
        float y = new_points_.index({i, 1}).item<float>(); // 获取y坐标
        float z = new_points_.index({i, 2}).item<float>(); // 获取z坐标
//        printf("vertex %d is(%lf.%lf,%lf)\n",i,x,y,z);
        new_points_vec.emplace_back(x, y, z); // 将点存入vector
    }
        printf("vertex 0 is(%lf.%lf,%lf)\n",new_points_vec[0].x(),new_points_vec[0].y(),new_points_vec[0].z());
    auto cells = this->batch_move_wrapper(vertices_to_move_vec, new_points_vec);
    auto cells_out = torch::empty({(long)cells.size(), 4}, torch::dtype(torch::kInt32).device(torch::kCPU));
    memcpy(cells_out.data_ptr(), reinterpret_cast<void*>(cells.data()), cells.size() * sizeof(uint4));
    return cells_out.to(vertices_to_move.device());
}

// PYBIND11_MODULE 注册模块和绑定
PYBIND11_MODULE(tetranerf_cpp_extension, m) {
    py::class_<TriangulationWrapper>(m, "TriangulationWrapper")
            .def(py::init<>())
            .def("triangulate", &TriangulationWrapper::py_triangulate)
            .def("batch_move", &TriangulationWrapper::py_batch_move);
//            .def("parallel_move", &py_parallel_move);
} 