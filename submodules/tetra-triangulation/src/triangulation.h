#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/compute_average_spacing.h>
#include <torch/extension.h>  // 添加 PyTorch 头文件

//#include <tbb/parallel_for.h>
//#include <tbb/blocked_range.h>


typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_with_info_3<unsigned int, K> Vb;
typedef CGAL::Triangulation_data_structure_3<Vb> Tds;
typedef CGAL::Delaunay_triangulation_3<K, Tds, CGAL::Fast_location> Triangulation;

typedef Triangulation::Point Point;
typedef Triangulation::Vertex_handle Vertex_handle; // 定义 Vertex_handle


// 函数声明
//std::vector<uint4> triangulate(size_t num_points, float3 *points); // xxchange 1022
std::vector<uint4> triangulate(size_t num_points, float3* points, std::vector<Vertex_handle>& vertex_map, Triangulation& T);
std::vector<uint4> batch_move(Triangulation& T, const std::vector<unsigned int>& vertices_to_move, const std::vector<Point>& new_points, std::vector<Vertex_handle>& vertex_map);
//void parallel_move(Triangulation& T, const std::vector<unsigned int>& vertices_to_move, const std::vector<Point>& new_points, std::vector<Vertex_handle>& vertex_map);


// Wrapper 类声明
class TriangulationWrapper {
public:
    Triangulation T;                     // 三角剖分对象
    std::vector<Vertex_handle> vertex_map;  // 顶点映射

    // 构造函数
    TriangulationWrapper() {}

    // 包装函数
    std::vector<uint4> triangulate_wrapper(size_t num_points, float3* points);
    std::vector<uint4> batch_move_wrapper(const std::vector<unsigned int>& vertices_to_move, const std::vector<Point>& new_points);
//    void parallel_move_wrapper(const std::vector<unsigned int>& vertices_to_move, const std::vector<Point>& new_points);

 // 添加 py_triangulate 函数的声明
    torch::Tensor py_triangulate(const torch::Tensor &points);
    torch::Tensor py_batch_move(const torch::Tensor &vertices_to_move, const torch::Tensor &new_points);
};