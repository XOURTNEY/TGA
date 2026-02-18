#include "triangulation.h"

#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/compute_average_spacing.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <limits>

#include "utils/exception.h"

//typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
//typedef CGAL::Triangulation_vertex_base_with_info_3<unsigned int, K> Vb;
//typedef CGAL::Triangulation_data_structure_3<Vb> Tds;
//typedef CGAL::Delaunay_triangulation_3<K, Tds> Triangulation;

//typedef Triangulation::Point Point;

std::vector<uint4> triangulate(size_t num_points, float3* points, std::vector<Vertex_handle>& vertex_map, Triangulation& T) {
    std::vector<std::pair<Point, unsigned int>> L(num_points); // 创建向量 L，每个元素是Point和对应的索引的二元组
    for (size_t i = 0; i < num_points; i++) {
        const auto p = Point(
            points[i].x,
            points[i].y,
            points[i].z);
        L[i] = std::make_pair(p, i);
    }


    // Fix locality
    // xxchange 1021
//     const unsigned int desired_cluster_size = 128;
//     int max_depth = std::ceil(std::log2(L.size() / desired_cluster_size));
//     std::shared_ptr<OctreeNode> octree = build_octree(L,
//                                                       std::numeric_limits<float>.min(),
//                                                       std::numeric_limits<float>.max(),
//                                                       std::numeric_limits<float>.min(),
//                                                       std::numeric_limits<float>.max(),
//                                                       std::numeric_limits<float>.min(),
//                                                       std::numeric_limits<float>.max(),
//                                                       0, max_depth, desired_cluster_size);

//    Triangulation T; // 空的 Delaunay 三角剖分
//    for (const auto& point : L) {
//        // 找到当前点附近的点
//        std::vector<Point> nearby_points = find_nearby_points(point.first, octree, search_radius);
//        T.insert(nearby_points.begin(), nearby_points.end());// 在局部点集中执行 Delaunay 插入
//    }

    // xxchange 1021
    Triangulation TT(L.begin(), L.end()); // todo cgal会对L中的点构造Delaunay剖分 看看是38/48行慢 保存时间戳看看
    T = TT;
    //T.insert(L.begin(), L.end()); // xxchange 1021

    if (!T.is_valid()) {
        throw Exception("Triangulation failed");
    }

     // Create a map from idx to Vertex_handle.
     vertex_map.resize(num_points);
     for (auto v = T.finite_vertices_begin(); v != T.finite_vertices_end(); ++v) { // ;++v替换v = std::next(v)
         unsigned int idx = v->info();  // 顶点索引
         vertex_map[idx] = v;           // vertex_handle存相应的idx
     }

//    const char* filename = "vertexmap0.txt";
//    std::remove(filename);
//    std::ofstream outfile(filename);
//    if (outfile.is_open()) {
//         // Create a map from idx to Vertex_handle.
//         vertex_map.resize(num_points);
//         for (auto v = T.finite_vertices_begin(); v != T.finite_vertices_end(); ++v) { // ;++v替换v = std::next(v)
//             unsigned int idx = v->info();  // 顶点索引
//             vertex_map[idx] = v;           // vertex_handle存相应的idx
//             outfile << idx << std::endl;
//         }
//
//        printf("%zu\n", vertex_map.size());
//        for (size_t i = 0; i < vertex_map.size(); ++i) {
//            Vertex_handle& v = vertex_map[i];
//            printf("xxxx\n");
//            //outfile << v->info() << " ";
//            try{
//                outfile <<v->info()<<"";
//            } catch (...) {
//                 std::cout<<i<<std::endl;
//                 //std::cout<<v<<endl;
//            }
//            printf("xxxx1\n");
//        }
//        outfile.close();
//        std::cout << "Data written to file successfully!" << std::endl;
//    } else {
//        std::cerr << "Failed to open file." << std::endl;
//    }


//    std::ofstream file("point_v0.txt");
//    std::ofstream outfile("point_v1.txt");
//    for (size_t i = 0; i < num_points; ++i) {
//        file << "points[" << i << "] -> " << points[i].x << ", " << points[i].y << ", " << points[i].z << "\n";
//    }
//    for (auto v = T.finite_vertices_begin(); v != T.finite_vertices_end(); ++v) { // ;++v替换v = std::next(v)
//             unsigned int idx = v->info();  // 顶点索引
//             vertex_map[idx] = v;           // vertex_handle存相应的idx
//             outfile << "vertices[" << idx << "] -> " << v->point().x() << ", " << v->point().y() << ", " << v->point().z() << "\n";
//         }
//    file.close();
//    outfile.close();

    // Export
    std::vector<uint4> cells(T.number_of_finite_cells()); //cells用来存储生成的四面体单元 每个单元有4个顶点索引
    unsigned int* cells_uint = reinterpret_cast<unsigned int*>(cells.data());

    size_t i = 0;
    for (auto cell : T.finite_cell_handles()) { // 遍历三角剖分中的所有有限单元
        for (int j = 0; j < 4; ++j) { // 将每个单元的 4 个顶点的索引存储到 cells 中
            cells_uint[i * 4 + j] = cell->vertex(j)->info();
        }
        i++;
    }
    return cells; // 四元组的向量，表示生成的四面体网格的顶点索引
}

/* 几种batch move思路
1. 批量删除+新增：错误
2. 批量删除，插入时使用旧的 vertex_handle 来保持它们的序号不变
3. tbb并行+cgal的单点move
4. 更新顶点坐标+cgal的三角剖分refine:delaunay没有refine
*/
//std::vector<uint4> batch_move(Triangulation& T,
//                const std::vector<unsigned int>& vertices_to_move,  // 需要移动的顶点索引
//                const std::vector<Point>& new_points,               // 新的位置
//                std::vector<Vertex_handle>& vertex_map)             // 顶点映射，保存 vertex_handle
//{
//    if (vertices_to_move.size() != new_points.size()) {
//        throw std::runtime_error("The number of vertices to move must match the number of new points.");
//    }
//
//    // 打印顶点索引和新位置
//    for (size_t i = 0; i < 3; ++i) { //vertices_to_move.size()
//        unsigned int idx = vertices_to_move[i];
//        const Point& point = new_points[i];
//        printf("Moving vertex %u to new position %lf %lf %lf\n", idx, point.x(), point.y(), point.z());
//        // 进一步输出到文件或日志
//    }
//
//    // Step 1: Collect the Vertex_handles that correspond to the vertices to move
//    std::vector <Vertex_handle> handles_to_remove;
//    for (unsigned int idx: vertices_to_move) {
//        handles_to_remove.push_back(vertex_map[idx]);
//    }
//
//    printf("222\n");
//    // Step 2: Batch remove the vertices
//    T.remove(handles_to_remove.begin(), handles_to_remove.end());
//    printf("333\n");
//    // Step 3: Prepare pairs of new points and their indices
//    std::vector <std::pair<Point, unsigned int>> points_with_info;
//    for (size_t i = 0; i < new_points.size(); ++i) {
//        points_with_info.emplace_back(new_points[i], vertices_to_move[i]);
//    }
//    printf("444\n");
//    // Step 4: Batch insert the new points with their original indices
//    T.insert(points_with_info.begin(), points_with_info.end());
//    printf("555\n");
//
//    // Step 5: Update the vertex_map with the new Vertex_handles
//    for (auto v = T.finite_vertices_begin(); v != T.finite_vertices_end(); ++v) {
//        unsigned int idx = v->info();
//        vertex_map[idx] = v;  // 更新 vertex_map，原来的Vertex_handle 对应新的位置
//    }
//
//    // Export
//    std::vector <uint4> cells(T.number_of_finite_cells()); //cells用来存储生成的四面体单元 每个单元有4个顶点索引
//    unsigned int *cells_uint = reinterpret_cast<unsigned int *>(cells.data());
//
//    size_t i = 0;
//    for (auto cell: T.finite_cell_handles()) { // 遍历三角剖分中的所有有限单元
//        for (int j = 0; j < 4; ++j) { // 将每个单元的 4 个顶点的索引存储到 cells 中
//            cells_uint[i * 4 + j] = cell->vertex(j)->info();
//        }
//        i++;
//    }
//    return cells;
//}

std::vector<uint4> batch_move(Triangulation& T,
                               const std::vector<unsigned int>& vertices_to_move,  // 需要移动的顶点索引
                               const std::vector<Point>& new_points,               // 新的位置
                               std::vector<Vertex_handle>& vertex_map)             // 顶点映射，保存 vertex_handle
{
    if (vertices_to_move.size() != new_points.size()) {
        throw std::runtime_error("The number of vertices to move must match the number of new points.");
    }


//    // 打开文件用于写入
//    std::ofstream output_file("batch_move_output.txt", std::ios::out);
//    if (!output_file.is_open()) {
//        throw std::runtime_error("Could not open the output file.");
//    }
//
//    // 写入文件头部（可以根据需要修改格式）
//    output_file << "Moving vertices:\n";
//    output_file << "Index, New X, New Y, New Z\n";
//
//    // 打印顶点索引和新位置，并输出到文件
//    for (size_t i = 0; i < vertices_to_move.size(); ++i) {
//        unsigned int idx = vertices_to_move[i];
//        const Point& point = new_points[i];
//        // 输出到控制台
//        //printf("Moving vertex %u to new position %lf %lf %lf\n", idx, point.x(), point.y(), point.z());
//        // 输出到文件
//        output_file << idx << ", " << point.x() << ", " << point.y() << ", " << point.z() << "\n";
//    }
//    output_file.close();
//    printf("111\n");
    // Step 1: Collect the Vertex_handles that correspond to the vertices to move
    for (size_t i = 0; i < vertices_to_move.size(); ++i) { // Step 2: Update the vertex's position instead of deleting it
        //printf("vertices to move num is %ld\n",vertices_to_move.size());
        unsigned int idx = vertices_to_move[i];
        Vertex_handle vhandle = vertex_map[idx];
//        printf("(%u %lf)",idx,new_points[i].x());
//        printf("%lf=>",vhandle->point().x());
        vhandle->set_point(new_points[i]);
//        printf(",now is %lf",vhandle->point().x());
//        printf(".\n");
    }
//    printf("222\n");
    // Step 3: Update the vertex_map with the new Vertex_handles (in case the vertex_handle has been updated)
//    for (auto v = T.finite_vertices_begin(); v != T.finite_vertices_end(); ++v) {
//        unsigned int idx = v->info();
//        vertex_map[idx] = v;  // Update vertex_map with new Vertex_handle if necessary
//    }

    // Step 4: Export the updated cells (to ensure the cell information is correct)
    std::vector<uint4> cells(T.number_of_finite_cells());
    unsigned int *cells_uint = reinterpret_cast<unsigned int *>(cells.data());

    size_t i = 0;
    for (auto cell: T.finite_cell_handles()) { // 遍历三角剖分中的所有有限单元
        for (int j = 0; j < 4; ++j) { // 将每个单元的 4 个顶点的索引存储到 cells 中
            cells_uint[i * 4 + j] = cell->vertex(j)->info();
        }
        i++;
    }

    return cells;
}


// TriangulationWrapper 的包装函数实现
std::vector<uint4> TriangulationWrapper::triangulate_wrapper(size_t num_points, float3* points) {
    return triangulate(num_points, points, vertex_map, T);
}

std::vector<uint4> TriangulationWrapper::batch_move_wrapper(const std::vector<unsigned int>& vertices_to_move, const std::vector<Point>& new_points) {
    printf("TriangulationWrapper::batch_move_wrapper\n");
    return batch_move(T, vertices_to_move, new_points, vertex_map);
}