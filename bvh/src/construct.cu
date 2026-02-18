#include <construct.cuh>
#include <cmath>
#include <stdio.h>


__device__ __host__
inline std::uint32_t expand_bits(std::uint32_t v) noexcept
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__
inline int common_upper_bits(const uint64_t lhs, const uint64_t rhs) noexcept
{
    return ::__clzll(lhs ^ rhs);
}

__device__ __host__
inline std::uint32_t morton_code(float3 xyz, float resolution = 1024.0f) noexcept // 不会抛出异常
{ // xyz范围是0-1，默认分辨率为 1024.0，表示 xyz 的每个坐标都会被放大 1024 倍
    xyz.x = ::fminf(::fmaxf(xyz.x * resolution, 0.0f), resolution - 1.0f); // fmaxf 和 fminf 将每个坐标值限制在 [0, resolution - 1]
    xyz.y = ::fminf(::fmaxf(xyz.y * resolution, 0.0f), resolution - 1.0f);
    xyz.z = ::fminf(::fmaxf(xyz.z * resolution, 0.0f), resolution - 1.0f);
    const std::uint32_t xx = expand_bits(static_cast<std::uint32_t>(xyz.x)); // expand_bits位扩展：通过在每个位之间插入两个零，将10位的整数扩展为30位的摩顿码格式。将三维坐标的每一位“交织”在一起。
    const std::uint32_t yy = expand_bits(static_cast<std::uint32_t>(xyz.y)); // 仍是十进制
    const std::uint32_t zz = expand_bits(static_cast<std::uint32_t>(xyz.z));
    return xx * 4 + yy * 2 + zz; // xx * 4 将 xx 左移两位。yy * 2 将 yy 左移一位。zz 保持不变。
}

struct morton_code_calculator // 包围盒 whole 用于归一化每个物体的位置，将它们映射到一个标准的坐标范围内
{
    morton_code_calculator(aabb_type w): whole(w) {} // 使用构造函数参数 w 初始化成员变量 whole

    __device__ __host__ // 可以在 GPU（__device__）和 CPU（__host__）上调用
    inline uint32_t operator()(const aabb_type& box) noexcept // 质心归一化
    // inline 提示编译器将函数“内联化” 将函数插入调用的地方 而不是跳转到另一个内存位置执行
    // operator()：重载的函数调用运算符，使得morton_code_calculator可以像函数一样被调用
    {
        auto p = centroid(box); // 质心
        p.x -= whole.lower.x;
        p.y -= whole.lower.y;
        p.z -= whole.lower.z;
        p.x /= (whole.upper.x - whole.lower.x);
        p.y /= (whole.upper.y - whole.lower.y);
        p.z /= (whole.upper.z - whole.lower.z);
        return morton_code(p); // p是0-1
    }

    aabb_type whole; // 成员变量
};

__device__
inline uint2 determine_range(const uint64_t* node_code,
                             const uint32_t num_leaves, uint32_t idx)
{
    if(idx == 0)
    {
        return make_uint2(0, num_leaves-1);
    }

    // determine direction of the range
    uint64_t self_code = node_code[idx];
    const int L_delta = common_upper_bits(self_code, node_code[idx-1]);
    const int R_delta = common_upper_bits(self_code, node_code[idx+1]);
    const int d = (R_delta > L_delta) ? 1 : -1;

    // Compute upper bound for the length of the range

    const int delta_min = thrust::min(L_delta, R_delta);
    int l_max = 2;
    int delta = -1;
    int i_tmp = idx + d * l_max;
    if(0 <= i_tmp && i_tmp < num_leaves)
    {
        delta = common_upper_bits(self_code, node_code[i_tmp]);
    }
    while(delta > delta_min)
    {
        l_max <<= 1;
        i_tmp = idx + d * l_max;
        delta = -1;
        if(0 <= i_tmp && i_tmp < num_leaves)
        {
            delta = common_upper_bits(self_code, node_code[i_tmp]);
        }
    }

    // Find the other end by binary search
    int l = 0;
    int t = l_max >> 1;
    while(t > 0)
    {
        i_tmp = idx + (l + t) * d;
        delta = -1;
        if(0 <= i_tmp && i_tmp < num_leaves)
        {
            delta = common_upper_bits(self_code, node_code[i_tmp]);
        }
        if(delta > delta_min)
        {
            l += t;
        }
        t >>= 1;
    }
    uint32_t jdx = idx + l * d;
    if(d < 0)
    {
        thrust::swap(idx, jdx); // make it sure that idx < jdx
    }
    return make_uint2(idx, jdx);
}

__device__
inline int32_t find_split(uint64_t const* node_code, const int32_t num_leaves,
                               const int32_t first, const int32_t last) noexcept
{
    const uint64_t first_code = node_code[first];
    const uint64_t last_code  = node_code[last];
    if (first_code == last_code)
    {
        return (first + last) >> 1;
    }
    const int32_t delta_node = common_upper_bits(first_code, last_code);

    // binary search...
    int32_t split  = first;
    int32_t stride = last - first;
    do
    {
        stride = (stride + 1) >> 1;
        const int middle = split + stride;
        if (middle < last)
        {
            const int32_t delta = common_upper_bits(first_code, node_code[middle]);
            if (delta > delta_node)
            {
                split = middle;
            }
        }
    }
    while(stride > 1);

    return split;
}

void construct_bvh(
        int32_t num_objects,
        const float* means3D,
        const float* scales,
        const float* rotations,
        int32_t* nodes,
        float* aabbs,
        uint64_t* morton){
    const int32_t num_internal_nodes = num_objects - 1;
    const int32_t num_nodes          = num_objects * 2 - 1;
    auto* aabbs_internal = reinterpret_cast<aabb_type*>(aabbs);

    // 1. 计算所有物体AABB
    aabb_type default_aabb{}; // lower upper
    default_aabb.upper.x = -100000.f; default_aabb.lower.x = 100000.f;
    default_aabb.upper.y = -100000.f; default_aabb.lower.y = 100000.f;
    default_aabb.upper.z = -100000.f; default_aabb.lower.z = 100000.f;
    thrust::device_ptr<aabb_type> aabbs_ptr = thrust::device_pointer_cast(aabbs_internal);
    const auto aabb_whole = thrust::reduce( // 二叉树中所有叶节点的 AABB
            aabbs_ptr + num_internal_nodes, aabbs_ptr + num_nodes, default_aabb,
            [] __device__ (const aabb_type& lhs, const aabb_type& rhs) {
                return merge(lhs, rhs);
            });

    // 2. 将三维空间中的物体映射到一维坐标，以便于后续的排序
    thrust::device_vector<uint32_t> morton32(num_objects); // 创建一个大小为 num_objects 的设备向量 morton32
    thrust::transform(aabbs_ptr + num_internal_nodes, aabbs_ptr + num_nodes, // 输入范围（所有叶节点）
                      morton32.begin(), // 输出起始位置
                      morton_code_calculator(aabb_whole)); // 一个仿函数，调用普通函数一样调用对象本身
    // transform将输入范围内的每一个元素传入仿函数，并将结果存储到输出范围内

    // 3. 排序
    thrust::device_vector<uint32_t> indices(num_objects); // 索引列表，追踪物体的位置
    // copy要求传连个迭代器 [begin, end)
    thrust::copy(thrust::make_counting_iterator<uint32_t>(0), // 递增序列迭代器是惰性计算，只在需要时才生成数值。避免了提前计算和存储所有值
                 thrust::make_counting_iterator<uint32_t>(num_objects),
                 indices.begin()); // 实际的数据容器来存储具体的数据


    // 通过“键”进行排序，同时重新排列与之相关联的“值”
    thrust::stable_sort_by_key(morton32.begin(), morton32.end(), // 排序的“键”：每个物体的摩顿码
                               thrust::make_zip_iterator(
                                       thrust::make_tuple(aabbs_ptr + num_internal_nodes,
                                                          indices.begin())));

    // 4. 转换 32 位摩顿码为 64 位摩顿码
    thrust::device_ptr<uint64_t> morton_ptr = thrust::device_pointer_cast(morton);
    thrust::transform(morton32.begin(), morton32.end(), indices.begin(),
                      morton_ptr,
                      [] __device__ (const uint32_t m, const uint32_t idx)
                      {
                          uint64_t m64 = m;
                          m64 <<= 31;
                          m64 |= idx;
                          return m64;
                      });
    thrust::device_ptr<int32_t> nodes_ptr = thrust::device_pointer_cast(nodes);

    // 5. 设置叶节点
    uint32_t* indices_ptr = indices.data().get();
    thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<int32_t>(0),
                     thrust::make_counting_iterator<int32_t>(num_objects),
                     [num_objects, indices_ptr, nodes] __device__ (const int32_t idx){
                        nodes[(num_objects - 1 + idx) * 5 + 3] = indices_ptr[idx];
                     });

    // 6. 构建内部节点
    thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<int32_t>(0),
                     thrust::make_counting_iterator<int32_t>(num_objects - 1),
                     [num_objects, nodes, morton] __device__ (const int32_t idx)
                     {
                        int32_t* node = nodes + idx * 5;
                         node[3] = -1; //  internal nodes

                         const uint2 ij  = determine_range(morton, num_objects, idx);
                         const int32_t gamma = find_split(morton, num_objects, ij.x, ij.y);

                         node[1] = gamma;
                         node[2] = gamma + 1;
                         if(thrust::min(ij.x, ij.y) == gamma)
                         {
                             node[1] += num_objects - 1;
                         }
                         if(thrust::max(ij.x, ij.y) == gamma + 1)
                         {
                             node[2] += num_objects - 1;
                         }
                         nodes[node[1]*5]  = idx;
                         nodes[node[2]*5] = idx;
                         return;
                     });

    thrust::device_vector<int> flag_container(num_internal_nodes, 0);
    const auto flags = flag_container.data().get();

    // 7. 递归合并 AABB
    thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<int32_t>(num_internal_nodes),
                     thrust::make_counting_iterator<int32_t>(num_nodes),
                     [nodes, aabbs_internal, flags] __device__ (int32_t idx)
                     {
                         int32_t num = 1;
                         int32_t parent = *(nodes + idx * 5);
                         while(parent != -1) // means idx == 0
                         {
                             atomicAdd(nodes + parent * 5 + 4, num);
                             const int old = atomicCAS(flags + parent, 0, 1);
                             if(old == 0)
                             {
                                 // this is the first thread entered here.
                                 // wait the other thread from the other child node.
                                 return;
                             }
                             assert(old == 1);
                             // here, the flag has already been 1. it means that this
                             // thread is the 2nd thread. merge AABB of both childlen.
                             int32_t* parent_node = nodes + parent * 5;

                             const auto lidx = *(parent_node + 1);
                             const auto ridx = *(parent_node + 2);
                             const auto lbox = aabbs_internal[lidx];
                             const auto rbox = aabbs_internal[ridx];
                             aabbs_internal[parent] = merge(lbox, rbox);
                             num = nodes[parent*5+4];

                             // look the next parent...
                             parent = *parent_node;
                         }
                         return;
                     });
}