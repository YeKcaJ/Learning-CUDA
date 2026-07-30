#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

// 显存池：算子按 host 接口逐次调用，每次都 cudaMalloc/cudaFree 的开销
// 远超 kernel 本身。这里跨调用复用一整块显存，按需增长、不归还。
// 单线程使用（测试框架串行调用），未加锁。
namespace {

class DeviceArena {
 public:
  // 保证每段起始地址 256B 对齐，满足任意类型的向量化访存要求
  static size_t align(size_t bytes) { return (bytes + 255) & ~static_cast<size_t>(255); }

  // 容量不足时整块重分配；旧数据无需保留，故直接 free 再 malloc
  void reserve(size_t bytes) {
    if (bytes <= capacity_) return;
    if (base_) RUNTIME_CHECK(cudaFree(base_));
    RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&base_), bytes));
    capacity_ = bytes;
  }

  char* at(size_t offset) const { return base_ + offset; }

 private:
  char* base_ = nullptr;
  size_t capacity_ = 0;
  // 析构时刻意不 cudaFree：静态对象销毁可能晚于 CUDA runtime 卸载，
  // 此时调用会报错。进程退出由驱动统一回收。
};

DeviceArena g_arena;  // float / half 两份实例化共用同一块

}  // namespace

// *********************************************************************
// 题目一：RMSNorm
// *********************************************************************

static constexpr int kWarpSize = 32;

// 类型转换
__device__ __forceinline__ float loadAsFloat(float x) { return x; }
__device__ __forceinline__ float loadAsFloat(half x) { return __half2float(x); }
__device__ __forceinline__ void storeFromFloat(float v, float& dst) { dst = v; }
__device__ __forceinline__ void storeFromFloat(float v, half& dst) {
  dst = __float2half(v);
}

// warp 内求和
__device__ __forceinline__ float warpReduceSum(float v) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

// block 内求和
template <int BLOCK>
__device__ __forceinline__ float blockReduceSum(float v, float* smem) 
{
  constexpr int kNumWarps = BLOCK / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int wid = threadIdx.x / kWarpSize;

  v = warpReduceSum(v);
  if (lane == 0) smem[wid] = v;
  __syncthreads();

  // 部分和至多 32 个，一个 warp 收完；越界 lane 补 0
  v = (threadIdx.x < kNumWarps) ? smem[threadIdx.x] : 0.0f;
  if (wid == 0) v = warpReduceSum(v);
  return v;
}

// RMSNorm kernel
// USE_SMEM=true 时第一趟顺手把整行缓存进 shared memory，第二趟不再访问 global；
// 整行装不下 shared memory 时置 false 回退为重读
template <typename T, int BLOCK, bool USE_SMEM>
__global__ void rmsNormKernel(const T* __restrict__ input,const T* __restrict__ weight,T* __restrict__ output, size_t hidden_dim,float eps)
{

  const size_t row = blockIdx.x;
  const T* in_row = input + row * hidden_dim;
  T* out_row = output + row * hidden_dim;

  __shared__ float s_partial[BLOCK / kWarpSize];
  __shared__ float s_scale;
  // 动态分配，长度 hidden_dim；排在静态 smem 之后，不会与上面两个重叠
  extern __shared__ float s_row[];

  // 第一趟求平方和，步长循环天然覆盖 hidden_dim 非 BLOCK 整数倍的情况
  float local_sum = 0.0f;
  for (size_t j = threadIdx.x; j < hidden_dim; j += BLOCK)
  {
    const float x = loadAsFloat(in_row[j]);
    // 缓存已转好的 fp32 值，第二趟省一次 global 读 + 一次类型转换
    if (USE_SMEM) s_row[j] = x;
    local_sum += x * x;
  }

  const float row_sum = blockReduceSum<BLOCK>(local_sum, s_partial);

  // 归约结果只在 thread 0 手里，须经 smem 广播给整个 block
  if (threadIdx.x == 0) 
  {
    const float mean_square = row_sum / static_cast<float>(hidden_dim);
    s_scale = rsqrtf(mean_square + eps);


  const float scale = s_scale;

  // 第二趟缩放写出。weight 与行无关、天然驻留 L2，不必缓存
  for (size_t j = threadIdx.x; j < hidden_dim; j += BLOCK)
  {
    const float x = USE_SMEM ? s_row[j] : loadAsFloat(in_row[j]);
    storeFromFloat(x * scale * loadAsFloat(weight[j]), out_row[j]);
  }
}

/**
 * @brief Computes RMSNorm over the last dimension of a 2D tensor.
 *
 * The input is a row-major matrix with shape [rows, hidden_dim]. For each row
 * i and column j:
 *
 *   output[i, j] = input[i, j] * rsqrt(mean(input[i, :]^2) + eps) * weight[j]
 *
 * The output vector is preallocated with rows * hidden_dim elements.
 *
 * @tparam T Data type of input, weight, and output tensors.
 * @param[in] h_input Flattened input matrix of shape [rows, hidden_dim].
 * @param[in] h_weight Per-column scale vector of shape [hidden_dim].
 * @param[out] h_output Flattened output matrix of shape [rows, hidden_dim].
 * @param[in] rows Number of rows/tokens.
 * @param[in] hidden_dim Size of the normalized dimension.
 * @param[in] eps Numerical stability epsilon.
 */

template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
              std::vector<T>& h_output, size_t rows, size_t hidden_dim,
              float eps) 
{
  const size_t total = rows * hidden_dim;
  if (total == 0) return;  // 0 不能作为 grid 维度
  if (h_input.size() < total || h_weight.size() < hidden_dim) return;
  if (h_output.size() < total) h_output.resize(total);

  // 三段布局在同一块显存里：input | weight | output
  const size_t off_input = 0;
  const size_t off_weight = DeviceArena::align(total * sizeof(T));
  const size_t off_output = off_weight + DeviceArena::align(hidden_dim * sizeof(T));
  g_arena.reserve(off_output + DeviceArena::align(total * sizeof(T)));

  T* d_input = reinterpret_cast<T*>(g_arena.at(off_input));
  T* d_weight = reinterpret_cast<T*>(g_arena.at(off_weight));
  T* d_output = reinterpret_cast<T*>(g_arena.at(off_output));

  RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), total * sizeof(T),cudaMemcpyHostToDevice));

  RUNTIME_CHECK(cudaMemcpy(d_weight, h_weight.data(), hidden_dim * sizeof(T),cudaMemcpyHostToDevice));

  // 整行缓存需 hidden_dim 个 float；留 1KB 给静态 smem 与编译器余量
  int max_smem = 0;
  int device = 0;
  RUNTIME_CHECK(cudaGetDevice(&device));
  RUNTIME_CHECK(cudaDeviceGetAttribute(
      &max_smem, cudaDevAttrMaxSharedMemoryPerBlock, device));
  const size_t smem_bytes = hidden_dim * sizeof(float);
  const bool use_smem = smem_bytes + 1024 <= static_cast<size_t>(max_smem);

  // block 大小按 hidden_dim 分档，模板参数保证归约循环编译期展开
  const unsigned int grid = static_cast<unsigned int>(rows);
  if (hidden_dim <= 128)
  {
    // 该档 smem 至多 512B，恒定装得下，不必生成回退版本
    rmsNormKernel<T, 128, true><<<grid, 128, smem_bytes>>>(d_input, d_weight, d_output,hidden_dim, eps);
  }
  else if (hidden_dim <= 1024)
  {
    rmsNormKernel<T, 256, true><<<grid, 256, smem_bytes>>>(d_input, d_weight, d_output,hidden_dim, eps);
  }
  else if (use_smem)
  {
    rmsNormKernel<T, 512, true><<<grid, 512, smem_bytes>>>(d_input, d_weight, d_output,hidden_dim, eps);
  }
  else
  {
    // 超大 hidden_dim：装不进 smem，退回第二趟重读 global
    rmsNormKernel<T, 512, false><<<grid, 512>>>(d_input, d_weight, d_output,hidden_dim, eps);
  }

  RUNTIME_CHECK(cudaGetLastError());

  // 默认流上 cudaMemcpy 阻塞并隐式等 kernel，无需额外同步
  RUNTIME_CHECK(cudaMemcpy(h_output.data(), d_output, total * sizeof(T),cudaMemcpyDeviceToHost));
}



















/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  // TODO: Implement the flash attention function
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template void rmsNorm<float>(const std::vector<float>&, const std::vector<float>&,
  std::vector<float>&, size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half>&, const std::vector<half>&,
  std::vector<half>&, size_t, size_t, float);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
