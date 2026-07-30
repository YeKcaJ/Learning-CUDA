#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

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
template <typename T, int BLOCK>
__global__ void rmsNormKernel(const T* __restrict__ input,const T* __restrict__ weight,T* __restrict__ output, size_t hidden_dim,float eps)
{

  const size_t row = blockIdx.x;
  const T* in_row = input + row * hidden_dim;
  T* out_row = output + row * hidden_dim;

  __shared__ float s_partial[BLOCK / kWarpSize];
  __shared__ float s_scale;

  // 第一趟求平方和，步长循环天然覆盖 hidden_dim 非 BLOCK 整数倍的情况
  float local_sum = 0.0f;
  for (size_t j = threadIdx.x; j < hidden_dim; j += BLOCK) 
  {
    const float x = loadAsFloat(in_row[j]);
    local_sum += x * x;
  }

  const float row_sum = blockReduceSum<BLOCK>(local_sum, s_partial);

  // 归约结果只在 thread 0 手里，须经 smem 广播给整个 block
  if (threadIdx.x == 0) 
  {
    const float mean_square = row_sum / static_cast<float>(hidden_dim);
    s_scale = rsqrtf(mean_square + eps);
  }
  __syncthreads();

  const float scale = s_scale;

  // 第二趟缩放写出
  for (size_t j = threadIdx.x; j < hidden_dim; j += BLOCK) 
  {
    const float v = loadAsFloat(in_row[j]) * scale * loadAsFloat(weight[j]);
    storeFromFloat(v, out_row[j]);
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

  T *d_input = nullptr, *d_weight = nullptr, *d_output = nullptr;
  RUNTIME_CHECK(cudaMalloc(&d_input, total * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_weight, hidden_dim * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_output, total * sizeof(T)));

  RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), total * sizeof(T),cudaMemcpyHostToDevice));

  RUNTIME_CHECK(cudaMemcpy(d_weight, h_weight.data(), hidden_dim * sizeof(T),cudaMemcpyHostToDevice));

  // block 大小按 hidden_dim 分档，模板参数保证归约循环编译期展开
  const unsigned int grid = static_cast<unsigned int>(rows);
  if (hidden_dim <= 128) 
  {
    rmsNormKernel<T, 128><<<grid, 128>>>(d_input, d_weight, d_output,hidden_dim, eps);
  } 
  else if (hidden_dim <= 1024) 
  {
    rmsNormKernel<T, 256><<<grid, 256>>>(d_input, d_weight, d_output,hidden_dim, eps);
  } 
  else
  {
    rmsNormKernel<T, 512><<<grid, 512>>>(d_input, d_weight, d_output,hidden_dim, eps);
  }

  RUNTIME_CHECK(cudaGetLastError());

  // 默认流上 cudaMemcpy 阻塞并隐式等 kernel，无需额外同步
  RUNTIME_CHECK(cudaMemcpy(h_output.data(), d_output, total * sizeof(T),cudaMemcpyDeviceToHost));

  RUNTIME_CHECK(cudaFree(d_input));
  RUNTIME_CHECK(cudaFree(d_weight));
  RUNTIME_CHECK(cudaFree(d_output));
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
