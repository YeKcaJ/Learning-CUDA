#include <vector>
#include <cstdint>
#include <cuda_fp16.h>

#include "../tester/utils.h"


// 显存池
namespace
{
constexpr size_t kAlign = 256;  // cudaMalloc 基址对齐粒度，兼容任意向量化访存

// 尺寸计算一律经这两个函数，避免 size_t 回绕后申请到一小块显存
inline bool mulOverflow(size_t a, size_t b, size_t& out)
{
  if (a != 0 && b > SIZE_MAX / a) return false;
  out = a * b;
  return true;
}

inline bool alignUp(size_t bytes, size_t& out)
{
  if (bytes > SIZE_MAX - (kAlign - 1)) return false;
  out = (bytes + kAlign - 1) & ~(kAlign - 1);
  return true;
}

class DeviceArena
{
  public:
  // 绑定所属设备与流。use_pool 为真时走 cudaMallocAsync（CUDA 11.2+ 内存池），
  // 由调用方查询 cudaDevAttrMemoryPoolsSupported 后传入
  void bind(int device, cudaStream_t stream, bool use_pool)
  {
    device_ = device;
    stream_ = stream;
    use_pool_ = use_pool;
  }

  // 容量不足时整块重分配；旧数据无需保留，故直接释放再申请。
  // 按 1.5 倍几何增长，避免测例规模递增时反复重分配。
  bool reserve(size_t bytes)
  {
    if (bytes <= capacity_) return true;

    size_t grown = capacity_ + capacity_ / 2;
    if (grown < capacity_) return false;      // 回绕
    size_t want = grown > bytes ? grown : bytes;
    if (!alignUp(want, want)) return false;

    release();
    if (use_pool_)
    {
      // 流序分配：与同流上的 kernel、memcpy 自动保序，无需额外同步
      RUNTIME_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&base_), want, stream_));
    }
    else
    {
      RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&base_), want));
    }
    capacity_ = want;
    return true;
  }

  void release()
  {
    if (!base_) return;
    if (use_pool_) RUNTIME_CHECK(cudaFreeAsync(base_, stream_));
    else           RUNTIME_CHECK(cudaFree(base_));
    base_ = nullptr;
    capacity_ = 0;
  }

  char* at(size_t offset) const
  {
    return base_ + offset;
  }

  private:
  char* base_ = nullptr;
  size_t capacity_ = 0;
  int device_ = -1;
  cudaStream_t stream_ = nullptr;
  bool use_pool_ = false;
  // 析构时刻意不释放：静态对象销毁可能晚于 CUDA runtime 卸载，
  // 那时调用 cudaFree 会报错。进程退出由驱动统一回收。
};

}


// RMSNorm
static constexpr int kWarpSize = 32;

namespace
{
// rmsNorm 专属上下文：自带 arena 与 stream，并缓存只需查一次的设备属性。
// 每个实例化类型一份（见 rmsNormContext<T>()），与其他算子互不干扰。
struct RmsNormContext
{
  DeviceArena arena;
  cudaStream_t stream = nullptr;
  int device = -1;
  int max_smem = 0;      // 单 block 可用 shared memory 上限
  int max_grid_x = 0;    // grid.x 上限，用于校验 rows

  // 首次调用或当前设备变更时（重）初始化
  void ensureReady()
  {
    int cur = 0;
    RUNTIME_CHECK(cudaGetDevice(&cur));
    if (cur == device) return;

    // 换设备：旧 arena 与 stream 属于旧设备，须切回旧设备销毁再切回来
    if (device >= 0)
    {
      RUNTIME_CHECK(cudaSetDevice(device));
      arena.release();
      RUNTIME_CHECK(cudaStreamDestroy(stream));
      RUNTIME_CHECK(cudaSetDevice(cur));
    }

    device = cur;
    RUNTIME_CHECK(cudaStreamCreate(&stream));
    RUNTIME_CHECK(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlock, device));
    RUNTIME_CHECK(cudaDeviceGetAttribute(&max_grid_x, cudaDevAttrMaxGridDimX, device));

    // 内存池须运行时确认：CUDA 11.2+ 编译不代表当前设备支持
    int pools = 0;
    RUNTIME_CHECK(cudaDeviceGetAttribute(&pools, cudaDevAttrMemoryPoolsSupported, device));
#if CUDART_VERSION >= 11020
    arena.bind(device, stream, pools != 0);
#else
    arena.bind(device, stream, false);
#endif
  }
};

// 每个 T 一份，float / half 各自增长，互不挤占
template <typename T>
RmsNormContext& rmsNormContext()
{
  static RmsNormContext ctx;
  return ctx;
}

}

// 读取：统一转换为float
__device__ __forceinline__ float loadAsFloat(float x)
{
  return x;
} 
__device__ __forceinline__ float loadAsFloat(half x) 
{ 
  return __half2float(x); 
}

//写入：从float转换成目标类型
__device__ __forceinline__ void storeFromFloat(float v, float& dst) 
{ 
  dst = v; 
}
__device__ __forceinline__ void storeFromFloat(float v, half& dst) 
{
  dst = __float2half(v);
}

// warp 内求和
__device__ __forceinline__ float warpReduceSum(float v) 
{
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) 
  {
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

  if (wid == 0) 
  v = warpReduceSum(v);
  
  return v;
}

// RMSNorm kernel
template <typename T, int BLOCK, bool USE_SMEM>
__global__ void rmsNormKernel(const T* __restrict__ input,const T* __restrict__ weight,T* __restrict__ output, size_t hidden_dim,float eps)
{

  const size_t row = blockIdx.x;
  const T* in_row = input + row * hidden_dim;
        T* out_row = output + row * hidden_dim;

  __shared__ float s_partial[BLOCK / kWarpSize];
  __shared__ float s_scale;

  extern __shared__ float s_row[];

  // 第一趟求平方和
  float local_sum = 0.0f;
  for (size_t j = threadIdx.x; j < hidden_dim; j += BLOCK)
  {
    const float x = loadAsFloat(in_row[j]);// 缓存已转好的 fp32 值
    
    if (USE_SMEM) s_row[j] = x;
    local_sum += x * x;
  }

  const float row_sum = blockReduceSum<BLOCK>(local_sum, s_partial);//块内规约

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
  size_t total = 0;
  if (!mulOverflow(rows, hidden_dim, total)) return;
  if (total == 0) return;  // 0 不能作为 grid 维度

  if (h_input.size() < total || h_weight.size() < hidden_dim) return;
  if (h_output.size() < total) h_output.resize(total);

  RmsNormContext& ctx = rmsNormContext<T>();
  ctx.ensureReady();

  if (rows > static_cast<size_t>(ctx.max_grid_x)) return;  // 一行一 block，rows 即 grid.x

  // 三段布局在同一块显存里；每步都查回绕
  size_t bytes_total = 0, bytes_weight = 0;
  if (!mulOverflow(total, sizeof(T), bytes_total)) return;
  if (!mulOverflow(hidden_dim, sizeof(T), bytes_weight)) return;

  size_t seg_total = 0, seg_weight = 0;
  if (!alignUp(bytes_total, seg_total)) return;
  if (!alignUp(bytes_weight, seg_weight)) return;

  const size_t off_input = 0;
  const size_t off_weight = seg_total;
  if (off_weight > SIZE_MAX - seg_weight) return;
  const size_t off_output = off_weight + seg_weight;
  if (off_output > SIZE_MAX - seg_total) return;
  if (!ctx.arena.reserve(off_output + seg_total)) return;

  T* d_input = reinterpret_cast<T*>(ctx.arena.at(off_input));
  T* d_weight = reinterpret_cast<T*>(ctx.arena.at(off_weight));
  T* d_output = reinterpret_cast<T*>(ctx.arena.at(off_output));

  RUNTIME_CHECK(cudaMemcpyAsync(d_input, h_input.data(), bytes_total,cudaMemcpyHostToDevice, ctx.stream));
  RUNTIME_CHECK(cudaMemcpyAsync(d_weight, h_weight.data(), bytes_weight,cudaMemcpyHostToDevice, ctx.stream));

  const size_t smem_bytes = hidden_dim * sizeof(float);
  const bool use_smem = smem_bytes + 1024 <= static_cast<size_t>(ctx.max_smem);

  // block 大小按 hidden_dim 分档
  const unsigned int grid = static_cast<unsigned int>(rows);
  if (hidden_dim <= 128)
  {
    rmsNormKernel<T, 128, true><<<grid, 128, smem_bytes, ctx.stream>>>(d_input, d_weight, d_output,hidden_dim, eps);
  }
  else if (hidden_dim <= 1024)
  {
    rmsNormKernel<T, 256, true><<<grid, 256, smem_bytes, ctx.stream>>>(d_input, d_weight, d_output,hidden_dim, eps);
  }
  else if (use_smem)
  {
    rmsNormKernel<T, 512, true><<<grid, 512, smem_bytes, ctx.stream>>>(d_input, d_weight, d_output,hidden_dim, eps);
  }
  else
  {
    rmsNormKernel<T, 512, false><<<grid, 512, 0, ctx.stream>>>(d_input, d_weight, d_output,hidden_dim, eps);
  }

  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaMemcpyAsync(h_output.data(), d_output, bytes_total,cudaMemcpyDeviceToHost, ctx.stream));
  // 非默认流上的异步拷贝不阻塞 host，须显式同步后 h_output 才可读
  RUNTIME_CHECK(cudaStreamSynchronize(ctx.stream));
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
