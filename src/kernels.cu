#include <vector>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>
#include <cuda_fp16.h>
#include "../tester/utils.h"


// *****公共工具***** //
namespace
{
constexpr size_t kAlign = 256;  //cudaMalloc 基址对齐粒度，兼容任意向量化访存

//尺寸计算一律经这两个函数，避免 size_t 回绕后申请到一小块显存
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
    //绑定所属设备与流。use_pool 为真时走 cudaMallocAsync（CUDA 11.2+ 内存池），
    //由调用方查询 cudaDevAttrMemoryPoolsSupported 后传入
    void bind(int device, cudaStream_t stream, bool use_pool)
    {
        device_ = device;
        stream_ = stream;
        use_pool_ = use_pool;
    }

    //容量不足时整块重分配；旧数据无需保留，故直接释放再申请。
    //按 1.5 倍几何增长，避免测例规模递增时反复重分配。
    bool reserve(size_t bytes)
    {
        if (bytes <= capacity_) return true;

        size_t grown = capacity_ + capacity_ / 2;
        if (grown < capacity_) return false; //回绕
        size_t want = grown > bytes ? grown : bytes;
        if (!alignUp(want, want)) return false;

        release();
        if (use_pool_)
          {//流序分配：与同流上的 kernel、memcpy 自动保序，无需额外同步
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
    //析构时刻意不释放：静态对象销毁可能晚于 CUDA runtime 卸载，
    //那时调用 cudaFree 会报错。进程退出由驱动统一回收。
};

}


static constexpr int kWarpSize = 32;
namespace
{
//算子上下文：自带 arena 与 stream，并缓存只需查一次的设备属性。
//每个算子、每个实例化类型各一份（见下面两个 xxxContext<T>()），互不挤占显存。
struct OpContext
{
    DeviceArena arena;
    cudaStream_t stream = nullptr;
    int device = -1;
    int max_smem = 0;   //单 block 可用 shared memory 上限
    int max_grid_x = 0; //grid.x 上限，用于校验 rows

    //首次调用或当前设备变更时（重）初始化
    void ensureReady()
    {
        int cur = 0;
        RUNTIME_CHECK(cudaGetDevice(&cur));
        if (cur == device) return;

        if (device >= 0)
          {//换设备：旧 arena 与 stream 属于旧设备，须切回旧设备销毁再切回来
            RUNTIME_CHECK(cudaSetDevice(device));
            arena.release();
            RUNTIME_CHECK(cudaStreamDestroy(stream));
            RUNTIME_CHECK(cudaSetDevice(cur));
          }

        device = cur;
        RUNTIME_CHECK(cudaStreamCreate(&stream));
        RUNTIME_CHECK(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlock, device));
        RUNTIME_CHECK(cudaDeviceGetAttribute(&max_grid_x, cudaDevAttrMaxGridDimX, device));

        //内存池须运行时确认：CUDA 11.2+ 编译不代表当前设备支持
        int pools = 0;
        RUNTIME_CHECK(cudaDeviceGetAttribute(&pools, cudaDevAttrMemoryPoolsSupported, device));
        #if CUDART_VERSION >= 11020
            arena.bind(device, stream, pools != 0);
        #else
            arena.bind(device, stream, false);
        #endif
    }
};

    //每个 T 一份，float / half 各自增长，互不挤占
    template <typename T>
    OpContext& rmsNormContext()
    {
      static OpContext ctx;
      return ctx;
    }

    template <typename T>
    OpContext& flashAttentionContext()
    {
      static OpContext ctx;
      return ctx;
    }

}

//读取：统一转换为float
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

//warp 内求和
__device__ __forceinline__ float warpReduceSum(float v)
{
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
      {
        v += __shfl_down_sync(0xffffffffu, v, offset);
      }
    return v;
}

//warp 内求最大值，蝶形交换版：结束后每个 lane 都持有完整结果，省掉一次广播。
//flash attention 求本行 score 最大值时要全 lane 拿到 m。
__device__ __forceinline__ float warpAllReduceMax(float v)
{
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
      {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
      }
    return v;
}

//block 内求和
template <int BLOCK>
__device__ __forceinline__ float blockReduceSum(float v, float* smem)
{
    constexpr int kNumWarps = BLOCK / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int wid = threadIdx.x / kWarpSize;

    v = warpReduceSum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();

    //部分和至多 32 个，一个 warp 收完；越界 lane 补 0
    v = (threadIdx.x < kNumWarps) ? smem[threadIdx.x] : 0.0f;
    if (wid == 0)
      {
        v = warpReduceSum(v);
      }
    return v;
}

// *****Part 1: RMSNorm***** //
//codeOnGPU
template <typename T, int BLOCK, bool USE_SMEM>
__global__ void rmsNormKernel(const T* __restrict__ input,
                              const T* __restrict__ weight,
                              T* __restrict__ output,
                              size_t hidden_dim,
                              float eps) {
//1.定位本 block 负责的行
    const size_t row = blockIdx.x;
    const T* in_row = input + row * hidden_dim;
    T* out_row = output + row * hidden_dim;

    __shared__ float s_partial[BLOCK / kWarpSize];
    __shared__ float s_scale;
    extern __shared__ float s_row[]; //长度 hidden_dim，缓存 fp32 值

//2.第一趟：读取、转换、缓存、累加平方和
    float local_sum = 0.0f;
    for (size_t j = threadIdx.x; j < hidden_dim; j += BLOCK)
    {
      const float x = loadAsFloat(in_row[j]);
      if (USE_SMEM) s_row[j] = x; //缓存已转好的 fp32 值
      local_sum += x * x;
    }

//3.块内规约求整行平方和
    const float row_sum = blockReduceSum<BLOCK>(local_sum, s_partial);

//4.thread 0 算缩放因子，经 smem 广播给全 block
    if (threadIdx.x == 0)
    {
      const float mean_square = row_sum / static_cast<float>(hidden_dim);
      s_scale = rsqrtf(mean_square + eps);
    }
    __syncthreads();
    const float scale = s_scale;

//5.第二趟：缩放并写出
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

//codeOnCPU
template <typename T>
void rmsNorm(const std::vector<T>& h_input,
             const std::vector<T>& h_weight,
             std::vector<T>& h_output,
             size_t rows,
             size_t hidden_dim,
             float eps) {
//1.参数检验
    size_t total = 0;
    if (!mulOverflow(rows, hidden_dim, total)) return;
    if (total == 0) return; //0 不能作为 grid 维度

    if (h_input.size() < total || h_weight.size() < hidden_dim) return;
    if (h_output.size() < total) h_output.resize(total);

//2.取上下文，确认它对应当前设备
    OpContext& ctx = rmsNormContext<T>();
    ctx.ensureReady();

    if (rows > static_cast<size_t>(ctx.max_grid_x)) return; //一行一 block，rows 即 grid.x

//3.算三段布局：input | weight | output，每步都查回绕
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

//4.拷贝数据到GPU
    RUNTIME_CHECK(cudaMemcpyAsync(d_input, h_input.data(), bytes_total,
                                  cudaMemcpyHostToDevice, ctx.stream));
    RUNTIME_CHECK(cudaMemcpyAsync(d_weight, h_weight.data(), bytes_weight,
                                  cudaMemcpyHostToDevice, ctx.stream));

//5.启动kernel，block 大小按 hidden_dim 分档
    //smem 缓存的是 fp32，故按 sizeof(float) 算；留 1KB 给静态 smem
    const size_t smem_bytes = hidden_dim * sizeof(float);
    const bool use_smem = smem_bytes + 1024 <= static_cast<size_t>(ctx.max_smem);

    const unsigned int grid = static_cast<unsigned int>(rows);
    if (hidden_dim <= 128)
    {
      rmsNormKernel<T, 128, true><<<grid, 128, smem_bytes, ctx.stream>>>
      (d_input, d_weight, d_output, hidden_dim, eps);
    }
    else if (hidden_dim <= 1024)
    {
      rmsNormKernel<T, 256, true><<<grid, 256, smem_bytes, ctx.stream>>>
      (d_input, d_weight, d_output, hidden_dim, eps);
    }
    else if (use_smem)
    {
      rmsNormKernel<T, 512, true><<<grid, 512, smem_bytes, ctx.stream>>>
      (d_input, d_weight, d_output, hidden_dim, eps);
    }
    else
    {
      //整行装不进 smem，退回第二趟重读 global
      rmsNormKernel<T, 512, false><<<grid, 512, 0, ctx.stream>>>
      (d_input, d_weight, d_output, hidden_dim, eps);
    }

    RUNTIME_CHECK(cudaGetLastError());

//6.拷贝结果回CPU
    RUNTIME_CHECK(cudaMemcpyAsync(h_output.data(), d_output, bytes_total,cudaMemcpyDeviceToHost, ctx.stream));
    //非默认流上的异步拷贝不阻塞 host，须显式同步后 h_output 才可读
    RUNTIME_CHECK(cudaStreamSynchronize(ctx.stream));
}




// *****Part 2: Flash Attention***** //
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

//codeOnGPU
//一次迭代处理的 k/v 行数。必须等于 warp 大小：kernel 里「一个 warp 管一个 q 行、
//lane 号即 K 的列号」的映射靠这个前提，行内规约才能全走 shuffle，不落 smem。
static constexpr int kFaTileN = kWarpSize;

//K/V 分块加载：BM 个 warp 分摊 BN 行，ty 跨步覆盖（BM 可能小于 BN）。
//row_limit 是整个序列的长度，不是某一行的 causal 上界：tile 由全 block 共用。
//越界行填 0，配合 p=0 使 fmaf 贡献恰好为 0。
template <typename T, int BM, int BN>
__device__ __forceinline__ void loadKTile(float* K_tile, const T* k_base,
                                          int kv_start, int row_limit,
                                          int kv_stride, int head_dim,
                                          int ty, int lane)
{
    for (int r = ty; r < BN; r += BM)
      {
        const int kv_row = kv_start + r;
        const bool ok = kv_row < row_limit;
        for (int d = lane; d < head_dim; d += BN)
          {
            K_tile[r * head_dim + d] =
                ok ? loadAsFloat(k_base[(size_t)kv_row * kv_stride + d]) : 0.0f;
          }
      }
}

template <typename T, int BM, int BN>
__device__ __forceinline__ void loadKVTile(float* K_tile, float* V_tile,
                                           const T* k_base, const T* v_base,
                                           int kv_start, int row_limit,
                                           int kv_stride, int head_dim,
                                           int ty, int lane)
{
    for (int r = ty; r < BN; r += BM)
      {
        const int kv_row = kv_start + r;
        const bool ok = kv_row < row_limit;
        for (int d = lane; d < head_dim; d += BN)
          {
            const size_t off = (size_t)kv_row * kv_stride + d;
            K_tile[r * head_dim + d] = ok ? loadAsFloat(k_base[off]) : 0.0f;
            V_tile[r * head_dim + d] = ok ? loadAsFloat(v_base[off]) : 0.0f;
          }
      }
}

//BM = 一个 block 负责的 q 行数（即 warp 数），BN = 一次迭代的 k/v 行数。
//float 与 half 走完全相同的这一条路径，只在加载/写回处做一次类型转换。
template <typename T, int BM, int BN>
__global__ void flashAttentionKernel( const T* __restrict__ d_q,
                                      const T* __restrict__ d_k,
                                      const T* __restrict__ d_v,
                                      T* __restrict__ d_o,
                                      int target_seq_len,
                                      int src_seq_len,
                                      int query_heads,
                                      int kv_heads,
                                      int head_dim,
                                      int kv_groups,
                                      bool is_causal){
    static_assert(BN == kWarpSize, "BN 必须等于 warp 大小");
//1.共享内存分区：全部按 head_dim 动态定尺
    //统一存 fp32：half 输入在加载时转换一次，后续算术无需反复转型
    extern __shared__ char smem_raw[];
    float* Q_tile = reinterpret_cast<float*>(smem_raw); //[BM][head_dim]
    float* K_tile = Q_tile + BM * head_dim;             //[BN][head_dim]
    float* V_tile = K_tile + BN * head_dim;             //[BN][head_dim]
    float* O_tile = V_tile + BN * head_dim;             //[BM][head_dim] fp32 累加器
    float* P_tile = O_tile + BM * head_dim;             //[BM][BN] 本块概率

    const int lane = threadIdx.x; //0..BN-1，兼作 K/V 的行号
    const int ty   = threadIdx.y; //0..BM-1，本 warp 负责的 q 行（块内偏移）

//2.当前 block 处理的 (batch, query_head) 与 Q 的行分块
    const int batch_id      = blockIdx.y / query_heads;
    const int query_head_id = blockIdx.y % query_heads;
    const int kv_head_id    = query_head_id / kv_groups; //GQA：多个 q head 共用一组 kv

    const int q_row    = blockIdx.x * BM + ty;    //本 warp 的全局 q 行号
    const bool q_valid = q_row < target_seq_len;  //尾块可能不满 BM

    //[batch, seq, head, dim] 布局：相邻 token 间跨度 = heads * head_dim
    const int q_stride  = query_heads * head_dim;
    const int kv_stride = kv_heads * head_dim;
    const T* q_base = d_q + (size_t)batch_id * target_seq_len * q_stride
                          + (size_t)query_head_id * head_dim;
    const T* k_base = d_k + (size_t)batch_id * src_seq_len * kv_stride
                          + (size_t)kv_head_id * head_dim;
    const T* v_base = d_v + (size_t)batch_id * src_seq_len * kv_stride
                          + (size_t)kv_head_id * head_dim;

//3.加载 Q_tile，清零 O_tile
    for (int d = lane; d < head_dim; d += BN)
      {
        Q_tile[ty * head_dim + d] =
            q_valid ? loadAsFloat(q_base[(size_t)q_row * q_stride + d]) : 0.0f;
        O_tile[ty * head_dim + d] = 0.0f;
      }

    //causal 下本行只看 k <= q_row
    const int key_end = is_causal ? min(src_seq_len, q_row + 1) : src_seq_len;
    //循环上界必须与 q_row 无关：块内各 warp 的 key_end 不同，若据此定次数，
    //循环体里的 __syncthreads() 就会分叉，K_tile 会被提前退出的 warp 覆写。
    const int num_kv_blocks = (src_seq_len + BN - 1) / BN;
    __syncthreads();

    //scale 在 device 端算：sqrt.rn + rcp.rn
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    //lane 号即 k 的块内偏移；dot 按 d 升序串行 FMA。返回未乘 scale 的原始值，
    //好让下面把「乘 scale」和「减 m」交给同一条 fmaf。
    auto dotOf = [&]() -> float {
      float dot = 0.0f;
      for (int d = 0; d < head_dim; d++)
        {
          dot = fmaf(Q_tile[ty * head_dim + d], K_tile[lane * head_dim + d], dot);
        }
      return dot;
    };

//4.第一趟：求本行 score 最大值（只读 K）
    float m_i = -INFINITY;
    for (int kb = 0; kb < num_kv_blocks; kb++)
      {
        const int kv_start = kb * BN;
        //加载上界用 src_seq_len：tile 是块内共享的，不能按本 warp 的 key_end 截断
        loadKTile<T, BM, BN>(K_tile, k_base, kv_start, src_seq_len, kv_stride,
                             head_dim, ty, lane);
        __syncthreads();
        //m 取的是已舍入的 dot*scale，这一步不能融合（后面是 max，不是加法）
        const bool valid = (kv_start + lane) < key_end;
        const float s = valid ? dotOf() * scale : -INFINITY;
        m_i = fmaxf(m_i, warpAllReduceMax(s));
        __syncthreads();
      }

//5.第二趟：求 exp 和（只读 K）
    float l_i = 0.0f;
    for (int kb = 0; kb < num_kv_blocks; kb++)
      {
        const int kv_start = kb * BN;
        loadKTile<T, BM, BN>(K_tile, k_base, kv_start, src_seq_len, kv_stride,
                             head_dim, ty, lane);
        __syncthreads();
        //exp 的参数必须是一条 fmaf：dot*scale 若先落地再减 m 就是两次舍入，
        //head_dim 非 4 的幂时（8、32）scale 不是 2 的幂，两种写法差 1 ulp。
        //mask 用整数判断，不去比较 score，否则乘积被迫落地、融合就没了。
        const bool valid = (kv_start + lane) < key_end;
        const float p = valid ? expf(fmaf(dotOf(), scale, -m_i)) : 0.0f;
        //蝶形规约会改变加法顺序；各 lane 冗余地按 k 升序串行累加才能保持同序
        P_tile[ty * BN + lane] = p;
        __syncwarp();
        float acc = l_i;
        for (int k = 0; k < BN; k++) acc += P_tile[ty * BN + k];
        l_i = acc;
        __syncthreads();
      }

    //l_i 为 0 说明该行全被 mask，按 0 输出
    const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;

//6.第三趟：累加 O。p 先归一化再 FMA，累加量级保持 O(1)
    for (int kb = 0; kb < num_kv_blocks; kb++)
      {
        const int kv_start = kb * BN;
        loadKVTile<T, BM, BN>(K_tile, V_tile, k_base, v_base, kv_start,
                              src_seq_len, kv_stride, head_dim, ty, lane);
        __syncthreads();
        const bool valid = (kv_start + lane) < key_end;
        P_tile[ty * BN + lane] =
            valid ? inv_l * expf(fmaf(dotOf(), scale, -m_i)) : 0.0f;
        __syncwarp(); //本行的 P 由本 warp 写、本 warp 读，warp 级同步即可

        //lane 负责 head_dim 的一段，各自按 k 升序遍历本块
        for (int d = lane; d < head_dim; d += BN)
          {
            float acc = O_tile[ty * head_dim + d];
            for (int k = 0; k < BN; k++)
              {
                acc = fmaf(P_tile[ty * BN + k], V_tile[k * head_dim + d], acc);
              }
            O_tile[ty * head_dim + d] = acc;
          }
        __syncthreads();
      }

//7.写回输出
    if (!q_valid) return;
    T* o_row = d_o + (size_t)batch_id * target_seq_len * q_stride
                   + (size_t)q_row * q_stride
                   + (size_t)query_head_id * head_dim;
    for (int d = lane; d < head_dim; d += BN)
      {
        storeFromFloat(O_tile[ty * head_dim + d], o_row[d]);
      }
}

//codeOnCPU
template <typename T>
void flashAttention(const std::vector<T>& h_q, 
                    const std::vector<T>& h_k,
                    const std::vector<T>& h_v, 
                    std::vector<T>& h_o,
                    int batch_size, 
                    int target_seq_len, 
                    int src_seq_len, 
                    int query_heads, 
                    int kv_heads, 
                    int head_dim, 
                    bool is_causal){       
//1.参数检验
    if (batch_size <= 0 || target_seq_len <= 0 || src_seq_len <= 0 ||
        query_heads <= 0 || kv_heads <= 0 || head_dim <= 0)
      {
        throw std::runtime_error("Non-positive dimension in flashAttention");
      }
    if (query_heads % kv_heads != 0)
      {//GQA 要求 q head 数能被 kv head 数整除
        throw std::runtime_error("query_heads not divisible by kv_heads");
      }

    const size_t q_size = (size_t)batch_size * target_seq_len * query_heads * head_dim;
    const size_t k_size = (size_t)batch_size * src_seq_len * kv_heads * head_dim;
    const size_t v_size = k_size;
    const size_t o_size = q_size;

    if (h_q.size() != q_size || h_k.size() != k_size || h_v.size() != v_size || h_o.size() != o_size)
      {
        throw std::runtime_error("Tensor size mismatch in flashAttention");
      }

//2.取上下文，确认它对应当前设备
    OpContext& ctx = flashAttentionContext<T>();
    ctx.ensureReady();

//3.算四段布局：Q | K | V | O，每步都查回绕
    size_t q_bytes = 0, k_bytes = 0, o_bytes = 0;
    if (!mulOverflow(q_size, sizeof(T), q_bytes) ||
        !mulOverflow(k_size, sizeof(T), k_bytes) ||
        !mulOverflow(o_size, sizeof(T), o_bytes))
      {
        throw std::runtime_error("Tensor byte size overflow in flashAttention");
      }
    const size_t v_bytes = k_bytes;

    size_t seg_q = 0, seg_k = 0, seg_o = 0;
    if (!alignUp(q_bytes, seg_q) || !alignUp(k_bytes, seg_k) ||
        !alignUp(o_bytes, seg_o))
      {
        throw std::runtime_error("Segment alignment overflow in flashAttention");
      }

    const size_t off_q = 0;
    const size_t off_k = seg_q;
    const size_t off_v = off_k + seg_k;
    const size_t off_o = off_v + seg_k;
    //逐段累加时查回绕，任一步溢出即不可能装得下
    if (off_k < seg_q || off_v < off_k || off_o < off_v ||
        off_o > SIZE_MAX - seg_o || !ctx.arena.reserve(off_o + seg_o))
      {
        throw std::runtime_error("Device memory reserve failed in flashAttention");
      }

    T* d_q = reinterpret_cast<T*>(ctx.arena.at(off_q));
    T* d_k = reinterpret_cast<T*>(ctx.arena.at(off_k));
    T* d_v = reinterpret_cast<T*>(ctx.arena.at(off_v));
    T* d_o = reinterpret_cast<T*>(ctx.arena.at(off_o));

//4.拷贝数据到GPU
    RUNTIME_CHECK(cudaMemcpyAsync(d_q, h_q.data(), q_bytes,
                                  cudaMemcpyHostToDevice, ctx.stream));
    RUNTIME_CHECK(cudaMemcpyAsync(d_k, h_k.data(), k_bytes,
                                  cudaMemcpyHostToDevice, ctx.stream));
    RUNTIME_CHECK(cudaMemcpyAsync(d_v, h_v.data(), v_bytes,
                                  cudaMemcpyHostToDevice, ctx.stream));
    //arena 会被复用，上一次调用的残留必须清掉
    RUNTIME_CHECK(cudaMemsetAsync(d_o, 0, o_bytes, ctx.stream));

//5.确定分块大小和kernel启动参数
    //smem 用量 = Q、O、K、V 四块（随 head_dim）加 P 一块（BM*BN，与 head_dim 无关）。
    //head_dim 大时 BM=32 可能装不下，逐档降到 8；BN 固定为 warp 大小不可调。
    auto smemFor = [head_dim](int bm) -> size_t {
      const size_t qkvo = (size_t)(2 * bm + 2 * kFaTileN) * head_dim; //Q O K V
      const size_t p    = (size_t)bm * kFaTileN;                      //P
      return (qkvo + p) * sizeof(float);
    };

    int bm = 0;
    for (int cand : {32, 16, 8})
      {
        if (smemFor(cand) <= (size_t)ctx.max_smem) { bm = cand; break; }
      }
    if (bm == 0)
      {
        throw std::runtime_error("head_dim exceeds maximum");
      }

    const int kv_groups = query_heads / kv_heads; //GQA：每组 kv 服务多少个 q head
    const dim3 grid((target_seq_len + bm - 1) / bm, batch_size * query_heads, 1);
    const dim3 block(kFaTileN, bm, 1); //x = lane（对应 K 列），y = warp（对应 q 行）

//6.启动kernel
    //float 与 half 共用同一个 kernel，只有 BM 随 smem 余量分档
    switch (bm)
      {
        case 32:
          flashAttentionKernel<T, 32, kFaTileN><<<grid, block, smemFor(32), ctx.stream>>>
          (d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads,
           kv_heads, head_dim, kv_groups, is_causal);
          break;
        case 16:
          flashAttentionKernel<T, 16, kFaTileN><<<grid, block, smemFor(16), ctx.stream>>>
          (d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads,
           kv_heads, head_dim, kv_groups, is_causal);
          break;
        default:
          flashAttentionKernel<T, 8, kFaTileN><<<grid, block, smemFor(8), ctx.stream>>>
          (d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads,
           kv_heads, head_dim, kv_groups, is_causal);
          break;
      }

    //检查kernel启动是否成功
    RUNTIME_CHECK(cudaGetLastError());

//7.拷回结果并同步。显存由 arena 持有、跨调用复用，此处不释放
    RUNTIME_CHECK(cudaMemcpyAsync(h_o.data(), d_o, o_bytes,
                                  cudaMemcpyDeviceToHost, ctx.stream));
    RUNTIME_CHECK(cudaStreamSynchronize(ctx.stream));
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
