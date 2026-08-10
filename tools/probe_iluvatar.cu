// 天数智芯(Iluvatar CoreX)平台探测：确认 warp 宽度、shuffle 掩码语义、
// smem/线程上限，以及内存池/occupancy API 是否存在。
// 编译： make PLATFORM=iluvatar   （用训练营提供的环境：clang++ + CoreX libcudart）
// 或手动：clang++ -std=c++17 -O2 -DPLATFORM_ILUVATAR \
//           -I/usr/local/corex/include -L/usr/local/corex/lib64 -lcudart \
//           tools/probe_iluvatar.cu -o probe_iluvatar
// 可选再加 -DPROBE_MASK64 / -DPROBE_POOL / -DPROBE_OCC
#include <cstdio>
#include <vector>
#include <cuda_fp16.h>
#include "../tester/utils.h"

//蝶形 all-reduce，确认 shuffle 掩码覆盖整个 warp
__global__ void probeReduce32(float* out, int ws)
{
    float v = 1.0f;
    for (int o = ws / 2; o > 0; o >>= 1)
      {
        v += __shfl_xor_sync(0xffffffffu, v, o);
      }
    out[threadIdx.x] = v;
}

#ifdef PROBE_MASK64
__global__ void probeReduce64(float* out, int ws)
{
    float v = 1.0f;
    for (int o = ws / 2; o > 0; o >>= 1)
      {
        v += __shfl_xor_sync(0xffffffffffffffffull, v, o);
      }
    out[threadIdx.x] = v;
}
#endif

//half 转换、__syncwarp、动态 smem、fmaf/expf/rsqrtf
__global__ void probeMisc(float* out, const half* h_in)
{
    extern __shared__ float s[];
    const int t = threadIdx.x;
    s[t] = __half2float(h_in[t]);
    __syncwarp();
    float acc = fmaf(s[t], 2.0f, 1.0f);
    acc += expf(0.0f) + rsqrtf(4.0f);
    out[t] = acc;
    if (t == 0) out[0] = __half2float(__float2half(1.5f));
}

int main()
{
    int dev = 0;
    RUNTIME_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp props;
    RUNTIME_CHECK(cudaGetDeviceProperties(&props, dev));

    printf("=== 设备 ===\n");
    printf("  name                     : %s\n", props.name);
    printf("  warpSize                 : %d   <<< 最关键\n", props.warpSize);
    printf("  sharedMemPerBlock        : %zu B\n", (size_t)props.sharedMemPerBlock);
    printf("  maxThreadsPerBlock       : %d\n", props.maxThreadsPerBlock);
    printf("  maxThreadsPerMultiProcessor: %d\n", props.maxThreadsPerMultiProcessor);
    printf("  multiProcessorCount      : %d\n", props.multiProcessorCount);
    printf("  maxGridSize[0]           : %d\n", props.maxGridSize[0]);
    printf("  regsPerBlock             : %d\n", props.regsPerBlock);

    const int ws = props.warpSize;

    //shuffle 正确性：全 warp 每个 lane 持 1.0，全归约后应等于 ws
    float* d_out = nullptr;
    RUNTIME_CHECK(cudaMalloc((void**)&d_out, sizeof(float) * 256));
    std::vector<float> h_out(256, -1.0f);

    printf("=== shuffle 掩码语义 (每 lane 1.0, 全归约应得 %d) ===\n", ws);
    probeReduce32<<<1, ws>>>(d_out, ws);
    RUNTIME_CHECK(cudaGetLastError());
    RUNTIME_CHECK(cudaDeviceSynchronize());
    RUNTIME_CHECK(cudaMemcpy(h_out.data(), d_out, sizeof(float) * ws,
                             cudaMemcpyDeviceToHost));
    printf("  掩码 0xffffffffu  : lane0=%.1f  lane%d=%.1f  %s\n",
           h_out[0], ws - 1, h_out[ws - 1],
           (h_out[0] == (float)ws) ? "正确" : "!! 错误，掩码没覆盖整个 warp");

#ifdef PROBE_MASK64
    probeReduce64<<<1, ws>>>(d_out, ws);
    RUNTIME_CHECK(cudaGetLastError());
    RUNTIME_CHECK(cudaDeviceSynchronize());
    RUNTIME_CHECK(cudaMemcpy(h_out.data(), d_out, sizeof(float) * ws,
                             cudaMemcpyDeviceToHost));
    printf("  掩码 64 位全 1    : lane0=%.1f  lane%d=%.1f  %s\n",
           h_out[0], ws - 1, h_out[ws - 1],
           (h_out[0] == (float)ws) ? "正确" : "!! 错误");
#else
    printf("  (未测 64 位掩码，加 -DPROBE_MASK64 再编一次)\n");
#endif

    //half / __syncwarp / 动态 smem
    half* d_h = nullptr;
    RUNTIME_CHECK(cudaMalloc((void**)&d_h, sizeof(half) * 256));
    std::vector<half> h_h(256, __float2half(3.0f));
    RUNTIME_CHECK(cudaMemcpy(d_h, h_h.data(), sizeof(half) * 256,
                             cudaMemcpyHostToDevice));
    probeMisc<<<1, ws, sizeof(float) * ws>>>(d_out, d_h);
    RUNTIME_CHECK(cudaGetLastError());
    RUNTIME_CHECK(cudaDeviceSynchronize());
    RUNTIME_CHECK(cudaMemcpy(h_out.data(), d_out, sizeof(float) * ws,
                             cudaMemcpyDeviceToHost));
    printf("=== half / __syncwarp / 动态 smem ===\n");
    printf("  half(1.5) 回读     : %.3f  (应为 1.500)\n", h_out[0]);
    printf("  lane1 = 3*2+1+1+0.5: %.3f  (应为 8.500)\n", h_out[1]);

    //流
    cudaStream_t stream = nullptr;
    RUNTIME_CHECK(cudaStreamCreate(&stream));
    RUNTIME_CHECK(cudaStreamSynchronize(stream));
    RUNTIME_CHECK(cudaStreamDestroy(stream));
    printf("=== 流 API: 可用 ===\n");

#ifdef PROBE_POOL
    int pools = 0;
    RUNTIME_CHECK(cudaDeviceGetAttribute(&pools, cudaDevAttrMemoryPoolsSupported, dev));
    printf("=== 内存池: supported=%d ===\n", pools);
    if (pools)
      {
        void* p = nullptr;
        RUNTIME_CHECK(cudaMallocAsync(&p, 1024, (cudaStream_t)0));
        RUNTIME_CHECK(cudaFreeAsync(p, (cudaStream_t)0));
        RUNTIME_CHECK(cudaStreamSynchronize((cudaStream_t)0));
        printf("  cudaMallocAsync/cudaFreeAsync: 可用\n");
      }
#else
    printf("=== (未测内存池，加 -DPROBE_POOL 再编一次) ===\n");
#endif

#ifdef PROBE_OCC
    int nb = 0;
    RUNTIME_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &nb, (const void*)probeReduce32, ws, 0));
    printf("=== occupancy API 可用, blocks/SM=%d ===\n", nb);
#else
    printf("=== (未测 occupancy API，加 -DPROBE_OCC 再编一次) ===\n");
#endif

    RUNTIME_CHECK(cudaFree(d_out));
    RUNTIME_CHECK(cudaFree(d_h));
    printf("=== 探测结束 ===\n");
    return 0;
}
