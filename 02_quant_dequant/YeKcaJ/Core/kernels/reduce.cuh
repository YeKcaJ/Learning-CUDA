#pragma once

// NVFP4 及 tensor 模式的全局最大值两级归约。
#include "common/types.h"
#include <cuda_runtime.h>

namespace pipeline {

// Kernel 1：全张量绝对值最大值的第一阶段，用于 NVFP4 或 tensor scale 模式。
// 每个 CUDA 线程块固定 256 线程，跨步扫描 FP32 input，再在共享内存中归约。
// 输出 partial[blockIdx.x]：该线程块负责的数据的最大绝对值。
__global__ void maximum(const float* input, float* partial, std::size_t n) {
  __shared__ float s[256];
  float v = 0;
  for (std::size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += gridDim.x * 256ull)
    v = fmaxf(v, fabsf(input[i]));
  s[threadIdx.x] = v;
  __syncthreads();

  for (unsigned step = 128; step; step >>= 1) {
    if (threadIdx.x < step)
      s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + step]);
    __syncthreads();
  }
  if (threadIdx.x == 0) partial[blockIdx.x] = s[0];
}

// Kernel 2：用一个 256 线程块归约 partial，得到全张量最大绝对值 M。
// 输出 state[0]=M；NVFP4 的 state[1]=global_scale=M/(6*448)，其他格式为 1。
// 全零输入使用 global_scale=1；正数下溢时保留最小正 FP32 值，避免除零。
__global__ void finalize_max(const float* partial, float* state, unsigned n) {
  __shared__ float s[256];
  float v = 0;
  for (unsigned i = threadIdx.x; i < n; i += 256) v = fmaxf(v, partial[i]);
  s[threadIdx.x] = v;
  __syncthreads();

  for (unsigned step = 128; step; step >>= 1) {
    if (threadIdx.x < step)
      s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + step]);
    __syncthreads();
  }
  if (!threadIdx.x) {
    state[0] = s[0];
    state[1] = nvfp4 && s[0] > 0 ? fmaxf(s[0] / 2688.0f, __int_as_float(1)) : 1.0f;
  }
}

}  // namespace pipeline
