#pragma once

// MXFP8 默认量化算子：block + nearest。
#include "kernels/scale.cuh"

namespace pipeline {

// MXFP8 block+nearest 融合 kernel：一个 warp 处理 32 元素，输入只读取一次。
// 尾部补零并参与所有 shuffle；组首计算 scale 后广播，复用寄存器中的原值编码。
// scale 公式及 FP32 除法保持原样，不引入 fast-math 或倒数近似。
__global__ void mxfp8_quantize_fused_kernel(const float* input, std::uint8_t* data,
                                          std::uint8_t* scales, std::size_t n) {
  const std::size_t i = blockIdx.x * 256ull + threadIdx.x;
  const unsigned lane = threadIdx.x & 31;
  const float value = i < n ? input[i] : 0.0f;
  float maximum = fabsf(value);
  for (unsigned step = 16; step; step >>= 1)
    maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, step));

  unsigned code = 127;
  if (lane == 0 && i < n) {
    code = scale_code(maximum, 1.0f);
    scales[i / 32] = code;
  }
  code = __shfl_sync(0xffffffffu, code, 0);
  if (i < n) {
    const float scale = scalbnf(1.0f, static_cast<int>(code) - 127);
    data[i] = encode_e4m3_nearest(value / scale);
  }
}

}  // namespace pipeline
