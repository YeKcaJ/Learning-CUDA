// Moore 最小复现：直接调 rmsNorm<float>，绕过 tester
// 编译：mcc -std=c++11 -O2 -DPLATFORM_MOORE tools/debug_moore.mu src/kernels.mu -I. -o debug_moore -lmusart -L/usr/local/musa/lib
#include <cstdio>
#include <vector>
#include <musa_fp16.h>
#include "tester/utils.h"

// 复用 kernels.mu 的 rmsNorm<float>（外部显式实例化）
template <typename T>
void rmsNorm(const std::vector<T>& h_input,
             const std::vector<T>& h_weight,
             std::vector<T>& h_output,
             size_t rows,
             size_t hidden_dim,
             float eps);

int main()
{
    // 最简单测例：1行 × 4维
    const size_t rows = 1, hidden_dim = 4;
    std::vector<float> input  = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> weight = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> output(hidden_dim, -999.0f); // 初始化为不可能值

    printf("=== 调用前: output=[%.4f, %.4f, %.4f, %.4f]\n",
           output[0], output[1], output[2], output[3]);

    rmsNorm<float>(input, weight, output, rows, hidden_dim, 1e-5f);

    printf("=== 调用后: output=[%.4f, %.4f, %.4f, %.4f]\n",
           output[0], output[1], output[2], output[3]);

    // 手算期望：
    // mean_square = (1+4+9+16)/4 = 7.5
    // rsqrt(7.5 + 1e-5) ≈ 0.365148
    // output[i] = input[i] * 0.365148 * 1.0
    const float expected_scale = 1.0f / sqrtf(7.5f + 1e-5f);
    printf("=== 期望: scale=%.6f, output=[%.4f, %.4f, %.4f, %.4f]\n",
           expected_scale,
           1.0f * expected_scale,
           2.0f * expected_scale,
           3.0f * expected_scale,
           4.0f * expected_scale);

    // 如果 output 仍然是 -999，说明 kernel 根本没跑
    bool ok = true;
    for (size_t i = 0; i < hidden_dim; i++) {
        float expect = input[i] * expected_scale;
        float diff = fabsf(output[i] - expect);
        if (diff > 0.001f) ok = false;
    }
    printf("=== %s ===\n", ok ? "PASS" : "FAIL");

    // 额外：确认不是 memset 或 memcpy 的假数据
    printf("\n=== 原始 output (hex): ");
    for (size_t i = 0; i < hidden_dim; i++) {
        unsigned int bits = *(unsigned int*)&output[i];
        printf("%08x ", bits);
    }
    printf("\n");
    return ok ? 0 : 1;
}
