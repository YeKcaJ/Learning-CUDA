# CPU Reference Specification

This specification freezes the deterministic CPU behavior used by the CUDA
implementation. All integer fields are little-endian and all matrices are
row-major flattened arrays.

## Common input file

`tests/data/*.fp32` uses the following header:

```text
magic[8]   = FP32INP1
version    = uint32(1)
rows       = uint64
cols       = uint64
values     = rows * cols little-endian float32 values
```

## MXFP8

- Data format: E4M3FN, one byte per element.
- Block size: 32 elements.
- Scale format: E8M0 exponent byte, bias 127.
- For a block maximum `m`, choose `e = clamp(ceil(log2(m / 448)) + 127, 0, 254)`;
  use `scale = 2^(e - 127)`. An all-zero block uses `e = 127`.
- Quantization is nearest finite E4M3FN with saturation at `[-448, 448]`.
- Dequantization is `decode_e4m3(code) * scale`.
- Quantized file header: `magic[8]=MXFP8Q1`, `version=uint32(1)`, then
  `rows:uint64, cols:uint64, block:uint32, data_bytes:uint64,
  scale_bytes:uint64, data, scales`.

## NVFP4

- Data format: E2M1 nibble. Magnitudes are `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`;
  bit 3 is the sign and bits 0-2 select the magnitude.
- Block size: 16 elements.
- Two elements are packed per byte: even index in the low nibble, odd index in
  the high nibble.
- The tensor scale is FP32: `global_scale = M / (6 * 448)`, where `M` is the
  tensor maximum absolute value; all-zero tensors use `1`.
- For block maximum `m_b`, encode `block_scale = E4M3(m_b /
  (6 * global_scale))` using nearest rounding.
- Effective scale is `global_scale * decode_e4m3(block_scale)`.
- Quantization is nearest E2M1; dequantization is
  `decode_e2m1(nibble) * effective_scale`.
- Quantized file header: `magic[8]=NVFP4Q1`, `version=uint32(1)`, then
  `rows:uint64, cols:uint64, block:uint32, global_scale:float32,
  data_bytes:uint64, scale_bytes:uint64, packed_data, block_scales`.

## Dequantized golden file

`tests/golden/*.dequant.fp32` uses:

```text
magic[8]   = FP32DEQ1
version    = uint32(1)
rows       = uint64
cols       = uint64
count      = uint64 (rows * cols)
values     = count little-endian float32 values
```

Golden validation requires quantized metadata/data to match byte-for-byte and
dequantized values to have maximum absolute difference no greater than `1e-6`.
Run `--freeze` once to create the fixed corpus; subsequent checks must use
`--verify`, which never overwrites the existing golden files.
