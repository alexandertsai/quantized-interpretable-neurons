# NVFP4 reference pilot (in progress)

Tests the existing RedHatAI/Qwen3.6-35B-A3B-NVFP4 release at revision
`e94a930c8e9b8a7d3763e449289c261c9c7d3e5e`. This is a Red Hat release,
not an official Qwen quantization. No calibration or training is performed.

Packed 4-bit weights are decoded with compressed-tensors 0.20.1. Its published
activation scheme is emulated through quantize/dequantize operations, with
BF16 matrix multiplications. This is not native FP4 inference and cannot
establish native kernel accuracy, memory use or speed. A W4A16 control omits
activation rounding while retaining the same released weight values.

Run `bash setup.sh` inside the recorded PyTorch 2.10 CUDA 12.8 runtime with
a 96 GB GPU. It regenerates the exact 32 prior evaluation articles (hash
checked), runs validation, then BF16 and FP4 reference experiments. The mask
logic matches the previous transfer pilot. No results are claimed yet.

CPU preflight matched the official compressor for one downloaded expert and
matched activation QDQ on random rows, including zeros. GPU validation also
checks the MoE reference against unmodified BF16 grouped execution, activation
QDQ against official per-row calls, and random packed weight decompression.
