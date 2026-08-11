# Device NVSHMEM Variant 24: CuTile dX GEMM

## Hypothesis

**Target bottleneck:** Variant 22 hides communication under dW, leaving the
serialized dX GEMM as the next exposed stage. The tuned Blackwell CuTile dX
kernel measured about 2% faster than `torch.mm` in an isolated
BT16K/H4096/Vlocal8K microbenchmark.

Launching CuTile directly into symmetric low-precision output may reduce the
exposed dX stage without changing communication or dW.

## Changes

- Restore Variant 22's 4,096-block communication cap.
- Add a CuTile dX launcher using the previously tuned 1/2-CTA tile crossover.
- Write CuTile output directly into the symmetric dX source buffer.
- Keep the FP32 and unsupported-shape fallbacks unchanged.

## Expected Outcome

Improve realistic full-step latency by 0.5-1.5% while preserving low-BT
performance.

## Actual Results

CuTile improved low-BT TP4/V32K full-step latency to 0.4327 ms at BT512
and 0.4687 ms at BT2048. At BT16K it regressed backward from Variant 22's
1.5220 ms to 1.5724 ms and full-step from 2.5483 to 2.6153 ms.

## Verdict

**REJECTED AS A GLOBAL REPLACEMENT.** Retain CuTile only below the measured
shape crossover.
