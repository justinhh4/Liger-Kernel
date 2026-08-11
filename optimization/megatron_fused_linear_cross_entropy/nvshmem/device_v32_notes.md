# Device NVSHMEM Variant 32: Shape-Gated L2 Peer Traffic

## Hypothesis

**Target bottleneck:** Variant 31 improves realistic TP8 but regresses BT512.
The same BT2048 crossover already selects CuTile versus cuBLAS dX. Applying
the peer cache policy at that boundary should retain both wins.

## Changes

- Restore ordinary 16-byte direct loads/stores through 2,048 tokens.
- Instantiate `.cg` peer-load/store kernels above 2,048 tokens.
- Preserve v30's parallel readiness handshake.

## Expected Outcome

Match v30 at low BT and v31 at realistic shapes.

## Actual Results

TP8 full-step latency was 0.4338/0.4263 ms at BT512/2048 and
1.6830/3.0475 ms at BT16K/32K. TP4 remained within run-to-run noise at
BT16K and improved slightly at BT32K.

## Verdict

**ACCEPTED.**
