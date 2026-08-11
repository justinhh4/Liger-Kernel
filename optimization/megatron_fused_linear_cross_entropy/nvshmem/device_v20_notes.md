# Device NVSHMEM Variant 20: TP-Specialized Peer Loops

## Hypothesis

**Target bottleneck:** Variant 19 reduces the two communication phases from
about 913 us to about 902 us, but its fused kernel still takes 887 us. The
peer count is runtime-dynamic, preventing guaranteed unrolling of three TP4 or
seven TP8 independent peer operations.

Compiling dedicated TP2/TP4/TP8 kernels should remove loop control and expose
more direct-peer memory-level parallelism to the compiler.

## Changes

- Template the vectorized reduce-and-broadcast kernel on peer count.
- Dispatch dedicated 2-, 4-, and 8-rank instantiations.
- Keep the dynamic implementation for other supported world sizes.

## Expected Outcome

Reduce the fused communication kernel by 3-8% and improve full-step TP4/TP8
V32K latency by 1-3%.

## Actual Results

At TP4/BT16K/V32K, backward improved from Variant 19's 1.6843 to 1.5623 ms
and full-step improved from 2.6897 to 2.5766 ms. TP2 correctness also passed.

## Verdict

**ACCEPTED.** Compile-time TP count exposes substantially more peer-memory
parallelism.
