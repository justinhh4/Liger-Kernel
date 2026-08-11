"""Materialized tensor-parallel fused linear cross entropy for Megatron.

Each tensor-parallel rank owns a contiguous vocabulary shard. Forward performs
one local projection GEMM, computes globally normalized cross entropy, and saves
shifted exponentials in the projection dtype. Backward converts that buffer to
dlogits in-place, avoiding projection recomputation before forming dX and dW.
"""

from __future__ import annotations

import operator

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from liger_kernel.ops.utils import compare_version

_SUPPORTS_OUT_DTYPE = compare_version("torch", operator.ge, "2.8.0")


def _cutile_dx_out(a: torch.Tensor, b: torch.Tensor, output: torch.Tensor) -> None:
    import cuda.tile as ct

    from liger_kernel.ops.cutile.ops.megatron_fused_linear_cross_entropy import _matmul_1cta_kernel
    from liger_kernel.ops.cutile.ops.megatron_fused_linear_cross_entropy import _matmul_2cta_kernel

    if a.shape[0] <= 1024:
        kernel, tile = _matmul_1cta_kernel, (128, 128, 64)
    elif a.shape[1] > 16000 or a.shape[0] > 16384:
        kernel, tile = _matmul_2cta_kernel, (512, 256, 64)
    else:
        kernel, tile = _matmul_1cta_kernel, (256, 256, 64)
    tile_m, tile_n, tile_k = tile
    grid = (
        ct.cdiv(a.shape[0], tile_m) * ct.cdiv(b.shape[1], tile_n),
        1,
        1,
    )
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        kernel,
        (
            a,
            b,
            output,
            output,
            tile_m,
            tile_n,
            tile_k,
            False,
            False,
        ),
    )


@triton.jit
def _local_logits_max_kernel(
    logits_ptr,
    logits_stride,
    output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(
        logits_ptr + row * logits_stride + offsets,
        mask=offsets < n_cols,
        other=-float("inf"),
    ).to(tl.float32)
    tl.store(output_ptr + row, tl.max(values, axis=0))


def _tp_rank_and_world(tp_group) -> tuple[int, int]:
    if tp_group is None:
        return 0, 1
    world = dist.get_world_size(tp_group)
    if world == 1:
        return 0, 1
    return dist.get_rank(tp_group), world


def _materialized_backward(ctx, grad_output: torch.Tensor):
    """Convert saved CE state to dlogits and form projection gradients."""
    from liger_kernel.ops.vocab_parallel_cross_entropy import _get_num_warps
    from liger_kernel.ops.vocab_parallel_cross_entropy import liger_vocab_parallel_ce_backward_kernel

    hidden, weight, exp_buf, sum_exp_global, target = ctx.saved_tensors
    grad_out = grad_output.contiguous().reshape(-1).float()
    num_warps = _get_num_warps(ctx.ce_block_size)
    liger_vocab_parallel_ce_backward_kernel[(hidden.shape[0],)](
        EXP_ptr=exp_buf,
        EXP_stride=exp_buf.stride(0),
        sum_exp_ptr=sum_exp_global,
        Y_ptr=target,
        grad_out_ptr=grad_out,
        vocab_start=ctx.vocab_start,
        n_cols=weight.shape[0],
        ignore_index=ctx.ignore_index,
        alpha_eff=0.0,
        eps_eff=0.0,
        HAS_LABEL_SMOOTHING=False,
        BLOCK_SIZE=ctx.ce_block_size,
        num_warps=num_warps,
    )

    workspace = ctx.nvshmem_workspace
    if workspace is not None and ctx.tp_world > 1:
        grad_hidden_source, grad_hidden_destination = workspace.dx_buffers(
            hidden.shape[0],
            hidden.shape[1],
        )
        if grad_hidden_source.dtype == hidden.dtype:
            if hidden.shape[0] <= 2048:
                _cutile_dx_out(exp_buf, weight, grad_hidden_source)
            else:
                torch.mm(
                    exp_buf,
                    weight,
                    out=grad_hidden_source,
                )
        elif _SUPPORTS_OUT_DTYPE:
            torch.mm(
                exp_buf,
                weight,
                out=grad_hidden_source,
                out_dtype=torch.float32,
            )
        else:
            grad_hidden_source.copy_(exp_buf.float() @ weight.float())
        workspace.dx_ready.record(torch.cuda.current_stream())
        workspace.overlap_stream.wait_event(workspace.dx_ready)
        with torch.cuda.stream(workspace.overlap_stream):
            grad_weight = exp_buf.t() @ hidden
            grad_bias = exp_buf.sum(dim=0, dtype=torch.float32).to(ctx.bias_dtype) if ctx.has_bias else None
        workspace.sum_reduce(grad_hidden_source, grad_hidden_destination)
        torch.cuda.current_stream().wait_stream(workspace.overlap_stream)
        grad_hidden = grad_hidden_destination
    elif _SUPPORTS_OUT_DTYPE:
        grad_hidden = torch.mm(exp_buf, weight, out_dtype=torch.float32)
        grad_weight = exp_buf.t() @ hidden
        grad_bias = exp_buf.sum(dim=0, dtype=torch.float32).to(ctx.bias_dtype) if ctx.has_bias else None
    else:
        grad_hidden = exp_buf.float() @ weight.float()
        grad_weight = exp_buf.t() @ hidden
        grad_bias = exp_buf.sum(dim=0, dtype=torch.float32).to(ctx.bias_dtype) if ctx.has_bias else None

    if workspace is None and ctx.tp_world > 1:
        dist.all_reduce(grad_hidden, op=dist.ReduceOp.SUM, group=ctx.tp_group)
    grad_hidden = grad_hidden.to(ctx.hidden_dtype).reshape(ctx.original_hidden_shape)
    return grad_hidden, grad_weight, grad_bias


class LigerMegatronFusedLinearCrossEntropyFunction(torch.autograd.Function):
    """Hidden-to-loss tensor-parallel FLCE with saved low-precision CE state."""

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        bias: torch.Tensor | None,
        tp_group,
        ignore_index: int,
        nvshmem_workspace,
    ) -> torch.Tensor:
        if hidden.ndim < 2:
            raise ValueError(f"hidden must have at least 2 dimensions, got shape {tuple(hidden.shape)}.")
        if weight.ndim != 2:
            raise ValueError(f"weight must be 2-D [V_local, H], got shape {tuple(weight.shape)}.")
        if tuple(target.shape) != tuple(hidden.shape[:-1]):
            raise ValueError(
                f"target shape must equal hidden.shape[:-1]; got target={tuple(target.shape)}, "
                f"hidden={tuple(hidden.shape)}."
            )
        if hidden.shape[-1] != weight.shape[1]:
            raise ValueError(f"hidden size mismatch: hidden has H={hidden.shape[-1]}, weight has H={weight.shape[1]}.")
        if hidden.dtype != weight.dtype:
            raise TypeError(f"hidden and weight must have the same dtype, got {hidden.dtype} and {weight.dtype}.")
        if hidden.device != weight.device or hidden.device != target.device:
            raise ValueError("hidden, weight, and target must be on the same device.")
        if bias is not None:
            if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
                raise ValueError(f"bias must have shape ({weight.shape[0]},), got {tuple(bias.shape)}.")
            if bias.device != hidden.device or bias.dtype != hidden.dtype:
                raise TypeError("bias must have the same device and dtype as hidden.")
        if hidden.device.type != "cuda" or hidden.dtype not in (torch.bfloat16, torch.float16):
            raise RuntimeError("Megatron FLCE requires a CUDA GPU and float16 or bfloat16 inputs.")

        tp_rank, tp_world = _tp_rank_and_world(tp_group)
        vocab_local = weight.shape[0]
        vocab_global = vocab_local * tp_world
        vocab_start = tp_rank * vocab_local

        flat_target = target.reshape(-1).to(torch.int64).contiguous()
        valid = flat_target != ignore_index
        invalid = valid & ((flat_target < 0) | (flat_target >= vocab_global))
        valid_targets = ~torch.any(invalid)
        if hasattr(torch, "_assert_async"):
            torch._assert_async(valid_targets, f"non-ignored targets must be in [0, {vocab_global}).")
        elif not valid_targets.item():
            raise ValueError(f"non-ignored targets must be in [0, {vocab_global}).")

        original_hidden_shape = hidden.shape
        hidden_2d = hidden.reshape(-1, hidden.shape[-1]).contiguous()
        weight_2d = weight.contiguous()
        bias_1d = bias.contiguous() if bias is not None else None

        logits = torch.mm(hidden_2d, weight_2d.t())
        if bias_1d is not None:
            logits.add_(bias_1d)

        if nvshmem_workspace is not None and tp_world > 1:
            logits_max_source, logits_max = nvshmem_workspace.max_buffers(hidden_2d.shape[0])
            local_max_block = triton.next_power_of_2(vocab_local)
            local_max_warps = 8 if local_max_block <= 8192 else 16
            _local_logits_max_kernel[(hidden_2d.shape[0],)](
                logits,
                logits.stride(0),
                logits_max_source,
                vocab_local,
                BLOCK_SIZE=local_max_block,
                num_warps=local_max_warps,
            )
            nvshmem_workspace.max_reduce(logits_max_source, logits_max)
        else:
            logits_max = logits.amax(dim=-1).float()
        if nvshmem_workspace is None and tp_world > 1:
            dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)

        from liger_kernel.ops.vocab_parallel_cross_entropy import _get_num_warps
        from liger_kernel.ops.vocab_parallel_cross_entropy import _select_block_size
        from liger_kernel.ops.vocab_parallel_cross_entropy import liger_vocab_parallel_ce_forward_kernel

        exp_buf = torch.empty(
            hidden_2d.shape[0],
            vocab_local,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        if nvshmem_workspace is not None and tp_world > 1:
            predicted_logit, sum_exp, global_predicted_logit, global_sum_exp = nvshmem_workspace.stats_buffers(
                hidden_2d.shape[0]
            )
        else:
            predicted_logit = torch.empty(hidden_2d.shape[0], device=hidden.device, dtype=torch.float32)
            sum_exp = torch.empty_like(predicted_logit)
        ce_block_size = _select_block_size(vocab_local)
        num_warps = _get_num_warps(ce_block_size)
        liger_vocab_parallel_ce_forward_kernel[(hidden_2d.shape[0],)](
            X_ptr=logits,
            X_stride=logits.stride(0),
            EXP_ptr=exp_buf,
            EXP_stride=exp_buf.stride(0),
            logits_max_ptr=logits_max,
            Y_ptr=flat_target,
            pred_ptr=predicted_logit,
            sum_exp_ptr=sum_exp,
            vocab_start=vocab_start,
            n_cols=vocab_local,
            ignore_index=ignore_index,
            BLOCK_SIZE=ce_block_size,
            num_warps=num_warps,
        )
        if nvshmem_workspace is not None and tp_world > 1:
            count = 2 * hidden_2d.shape[0]
            nvshmem_workspace.sum_reduce(
                nvshmem_workspace.stats_source[:count],
                nvshmem_workspace.stats_destination[:count],
            )
            predicted_logit = global_predicted_logit
            sum_exp = global_sum_exp
        elif tp_world > 1:
            dist.all_reduce(predicted_logit, op=dist.ReduceOp.SUM, group=tp_group)
            dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=tp_group)

        loss = torch.log(sum_exp) - predicted_logit
        loss = torch.where(valid, loss, torch.zeros_like(loss))

        ctx.save_for_backward(hidden_2d, weight_2d, exp_buf, sum_exp, flat_target)
        ctx.has_bias = bias is not None
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.tp_group = tp_group
        ctx.tp_world = tp_world
        ctx.vocab_start = vocab_start
        ctx.ignore_index = ignore_index
        ctx.ce_block_size = ce_block_size
        ctx.original_hidden_shape = original_hidden_shape
        ctx.hidden_dtype = hidden.dtype
        ctx.nvshmem_workspace = nvshmem_workspace
        return loss.reshape(target.shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_hidden, grad_weight, grad_bias = _materialized_backward(ctx, grad_output)
        return grad_hidden, grad_weight, None, grad_bias, None, None, None


def liger_megatron_fused_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: torch.Tensor | None = None,
    tp_group=None,
    ignore_index: int = -100,
    nvshmem_workspace=None,
) -> torch.Tensor:
    """Compute per-token loss from replicated hidden states and a local vocab shard."""
    return LigerMegatronFusedLinearCrossEntropyFunction.apply(
        hidden,
        weight,
        target,
        bias,
        tp_group,
        ignore_index,
        nvshmem_workspace,
    )
