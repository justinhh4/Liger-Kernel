"""Materialized tensor-parallel fused linear cross entropy for Megatron.

Each tensor-parallel rank owns a contiguous vocabulary shard. Forward performs
one local projection GEMM, computes globally normalized cross entropy, and saves
shifted exponentials in the projection dtype. Backward converts that buffer to
dlogits in-place, avoiding projection recomputation before forming dX and dW.
"""

from __future__ import annotations

import operator

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
import torch.nn.functional as F

from optimization.megatron_fused_linear_cross_entropy.nvshmem.cutedsl_vocab_parallel_cross_entropy import (
    backward as cutedsl_ce_backward,
)
from optimization.megatron_fused_linear_cross_entropy.nvshmem.cutedsl_vocab_parallel_cross_entropy import (
    forward as cutedsl_ce_forward,
)
from optimization.megatron_fused_linear_cross_entropy.nvshmem.cutedsl_vocab_parallel_cross_entropy import (
    loss as cutedsl_ce_loss,
)
from optimization.megatron_fused_linear_cross_entropy.nvshmem.cutedsl_vocab_parallel_cross_entropy import (
    row_max as cutedsl_row_max,
)

from liger_kernel.ops.cutedsl.ops._sm100_gemm import K_ALIGNMENT
from liger_kernel.ops.cutedsl.ops._sm100_gemm import run_epilogue_gemm
from liger_kernel.ops.utils import compare_version

_GEMM_ROW_CHUNK_SIZE = 4096
_SUPPORTS_OUT_DTYPE = compare_version("torch", operator.ge, "2.8.0")


@cute.jit
def _identity_epilogue(accumulator, output):
    output_dtype = output.element_type
    for element in cutlass.range_constexpr(cute.size(accumulator)):
        output[element] = accumulator[element].to(output_dtype)


def _cutedsl_gemm(a: torch.Tensor, b: torch.Tensor, output: torch.Tensor | None = None) -> torch.Tensor:
    padding = (-a.shape[1]) % K_ALIGNMENT
    if padding:
        a = F.pad(a, (0, padding))
        b = F.pad(b, (0, padding))
    if output is None:
        output = torch.empty(a.shape[0], b.shape[0], device=a.device, dtype=a.dtype)
    for row_start in range(0, a.shape[0], _GEMM_ROW_CHUNK_SIZE):
        row_end = min(row_start + _GEMM_ROW_CHUNK_SIZE, a.shape[0])
        run_epilogue_gemm(
            a[row_start:row_end],
            b,
            output[row_start:row_end],
            _identity_epilogue,
            swizzle_size=1,
        )
    return output


def _matmul_out(a: torch.Tensor, b: torch.Tensor, output: torch.Tensor) -> None:
    if output.dtype == a.dtype:
        torch.mm(a, b, out=output)
    elif _SUPPORTS_OUT_DTYPE:
        torch.mm(a, b, out=output, out_dtype=output.dtype)
    else:
        output.copy_(a.float() @ b.float())


def _tp_rank_and_world(tp_group) -> tuple[int, int]:
    if tp_group is None:
        return 0, 1
    world = dist.get_world_size(tp_group)
    if world == 1:
        return 0, 1
    return dist.get_rank(tp_group), world


def _materialized_backward(ctx, grad_output: torch.Tensor):
    """Convert saved CE state to dlogits and form projection gradients."""
    hidden, weight, exp_buf, sum_exp_global, target = ctx.saved_tensors
    grad_out = grad_output.contiguous().reshape(-1).float()
    cutedsl_ce_backward(
        exp_buf,
        target,
        sum_exp_global,
        grad_out,
        ctx.vocab_start,
        ctx.ignore_index,
    )

    workspace = ctx.nvshmem_workspace
    if workspace is not None and ctx.tp_world > 1:
        grad_hidden_source, grad_hidden_destination = workspace.dx_buffers(
            hidden.shape[0],
            hidden.shape[1],
        )
        _matmul_out(exp_buf, weight, grad_hidden_source)
        workspace.dx_ready.record(torch.cuda.current_stream())
        workspace.overlap_stream.wait_event(workspace.dx_ready)
        with torch.cuda.stream(workspace.overlap_stream):
            grad_weight = exp_buf.t() @ hidden
            grad_bias = exp_buf.sum(dim=0, dtype=torch.float32).to(ctx.bias_dtype) if ctx.has_bias else None
        workspace.sum_reduce(grad_hidden_source, grad_hidden_destination)
        torch.cuda.current_stream().wait_stream(workspace.overlap_stream)
        grad_hidden = grad_hidden_destination
    else:
        if _SUPPORTS_OUT_DTYPE:
            grad_hidden = torch.mm(exp_buf, weight, out_dtype=torch.float32)
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
        vocab_alignment = 16 // hidden.element_size()
        if weight.shape[0] % vocab_alignment:
            raise ValueError(
                f"local vocabulary size must be divisible by {vocab_alignment} for aligned CuTe loads, "
                f"got {weight.shape[0]}."
            )

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

        logits = _cutedsl_gemm(hidden_2d, weight_2d)
        if bias_1d is not None:
            logits.add_(bias_1d)

        if nvshmem_workspace is not None and tp_world > 1:
            logits_max_source, logits_max = nvshmem_workspace.max_buffers(hidden_2d.shape[0])
            cutedsl_row_max(logits, logits_max_source)
            nvshmem_workspace.max_reduce(logits_max_source, logits_max)
        else:
            logits_max = logits.amax(dim=-1).float()
        if nvshmem_workspace is None and tp_world > 1:
            dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)

        exp_buf = logits
        if nvshmem_workspace is not None and tp_world > 1:
            predicted_logit, sum_exp, global_predicted_logit, global_sum_exp = nvshmem_workspace.stats_buffers(
                hidden_2d.shape[0]
            )
        else:
            predicted_logit = torch.empty(hidden_2d.shape[0], device=hidden.device, dtype=torch.float32)
            sum_exp = torch.empty_like(predicted_logit)
        cutedsl_ce_forward(
            exp_buf,
            flat_target,
            logits_max,
            predicted_logit,
            sum_exp,
            vocab_start,
            ignore_index,
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

        loss = cutedsl_ce_loss(sum_exp, predicted_logit, flat_target, ignore_index)

        ctx.save_for_backward(hidden_2d, weight_2d, exp_buf, sum_exp, flat_target)
        ctx.has_bias = bias is not None
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.tp_group = tp_group
        ctx.tp_world = tp_world
        ctx.vocab_start = vocab_start
        ctx.ignore_index = ignore_index
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
