from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from cutlass import Float32
from cutlass import Int32

from liger_kernel.ops.cutedsl.ops.cross_entropy import LN2
from liger_kernel.ops.cutedsl.ops.cross_entropy import LOG2_E
from liger_kernel.ops.cutedsl.ops.cross_entropy import NEG_INF_F32
from liger_kernel.ops.cutedsl.ops.cross_entropy import _cute_stream
from liger_kernel.ops.cutedsl.ops.cross_entropy import fmax
from liger_kernel.ops.cutedsl.ops.utils import to_cute_tensor

_NUM_WARPS = 8
_THREADS = 32 * _NUM_WARPS
_COMPILE_CACHE = {}


@cute.jit
def _warp_sum(value: Float32):
    for index in cutlass.range_constexpr(5):
        value = value + cute.arch.shuffle_sync_bfly(value, offset=1 << index)
    return value


@cute.jit
def _warp_max(value: Float32):
    for index in cutlass.range_constexpr(5):
        value = fmax(value, cute.arch.shuffle_sync_bfly(value, offset=1 << index))
    return value


@cute.kernel
def _row_max_kernel(
    mX: cute.Tensor,
    mMaximum: cute.Tensor,
    NUM_WARPS: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    lane = tid % 32
    warp = tid // 32

    scratch = cutlass.utils.SmemAllocator().allocate_tensor(
        Float32,
        cute.make_layout(NUM_WARPS),
        byte_alignment=4,
    )
    gX = mX[row, None]
    V = gX.shape[0]
    gX = cute.make_tensor(
        cute.make_ptr(mX.element_type, gX.iterator.toint(), cute.AddressSpace.gmem, assumed_align=16),
        cute.make_layout((V,)),
    )
    VEC = 128 // gX.element_type.width
    gXv = cute.tiled_divide(gX, (VEC,))
    num_vec = V // VEC
    fragment = cute.make_rmem_tensor((VEC,), gX.element_type)
    maximum = Float32(NEG_INF_F32)

    for index in cutlass.range(0, cute.ceil_div(num_vec, _THREADS)):
        vector_index = tid + index * _THREADS
        if vector_index < num_vec:
            cute.autovec_copy(gXv[None, vector_index], fragment)
            value = fragment.load().to(Float32).reduce(cute.ReductionOp.MAX, Float32(NEG_INF_F32), 0)
            maximum = fmax(maximum, value)

    maximum = _warp_max(maximum)
    if lane == 0:
        scratch[warp] = maximum
    cute.arch.barrier()

    maximum = Float32(NEG_INF_F32)
    for index in cutlass.range_constexpr(NUM_WARPS):
        maximum = fmax(maximum, scratch[index])
    if tid == 0:
        mMaximum[row] = maximum


@cute.kernel
def _forward_kernel(
    mX: cute.Tensor,
    mTarget: cute.Tensor,
    mMaximum: cute.Tensor,
    mPredicted: cute.Tensor,
    mSumExp: cute.Tensor,
    vocab_start: Int32,
    ignore_index: Int32,
    NUM_WARPS: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    lane = tid % 32
    warp = tid // 32

    scratch = cutlass.utils.SmemAllocator().allocate_tensor(
        Float32,
        cute.make_layout(NUM_WARPS),
        byte_alignment=4,
    )
    gX = mX[row, None]
    V = gX.shape[0]
    gX = cute.make_tensor(
        cute.make_ptr(mX.element_type, gX.iterator.toint(), cute.AddressSpace.gmem, assumed_align=16),
        cute.make_layout((V,)),
    )
    target = mTarget[row]
    maximum = mMaximum[row].to(Float32)

    if tid == 0:
        predicted = Float32(0.0)
        target_local = target - vocab_start
        if target != ignore_index:
            if target_local >= 0:
                if target_local < V:
                    predicted = gX[target_local].to(Float32) - maximum
        mPredicted[row] = predicted
    cute.arch.barrier()

    VEC = 128 // gX.element_type.width
    gXv = cute.tiled_divide(gX, (VEC,))
    num_vec = V // VEC
    fragment = cute.make_rmem_tensor((VEC,), gX.element_type)
    partial = Float32(0.0)
    is_ignored = target == ignore_index

    for index in cutlass.range(0, cute.ceil_div(num_vec, _THREADS)):
        vector_index = tid + index * _THREADS
        if vector_index < num_vec:
            cute.autovec_copy(gXv[None, vector_index], fragment)
            exponentials = cute.math.exp2((fragment.load().to(Float32) - maximum) * LOG2_E, fastmath=True)
            if is_ignored:
                exponentials = exponentials * Float32(0.0)
            partial = partial + exponentials.reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
            fragment.store(exponentials.to(gX.element_type))
            cute.autovec_copy(fragment, gXv[None, vector_index])

    partial = _warp_sum(partial)
    if lane == 0:
        scratch[warp] = partial
    cute.arch.barrier()

    total = Float32(0.0)
    for index in cutlass.range_constexpr(NUM_WARPS):
        total = total + scratch[index]
    if tid == 0:
        mSumExp[row] = total


@cute.kernel
def _backward_kernel(
    mExp: cute.Tensor,
    mTarget: cute.Tensor,
    mSumExp: cute.Tensor,
    mGradOutput: cute.Tensor,
    vocab_start: Int32,
    ignore_index: Int32,
):
    tid, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    gExp = mExp[row, None]
    V = gExp.shape[0]
    gExp = cute.make_tensor(
        cute.make_ptr(mExp.element_type, gExp.iterator.toint(), cute.AddressSpace.gmem, assumed_align=16),
        cute.make_layout((V,)),
    )
    target = mTarget[row]
    is_ignored = target == ignore_index
    denominator = mSumExp[row].to(Float32)
    upstream = mGradOutput[row].to(Float32)
    if is_ignored:
        denominator = Float32(1.0)
        upstream = Float32(0.0)
    VEC = 128 // gExp.element_type.width
    gExpV = cute.tiled_divide(gExp, (VEC,))
    num_vec = V // VEC
    fragment = cute.make_rmem_tensor((VEC,), gExp.element_type)

    for index in cutlass.range(0, cute.ceil_div(num_vec, _THREADS)):
        vector_index = tid + index * _THREADS
        if vector_index < num_vec:
            cute.autovec_copy(gExpV[None, vector_index], fragment)
            gradient = fragment.load().to(Float32) / denominator * upstream
            fragment.store(gradient.to(gExp.element_type))
            cute.autovec_copy(fragment, gExpV[None, vector_index])

    cute.arch.barrier()
    if tid == 0:
        target_local = target - vocab_start
        if target != ignore_index:
            if target_local >= 0:
                if target_local < V:
                    gExp[target_local] = (gExp[target_local].to(Float32) - upstream).to(gExp.element_type)


@cute.kernel
def _loss_kernel(
    mSumExp: cute.Tensor,
    mPredicted: cute.Tensor,
    mTarget: cute.Tensor,
    mLoss: cute.Tensor,
    ignore_index: Int32,
):
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    index = block * _THREADS + tid
    if index < mLoss.shape[0]:
        loss = cute.math.log2(mSumExp[index].to(Float32), fastmath=True) * LN2 - mPredicted[index].to(Float32)
        if mTarget[index] == ignore_index:
            loss = Float32(0.0)
        mLoss[index] = loss


@cute.jit
def _row_max_host(
    mX: cute.Tensor,
    mMaximum: cute.Tensor,
    NUM_WARPS: cutlass.Constexpr,
    stream: cuda.CUstream = None,
):
    _row_max_kernel(mX, mMaximum, NUM_WARPS).launch(
        grid=[mX.shape[0], 1, 1],
        block=[32 * NUM_WARPS, 1, 1],
        smem=NUM_WARPS * 4,
        stream=stream,
    )


@cute.jit
def _forward_host(
    mX: cute.Tensor,
    mTarget: cute.Tensor,
    mMaximum: cute.Tensor,
    mPredicted: cute.Tensor,
    mSumExp: cute.Tensor,
    vocab_start: Int32,
    ignore_index: Int32,
    NUM_WARPS: cutlass.Constexpr,
    stream: cuda.CUstream = None,
):
    _forward_kernel(
        mX,
        mTarget,
        mMaximum,
        mPredicted,
        mSumExp,
        vocab_start,
        ignore_index,
        NUM_WARPS,
    ).launch(
        grid=[mX.shape[0], 1, 1],
        block=[32 * NUM_WARPS, 1, 1],
        smem=NUM_WARPS * 4,
        stream=stream,
    )


@cute.jit
def _backward_host(
    mExp: cute.Tensor,
    mTarget: cute.Tensor,
    mSumExp: cute.Tensor,
    mGradOutput: cute.Tensor,
    vocab_start: Int32,
    ignore_index: Int32,
    stream: cuda.CUstream = None,
):
    _backward_kernel(
        mExp,
        mTarget,
        mSumExp,
        mGradOutput,
        vocab_start,
        ignore_index,
    ).launch(
        grid=[mExp.shape[0], 1, 1],
        block=[_THREADS, 1, 1],
        stream=stream,
    )


@cute.jit
def _loss_host(
    mSumExp: cute.Tensor,
    mPredicted: cute.Tensor,
    mTarget: cute.Tensor,
    mLoss: cute.Tensor,
    ignore_index: Int32,
    stream: cuda.CUstream = None,
):
    _loss_kernel(mSumExp, mPredicted, mTarget, mLoss, ignore_index).launch(
        grid=[cute.ceil_div(mLoss.shape[0], _THREADS), 1, 1],
        block=[_THREADS, 1, 1],
        stream=stream,
    )


def _tensor(tensor: torch.Tensor, assumed_align: int):
    return to_cute_tensor(tensor, assumed_align=assumed_align)


def row_max(x: torch.Tensor, output: torch.Tensor) -> None:
    stream = _cute_stream()
    x_ct = _tensor(x, 16)
    output_ct = _tensor(output, 4)
    key = ("max", x.device, x.dtype)
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        compiled = cute.compile(_row_max_host, x_ct, output_ct, _NUM_WARPS, stream)
        _COMPILE_CACHE[key] = compiled
    compiled(x_ct, output_ct, stream)


def forward(
    logits: torch.Tensor,
    target: torch.Tensor,
    global_max: torch.Tensor,
    predicted: torch.Tensor,
    sum_exp: torch.Tensor,
    vocab_start: int,
    ignore_index: int,
) -> None:
    stream = _cute_stream()
    args = (
        _tensor(logits, 16),
        _tensor(target, 8),
        _tensor(global_max, 4),
        _tensor(predicted, 4),
        _tensor(sum_exp, 4),
    )
    key = ("forward", logits.device, logits.dtype, target.dtype)
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        compiled = cute.compile(
            _forward_host,
            *args,
            int(vocab_start),
            int(ignore_index),
            _NUM_WARPS,
            stream,
        )
        _COMPILE_CACHE[key] = compiled
    compiled(*args, int(vocab_start), int(ignore_index), stream)


def backward(
    exp_buffer: torch.Tensor,
    target: torch.Tensor,
    global_sum_exp: torch.Tensor,
    grad_output: torch.Tensor,
    vocab_start: int,
    ignore_index: int,
) -> None:
    stream = _cute_stream()
    args = (
        _tensor(exp_buffer, 16),
        _tensor(target, 8),
        _tensor(global_sum_exp, 4),
        _tensor(grad_output, 4),
    )
    key = ("backward", exp_buffer.device, exp_buffer.dtype, target.dtype, grad_output.dtype)
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        compiled = cute.compile(
            _backward_host,
            *args,
            int(vocab_start),
            int(ignore_index),
            stream,
        )
        _COMPILE_CACHE[key] = compiled
    compiled(*args, int(vocab_start), int(ignore_index), stream)


def loss(
    global_sum_exp: torch.Tensor,
    global_predicted: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    output = torch.empty_like(global_sum_exp)
    stream = _cute_stream()
    args = (
        _tensor(global_sum_exp, 4),
        _tensor(global_predicted, 4),
        _tensor(target, 8),
        _tensor(output, 4),
    )
    key = ("loss", output.device, target.dtype)
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        compiled = cute.compile(_loss_host, *args, int(ignore_index), stream)
        _COMPILE_CACHE[key] = compiled
    compiled(*args, int(ignore_index), stream)
    return output
