"""CuTe DSL Fused-Linear-Cross-Entropy (Blackwell sm_100).

Two pieces live here:

1. The fully-fused FORWARD loss kernel (``fused_ce_loss_lse``): streams the vocabulary in
   V-blocks, GEMMs each (BLOCK_M, BLOCK_V) logits tile with tcgen05 UMMA into TMEM, and reduces
   online — the (chunk, V) logits tile NEVER touches HBM. Flash-decoding V-split fills the SMs;
   a small torch combine merges per-split partials into the loss and the log-sum-exp (LSE).
   Core path only (loss + ignore_index), bf16, Blackwell.

2. The FLCE wrapper (``fused_linear_cross_entropy_forward`` + ``LigerFusedLinearCrossEntropyFunction``),
   signature-compatible with the Triton FLCE. The core config dispatches to the fused kernel
   above (+ a chunked cuBLAS backward that recomputes dlogits from the saved LSE); everything
   else (softcap / label-smoothing / class-weight / z-loss / bias / fp16 / non-Blackwell) falls
   back to the chunked path built on the CuTe DSL CE kernel.
"""

import math

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch

from cutlass import Float32
from cutlass import const_expr
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack
from packaging.version import Version

from liger_kernel.ops.cutedsl.ops.cross_entropy import _launch_ce_fwd
from liger_kernel.ops.cutedsl.ops.utils import _next_power_of_2
from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd

# =============================================================================
# Fused forward loss kernel (logits never touch HBM)
# =============================================================================
acc_dtype = cutlass.Float32  # accumulation / reduction dtype (io dtype is derived per-call)

BLOCK_M = 128  # token tile == threads/CTA (one row per thread in the epilogue)
BLOCK_V = 256  # vocab (N) tile streamed for online softmax
BLOCK_K = 64  # H reduction tile
mma_inst_shape_mnk = (128, 256, 16)
mma_tiler_mnk = (BLOCK_M, BLOCK_V, BLOCK_K)
threads_per_cta = BLOCK_M

ab_stages = 2
acc_stage = 1

LOG2_E = 1.4426950408889634
NEG_INF = -1.0e38
IGNORE_INDEX = -100


@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stage * 2]
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mX_mkl: cute.Tensor,  # (BT, H, 1)  A operand (K-major)
    tma_atom_b: cute.CopyAtom,
    mW_nkl: cute.Tensor,  # (V, H, 1)   B operand (K-major)
    mTarget: cute.Tensor,  # (BT,)  int64
    mBias: cute.Tensor,  # (V_real,)  logit bias added per-column before softcap; used only if HAS_BIAS
    mCeW: cute.Tensor,  # (V_real,) fp32 class weights; used only if HAS_SMOOTHW (weighted smoothing sum)
    mMpart: cute.Tensor,  # (BT, num_splits) fp32  per-split running max
    mDpart: cute.Tensor,  # (BT, num_splits) fp32  per-split running sum-exp (base-2)
    mXtpart: cute.Tensor,  # (BT, num_splits) fp32  per-split captured target logit (0 if none)
    mSXpart: cute.Tensor,  # (BT, num_splits) fp32  per-split sum of (capped) logits (label smoothing)
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
    softcap: Float32,  # logit soft-cap threshold; used only if HAS_SOFTCAP
    io_dtype: cutlass.Constexpr,  # bf16 or fp16 (operand/staging dtype)
    HAS_SOFTCAP: cutlass.Constexpr,  # apply softcap*tanh(x/softcap) to logits before the reduction
    HAS_BIAS: cutlass.Constexpr,  # add mBias[col] to each logit before softcap
    NEED_SUMX: cutlass.Constexpr,  # accumulate sum of (capped) logits per row (label smoothing)
    HAS_SMOOTHW: cutlass.Constexpr,  # weight the sum_x accumulation by mCeW[col] (weighted smoothing)
    vbs_per_split: cutlass.Constexpr,  # V-blocks this CTA streams
    V_real: cutlass.Constexpr,  # true vocab; columns >= V_real are zero-pad -> masked
    H: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    m_tile, split, _ = cute.arch.block_idx()  # grid = (num_m_tiles, num_splits, 1)

    # ---- smem ----
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA = smem.allocate_tensor(io_dtype, a_smem_layout.outer, 128, swizzle=a_smem_layout.inner)
    sB = smem.allocate_tensor(io_dtype, b_smem_layout.outer, 128, swizzle=b_smem_layout.inner)
    # (BLOCK_M, BLOCK_V) fp32 staging tile for the per-row softmax reduction
    sTile = smem.allocate_tensor(acc_dtype, cute.make_layout((BLOCK_M, BLOCK_V)), 128)

    # ---- TMEM ----
    tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=threads_per_cta)
    tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_alloc_barrier)
    tmem.allocate(512)

    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)

    num_tma_bytes = cute.size_in_bytes(io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2])) + cute.size_in_bytes(
        io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2])
    )
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=ab_stages,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        tx_count=num_tma_bytes,
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
    ).make_participants()
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=acc_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, threads_per_cta),
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
    ).make_participants()

    # A operand is independent of the V-block (proj drops N), but we recompute its tile per
    # V-block from the shared (m_tile, v, None) coord to mirror the tutorial's local_tile usage.
    thr_mma = tiled_mma.get_slice(0)
    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)

    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(acc_dtype)
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

    # TMEM -> RMEM -> SMEM epilogue, mirroring fp16_gemm_0.py but targeting smem so each
    # thread can then read its own token row. The smem staging tile is partitioned as the
    # MMA-C accumulator (via partition_C), sub-tiled for ILP, exactly like the tutorial's gC.
    tCsC = thr_mma.partition_C(sTile)  # MMA-C view over the (BLOCK_M, BLOCK_V) smem tile
    # Ld32x32b.x64 loads 64 fp32/thread, so each epi subtile must be (BLOCK_M, 64):
    # BLOCK_M(=128 threads) * 64 = BLOCK_M * epi_width -> epi_width = 64 -> subtile_cnt = BLOCK_V/64.
    subtile_cnt = BLOCK_V // 64
    epi_tiler = ((cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1]) // subtile_cnt),)
    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
    sC_epi = cute.zipped_divide(tCsC, epi_tiler)

    tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), acc_dtype)
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)
    tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)  # (Cpy, NumCpy, NumTiles)
    tDsC = tmem_thr_copy.partition_D(sC_epi)  # (Cpy, NumCpy, NumTiles)
    tCrAcc = cute.make_rmem_tensor(tDsC[None, None, 0].shape, acc_dtype)

    num_k_tiles: cutlass.Constexpr = H // BLOCK_K

    # per-thread (== per token row) online-softmax running state
    row_max = acc_dtype(NEG_INF)
    row_sum = acc_dtype(0.0)
    x_tgt = acc_dtype(0.0)
    sum_x = acc_dtype(0.0)  # sum of (capped) logits over real columns (label smoothing)
    global_row = m_tile * BLOCK_M + tidx
    tgt_i = cutlass.Int32(mTarget[global_row])

    # ---- stream this split's V-blocks ----
    for v_local in cutlass.range(vbs_per_split):
        v = split * vbs_per_split + v_local  # global V-block index
        gA = cute.local_tile(mX_mkl, mma_tiler_mnk, (m_tile, v, None), proj=(1, None, 1))
        gB = cute.local_tile(mW_nkl, mma_tiler_mnk, (m_tile, v, None), proj=(None, 1, 1))
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        if warp_idx == 0:
            acc_empty = acc_producer.acquire_and_advance()
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k in cutlass.range(num_k_tiles):
                ab_empty = ab_producer.acquire_and_advance()
                # index the gmem source by the LOCAL k-tile (tAgA/tBgB are recomputed per
                # V-block with num_k_tiles entries); the pipeline .index drives the smem ring.
                cute.copy(tma_atom_a, tAgA[(None, k)], tAsA[(None, ab_empty.index)], tma_bar_ptr=ab_empty.barrier)
                cute.copy(tma_atom_b, tBgB[(None, k)], tBsB[(None, ab_empty.index)], tma_bar_ptr=ab_empty.barrier)
                ab_full = ab_consumer.wait_and_advance()
                for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                    coord = (None, None, kb, ab_full.index)
                    cute.gemm(tiled_mma, tCtAcc, tCrA[coord], tCrB[coord], tCtAcc)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                ab_full.release()
            acc_empty.commit()

        # epilogue: pull the logits tile into smem (TMEM->RMEM->SMEM), then reduce per row
        acc_full = acc_consumer.wait_and_advance()
        for i in cutlass.range_constexpr(cute.size(tDtC, mode=[2])):
            cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)
            cute.autovec_copy(tCrAcc, tDsC[None, None, i])
        acc_full.release()
        cute.arch.barrier()

        # one thread == one row: SINGLE-PASS online softmax over this block's BLOCK_V columns.
        # Read columns rotated by tidx (c+tidx & (BLOCK_V-1)) so the 128 threads hit distinct
        # smem banks each step instead of all colliding on bank (c % 32) -> conflict-free.
        # Fold the target-logit capture in branchlessly (exactly one global col == tgt).
        base_col = v * BLOCK_V
        lm = acc_dtype(NEG_INF)  # this tile's running max
        ld = acc_dtype(0.0)  # this tile's running sum-exp (base-2)
        for c in cutlass.range_constexpr(BLOCK_V):
            cc = (c + tidx) & (BLOCK_V - 1)
            gcol = base_col + cc
            # zero-padded columns (gcol >= V_real) must not affect softmax: force them to -inf
            # so exp2 -> 0 and fmax ignores them. Targets are always < V_real, so no false capture.
            is_pad = acc_dtype(gcol >= V_real)
            not_pad = acc_dtype(1.0) - is_pad
            raw = sTile[tidx, cc]
            if const_expr(HAS_BIAS or HAS_SMOOTHW):
                # branchless in-bounds index for the (V_real,) side vectors (bias / ce_weight):
                # padded cols read index 0, then get masked out (contribution × 1-is_pad). Only
                # computed when a side-vector is actually read (keeps the core loop ALU-lean).
                pad_i = cutlass.Int32(gcol >= V_real)
                gcol_safe = gcol * (cutlass.Int32(1) - pad_i)
            if const_expr(HAS_BIAS):
                # add the per-column logit bias BEFORE softcap (matches CE: logits = X@W^T + bias).
                raw = raw + mBias[gcol_safe].to(acc_dtype) * not_pad
            if const_expr(HAS_SOFTCAP):
                # softcap*tanh(x/softcap), applied per-logit before the softmax (matches CE kernel).
                raw = softcap * cute.math.tanh(raw / softcap, fastmath=True)
            val = raw * not_pad + NEG_INF * is_pad
            nm = cute.arch.fmax(lm, val)
            ld = ld * cute.math.exp2((lm - nm) * LOG2_E, fastmath=True) + cute.math.exp2(
                (val - nm) * LOG2_E, fastmath=True
            )
            lm = nm
            x_tgt = x_tgt + val * acc_dtype(gcol == tgt_i)
            if const_expr(NEED_SUMX):
                # sum of (capped) logits over REAL columns only (pad cols excluded). For WEIGHTED
                # label smoothing (HAS_SMOOTHW) accumulate sum_v ce_weight[v]*x_capped_v instead.
                sw = acc_dtype(1.0)
                if const_expr(HAS_SMOOTHW):
                    sw = mCeW[gcol_safe].to(acc_dtype)
                sum_x = sum_x + raw * not_pad * sw
        # merge this tile's (lm, ld) into the running (row_max, row_sum)
        m_new = cute.arch.fmax(row_max, lm)
        row_sum = row_sum * cute.math.exp2((row_max - m_new) * LOG2_E, fastmath=True) + ld * cute.math.exp2(
            (lm - m_new) * LOG2_E, fastmath=True
        )
        row_max = m_new
        cute.arch.barrier()

    tmem.relinquish_alloc_permit()

    # Write this split's partial online-softmax stats; a tiny torch combine merges splits into
    # the final loss (LSE = M + log(sum_s d_s*exp(m_s-M)); x_tgt = sum_s xt_s; loss=LSE-x_tgt).
    mMpart[global_row, split] = row_max
    mDpart[global_row, split] = row_sum
    mXtpart[global_row, split] = x_tgt
    if const_expr(NEED_SUMX):
        mSXpart[global_row, split] = sum_x

    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.jit
def host_function(
    mX: cute.Tensor,
    mW: cute.Tensor,
    mTarget: cute.Tensor,
    mBias: cute.Tensor,
    mCeW: cute.Tensor,
    mMpart: cute.Tensor,
    mDpart: cute.Tensor,
    mXtpart: cute.Tensor,
    mSXpart: cute.Tensor,
    softcap: Float32,
    HAS_SOFTCAP: cutlass.Constexpr,
    HAS_BIAS: cutlass.Constexpr,
    NEED_SUMX: cutlass.Constexpr,
    HAS_SMOOTHW: cutlass.Constexpr,
    vbs_per_split: cutlass.Constexpr,
    num_splits: cutlass.Constexpr,
    V_real: cutlass.Constexpr,
    H: cutlass.Constexpr,
):
    op = tcgen05.MmaF16BF16Op(
        mX.element_type,
        acc_dtype,
        mma_inst_shape_mnk,
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        cute.nvgpu.OperandMajorMode.K,
        cute.nvgpu.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(op)
    a_smem_layout = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, mX.element_type, ab_stages)
    b_smem_layout = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, mW.element_type, ab_stages)
    a_one = cute.select(a_smem_layout, mode=[0, 1, 2])
    b_one = cute.select(b_smem_layout, mode=[0, 1, 2])

    op_tma = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(op_tma, mX, a_one, mma_tiler_mnk, tiled_mma)
    b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(op_tma, mW, b_one, mma_tiler_mnk, tiled_mma)

    BT = mX.layout.shape[0]
    grid = (cute.ceil_div(BT, BLOCK_M), num_splits, 1)
    kernel(
        tiled_mma,
        a_tma_atom,
        a_tma_tensor,
        b_tma_atom,
        b_tma_tensor,
        mTarget,
        mBias,
        mCeW,
        mMpart,
        mDpart,
        mXtpart,
        mSXpart,
        a_smem_layout,
        b_smem_layout,
        softcap,
        mX.element_type,
        HAS_SOFTCAP,
        HAS_BIAS,
        NEED_SUMX,
        HAS_SMOOTHW,
        vbs_per_split,
        V_real,
        H,
    ).launch(grid=grid, block=(threads_per_cta, 1, 1))


# =============================================================================
# Public entry point
# =============================================================================
NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count if torch.cuda.is_available() else 148

_fwd_compile_cache = {}


def _pick_splits(num_m_tiles, num_v_blocks, target_waves=None):
    """num_splits ~ target_waves*SMs/num_m_tiles. Any count allowed (W is zero-padded), so we
    are not limited to divisors of num_v_blocks. target_waves scales with sqrt(num_v_blocks):
    larger vocab -> more per-CTA V-work / latency to hide -> split more aggressively."""
    if target_waves is None:
        target_waves = 4.0 * max(1.0, math.sqrt(num_v_blocks / 125.0))
    want = max(1, round(target_waves * NUM_SMS / num_m_tiles))
    return min(want, num_v_blocks)


def fused_ce_loss_lse(X, W, target, ignore_index=-100, softcap=None, need_sumx=False, bias=None, smooth_weight=None):
    """Fused linear + cross-entropy forward (loss only), logits never materialized in HBM.

    Args:
        X: (BT, H) bf16/fp16 hidden states, K-major (H contiguous).
        W: (V, H) bf16/fp16 classifier weight, K-major (H contiguous).
        target: (BT,) int64 class indices; ``ignore_index`` rows contribute 0 loss.
        softcap: optional logit soft-cap; if set, logits become ``softcap*tanh(logit/softcap)``
            (applied per-logit before the softmax, matching the CE kernel).
        need_sumx: if True, also return the per-row sum of (capped) logits over the real vocab
            (used by label smoothing).
        bias: optional (V,) logit bias added per-column BEFORE softcap (matches CE:
            ``logits = X@Wᵀ + bias``).
    Returns:
        (loss_row, lse, x_tgt, sum_x): all fp32 (BT,). ``loss_row`` is the UNnormalized per-row
        CE (LSE - x_target, 0 on ignored rows); ``lse`` is logsumexp per row (for the backward);
        ``x_tgt`` is the (capped) target logit; ``sum_x`` is the per-row sum of capped logits
        (zeros if ``need_sumx`` is False).
    Requires BLOCK_M | BT and BLOCK_K | H. V may be arbitrary (W is zero-padded internally).
    """
    assert X.is_cuda and W.is_cuda, "fused_ce_loss_lse requires CUDA tensors"
    BT, H = X.shape
    V = W.shape[0]
    assert H % BLOCK_K == 0, f"H ({H}) must be a multiple of BLOCK_K ({BLOCK_K})"
    Xd = X.detach()
    Wd = W.detach()
    dev = X.device

    num_v_blocks = (V + BLOCK_V - 1) // BLOCK_V
    # M-tail: BT need NOT be a multiple of BLOCK_M. We launch ceil(BT/BLOCK_M) token tiles; the last
    # tile's out-of-range rows are handled by (a) TMA zero-filling the OOB X rows (harmless logits)
    # and (b) padding the target + per-split output buffers to BT_pad so the kernel's unconditional
    # reads/writes stay in bounds. The padded rows carry ignore_index and are sliced off before the
    # loss combine — so the reduction stays branch-free (no per-row guard on the common path).
    num_m_tiles = (BT + BLOCK_M - 1) // BLOCK_M
    BT_pad = num_m_tiles * BLOCK_M
    num_splits = _pick_splits(num_m_tiles, num_v_blocks)
    vbs = (num_v_blocks + num_splits - 1) // num_splits
    Vpad = num_splits * vbs * BLOCK_V
    if Vpad != V:
        Wpad = torch.zeros(Vpad, H, device=dev, dtype=Wd.dtype)
        Wpad[:V] = Wd
        Wk = Wpad
    else:
        Wk = Wd

    m_part = torch.full((BT_pad, num_splits), NEG_INF, device=dev, dtype=torch.float32)
    d_part = torch.zeros(BT_pad, num_splits, device=dev, dtype=torch.float32)
    xt_part = torch.zeros(BT_pad, num_splits, device=dev, dtype=torch.float32)
    # sum-of-logits partial (label smoothing). Allocate a real (BT_pad, num_splits) buffer only when
    # needed; otherwise a tiny (BT_pad, 1) dummy so the kernel arg is always a valid tensor.
    sx_part = (
        torch.zeros(BT_pad, num_splits, device=dev, dtype=torch.float32)
        if need_sumx
        else torch.zeros(BT_pad, 1, device=dev, dtype=torch.float32)
    )
    # target padded to BT_pad with ignore_index so the last tile's OOB rows read a valid (ignored)
    # target. No copy on the common (BT % BLOCK_M == 0) path.
    if BT_pad != BT:
        target_k = torch.full((BT_pad,), ignore_index, device=dev, dtype=target.dtype)
        target_k[:BT] = target
    else:
        target_k = target

    mX = (
        from_dlpack(Xd, assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=BLOCK_K)
    )
    mW = (
        from_dlpack(Wk, assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=BLOCK_K)
    )
    mT = from_dlpack(target_k, assumed_align=8).mark_layout_dynamic(leading_dim=0)
    mM = from_dlpack(m_part, assumed_align=4).mark_layout_dynamic(leading_dim=1)
    mD = from_dlpack(d_part, assumed_align=4).mark_layout_dynamic(leading_dim=1)
    mXt = from_dlpack(xt_part, assumed_align=4).mark_layout_dynamic(leading_dim=1)
    mSX = from_dlpack(sx_part, assumed_align=4).mark_layout_dynamic(leading_dim=1)
    # bias (V,) added per-column before softcap. fp32 for precision (the fused logits are already
    # fp32, more precise than Triton's dtype logits, so an fp32 bias add stays within tolerance).
    # A 1-elem fp32 dummy when absent (never read: the kernel gates on HAS_BIAS at compile time).
    has_bias = bias is not None
    bias_t = (
        bias.detach().to(torch.float32).contiguous() if has_bias else torch.zeros(1, device=dev, dtype=torch.float32)
    )
    mB = from_dlpack(bias_t, assumed_align=4).mark_layout_dynamic(leading_dim=0)
    # class-weight vector for WEIGHTED label smoothing (sum_v ce_weight[v]*x_capped_v). fp32,
    # length V_real. A 1-elem dummy when unused (kernel gates on HAS_SMOOTHW at compile time).
    has_smoothw = need_sumx and smooth_weight is not None
    smoothw_t = (
        smooth_weight.detach().to(torch.float32).contiguous()
        if has_smoothw
        else torch.zeros(1, device=dev, dtype=torch.float32)
    )
    mCeW = from_dlpack(smoothw_t, assumed_align=4).mark_layout_dynamic(leading_dim=0)

    has_softcap = softcap is not None
    softcap_f = float(softcap) if has_softcap else 0.0
    key = (Xd.dtype, Wk.dtype, H, num_v_blocks, num_splits, has_softcap, has_bias, need_sumx, has_smoothw)
    compiled = _fwd_compile_cache.get(key)
    if compiled is None:
        compiled = cute.compile(
            host_function,
            mX,
            mW,
            mT,
            mB,
            mCeW,
            mM,
            mD,
            mXt,
            mSX,
            softcap_f,
            has_softcap,
            has_bias,
            need_sumx,
            has_smoothw,
            vbs,
            num_splits,
            V,
            H,
        )
        _fwd_compile_cache[key] = compiled
    compiled(mX, mW, mT, mB, mCeW, mM, mD, mXt, mSX, softcap_f)

    # Slice off the padded tail rows ([BT:BT_pad] carry ignored garbage) before the combine.
    m_part = m_part[:BT]
    d_part = d_part[:BT]
    xt_part = xt_part[:BT]
    sx_part = sx_part[:BT]
    M = m_part.max(dim=1).values
    D = (d_part * torch.exp2((m_part - M.unsqueeze(1)) * LOG2_E)).sum(dim=1)
    lse = M + torch.log(D)
    x_tgt = xt_part.sum(dim=1)
    loss_row = (lse - x_tgt) * (target != ignore_index).to(torch.float32)
    sum_x = sx_part.sum(dim=1) if need_sumx else torch.zeros(BT, device=dev, dtype=torch.float32)
    return loss_row, lse, x_tgt, sum_x


# grad_weight accumulation uses torch.addmm(..., out_dtype=fp32, out=grad_weight) to accumulate
# bf16/fp16 operands into an fp32 buffer without a [V, H] fp32 temp — identical to the upstream
# Triton FLCE (PR #1239). out_dtype was added to torch.addmm in torch 2.8.0; earlier versions fall
# back to mm().float(). Kept byte-for-byte in sync with the Triton path so the ONLY difference
# between the two FLCE backends is the CE kernel.
_TORCH_VERSION = Version(torch.__version__.split("+")[0])
_ADDMM_SUPPORTS_OUT_DTYPE = _TORCH_VERSION >= Version("2.8.0")

_UNSUPPORTED = "cutedsl FLCE: {feat} is not supported."


def _cdiv(a, b):
    return (a + b - 1) // b


def _accum_grad_weight(grad_weight, dlogits_t, xc):
    """grad_weight += dlogits_tᵀ-view @ xc, mirroring the upstream Triton FLCE accumulation EXACTLY
    (PR #1239): when grad_weight is fp32 use torch.addmm(out_dtype=fp32) to fold the bf16/fp16
    operands into the fp32 buffer with NO transient [V, H] fp32 temp; otherwise (bf16/fp16
    grad_weight, the default when accum_dtype is None) accumulate in the weight dtype. Keeping this
    identical to Triton means the two FLCE backends differ ONLY in the CE/forward kernel — and the
    grad_weight buffer is the weight dtype by default, so peak memory matches Triton."""
    if (
        _ADDMM_SUPPORTS_OUT_DTYPE
        and grad_weight.device.type == "cuda"
        and grad_weight.dtype == torch.float32
        and dlogits_t.dtype in (torch.float16, torch.bfloat16)
    ):
        # addmm's out_dtype path doesn't participate in autocast operand casting, so under AMP
        # (fp32 params) xc can stay fp32 while dlogits is the autocast dtype; align dtypes first.
        if xc.dtype != dlogits_t.dtype:
            xc = xc.to(dlogits_t.dtype)
        torch.addmm(grad_weight, dlogits_t, xc, out_dtype=torch.float32, out=grad_weight)
    else:
        grad_weight += (dlogits_t @ xc).to(grad_weight.dtype)


def _fused_fwd_supported(_input, weight):
    """The fused forward kernel targets Blackwell (sm_100) with bf16/fp16 operands and H a multiple
    of BLOCK_K. BT is arbitrary (the M-tail is TMA-zero-filled + target/output padded to BLOCK_M).
    fp32 / non-Blackwell / H%BLOCK_K≠0 fall back to the chunked path (any GPU / dtype)."""
    if not (_input.is_cuda and _input.dtype in (torch.bfloat16, torch.float16)):
        return False
    if weight.dtype != _input.dtype:
        return False
    BT, H = _input.shape
    if H % BLOCK_K != 0:
        return False
    try:
        major = torch.cuda.get_device_capability(_input.device)[0]
    except Exception:
        return False
    return major >= 10  # Blackwell sm_100+


# Crossover heuristic: the fused forward wins when the GPU is NOT already saturated by the token
# tiles (so its flash-decoding V-split buys occupancy) AND the vocab is large enough that avoiding
# Triton's memory-bound CE pass over a materialized (chunk, V) tile pays off. Past the crossover the
# token tiles saturate the SMs, the fused forward's ~24%-of-peak MMA (1 CTA/SM, CtaGroup.ONE) becomes
# pure overhead vs the shared-logits chunked path (materialize once via cuBLAS, reuse for loss+grad —
# no recompute), and the fused path loses (measured 0.77x @ BT32768, 0.58x @ BT65536, V=128k). Both
# paths are already implemented + tested, so we just pick the winner by shape.
#
# Empirically (B200, H=4096) the crossover in num_m_tiles (= ceil(BT/BLOCK_M)) scales ~linearly with
# V and the SM count: crossover_m_tiles ≈ C * V * NUM_SMS. C is ~7x larger with grad (the backward
# recompute amortizes the fused forward) than without. Tuned to the fwd+bwd / no-grad BT sweeps:
#   grad:    fused if BT < ~V/5    (V=128256 -> ~24.6k;  V=32000 -> ~6.4k;  V=262144 -> ~50k)
#   no-grad: fused if BT < ~V/35   (fused has no backward to amortize its slow forward)
_CROSSOVER_C_GRAD = 1.0e-5
_CROSSOVER_C_NOGRAD = 1.5e-6


def _fused_beats_chunked(BT, V, requires_grad):
    """True when the fully-fused path is expected to beat the chunked (materialize-and-reuse) path
    for this shape; else the caller should fall through to the chunked path. Heuristic, Blackwell/
    H≈4096-tuned — conservative near the boundary (the two are within ~10% there anyway); the point
    is to avoid the far-side cliff at very large BT, not to nail the exact crossover."""
    num_m_tiles = _cdiv(BT, BLOCK_M)
    c = _CROSSOVER_C_GRAD if requires_grad else _CROSSOVER_C_NOGRAD
    return num_m_tiles < c * V * NUM_SMS


def _fused_forward_core(_input, weight, target, ignore_index, reduction, accum_dtype, bias=None):
    """Lean core-path FLCE (no CE features beyond an optional logit ``bias``): the fully-fused
    forward kernel (loss + LSE, logits never in HBM) + a minimal chunked cuBLAS backward (self-
    consistent softmax, single scatter for the target one-hot). Kept separate from the general
    ``_fused_forward`` because the extra feature machinery (per-element coef/correction/mask passes)
    measurably slows the common last-layer case. Eager-grad; returns (loss, None, None, None,
    grad_input, grad_weight, grad_bias)."""
    BT, H = _input.shape
    V = weight.shape[0]
    input_requires_grad = _input.requires_grad

    loss_row, lse, _x_tgt, _sum_x = fused_ce_loss_lse(_input, weight, target, ignore_index=ignore_index, bias=bias)

    valid = target != ignore_index
    n_non_ignore = int(valid.sum().item())
    inv_n = 1.0 / n_non_ignore if (reduction == "mean" and n_non_ignore > 0) else 1.0

    if reduction == "none":
        loss = loss_row
    else:
        loss = loss_row.sum() * (inv_n if reduction == "mean" else 1.0)

    if not input_requires_grad:
        return loss, None, None, None, None, None, None

    Xd, Wd = _input.detach(), weight.detach()
    bias_f = bias.detach().float() if bias is not None else None
    # grad_weight / grad_bias dtype: accum_dtype when given, else the param dtype — matching Triton
    # (torch.zeros_like(weight)). The param-dtype default keeps peak memory on par with Triton (an
    # fp32 (V, H) grad_weight would nearly double the classifier-grad footprint).
    gw_dtype = accum_dtype if accum_dtype is not None else weight.dtype
    grad_input = torch.empty_like(Xd)
    grad_weight = torch.zeros(weight.shape, device=weight.device, dtype=gw_dtype)
    grad_bias = None
    if bias is not None:
        gb_dtype = accum_dtype if accum_dtype is not None else bias.dtype
        grad_bias = torch.zeros(V, device=weight.device, dtype=gb_dtype)
    # Chunk sizing identical to the upstream Triton FLCE (inc_factor = ceil(V/H); chunk_size =
    # next_pow2(ceil(BT/inc_factor))) so the transient (chunk, V) softmax tile matches the (BT, H)
    # input footprint and peak memory stays on par with Triton (apples-to-apples).
    inc_factor = _cdiv(V, H)
    chunk = _next_power_of_2(_cdiv(BT, inc_factor))
    chunk = max(1, min(chunk, BT))
    for s in range(0, BT, chunk):
        e = min(s + chunk, BT)
        xc = Xd[s:e]
        tc = target[s:e]
        vmask = tc != ignore_index
        logits = (xc @ Wd.t()).float()
        if bias_f is not None:
            logits = logits + bias_f
        # Self-consistent softmax: recompute LSE from THESE (cuBLAS) logits rather than reusing the
        # fused-forward (CuTe/tcgen05) LSE — pairing the two GEMMs' outputs is inconsistent and, for
        # peaked softmax (large logits), blows up the gradient. The accurate fused LSE still feeds
        # the reported loss.
        lse_bwd = torch.logsumexp(logits, dim=1, keepdim=True)
        p = torch.exp(logits - lse_bwd)
        tgt_clamped = torch.where(vmask, tc, torch.zeros_like(tc))
        p.scatter_add_(
            1,
            tgt_clamped.unsqueeze(1),
            torch.where(vmask, -1.0, 0.0).to(p.dtype).unsqueeze(1),
        )
        p *= inv_n
        p[~vmask] = 0.0
        p = p.to(Xd.dtype)
        grad_input[s:e] = p @ Wd
        _accum_grad_weight(grad_weight, p.t(), xc)
        if grad_bias is not None:
            grad_bias += p.sum(dim=0).to(grad_bias.dtype)
    grad_weight = grad_weight.to(weight.dtype)
    if grad_bias is not None:
        grad_bias = grad_bias.to(bias.dtype)
    return loss, None, None, None, grad_input, grad_weight, grad_bias


def _fused_forward(
    _input,
    weight,
    target,
    ignore_index,
    reduction,
    accum_dtype,
    bias=None,
    ce_weight=None,
    lse_square_scale=0.0,
    label_smoothing=0.0,
    softcap=None,
    return_z_loss=False,
    use_token_scaling=False,
):
    """Feature-complete FLCE via the fully-fused forward kernel (loss/LSE/x_target/Σx-capped, logits
    never in HBM) + a chunked cuBLAS backward that recomputes dlogits with every CE feature folded
    in (softcap chain rule, label-smoothing additive term, z_loss factor, class-weight scaling,
    token scaling, bias). The per-row loss is assembled from the kernel stats mirroring the CuTe DSL
    CE host formulas (cross_entropy.py); the backward mirrors its in-place gradient. Eager-grad,
    returns (loss, z_loss, token_accuracy=None, predicted_tokens=None, grad_input, grad_weight,
    grad_bias). token_accuracy / predicted_tokens (argmax) are NOT produced here — configs that
    request them fall to the chunked path."""
    BT, H = _input.shape
    V = weight.shape[0]
    dev = _input.device
    input_requires_grad = _input.requires_grad
    has_smoothing = label_smoothing != 0.0
    has_weight = ce_weight is not None
    has_zloss = lse_square_scale != 0.0 or return_z_loss

    if has_weight:
        ce_weight = ce_weight.to(torch.float32)
        if ce_weight.stride(-1) != 1:
            ce_weight = ce_weight.contiguous()

    # ---- fused forward: per-row stats (lse, capped target logit, Σ (w·)x_capped) ----
    # Weighted smoothing accumulates Σ ce_weight[v]·x_capped_v; unweighted accumulates Σ x_capped_v.
    smooth_weight = ce_weight if (has_smoothing and has_weight) else None
    _lr, lse, x_tgt, sum_x = fused_ce_loss_lse(
        _input,
        weight,
        target,
        ignore_index=ignore_index,
        softcap=softcap,
        need_sumx=has_smoothing,
        bias=bias,
        smooth_weight=smooth_weight,
    )

    target_mask = target != ignore_index
    fmask = target_mask.to(torch.float32)
    total_n_non_ignore = int(target_mask.sum().item())

    # per-row class weight w_eff (1.0 when unweighted), the weighted normalizer, and the full
    # class-weight sum (weighted smoothing term) — mirrors the chunked path / standalone CE.
    if has_weight:
        tgt_safe = torch.where(target_mask, target, torch.zeros_like(target))
        w_eff = torch.where(target_mask, ce_weight[tgt_safe], torch.zeros_like(fmask))
        total_sum_non_ignore_ce_weight = float(w_eff.sum().item())
        ce_weight_sum = float(ce_weight.sum().item())
    else:
        w_eff = torch.ones(BT, device=dev, dtype=torch.float32)
        total_sum_non_ignore_ce_weight = float(total_n_non_ignore)
        ce_weight_sum = 0.0

    if reduction == "mean" and total_n_non_ignore > 0:
        if has_weight and total_sum_non_ignore_ce_weight > 0:
            inv_n_loss = 1.0 / total_sum_non_ignore_ce_weight
        else:
            inv_n_loss = 1.0 / total_n_non_ignore
        inv_n_z = 1.0 / total_n_non_ignore
    else:
        inv_n_loss = 1.0
        inv_n_z = 1.0

    # ---- per-row loss assembly (fp32), mirroring the CE host formulas ----
    main = (lse - x_tgt) * w_eff
    if has_smoothing:
        eps = label_smoothing / V
        g_sxs = -eps * sum_x
        if has_weight:
            smooth_loss = g_sxs + eps * lse * ce_weight_sum
        else:
            smooth_loss = g_sxs + label_smoothing * lse
        main = main * (1.0 - label_smoothing) + smooth_loss
    loss_row = main * inv_n_loss
    zl_row = None
    if has_zloss:
        zl_row = lse_square_scale * lse * lse * inv_n_z
        loss_row = loss_row + zl_row
    loss_row = loss_row * fmask
    if zl_row is not None:
        zl_row = zl_row * fmask

    # token scaling: per-row detached softmax prob at the (capped) target = exp(x_tgt - lse).
    # Computed once from the fused stats so the forward loss and backward grad share it exactly.
    scaling = None
    if use_token_scaling:
        scaling = (torch.exp(x_tgt - lse) * fmask).detach()
        loss_row = loss_row * scaling
        if zl_row is not None:
            zl_row = zl_row * scaling

    if reduction == "none":
        loss = loss_row
        z_loss = zl_row if return_z_loss else None
    else:
        loss = loss_row.sum()
        z_loss = zl_row.sum() if (return_z_loss and zl_row is not None) else None

    if not input_requires_grad:
        return loss, z_loss, None, None, None, None, None

    # ---- chunked cuBLAS backward: recompute logits, feature-correct dlogits ----
    Xd, Wd = _input.detach(), weight.detach()
    bias_f = bias.detach().float() if bias is not None else None
    # grad_weight / grad_bias default to the param dtype (accum_dtype when given) so peak memory
    # matches Triton — an fp32 (V, H) grad_weight would nearly double the classifier-grad footprint.
    gw_dtype = accum_dtype if accum_dtype is not None else weight.dtype
    grad_input = torch.empty_like(Xd)
    grad_weight = torch.zeros(weight.shape, device=dev, dtype=gw_dtype)
    grad_bias = None
    if bias is not None:
        gb_dtype = accum_dtype if accum_dtype is not None else bias.dtype
        grad_bias = torch.zeros(V, device=dev, dtype=gb_dtype)
    eps = (label_smoothing / V) if has_smoothing else 0.0
    # Chunk sizing identical to the upstream Triton FLCE (inc_factor = ceil(V/H); chunk_size =
    # next_pow2(ceil(BT/inc_factor))) so the transient (chunk, V) tensors match the (BT, H) input
    # footprint and peak memory stays on par with Triton (apples-to-apples), even though the feature
    # backward holds a few live (chunk, V) fp32 buffers (logits/capped, tanh, softmax, gradient).
    inc_factor = _cdiv(V, H)
    chunk = _next_power_of_2(_cdiv(BT, inc_factor))
    chunk = max(1, min(chunk, BT))
    for s in range(0, BT, chunk):
        e = min(s + chunk, BT)
        xc = Xd[s:e]
        tc = target[s:e]
        vmask = tc != ignore_index
        fm = vmask.to(torch.float32)
        tc_safe = torch.where(vmask, tc, torch.zeros_like(tc))
        logits = (xc @ Wd.t()).float()
        if bias_f is not None:
            logits = logits + bias_f
        if softcap is not None:
            tnh = torch.tanh(logits / softcap)
            capped = softcap * tnh
        else:
            capped = logits
        # self-consistent softmax (recompute LSE from these logits — see note in the core path).
        lse_bwd = torch.logsumexp(capped, dim=1, keepdim=True)
        softmax = torch.exp(capped - lse_bwd)

        # softmax-proportional coefficient (dloss + z_loss), zeroed on ignored rows.
        if has_weight:
            weff_c = torch.where(vmask, ce_weight[tc_safe], torch.zeros_like(fm))
            coef = (1.0 - label_smoothing) * weff_c * inv_n_loss
        else:
            weff_c = None
            coef = torch.full((e - s,), inv_n_loss, device=dev, dtype=torch.float32)
        if has_zloss:
            coef = coef + (2.0 * lse_square_scale * lse_bwd.squeeze(1)) * inv_n_z
        coef = coef * fm
        g = softmax * coef.unsqueeze(1)

        # additive label-smoothing term (not proportional to softmax).
        if has_smoothing:
            eps_g = (eps * inv_n_loss) * fm
            if has_weight:
                g = g + (softmax * ce_weight_sum - ce_weight.unsqueeze(0)) * eps_g.unsqueeze(1)
            else:
                g = g - eps_g.unsqueeze(1)

        # softcap chain rule applies to the full per-element gradient.
        if softcap is not None:
            g = g * (1.0 - tnh * tnh)

        # -(1-ls)·w_eff·inv_n_loss correction at the (non-ignored) target column.
        dxy = -(1.0 - label_smoothing) * (weff_c if has_weight else torch.ones_like(fm)) * inv_n_loss
        if softcap is not None:
            t_y = capped.gather(1, tc_safe.unsqueeze(1)).squeeze(1) / softcap
            dxy = dxy * (1.0 - t_y * t_y)
        ar_local = torch.arange(e - s, device=dev)
        g[ar_local, tc_safe] = g[ar_local, tc_safe] + dxy * fm

        # token scaling multiplies the final per-row gradient (matches the chunked path).
        if use_token_scaling:
            g = g * scaling[s:e].unsqueeze(1)

        g = g * fm.unsqueeze(1)  # ignored rows contribute no gradient
        g = g.to(Xd.dtype)
        grad_input[s:e] = g @ Wd
        _accum_grad_weight(grad_weight, g.t(), xc)
        if grad_bias is not None:
            grad_bias += g.sum(dim=0).to(grad_bias.dtype)

    grad_weight = grad_weight.to(weight.dtype)
    if grad_bias is not None:
        grad_bias = grad_bias.to(bias.dtype)
    return loss, z_loss, None, None, grad_input, grad_weight, grad_bias


# =============================================================================
# Forward
# =============================================================================
def fused_linear_cross_entropy_forward(
    _input,
    weight,
    target,
    ce_weight=None,
    bias=None,
    ignore_index=-100,
    lse_square_scale=0.0,
    label_smoothing=0.0,
    reduction="mean",
    softcap=None,
    return_z_loss=False,
    accum_dtype=None,
    use_token_scaling=False,
    return_token_accuracy=False,
    return_predicted_tokens=False,
):
    """CuTe DSL FLCE forward.

    Returns (loss, z_loss, token_accuracy, predicted_tokens, grad_input,
    grad_weight, grad_bias). Matches
    ``liger_kernel.ops.fused_linear_cross_entropy.fused_linear_cross_entropy_forward``.
    """
    # Every CE feature (ce_weight / softcap / label_smoothing / z_loss / token_accuracy /
    # predicted_tokens) is plumbed straight through to the CuTe DSL CE kernel below, which
    # already implements each one (validated by the standalone CE suite). token_scaling is
    # an FLCE-level transform (pure torch, around the kernel). The ONE feature the fused
    # design genuinely cannot support is reduction='none' WITH grad (guarded just below).
    assert reduction in ("mean", "sum", "none"), f"Unsupported reduction: {reduction}"

    device = _input.device
    input_requires_grad = _input.requires_grad
    if reduction == "none" and input_requires_grad:
        # The fused design accumulates grad_weight/grad_bias over tokens during the
        # forward pass, so backward can't re-weight them by a per-token upstream grad.
        # Refuse loudly rather than crash (the (BT,) grad_output mis-broadcasts against
        # (BT, H)) or silently return a wrong grad_weight (the Triton path scales every
        # grad by the scalar grad_output[0], itself incorrect for per-token 'none').
        raise NotImplementedError(_UNSUPPORTED.format(feat="reduction='none' with grad"))

    # ---- Fast path: fully-fused forward loss kernel (logits never touch HBM) -------------
    # Fires on Blackwell for bf16/fp16 with the tile-divisibility the kernel needs. Handles bias,
    # softcap, class weights, z_loss, label smoothing (weighted + unweighted) and token scaling —
    # the fused forward produces the per-row stats and the chunked cuBLAS backward recomputes the
    # feature-correct dlogits. token_accuracy / predicted_tokens fall through to the chunked path:
    # they reduce to a per-row argmax and predicted_tokens is compared bit-exactly to Triton, whose
    # argmax runs on bf16 logits — the fused kernel's fp32 logits flip near-ties, so exact parity
    # needs the bf16 chunked path. Other unsupported configs (fp32 / non-Blackwell / odd shapes /
    # reduction='none' with grad, handled above) also fall through.
    _fused_ok = not return_token_accuracy and not return_predicted_tokens
    if (
        _fused_ok
        and _fused_fwd_supported(_input, weight)
        and _fused_beats_chunked(_input.shape[0], weight.shape[0], input_requires_grad)
    ):
        _core_only = (
            ce_weight is None
            and softcap is None
            and not return_z_loss
            and lse_square_scale == 0.0
            and label_smoothing == 0.0
            and not use_token_scaling
        )
        if _core_only:
            # Lean core+bias path (minimal backward) — the common last-layer CE case.
            return _fused_forward_core(_input, weight, target, ignore_index, reduction, accum_dtype, bias=bias)
        return _fused_forward(
            _input,
            weight,
            target,
            ignore_index,
            reduction,
            accum_dtype,
            bias=bias,
            ce_weight=ce_weight,
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
            softcap=softcap,
            return_z_loss=return_z_loss,
            use_token_scaling=use_token_scaling,
        )

    # inputs: (BT, H); per-chunk materialized logits: (chunk_size, V).
    BT, H = _input.shape
    V = weight.shape[0]
    # Mirror the CE kernel's divisibility contract (cross_entropy_forward): 128-bit
    # vectorized loads need V % vec == 0, vec = 16 // element_size (8 bf16 / 4 fp32). The
    # CE kernel predicates its 256-thread tail, so no stronger multiple is required. Fail
    # fast here with a dtype-aware message instead of letting the inner CE assert fire
    # mid-loop. (vec from _input.dtype == the logits dtype on the common path.)
    vec = 16 // _input.element_size()
    assert V % vec == 0, (
        f"cutedsl FLCE requires V % {vec} == 0 for {_input.dtype} logits "
        f"(the CE kernel's 128-bit vectorized loads); got V={V}."
    )

    # ---- chunk sizing (memory-minimal, identical to the upstream Triton FLCE) ----
    # Partition the BT tokens so the transient (chunk_size, V) logit tile matches the
    # (BT, H) input footprint: inc_factor = ceil(V/H), chunk_size = next_pow2(ceil(BT/inc_factor)).
    # This is the conservative upstream rule — no free-memory-dependent chunk growth — so the
    # chunk count (hence grad accumulation order) is deterministic and the peak transient is
    # bounded by construction. Keeps the CuTe DSL FLCE apples-to-apples with the Triton path.
    inc_factor = _cdiv(V, H)
    chunk_size = _next_power_of_2(_cdiv(BT, inc_factor))
    num_chunks = _cdiv(BT, chunk_size)

    grad_input = torch.empty_like(_input, device=device)  # fully overwritten per-chunk below

    # fp32 (or accum_dtype) accumulators for the weight / bias gradients.
    if input_requires_grad:
        gw_dtype = accum_dtype if accum_dtype is not None else weight.dtype
        grad_weight = torch.zeros_like(weight, dtype=gw_dtype, device=device) if weight.requires_grad else None
        if bias is not None:
            gb_dtype = accum_dtype if accum_dtype is not None else bias.dtype
            grad_bias = torch.zeros_like(bias, dtype=gb_dtype, device=device)
        else:
            grad_bias = None
    else:
        grad_weight = None
        grad_bias = None

    # fp32 loss accumulator, matching the Triton path exactly. Safe now that the CE kernel's
    # compile cache keys on loss.dtype: FLCE's fp32 loss and the standalone CE's input-dtype
    # loss compile to separate kernels instead of colliding.
    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)
    # Aux per-row buffers (one slice handed to each chunk's CE launch, written in place).
    #   z_loss: fp32 (NOT _input.dtype like Triton) -> z matches the fp32 loss buffer, so the
    #     CE kernel never hits an untested loss.dtype != z.dtype compile combo, and the summed
    #     z_loss is more accurate. Within the bf16 z_loss tolerance vs the Triton oracle.
    #   token_accuracy: fp32 per-row 1.0/0.0; predicted_tokens: int64 argmax, -1 for ignored.
    z_loss_1d = torch.zeros(BT, dtype=torch.float32, device=device) if return_z_loss else None
    token_accuracy_1d = torch.zeros(BT, dtype=torch.float32, device=device) if return_token_accuracy else None
    predicted_tokens_1d = torch.full((BT,), -1, dtype=torch.int64, device=device) if return_predicted_tokens else None

    # Global non-ignored token count -> ONE normalizer applied to EVERY chunk, so each chunk's
    # loss/grad come out already mean-normalized (matches Triton, which passes the totals to
    # every per-chunk CE launch).
    target_mask = target != ignore_index
    total_n_non_ignore = target_mask.sum().item()
    assert (target * target_mask).max() < V, f"Target out of bounds. Expected < {V}"
    assert (target * target_mask).min() >= 0, "Target out of bounds. Expected >= 0"

    # Class weight: the mean denominator becomes the summed weight of non-ignored targets
    # (sum_non_ignore_weight) instead of the count; weight_sum (full-vector sum) feeds the
    # weighted label-smoothing term. The kernel reads weight as fp32 -> upcast here (exact
    # for fp32 weights). Mirrors the standalone CE forward.
    total_sum_non_ignore_ce_weight = float(total_n_non_ignore)
    ce_weight_sum = 0.0
    if ce_weight is not None:
        assert ce_weight.shape[0] == V, f"ce_weight must be a Tensor of size V={V}. Got: {tuple(ce_weight.shape)}"
        assert torch.is_floating_point(ce_weight), f"ce_weight must be floating point. Got: {ce_weight.dtype}"
        ce_weight = ce_weight.to(torch.float32)
        if ce_weight.stride(-1) != 1:
            ce_weight = ce_weight.contiguous()
        total_sum_non_ignore_ce_weight = torch.gather(ce_weight, 0, target.masked_select(target_mask)).sum().item()
        ce_weight_sum = ce_weight.sum().item()

    # mean -> 1/N per-row in-kernel; sum/none -> 1.0 (unnormalized); 1.0 when all-ignored
    # (avoids /0). The main loss/grad normalize by sum_non_ignore_weight when weighted; z_loss
    # is never weight-scaled, so it always uses the plain non-ignored count.
    if reduction == "mean" and total_n_non_ignore > 0:
        if ce_weight is not None and total_sum_non_ignore_ce_weight > 0:
            inv_n_loss = 1.0 / total_sum_non_ignore_ce_weight
        else:
            inv_n_loss = 1.0 / total_n_non_ignore
        inv_n_z = 1.0 / total_n_non_ignore
    else:
        inv_n_loss = 1.0
        inv_n_z = 1.0

    if target.stride(-1) != 1:
        target = target.contiguous()

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        _input_chunk = _input[start_idx:end_idx]  # (chunk, H)

        # logits in the original precision (cuBLAS), exactly like the Triton path.
        logits_chunk = _input_chunk @ weight.t()  # (chunk, V)
        if bias is not None:
            # in-place add avoids a second (chunk, V) temp; fall back to out-of-place
            # when dtypes differ (autocast: bf16 matmul + fp32 bias).
            if logits_chunk.dtype == bias.dtype:
                logits_chunk += bias
            else:
                logits_chunk = logits_chunk + bias

        target_chunk = target[start_idx:end_idx]  # (chunk,)

        # Token scaling: detached softmax prob of the target token, computed on the
        # (softcapped) logits BEFORE the CE kernel overwrites logits_chunk with the gradient.
        # Ignored rows get a 0 factor. Pure FLCE-level transform (the kernel is unaware).
        if use_token_scaling:
            logits_for_softmax = logits_chunk.detach().clone()
            if softcap is not None:
                logits_for_softmax = softcap * torch.tanh(logits_for_softmax / softcap)
            probs = torch.softmax(logits_for_softmax, dim=-1)
            valid_mask = target_chunk != ignore_index
            pred_probs = torch.zeros_like(target_chunk, dtype=probs.dtype)
            if valid_mask.any():
                valid_targets = target_chunk[valid_mask]
                pred_probs[valid_mask] = torch.gather(probs[valid_mask], -1, valid_targets.unsqueeze(-1)).squeeze(-1)
            scaling_factors = pred_probs.detach()  # (chunk,)

        loss_1d_slice = loss_1d[start_idx:end_idx]  # (chunk,), fp32
        z_loss_1d_slice = z_loss_1d[start_idx:end_idx] if return_z_loss else None
        token_accuracy_1d_slice = token_accuracy_1d[start_idx:end_idx] if return_token_accuracy else None
        predicted_tokens_1d_slice = predicted_tokens_1d[start_idx:end_idx] if return_predicted_tokens else None

        # CE kernel needs the row contiguous (it slices mX[row, None]); target 1D contiguous.
        logits_chunk = logits_chunk.contiguous()
        target_chunk = target_chunk.contiguous()

        # CuTe DSL CE kernel: per-row loss (+ z_loss / token_accuracy / predicted_tokens) and
        # the in-place gradient over logits_chunk. has_grad gates the gradient pass (mirrors
        # Triton's HAS_GRADIENTS=input_requires_grad). Every advanced feature is a pass-through:
        # the kernel bakes the flags into its compile key and implements each term itself.
        _launch_ce_fwd(
            logits_chunk,
            target_chunk,
            loss_1d_slice,
            inv_n_loss,
            ignore_index,
            input_requires_grad,
            lse_square_scale,
            z_loss_1d_slice,
            return_z_loss,
            softcap,
            label_smoothing=label_smoothing,
            weight=ce_weight,
            weight_sum=ce_weight_sum,
            return_token_accuracy=return_token_accuracy,
            return_predicted_tokens=return_predicted_tokens,
            token_acc_out=token_accuracy_1d_slice,
            pred_tok_out=predicted_tokens_1d_slice,
            inv_n_z=inv_n_z,
        )

        # Apply token scaling to the per-row loss / z_loss (out-of-place -> write back).
        if use_token_scaling:
            loss_1d[start_idx:end_idx] = loss_1d_slice * scaling_factors
            if return_z_loss:
                z_loss_1d[start_idx:end_idx] = z_loss_1d_slice * scaling_factors

        grad_logits_chunk = logits_chunk  # (chunk, V): the in-place CE gradient
        # ... and to the gradient, so grad_input/grad_weight reflect the scaled loss.
        if use_token_scaling:
            grad_logits_chunk = grad_logits_chunk * scaling_factors.unsqueeze(-1)

        if input_requires_grad:
            grad_input[start_idx:end_idx] = grad_logits_chunk @ weight

        if grad_weight is not None:
            # Mirror the upstream Triton FLCE grad_weight accumulation EXACTLY (PR #1239): use
            # torch.addmm(out_dtype=fp32) to accumulate bf16/fp16 operands into an fp32 grad_weight
            # without materializing a [V, H] fp32 temp; otherwise the original mm().float(). Keeping
            # this identical to the Triton path means the ONLY difference between the two FLCE
            # backends is the CE kernel.
            grad_logits_t = grad_logits_chunk.t()
            if (
                _ADDMM_SUPPORTS_OUT_DTYPE
                and grad_weight.device.type == "cuda"
                and grad_weight.dtype == torch.float32
                and grad_logits_t.dtype in (torch.float16, torch.bfloat16)
            ):
                # Unlike torch.mm, torch.addmm's out_dtype path does not participate in
                # autocast operand casting, so under AMP (fp32 params, no bias) _input_chunk
                # can stay fp32 while grad_logits is the autocast dtype. addmm requires mat1
                # and mat2 to share a dtype, so align _input_chunk before accumulating.
                input_chunk = _input_chunk
                if input_chunk.dtype != grad_logits_t.dtype:
                    input_chunk = input_chunk.to(grad_logits_t.dtype)
                torch.addmm(
                    grad_weight,
                    grad_logits_t,
                    input_chunk,
                    out_dtype=torch.float32,
                    out=grad_weight,
                )
            else:
                grad_weight += torch.mm(grad_logits_chunk.t(), _input_chunk).float()

        if grad_bias is not None:
            torch.add(
                input=grad_bias,
                other=grad_logits_chunk.sum(dim=0),
                out=grad_bias,
                alpha=1.0,
            )

    # Reduce the per-row buffers. reduction='none' returns the per-token loss vector; mean/sum
    # sum it (the per-row 1/N normalizer already applied the mean). token_accuracy always
    # reduces to the mean over non-ignored tokens for mean/sum (matches Triton + standalone CE);
    # predicted_tokens is always the per-row vector. (none+grad is refused above, so the 'none'
    # branch here is forward-only.)
    if reduction == "none":
        loss = loss_1d
        z_loss = z_loss_1d if return_z_loss else None
        token_accuracy = token_accuracy_1d if return_token_accuracy else None
    else:
        loss = torch.sum(loss_1d)
        z_loss = torch.sum(z_loss_1d) if return_z_loss else None
        token_accuracy = torch.sum(token_accuracy_1d) / total_n_non_ignore if return_token_accuracy else None
    predicted_tokens = predicted_tokens_1d if return_predicted_tokens else None

    # Cast accumulators back to the parameter dtype.
    grad_weight = grad_weight.to(weight.dtype) if grad_weight is not None else None
    grad_bias = grad_bias.to(bias.dtype) if grad_bias is not None else None

    return loss, z_loss, token_accuracy, predicted_tokens, grad_input, grad_weight, grad_bias


# =============================================================================
# Backward
# =============================================================================
def fused_linear_cross_entropy_backward(grad_output, grad_input, grad_weight, grad_bias):
    """Scale the saved grads by ``grad_output`` (chain rule from upstream)."""
    # FLCE is usually the last layer -> grad_output == 1.0; skip the scaling.
    if not torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        # reduction='none'+grad is refused in forward, so grad_output is a scalar here.
        # Cast each product back to its tensor's dtype: the summed loss may be a
        # higher-precision scalar, so multiplying would otherwise promote e.g. bf16 grads
        # to fp32 and autograd would reject the dtype mismatch against the bf16 inputs.
        # Fresh tensors (not in-place) also avoid the autograd anomalies the Triton path
        # sidesteps with its custom element_mul kernel (which preserves dtype in place).
        grad_input = (grad_input * grad_output).to(grad_input.dtype)
        if grad_weight is not None:
            grad_weight = (grad_weight * grad_output).to(grad_weight.dtype)
        if grad_bias is not None:
            grad_bias = (grad_bias * grad_output).to(grad_bias.dtype)
    return grad_input, grad_weight, grad_bias


class LigerFusedLinearCrossEntropyFunction(torch.autograd.Function):
    """
    CuTe DSL autograd wrapper for Fused-Linear-Cross-Entropy.

    Signature-compatible with
    ``liger_kernel.ops.fused_linear_cross_entropy.LigerFusedLinearCrossEntropyFunction``.
    """

    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        _input,
        weight,
        target,
        bias=None,
        ce_weight=None,
        ignore_index=-100,
        lse_square_scale=0.0,
        label_smoothing=0.0,
        reduction="mean",
        softcap=None,
        return_z_loss: bool = False,
        accum_dtype=None,
        use_token_scaling: bool = False,
        return_token_accuracy: bool = False,
        return_predicted_tokens: bool = False,
    ):
        # Memory-minimal chunking bounds the transient (chunk_size, V) logit tile to the
        # (BT, H) input footprint by construction, so no OOM-retry / chunk-growth fallback is
        # needed — call the forward directly (matches the upstream Triton FLCE control flow).
        loss, z_loss, token_accuracy, predicted_tokens, grad_input, grad_weight, grad_bias = (
            fused_linear_cross_entropy_forward(
                _input=_input,
                weight=weight,
                target=target,
                bias=bias,
                ce_weight=ce_weight,
                ignore_index=ignore_index,
                lse_square_scale=lse_square_scale,
                label_smoothing=label_smoothing,
                reduction=reduction,
                softcap=softcap,
                return_z_loss=return_z_loss,
                accum_dtype=accum_dtype,
                use_token_scaling=use_token_scaling,
                return_token_accuracy=return_token_accuracy,
                return_predicted_tokens=return_predicted_tokens,
            )
        )

        ctx.save_for_backward(
            grad_input.detach() if grad_input is not None else None,
            grad_weight.detach() if grad_weight is not None else None,
            grad_bias.detach() if grad_bias is not None else None,
        )
        ctx.return_z_loss = return_z_loss
        ctx.return_token_accuracy = return_token_accuracy
        ctx.return_predicted_tokens = return_predicted_tokens
        return loss, z_loss, token_accuracy, predicted_tokens

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output, grad_output2, grad_output3, grad_output4):
        if ctx.return_z_loss:
            del grad_output2  # z_loss is only for logging
        if ctx.return_token_accuracy:
            del grad_output3  # token_accuracy is only for metrics
        if ctx.return_predicted_tokens:
            del grad_output4  # predicted_tokens is only for metrics
        (grad_input, grad_weight, grad_bias) = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_backward(
            grad_output, grad_input, grad_weight, grad_bias
        )
        return (
            grad_input,
            grad_weight,
            None,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,  # use_token_scaling
            None,  # return_token_accuracy
            None,  # return_predicted_tokens
        )
