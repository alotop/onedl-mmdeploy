# Copyright (c) OpenMMLab. All rights reserved.
import logging

import torch
from torch import Tensor

from mmdeploy.core import FUNCTION_REWRITER, SYMBOLIC_REWRITER
from mmdeploy.utils import Backend, get_common_config, get_root_logger

logger = logging.getLogger(__name__)


@SYMBOLIC_REWRITER.register_symbolic(
    'mmcv.ops.multi_scale_deform_attn.MultiScaleDeformableAttnFunction')
def ms_deform_attn_default(
    g,
    value,
    value_spatial_shapes,
    value_level_start_index,
    sampling_locations,
    attention_weights,
    im2col_step=64,
):
    """Rewrite msda symbolic function for all backend."""
    return g.op(
        'mmdeploy::MMCVMultiScaleDeformableAttention',
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step_i=im2col_step,
    )


class TRTMSDeformAttnOp(torch.autograd.Function):
    """Emit `mmdeploy::MMCVMultiScaleDeformableAttention` during ONNX export.

    The plugin takes `value_level_start_index` explicitly; mmcv's PyTorch
    fallback does not, so it is derived and carried as an extra argument.

    `forward` returns a placeholder of the right shape rather than calling the
    real implementation: `FunctionRewriter.enter` patches the target function
    by replacing *every* reference to it (`_replace_all_obj`), so calling
    `multi_scale_deformable_attn_pytorch` from here would re-enter this
    rewriter and recurse until the export ran out of memory.  This mirrors
    mmdeploy's own `TRTBatchedNMSop`, whose `forward` also just returns a
    dummy.  The real computation is the plugin node emitted by `symbolic`.
    """

    @staticmethod
    def forward(ctx, value: Tensor, value_spatial_shapes: Tensor,
                value_level_start_index: Tensor, sampling_locations: Tensor,
                attention_weights: Tensor) -> Tensor:
        return value.new_zeros(value.shape[0], sampling_locations.shape[1],
                               value.shape[2] * value.shape[3])

    @staticmethod
    def symbolic(g, value, value_spatial_shapes, value_level_start_index,
                 sampling_locations, attention_weights):
        return g.op('mmdeploy::MMCVMultiScaleDeformableAttention', value,
                    value_spatial_shapes, value_level_start_index,
                    sampling_locations, attention_weights)


def _level_start_index(value_spatial_shapes: Tensor) -> Tensor:
    """Prefix sum of H*W per level, as int32 (the plugin requires kINT32).

    Built from Gather / Mul / Add / Concat only -- all first-class TensorRT
    layers.  The first implementation used `hw[:i].sum()`, which TensorRT's
    Myelin backend could not compile:

        Error Code: 9: Skipping tactic ... Operation
        '/encoder/layers_0/self_attn/ReduceSum' uses tensor
        '/encoder/layers_0/self_attn/Slice_output_0' which has not been defined

    (it fused the shape derivation into a foreign node together with the
    plugin).  The number of levels is static, so a small Python loop over
    `index_select` is both clearer and TRT-friendly.
    """
    shp = value_spatial_shapes
    one = torch.tensor([0], device=shp.device)
    h = torch.index_select(shp, 1, one)
    w = torch.index_select(shp, 1, one + 1)
    hw = (h * w).reshape(-1)  # (L,)

    def at(i):
        return torch.index_select(hw, 0, torch.tensor([i], device=hw.device))

    n_levels = int(hw.shape[0])
    # `at(0) * 0` and not `hw * 0`: the accumulator must stay a length-1
    # tensor, otherwise it broadcasts against the (1,) gathers and the
    # concatenation comes out n_levels times too long.
    acc = at(0) * 0
    parts = []
    for i in range(n_levels):
        if i > 0:
            acc = acc + at(i - 1)
        parts.append(acc)
    return torch.cat(parts).to(torch.int32)


@FUNCTION_REWRITER.register_rewriter(
    'mmcv.ops.multi_scale_deform_attn.multi_scale_deformable_attn_pytorch',
    backend=Backend.TENSORRT.value)
def multi_scale_deformable_attn_pytorch__tensorrt(
        value: Tensor, value_spatial_shapes: Tensor,
        sampling_locations: Tensor, attention_weights: Tensor) -> Tensor:
    """Route multi-scale deformable attention to mmdeploy's TensorRT plugin.

    Why
    ---
    mmdeploy's ONNX export runs the model on **CPU** (`patched_model.cpu()` in
    `apis/onnx/export.py`), so mmcv's `MultiScaleDeformableAttention.forward`
    takes the `multi_scale_deformable_attn_pytorch` branch: four
    `F.grid_sample` calls per layer, which mmdeploy exports as its
    **FLOAT-only** `grid_sampler` plugin.  DDQ-DETR has 48 (4-scale) / 60
    (5-scale) of those, i.e. a quarter of the engine that `--fp16` cannot
    touch, plus the attention-weight/rescale chains around them.

    mmdeploy ships `MMCVMultiScaleDeformableAttention` (a port of
    TensorRT-OSS's multiscaleDeformableAttnPlugin, **kHALF-capable**) but
    nothing in the mmdet path ever reaches it, because the symbolic registered
    for it targets `MultiScaleDeformableAttnFunction` -- the CUDA op, which
    never runs during a CPU export.  This rewriter closes that gap.

    The two implementations are numerically equivalent: the plugin's
    kernel uses `h = loc_h * spatial_h - 0.5` with bilinear + zeros outside,
    which is exactly the `align_corners=False` grid_sample mapping of the
    PyTorch fallback (verified bit-exact in `local/test_msda_plugin.py`).
    """
    ctx = FUNCTION_REWRITER.get_context()
    if not get_common_config(ctx.cfg).get('ms_deform_attn_plugin', True):
        return ctx.origin_func(value, value_spatial_shapes, sampling_locations,
                               attention_weights)
    try:
        dtype = value.dtype
        return TRTMSDeformAttnOp.apply(
            value,
            value_spatial_shapes.to(torch.int32),
            _level_start_index(value_spatial_shapes),
            sampling_locations.to(dtype),
            attention_weights.to(dtype),
        )
    except Exception:  # noqa: BLE001
        get_root_logger().warning(
            'ms_deform_attn plugin rewrite failed; falling back to the '
            'PyTorch implementation',
            exc_info=True)
        return ctx.origin_func(value, value_spatial_shapes, sampling_locations,
                               attention_weights)
