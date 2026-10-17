# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.utils.misc.padding_to', backend='tensorrt')
def padding_to__tensorrt(input_tensor, max_len=300):
    """Zero-pad `input_tensor` to a *provably static* length `max_len`.

    Background (DDQ-DETR batch>1 export):

    `mmdet.models.utils.misc.padding_to` is called by `align_tensor`, which
    `DDQDETR.pre_decoder` uses to stack the per-image query-selection results
    (`torch.stack([padding_to(item, topk) for item in inputs])`).

    The original implementation does

        torch.cat([x, x.new_zeros(max_len - len(x))], dim=0)

    which leaves the padded length a *data-dependent expression* -- for DDQ
    the item length is `min(topk, <NMS keep count of that image>)`.  When such
    tensors are stacked across images, ONNX/TensorRT require the operands to
    have identical shapes; the shape machine cannot prove
    `min(topk, n0) + (topk - min(topk, n0)) == min(topk, n1) + (topk - ...)`
    so it emits a runtime equality check, which fails as soon as two images
    keep different numbers of boxes (e.g. `CHECK_EQUAL 542 != 900`).  Batch=1
    graphs have a single stack operand and are unaffected.

    This rewriter builds the same tensor with a static shape instead:

        x_ext = cat([x, zeros(max_len)], dim=0)     # length = len(x) + K
        out   = x_ext[arange(max_len)]              # gather -> (K, ...)

    The gather output shape depends only on the *index* tensor (constant
    `arange(K)`), so it is `(max_len, *x.shape[1:])` by construction, while
    the values are identical to the original padding: entries `k < len(x)`
    take `x[k]`, the rest fall into the appended zero block.

    Assumes `len(input_tensor) <= max_len` (the original crashes on the
    opposite case: `new_zeros` with a negative size).
    """
    ctx = FUNCTION_REWRITER.get_context()  # noqa: F841
    if max_len is None:
        return input_tensor
    x = input_tensor
    K = int(max_len)
    zero_block = x.new_zeros((K, *x.shape[1:]))
    x_ext = torch.cat([x, zero_block], dim=0)
    idx = torch.arange(K, device=x.device, dtype=torch.long)
    return x_ext[idx]
