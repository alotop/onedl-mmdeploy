_base_ = ['./detection_tensorrt-fp16_dynamic-320x320-1344x1344.py']

# DDQ-DETR specific TensorRT settings.
#
# DDQ-DETR's two-stage query selection runs `batched_nms` over *all* encoder
# tokens (100x168 + 50x84 + 25x42 + 13x21 = 22323 boxes at 800x1344), and
# TensorRT's ONNX `NonMaxSuppression` implementation costs roughly 1.8 us per
# candidate box, i.e. ~44 ms of a ~150 ms engine.
#
# `nms_pre_topk` keeps only the top-k candidates by score before NMS -- exactly
# what mmdetection's own `multiclass_nms(pre_top_k=...)` does and what DINO
# ships as `nms_pre=3000`.  It is applied as
# k = min(nms_pre_topk, num_candidates) so it stays valid for every shape of
# the dynamic profile, and it is a no-op for the five decoder-side NMS calls
# whose candidate set never exceeds num_queries (900).
#
# Measured on DDQ-DETR R50-4scale @ 800x1344, fp16: total NMS time
# 46.6 ms -> 23.7 ms (k=3000) -> 15.8 ms (k=1350).
#
# `nms_efficient_keep_topk` goes one step further and hands the *pre-decoder*
# NMS to mmdeploy's `TRTBatchedNMS` (EfficientNMS) plugin, which runs its own
# GPU top-k (`nms_efficient_topk`, capped at 5120 by the plugin's
# `ASSERT(t_size <= 10)` with BS=512) and then NMS.  Standalone measurements at
# 22323 candidates: ONNX NonMaxSuppression 36.6 ms, pre-topk(3000)+ONNX NMS
# ~12 ms, TRTBatchedNMS(topk=3000, keep=900) 6.9 ms.  The plugin's index output
# references the original candidate array, so it is a drop-in replacement.
# The five decoder-side NMS calls stay on the ONNX path: their keep-index
# *count* is semantic (it builds the distinct-query mask) and the plugin pads
# unused slots with -1.
backend_config = dict(
    common_config=dict(
        max_workspace_size=1 << 30,
        nms_pre_topk=3000,
        nms_efficient_keep_topk=900,
        nms_efficient_topk=3000))
