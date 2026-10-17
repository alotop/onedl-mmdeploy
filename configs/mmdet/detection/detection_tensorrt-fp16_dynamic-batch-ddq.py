_base_ = ['./detection_tensorrt-fp16_dynamic-320x320-1344x1344-ddq.py']

# Dynamic-batch profile: ONE engine serves every batch size in [1, max].
#
# This only works because the two per-image python loops of DDQ-DETR
# (`DDQDETR.pre_decoder` and `DDQTransformerDecoder.select_distinct_queries`)
# are rewritten into batched ops -- see
# `mmdeploy/codebase/mmdet/models/detectors/ddq_detr.py`.  Without those
# rewrites the ONNX trace unrolls the loop once per traced image and the
# batch dimension stops being symbolic.
#
# `dqs_pre_topk` caps the candidates the TRTBatchedNMS plugin sorts before NMS
# (the plugin's own GPU top-k); 0 keeps every candidate (exact semantics).
backend_config = dict(
    common_config=dict(
        max_workspace_size=1 << 30,
        dqs_pre_topk=3000,
        # the NMS plugin now comes from the rewriter itself, there is no ONNX
        # NonMaxSuppression node left for the graph pass to patch
        nms_efficient_keep_topk=0,
        nms_efficient_all=False),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 320, 320],
                    opt_shape=[4, 3, 800, 1344],
                    max_shape=[4, 3, 1344, 1344])))
    ])
