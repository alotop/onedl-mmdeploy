_base_ = ['./detection_tensorrt_dynamic-320x320-1344x1344.py']

# fp32 counterpart of
# `detection_tensorrt-fp16_dynamic-320x320-1344x1344-ddq.py`.
#
# Notes for the FP32 case:
#   * `native_grid_sample` is left off on purpose -- the native GridSample
#     layer is a *regression* in an FP32 engine (184 -> 238 ms measured), so
#     the mmdeploy plugin stays.  With the MSDA plugin in place there are no
#     grid_sampler nodes left anyway.
#   * `msda_fp32` is meaningless here: an FP32 engine runs the plugin in FP32
#     regardless (`getOutputDataType`/`supportsFormatCombination` follow
#     input 0, which is FP32).
#   * The fp32 base config uses max_workspace_size=1<<31 (2 GiB); a 1 GiB pool
#     is not enough -- a tactic for the GridSample-fused region asks
#     for 1.4 GB.
backend_config = dict(
    common_config=dict(
        max_workspace_size=1 << 31,
        nms_efficient_keep_topk=900,
        nms_efficient_topk=3000))
