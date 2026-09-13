# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.utils import get_root_logger


def _run_optimize_pass(logger, name, func, *args):
    """Run one optional ts_optimizer pass.

    These passes come from the experimental `mmdeploy_torchscript_ops`
    extension.  Skip a failing pass instead of aborting the whole ONNX export:
    a stale/incompatible build of that extension (for example one compiled
    against another torch version, which makes every pass raise
    `TypeError: incompatible function arguments`) must not break deployment.
    """
    try:
        func(*args)
    except Exception as e:  # noqa: BLE001
        logger.warning(f'Skip onnx optimize pass `{name}` '
                       f'({type(e).__name__}: {str(e).splitlines()[0]})')


def optimize_onnx(ctx, graph, params_dict, torch_out):
    """The optimize callback of the onnx model."""
    logger = get_root_logger()
    logger.info('Execute onnx optimize passes.')
    try:
        from mmdeploy.backend.torchscript import ts_optimizer
    except ImportError:
        logger.warning(
            'Can not optimize model, please build torchscipt extension.\n'
            'More details: '
            'https://github.com/vbti-development/onedl-mmdeploy/tree/main/docs/en/experimental/onnx_optimizer.md'  # noqa
        )
        return graph, params_dict, torch_out

    _run_optimize_pass(logger, '_jit_pass_merge_shape_concate',
                       ts_optimizer.onnx._jit_pass_merge_shape_concate, graph)
    _run_optimize_pass(logger, '_jit_pass_onnx_peephole',
                       ts_optimizer.onnx._jit_pass_onnx_peephole, graph)
    _run_optimize_pass(logger, '_jit_pass_flatten_cls_head',
                       ts_optimizer.onnx._jit_pass_flatten_cls_head, graph)
    _run_optimize_pass(logger, '_jit_pass_fuse_select_assign',
                       ts_optimizer.onnx._jit_pass_fuse_select_assign, graph,
                       params_dict)
    _run_optimize_pass(logger, '_jit_pass_common_subgraph_elimination',
                       ts_optimizer.onnx._jit_pass_common_subgraph_elimination,
                       graph, params_dict)
    return graph, params_dict, torch_out
