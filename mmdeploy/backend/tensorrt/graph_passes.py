# Copyright (c) OpenMMLab. All rights reserved.
"""ONNX graph passes applied only when converting to TensorRT.

These run *after* the (valid) ONNX file has been written and only on the
in-memory model handed to the TensorRT builder, so `end2end.onnx` itself stays
spec-compliant for onnxruntime / onnxsim / other backends.

Two passes, both measured on DDQ-DETR R50-4scale @ 800x1344 (RTX 5070 Ti):

1. `use_native_grid_sample` -- mmdeploy exports `F.grid_sample` as its
   `mmdeploy::grid_sampler` plugin (`pytorch/symbolics/grid_sampler.py`), and
   that plugin is **FLOAT only**
   (`csrc/.../grid_sampler/trt_grid_sampler.cpp: supportsFormatCombination`
   rejects kHALF).  DDQ-DETR has 48 of these layers (6 encoder + 6 decoder
   layers x 4 feature levels); they are ~26% of the engine and completely
   unaffected by `--fp16`.  TensorRT's own GridSample layer is FP16 capable and
   much faster: 151.5 -> 127.0 ms (-16%).
   Caveat: only useful for FP16 engines -- in an FP32 engine the native layer
   is *slower* than the plugin (184 -> 238 ms), because its tactic gets fused
   into a bad giant kernel.  Hence the `fp16_only` gate.

2. `nms_pre_topk` -- `NonMaxSuppression` cost in TensorRT is ~1.8 us per
   candidate box (nearly linear, effectively serial), and DDQ-DETR feeds it
   **22323** encoder tokens.  Keeping only the top-k candidates by score first
   (exactly what mmdetection's own `multiclass_nms(pre_top_k=...)` and DINO's
   `nms_pre=3000` do) takes that layer from 46.6 -> ~24 ms at k=3000 and
   -> ~16 ms at k=1350.  The bound is applied as
   `k = min(pre_topk, num_candidates)` so it stays valid for every shape in a
   dynamic profile, and it is a no-op for the five decoder-side NMS calls whose
   candidate set already never exceeds `num_queries`.
"""
import logging

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

logger = logging.getLogger(__name__)

INTERP_MODE = {0: 'bilinear', 1: 'nearest', 2: 'bicubic'}
PADDING_MODE = {0: 'zeros', 1: 'border', 2: 'reflection'}


def _to_model(onnx_model):
    if isinstance(onnx_model, str):
        return onnx.load(onnx_model)
    return onnx_model


def _toposort(graph):
    """Stable Kahn topological sort (appending nodes breaks the order the ONNX
    checker and picky parsers require)."""
    available = {i.name for i in graph.input}
    available |= {i.name for i in graph.initializer}
    order, pending = [], list(graph.node)
    while pending:
        progressed, rest = False, []
        for n in pending:
            if all(i == '' or i in available for i in n.input):
                order.append(n)
                available.update(n.output)
                progressed = True
            else:
                rest.append(n)
        if not progressed:
            order.extend(rest)
            break
        pending = rest
    del graph.node[:]
    graph.node.extend(order)


# --------------------------------------------------------------------------
# pass 1: grid_sampler plugin -> native GridSample
# --------------------------------------------------------------------------
def swap_grid_sampler_for_native(model):
    """Replace `mmdeploy::grid_sampler` nodes with the ONNX/TensorRT native
    `GridSample` op.  Returns the number of nodes swapped.

    The exported graph declares opset 11, where `GridSample` does not exist in
    the ONNX schema -- TensorRT's parser keys off the node rather than the
    declared opset (verified in `local/test_native_gridsample.py`), and this
    model is only consumed by that parser.

    (Deliberately *not* named `use_native_grid_sample`: that is the name of the
    `apply_trt_graph_passes` argument, and shadowing it there silently turned
    the call into `int(...)` -> TypeError -> fallback to the unpatched model.)
    """
    n = 0
    for node in model.graph.node:
        if node.domain != 'mmdeploy' or node.op_type != 'grid_sampler':
            continue
        attrs = {a.name: a.i for a in node.attribute}
        mode = attrs.get('interpolation_mode', 0)
        pad = attrs.get('padding_mode', 0)
        if mode not in INTERP_MODE or pad not in PADDING_MODE:
            logger.warning('skipping grid_sampler %s: unsupported mode %s/%s',
                           node.name, mode, pad)
            continue
        del node.attribute[:]
        node.domain = ''
        node.op_type = 'GridSample'
        node.attribute.extend([
            helper.make_attribute('mode', INTERP_MODE[mode]),
            helper.make_attribute('padding_mode', PADDING_MODE[pad]),
            helper.make_attribute('align_corners',
                                  attrs.get('align_corners', 0)),
        ])
        n += 1
    if n:
        left = {x.domain for x in model.graph.node if x.domain}
        keep = [
            o for o in model.opset_import if o.domain == '' or o.domain in left
        ]
        del model.opset_import[:]
        model.opset_import.extend(keep)
        logger.info('TRT pass: swapped %d grid_sampler -> native GridSample',
                    n)
    return n


# --------------------------------------------------------------------------
# pass 2: bound the candidate count reaching NonMaxSuppression
# --------------------------------------------------------------------------
def _inputs(by_output, name, kinds):
    """Walk back through `kinds` (Cast/Reshape/Unsqueeze/...) to the tensor
    that holds the actual data."""
    seen = 0
    cur = name
    while seen < 4:
        node = by_output.get(cur)
        if node is None or node.op_type not in kinds:
            break
        cur = node.input[0]
        seen += 1
    return cur


def bound_nms_candidates(model, pre_topk):
    """Insert a score pre-topk in front of every `NonMaxSuppression` node.

    Returns the number of nodes patched.  `pre_topk <= 0` disables the pass.
    """
    if pre_topk is None or pre_topk <= 0:
        return 0
    g = model.graph
    by_output = {o: n for n in g.node for o in n.output}
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    added, patched = [], 0
    for nms in [
            n for n in g.node
            if n.op_type == 'NonMaxSuppression' and not n.domain
    ]:
        try:
            boxes_in, scores_in = nms.input[0], nms.input[1]
            boxes_src = _inputs(by_output, boxes_in, {'Cast', 'Identity'})
            scores_src = _inputs(by_output, scores_in, {'Cast', 'Identity'})
            bnode = by_output.get(boxes_src)
            snode = by_output.get(scores_src)
            if bnode is None or snode is None or \
                    bnode.op_type != 'Unsqueeze' or \
                    snode.op_type != 'Unsqueeze':
                raise ValueError('unexpected NMS input chain')
            # boxes (1,N,4): the Unsqueeze input is (N,4) or (N,4)-ish;
            # scores (1,1,N): the Unsqueeze input is (N,).
            p = f'trtpass_pretopk_{patched}'
            c_k = helper.make_node(
                'Constant', [], [p + '_k'],
                name=p + '_k',
                value=numpy_helper.from_array(
                    np.array([pre_topk], dtype=np.int64)))
            c_ax = helper.make_node(
                'Constant', [], [p + '_ax'],
                name=p + '_ax',
                value=numpy_helper.from_array(np.array([2], dtype=np.int64)))
            c_s1 = helper.make_node(
                'Constant', [], [p + '_s1'],
                name=p + '_s1',
                value=numpy_helper.from_array(np.array([-1], dtype=np.int64)))
            c_s2 = helper.make_node(
                'Constant', [], [p + '_s2'],
                name=p + '_s2',
                value=numpy_helper.from_array(
                    np.array([-1, 1], dtype=np.int64)))
            shape = helper.make_node(
                'Shape', [scores_in], [p + '_shape'], name=p + '_shape')
            dim = helper.make_node(
                'Gather', [p + '_shape', p + '_ax'], [p + '_n'],
                name=p + '_n',
                axis=0)
            kmin = helper.make_node(
                'Min', [p + '_n', p + '_k'], [p + '_kmin'], name=p + '_kmin')
            topk = helper.make_node(
                'TopK', [scores_in, p + '_kmin'], [p + '_val', p + '_idx3'],
                name=p + '_topk',
                axis=2,
                largest=1,
                sorted=1)
            flat = helper.make_node(
                'Reshape', [p + '_idx3', p + '_s1'], [p + '_idx'],
                name=p + '_flat')
            gb = helper.make_node(
                'Gather', [boxes_in, p + '_idx'], [p + '_boxes'],
                name=p + '_gb',
                axis=1)
            gs = helper.make_node(
                'Gather', [scores_in, p + '_idx'], [p + '_scores'],
                name=p + '_gs',
                axis=2)
            added += [
                c_k, c_ax, c_s1, c_s2, shape, dim, kmin, topk, flat, gb, gs
            ]
            nms.input[0] = p + '_boxes'
            nms.input[1] = p + '_scores'

            # The NMS index column is also captured *implicitly* by the
            # following ONNX `If` subgraphs, so a plain consumer rewrite would
            # leave those subgraphs indexing the narrowed array.  Renaming the
            # raw tensor and re-using its original name for the remapped one
            # fixes every reference, explicit or implicit.
            col = consumers[nms.output[0]][0]
            if col.op_type != 'Gather':
                raise ValueError('unexpected NMS output consumer')
            raw = p + '_local_raw'
            orig = col.output[0]
            col.output[0] = raw
            fl = helper.make_node(
                'Reshape', [raw, p + '_s1'], [p + '_loc'], name=p + '_fl')
            remap = helper.make_node(
                'Gather', [p + '_idx', p + '_loc'], [p + '_remap'],
                name=p + '_remap',
                axis=0)
            back = helper.make_node(
                'Reshape', [p + '_remap', p + '_s2'], [orig], name=p + '_back')
            added += [fl, remap, back]
            patched += 1
        except Exception as e:  # noqa: BLE001
            logger.warning('skipping NMS pre-topk for %s: %s', nms.name, e)

    if patched:
        g.node.extend(added)
        _toposort(g)
        logger.info(
            'TRT pass: bounded %d NonMaxSuppression node(s) to '
            'pre_topk=%d', patched, pre_topk)
    return patched


def _log():
    """Mmengine's root logger when available (so warnings show up in
    tools/deploy.py output), the stdlib logger otherwise."""
    try:
        from mmdeploy.utils import get_root_logger
        return get_root_logger()
    except Exception:  # noqa: BLE001
        return logger


def apply_trt_graph_passes(onnx_model,
                           fp16_mode=False,
                           nms_pre_topk=0,
                           use_native_grid_sample=None):
    """Apply every enabled TensorRT-only ONNX pass, in place.

    Args:
        onnx_model (str | onnx.ModelProto): exported model.
        fp16_mode (bool): whether an FP16 engine is being built.  The native
            GridSample pass is only a win for FP16 (see module docstring), so
            it defaults to `fp16_mode`.
        nms_pre_topk (int): candidate bound for NonMaxSuppression; <= 0 keeps
            the export untouched.
        use_native_grid_sample (bool | None): explicit override for the
            GridSample pass.  `None` (default) means "on iff fp16_mode".

    Returns:
        onnx.ModelProto
    """
    if use_native_grid_sample is None:
        use_native_grid_sample = fp16_mode
    if not use_native_grid_sample and (nms_pre_topk is None
                                       or nms_pre_topk <= 0):
        return onnx_model
    # work on a copy so a failure half-way cannot corrupt the caller's model
    model = onnx.ModelProto()
    model.CopyFrom(_to_model(onnx_model))
    try:
        if use_native_grid_sample:
            swap_grid_sampler_for_native(model)
        bound_nms_candidates(model, nms_pre_topk)
    except Exception:  # noqa: BLE001
        _log().exception('TRT graph passes failed; falling back to the '
                         'original ONNX model')
        return _to_model(onnx_model)
    return model
