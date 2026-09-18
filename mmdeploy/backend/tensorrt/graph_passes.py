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
from onnx import helper, numpy_helper

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
            # raw tensor and reusing its original name for the remapped one
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


def _const_scalar(by_output, graph, name, default=None):
    """Read a scalar constant from an initializer or a Constant node."""
    for init in graph.initializer:
        if init.name == name:
            return float(numpy_helper.to_array(init).reshape(-1)[0])
    node = by_output.get(name)
    if node is not None and node.op_type == 'Constant':
        return float(numpy_helper.to_array(node.attribute[0].t).reshape(-1)[0])
    return default


# allClassNMS.cu launches `kernel[t_size - 1] <<<num_classes, BS>>>` with
# BS = 512 and `ASSERT(t_size <= 10)`, so the plugin's own pre-NMS top-k is
# capped at 5120 candidates -- and with num_classes == 1 the whole NMS runs on
# a single thread block, which is why class-agnostic NMS is not free there.
MAX_EFFICIENT_TOPK = 10 * 512


def _restore_length(g, idx, out_name, keep_topk, p, dynamic):
    """Give the plugin's fixed-size `(1, keep_topk)` index tensor the length
    the ONNX `NonMaxSuppression` would have produced.

    The plugin pads unused slots with **-1** (`allClassNMS.cu`), and
    `ddq_detr_layers.py::select_distinct_queries` builds the distinct-query
    attention mask out of the *number* of kept indices
    (`attn_mask[real_keep_index] = False`).  Feeding the -1s straight through
    would silently unmask queries, so the tensor is trimmed to
    `count(index >= 0)` -- exactly the length the original NMS returned.
    """
    added = []

    def const(name, arr):
        node = helper.make_node(
            'Constant', [], [p + '_' + name],
            name=p + '_' + name,
            value=numpy_helper.from_array(np.asarray(arr)))
        added.append(node)
        return p + '_' + name

    if not dynamic:
        c_shape = const('shape', np.array([keep_topk, 1], dtype=np.int64))
        added.append(
            helper.make_node(
                'Reshape', [idx, c_shape], [out_name], name=p + '_out'))
        return added

    c_neg = const('neg1', np.array([[-1]], dtype=np.int64))
    c_axes = const('axes', np.array([1], dtype=np.int64))
    c_start = const('start', np.array([0], dtype=np.int64))
    c_step = const('step', np.array([1], dtype=np.int64))
    c_shape = const('shape', np.array([-1, 1], dtype=np.int64))
    valid = helper.make_node(
        'Greater', [idx, c_neg], [p + '_valid'], name=p + '_valid')
    cast32 = helper.make_node(
        'Cast', [p + '_valid'], [p + '_valid32'],
        name=p + '_valid32',
        to=int(onnx.TensorProto.INT32))
    cnt = helper.make_node(
        'ReduceSum', [p + '_valid32'], [p + '_cnt'],
        name=p + '_cnt',
        axes=[1],
        keepdims=0)
    cnt64 = helper.make_node(
        'Cast', [p + '_cnt'], [p + '_cnt64'],
        name=p + '_cnt64',
        to=int(onnx.TensorProto.INT64))
    sliced = helper.make_node(
        'Slice', [idx, c_start, p + '_cnt64', c_axes, c_step],
        [p + '_idx_dyn'],
        name=p + '_slice')
    flat = helper.make_node(
        'Reshape', [p + '_idx_dyn', c_shape], [out_name], name=p + '_out')
    added += [valid, cast32, cnt, cnt64, sliced, flat]
    return added


def swap_nms_for_efficient(model,
                           keep_topk,
                           pre_topk=3000,
                           all_nodes=False,
                           decoder_topk=0):
    """Replace `NonMaxSuppression` node(s) with `mmdeploy::TRTBatchedNMS`.

    Returns the number of nodes replaced.

    Why the plugin: TensorRT's ONNX `NonMaxSuppression` costs ~1.8 us per
    candidate (22323 -> ~36 ms standalone).  `TRTBatchedNMS` runs its own GPU
    top-k first (22323 candidates, topk=3000 -> 6.9 ms) and its index output
    references the **original** candidate array
    (`loc_bboxIndex = beforeNMS_index_array[...]` in allClassNMS.cu), so it
    drops straight into where the old index column was.

    Args:
        keep_topk (int): boxes to keep; <= 0 disables the pass.
        pre_topk (int): plugin-side candidate cap for the **pre-decoder** node,
            clamped to `MAX_EFFICIENT_TOPK` (5120 is the plugin's hard limit:
            `allClassNMS.cu` asserts `ceil(top_k / BS) <= 10` with BS = 512).
        all_nodes (bool): also replace the five **decoder-side** nodes.  Their
            candidate set never exceeds `num_queries`, so no approximation is
            needed (`decoder_topk=0` means "consider all candidates"), and the
            exactness of the keep-index *count* is preserved by
            `_restore_length`.
        decoder_topk (int): plugin-side candidate cap for the decoder nodes;
            0 = all.
    """
    if keep_topk <= 0:
        return 0
    g = model.graph
    nms_nodes = [
        n for n in g.node if n.op_type == 'NonMaxSuppression' and not n.domain
    ]
    if not nms_nodes:
        return 0
    targets = nms_nodes if all_nodes else nms_nodes[:1]
    if not all_nodes and len(nms_nodes) < 2:
        # A lone NMS is a normal detector's postprocess NMS: its output is the
        # graph output, where a fixed keep_topk would change the API rather
        # than just the internals.  Leave it to `bound_nms_candidates`.
        logger.info(
            'efficient-NMS pass: only %d NonMaxSuppression node(s); skipping '
            '(expected a pre-decoder one plus decoder ones)', len(nms_nodes))
        return 0
    by_output = {o: n for n in g.node for o in n.output}
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    replaced = 0
    for nms in targets:
        p = f'trtpass_effnms_{replaced}'
        try:
            iou = _const_scalar(by_output, g, nms.input[3], 0.8)
            topk = pre_topk if nms is nms_nodes[0] else decoder_topk
            topk = min(int(topk), MAX_EFFICIENT_TOPK)
            c_ax = helper.make_node(
                'Constant', [], [p + '_ax'],
                name=p + '_ax',
                value=numpy_helper.from_array(np.array([2], dtype=np.int64)))
            c_perm = helper.make_node(
                'Constant', [], [p + '_perm'],
                name=p + '_perm',
                value=numpy_helper.from_array(
                    np.array([0, 2, 1], dtype=np.int64)))
            boxes4 = helper.make_node(
                'Unsqueeze', [nms.input[0], p + '_ax'], [p + '_boxes'],
                name=p + '_boxes',
                axes=[2])
            scores_t = helper.make_node(
                'Transpose', [nms.input[1], p + '_perm'], [p + '_scores'],
                name=p + '_scores',
                perm=[0, 2, 1])
            plugin = helper.make_node(
                'TRTBatchedNMS', [p + '_boxes', p + '_scores'],
                [p + '_dets', p + '_labels', p + '_idx'],
                domain='mmdeploy',
                name=p,
                background_label_id=-1,
                num_classes=1,
                topk=int(topk),
                keep_topk=int(keep_topk),
                score_threshold=0.0,
                iou_threshold=float(iou),
                is_normalized=0,
                clip_boxes=0,
                return_index=1)
            # rename trick (same reason as in bound_nms_candidates): the
            # keep-index tensor is also captured implicitly by the following
            # ONNX `If` subgraphs, so the replacement must take over the
            # original name.
            col = [
                c for c in consumers[nms.output[0]] if c.op_type == 'Gather'
            ][0]
            orig = col.output[0]
            added = _restore_length(
                g, p + '_idx', orig, keep_topk, p, dynamic=all_nodes)
            g.node.extend([c_ax, c_perm, boxes4, scores_t, plugin] + added)
            # drop the old slow NMS + its index-column Gather
            g.node.remove(nms)
            g.node.remove(col)
            replaced += 1
        except Exception as e:  # noqa: BLE001
            logger.warning('efficient-NMS pass skipped %s: %s', nms.name, e)
    if replaced:
        _toposort(g)
        logger.info(
            'TRT pass: replaced %d/%d NonMaxSuppression node(s) with '
            'TRTBatchedNMS (keep_topk=%d, pre_topk=%d, decoder_topk=%d)',
            replaced, len(nms_nodes), keep_topk, pre_topk, decoder_topk)
    return replaced


def swap_first_nms_for_efficient(model, keep_topk, topk=3000):
    """Backwards-compatible wrapper: only the pre-decoder NMS is replaced."""
    return swap_nms_for_efficient(
        model, keep_topk, pre_topk=topk, all_nodes=False)


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
                           use_native_grid_sample=None,
                           nms_efficient_keep_topk=0,
                           nms_efficient_topk=3000,
                           nms_efficient_all=False):
    """Apply every enabled TensorRT-only ONNX pass, in place.

    Args:
        onnx_model (str | onnx.ModelProto): exported model.
        fp16_mode (bool): whether an FP16 engine is being built.  The native
            GridSample pass is only a win for FP16 (see module docstring), so
            it defaults to `fp16_mode`.
        nms_pre_topk (int): candidate bound for the *remaining*
            NonMaxSuppression nodes (`k = min(nms_pre_topk, candidates)`);
            <= 0 leaves them untouched.  This is what the five decoder-side
            NMS calls need (it is a no-op for them, their candidate set never
            exceeds num_queries).
        use_native_grid_sample (bool | None): explicit override for the
            GridSample pass.  `None` (default) means "on iff fp16_mode".
        nms_efficient_keep_topk (int): if > 0, replace the pre-decoder
            NonMaxSuppression with the TRTBatchedNMS plugin keeping this many
            boxes; <= 0 (default) disables the pass.
        nms_efficient_topk (int): the plugin's own pre-NMS top-k for the
            pre-decoder node, clamped to `MAX_EFFICIENT_TOPK` (5120, the
            plugin's hard limit).
        nms_efficient_all (bool): also replace the five decoder-side NMS nodes
            (their candidate set is small, so no top-k approximation is used
            and the keep-index count is preserved by a dynamic slice).
            Defaults to False.

    Returns:
        onnx.ModelProto
    """
    if use_native_grid_sample is None:
        use_native_grid_sample = fp16_mode
    nms_efficient_keep_topk = int(nms_efficient_keep_topk or 0)
    if nms_efficient_keep_topk > 0:
        nms_efficient_topk = min(
            int(nms_efficient_topk or 0), MAX_EFFICIENT_TOPK)
    if not use_native_grid_sample and \
            (nms_pre_topk is None or nms_pre_topk <= 0) and \
            nms_efficient_keep_topk <= 0:
        return onnx_model
    # work on a copy so a failure half-way cannot corrupt the caller's model
    model = onnx.ModelProto()
    model.CopyFrom(_to_model(onnx_model))
    try:
        if use_native_grid_sample:
            swap_grid_sampler_for_native(model)
        if nms_efficient_keep_topk > 0:
            swap_nms_for_efficient(
                model,
                nms_efficient_keep_topk,
                pre_topk=nms_efficient_topk,
                all_nodes=nms_efficient_all)
        bound_nms_candidates(model, nms_pre_topk)
    except Exception:  # noqa: BLE001
        _log().exception('TRT graph passes failed; falling back to the '
                         'original ONNX model')
        return _to_model(onnx_model)
    return model
