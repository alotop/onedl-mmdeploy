# Copyright (c) OpenMMLab. All rights reserved.
"""Batch-generic (dynamic-batch) rewrites for DDQ-DETR.

The stock ``DDQDETR.pre_decoder`` and ``DDQTransformerDecoder`` loop over the
batch in *python*::

    for img_id in range(num_imgs):
        _, keep_idxs = batched_nms(single_proposals, single_scores, ...)

``num_imgs`` is a python ``int``, so during ONNX export the loop is unrolled
once per *traced* image and the resulting graph is hard-wired to the batch size
used at export time.  A ``min=1 / opt=max=N`` TensorRT profile built on such a
graph is therefore useless: for ``B < N`` the trailing copies gather from
out-of-range batch indices, and for a graph traced with one image every output
collapses to ``(1, 300, 5)``.

Both loops are rewritten here with fully batched ops:

* the per-image ``batched_nms`` becomes one ``mmdeploy::TRTBatchedNMS`` node
  (``TRTBatchedNMSop`` with ``return_index=True``).  The plugin accepts a
  **dynamic** batch dimension and returns a **fixed length** ``[B, keep_topk]``
  index tensor padded with ``-1``;
* the gathered queries are zero-padded exactly like ``mmdet.utils.padding_to``
  (via ``align_tensor``) did, so the decoder receives numerically identical
  inputs while every branch keeps a shape that only depends on constants;
* the distinct-query attention mask is built with ``scatter`` instead of a
  python loop and expanded to ``(B * num_heads, Q, Q)`` with ``expand`` +
  ``reshape(-1, ...)`` so that the batch dimension stays symbolic.

The rewrites are only registered for the TensorRT backend (they emit an
``mmdeploy`` custom op) and only support inference.
"""
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from mmdeploy.core import FUNCTION_REWRITER


def _pre_topk_from_cfg(cfg, default: int = 0) -> int:
    """Candidate cap handed to the plugin for the pre-decoder NMS.

    Mirrors ``nms_pre_topk`` (mmdet's own ``multiclass_nms(pre_top_k=...)``
    does the same).  ``0`` keeps every candidate, i.e. exact semantics.

    NOTE: the value must stay within the plugin's hard limit -- its
    ``allClassNMS`` kernel is instantiated for ``ceil(top_k / 512) <= 10`` and
    asserts at *runtime* otherwise.
    """
    try:
        common = cfg['backend_config'].get('common_config', {}) or {}
        value = int(common.get('dqs_pre_topk', default))
    except Exception:  # noqa: BLE001
        value = default
    return max(0, min(value, 5120))


def _batched_nms_index(boxes: Tensor, scores: Tensor, pre_topk: int,
                       keep_topk: int, iou_threshold: float) -> Tensor:
    """One NMS for the whole batch -> ``(B, keep_topk)`` int64, ``-1`` padded.

    Args:
        boxes (Tensor): ``(B, L, 4)`` xyxy boxes.
        scores (Tensor): ``(B, L)`` scores.
        pre_topk (int): candidate cap before NMS (0 = all).
        keep_topk (int): fixed output length.
        iou_threshold (float): NMS IoU threshold.

    Returns:
        Tensor: keep indices into the original ``L`` candidates.
    """
    from mmdeploy.mmcv.ops.nms import TRTBatchedNMSop

    # The plugin wants (B, L, 1, 4) boxes and (B, L, 1) scores.
    idx = TRTBatchedNMSop.apply(
        boxes.unsqueeze(2), scores.unsqueeze(-1), 1, int(pre_topk),
        int(keep_topk), float(iou_threshold), 0.0, -1, True)[2]
    return idx.long()


def _gather_padded(x: Tensor, idx: Tensor) -> Tensor:
    """Gather dim 1 by ``idx`` and zero the ``-1`` (padding) slots.

    Equivalent to ``align_tensor([...], topk)`` in the original code, which
    stacks ``padding_to(item, topk)`` -- i.e. the kept rows followed by zeros.
    """
    valid = idx >= 0
    # NOTE: `idx.clamp(min=0)` exports as ONNX `Clip` whose bounds are float
    # constants while the input is int64, which TensorRT's parser rejects
    # ("MAX operation has incompatible input types Int64 and Float").
    safe = torch.where(valid, idx, torch.zeros_like(idx))
    out = torch.gather(x, 1, safe.unsqueeze(-1).expand(-1, -1, x.size(-1)))
    # NOTE: `out * valid` would turn the `inf`s that
    # `gen_encoder_output_proposals` produces for degenerate boxes into NaN
    # (0 * inf).  `where` drops them without touching the kept rows -- exactly
    # what `padding_to` did.
    return torch.where(valid.unsqueeze(-1), out, torch.zeros_like(out))


def select_distinct_queries_impl(self, reference_points: Tensor, query: Tensor,
                                 self_attn_mask: Tensor, layer_index: int):
    """Batched ``DDQTransformerDecoder.select_distinct_queries``.

    Same semantics as the python-loop version, expressed with one batched NMS
    and ``scatter`` so the batch dimension stays symbolic.
    """
    from mmdet.structures.bbox import bbox_cxcywh_to_xyxy

    num_heads = int(self.cache_dict['num_heads'])
    dis_start, num_dis = self.cache_dict['dis_query_info']
    dis_start, num_dis = int(dis_start), int(num_dis)

    dis_mask = self_attn_mask[:, dis_start:dis_start + num_dis,
                              dis_start:dis_start + num_dis]
    scores = self.cache_dict['cls_branches'][layer_index](
        query[:, dis_start:dis_start + num_dis]).sigmoid().max(-1).values
    proposals = bbox_cxcywh_to_xyxy(reference_points[:, dis_start:dis_start +
                                                     num_dis])

    # Which queries may still attend (identical for every head -> take head 0,
    # row 0, exactly like `~dis_mask[img_id * num_heads][0]`).
    active = ~dis_mask.reshape(-1, num_heads, num_dis, num_dis)[:, 0, 0, :]

    # Inactive candidates are pushed below `score_threshold` (0.0) so the
    # plugin drops them; sigmoid scores of active ones are >= 0, hence an
    # inactive box can never outrank (and suppress) an active one.
    masked_scores = scores.masked_fill(~active, -1.0)
    iou_threshold = float(self.cache_dict['dqs_cfg']['iou_threshold'])
    idx = _batched_nms_index(proposals, masked_scores, 0, num_dis,
                             iou_threshold)
    valid = idx >= 0
    keep = valid.to(scores.dtype)  # 1.0 -> mask stays, 0.0 -> unmask

    # Scatter "row/column stays masked?" into a (K+1, K+1) buffer.  Slot
    # `num_dis` is a sink for the padded (-1) entries, so they are no-ops; it
    # is sliced off afterwards.
    n = num_dis + 1
    sink = num_dis
    mask_f = torch.ones_like(scores[:, :1]).unsqueeze(-1).expand(-1, n, n)
    write = 1.0 - keep  # 0.0 = unmasks, 1.0 = padding no-op
    idx_row = torch.where(valid, idx, torch.full_like(idx, sink))
    # rows: mask[keep] = False
    mask_f = mask_f.scatter(1,
                            idx_row.unsqueeze(-1).expand(-1, -1, n),
                            write.unsqueeze(-1).expand(-1, -1, n))
    # columns: mask[:, keep] = False
    mask_f = mask_f.scatter(2,
                            idx_row.unsqueeze(1).expand(-1, n, -1),
                            write.unsqueeze(1).expand(-1, n, -1))
    attn_mask = mask_f[:, :num_dis, :num_dis] > 0.5

    # (B, K, K) -> (B * num_heads, K, K); -1 in expand/reshape keeps B symbolic
    attn_mask = attn_mask.unsqueeze(1).expand(-1, num_heads, -1, -1).reshape(
        -1, num_dis, num_dis)
    self.cache_dict['distinct_query_mask'].append(~attn_mask)

    if dis_start == 0 and num_dis == self_attn_mask.size(-1):
        # inference: the slice covers the whole mask, no index_put needed
        return attn_mask

    import copy
    self_attn_mask = copy.deepcopy(self_attn_mask)
    self_attn_mask[:, dis_start:dis_start + num_dis,
                   dis_start:dis_start + num_dis] = attn_mask
    return self_attn_mask


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.layers.transformer.ddq_detr_layers.'
    'DDQTransformerDecoder.select_distinct_queries',
    backend='tensorrt')
def ddq_transformer_decoder__select_distinct_queries__tensorrt(
        self, reference_points: Tensor, query: Tensor, self_attn_mask: Tensor,
        layer_index: int):
    """Rewrite `select_distinct_queries` for the TensorRT backend."""
    return select_distinct_queries_impl(self, reference_points, query,
                                        self_attn_mask, layer_index)


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.detectors.DDQDETR.pre_decoder', backend='tensorrt')
def ddqdetr__pre_decoder__tensorrt(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        batch_data_samples: Optional[list] = None) -> Tuple[Dict, Dict]:
    """Rewrite `pre_decoder` of `DDQDETR` for the TensorRT backend.

    Replaces the per-image ``batched_nms`` + ``align_tensor`` query selection
    with a single batched NMS and batched gathers, so that the graph is valid
    for *any* batch size instead of only the traced one.
    """
    if self.training:
        raise NotImplementedError(
            'the tensorrt rewrite of DDQDETR.pre_decoder only supports '
            'inference')

    from mmdet.structures.bbox import bbox_cxcywh_to_xyxy

    # NOTE: `get_context()` resolves the rewriter through the call stack, so it
    # has to be called from the rewritten function itself (not a helper).
    ctx = FUNCTION_REWRITER.get_context()
    pre_topk = _pre_topk_from_cfg(getattr(ctx, 'cfg', None))

    output_memory, output_proposals = self.gen_encoder_output_proposals(
        memory, memory_mask, spatial_shapes)
    enc_outputs_class = self.bbox_head.cls_branches[self.decoder.num_layers](
        output_memory)
    enc_outputs_coord_unact = self.bbox_head.reg_branches[
        self.decoder.num_layers](output_memory) + output_proposals

    topk = int(self.num_queries)
    proposals = bbox_cxcywh_to_xyxy(enc_outputs_coord_unact.sigmoid())
    scores = enc_outputs_class.max(-1)[0].sigmoid()

    iou_threshold = float(self.cache_dict['dqs_cfg']['iou_threshold'])
    idx = _batched_nms_index(proposals, scores, pre_topk, topk, iou_threshold)

    # `enc_outputs_class` is only consumed by the training branch, so it is
    # not gathered here.
    topk_coords_unact = _gather_padded(enc_outputs_coord_unact, idx)
    # `self.query_map` is a Linear on the last dim -> batch-vectorised
    query = _gather_padded(self.query_map(memory.detach()), idx)

    topk_coords_unact = topk_coords_unact.detach()
    self.cache_dict['dis_query_info'] = [0, topk]
    reference_points = topk_coords_unact.sigmoid()

    decoder_inputs_dict = dict(
        query=query,
        memory=memory,
        reference_points=reference_points,
        dn_mask=None)
    return decoder_inputs_dict, {}


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.layers.transformer.ddq_detr_layers.'
    'DDQTransformerDecoder.forward',
    backend='tensorrt')
def ddq_transformer_decoder__forward__tensorrt(
        self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
        self_attn_mask: Tensor, reference_points: Tensor,
        spatial_shapes: Tensor, level_start_index: Tensor,
        valid_ratios: Tensor, reg_branches: torch.nn.ModuleList,
        **kwargs) -> Tensor:
    """Rewrite `DDQTransformerDecoder.forward` for the TensorRT backend.

    Identical to the mmdet implementation except that the self-attention mask
    is expanded from ``(Q, Q)`` to ``(B * num_heads, Q, Q)`` with
    ``expand``/``reshape(-1, ...)``.  ``[None].repeat(len(query) * num_heads)``
    bakes the traced batch size into the graph and breaks dynamic batch.
    """
    if self.training:
        raise NotImplementedError(
            'the tensorrt rewrite of DDQTransformerDecoder.forward only '
            'supports inference')

    from mmdet.models.layers.transformer.utils import (coordinate_to_encoding,
                                                       inverse_sigmoid)

    num_heads = int(self.cache_dict['num_heads'])
    num_queries = query.size(1)
    intermediate = []
    intermediate_reference_points = [reference_points]
    self.cache_dict['distinct_query_mask'] = []

    if self_attn_mask is None:
        self_attn_mask = torch.zeros((num_queries, num_queries),
                                     device=query.device,
                                     dtype=torch.bool)
    # (Q, Q) -> (B, Q, Q): `query[:, :1, :1] * 0` carries the *dynamic* batch
    # dim, plain broadcasting does the rest.
    mask_f = self_attn_mask.to(torch.float32) + query[:, :1, :1] * 0.0
    self_attn_mask = (mask_f > 0.5).unsqueeze(1).expand(
        -1, num_heads, -1, -1).reshape(-1, num_queries, num_queries)

    for layer_index, layer in enumerate(self.layers):
        reference_points_input = reference_points[:, :, None] * torch.cat(
            [valid_ratios, valid_ratios], -1)[:, None]

        query_sine_embed = coordinate_to_encoding(
            reference_points_input[:, :, 0, :], num_feats=self.embed_dims // 2)
        query_pos = self.ref_point_head(query_sine_embed)

        query = layer(
            query,
            query_pos=query_pos,
            value=value,
            key_padding_mask=key_padding_mask,
            self_attn_mask=self_attn_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reference_points=reference_points_input,
            **kwargs)

        tmp = reg_branches[layer_index](query)
        new_reference_points = tmp + inverse_sigmoid(
            reference_points, eps=1e-3)
        new_reference_points = new_reference_points.sigmoid()
        reference_points = new_reference_points.detach()
        if layer_index < (len(self.layers) - 1):
            self_attn_mask = select_distinct_queries_impl(
                self, reference_points, query, self_attn_mask, layer_index)

        if self.return_intermediate:
            intermediate.append(self.norm(query))
            intermediate_reference_points.append(new_reference_points)

    if self.return_intermediate:
        return torch.stack(intermediate), torch.stack(
            intermediate_reference_points)

    return query, reference_points
