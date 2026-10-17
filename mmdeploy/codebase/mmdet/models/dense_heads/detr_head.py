# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torchvision
from torch import Tensor
from torch.nn import functional as F

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.dense_heads.DDQDETRHead.predict_by_feat')
@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.dense_heads.DeformableDETRHead.predict_by_feat')
@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.dense_heads.DETRHead.predict_by_feat')
def detrhead__predict_by_feat__default(self,
                                       all_cls_scores_list: List[Tensor],
                                       all_bbox_preds_list: List[Tensor],
                                       batch_img_metas: List[dict],
                                       rescale: bool = True):
    """Rewrite `predict_by_feat` of `FoveaHead` for default backend."""
    from mmdet.structures.bbox import bbox_cxcywh_to_xyxy

    cls_scores = all_cls_scores_list[-1]
    bbox_preds = all_bbox_preds_list[-1]
    img_shape = batch_img_metas[0]['img_shape']
    if isinstance(img_shape, list):
        img_shape = torch.tensor(
            img_shape, dtype=torch.long, device=cls_scores.device)
    img_shape = img_shape.unsqueeze(0)

    max_per_img = self.test_cfg.get('max_per_img', len(cls_scores[0]))

    # supports dynamical batch inference
    if self.loss_cls.use_sigmoid:
        cls_scores = cls_scores.sigmoid()
        scores, indexes = cls_scores.flatten(1).topk(max_per_img, dim=1)
        det_labels = indexes % self.num_classes
        # NOTE: `indexes // num_classes` is already the query index *within*
        # the image, so gather along dim 1 -- no cross-image offset needed.
        # The old `arange(batch_size) * max_per_img` offset looked correct,
        # but `cls_scores.size(0)` is a python int while tracing, so it
        # baked the *exported* batch size into the graph: for any other batch
        # size every image then gathered its boxes from image 0.
        bbox_index = indexes // self.num_classes
        bbox_preds = torch.gather(
            bbox_preds, 1,
            bbox_index.unsqueeze(-1).expand(-1, -1, bbox_preds.size(-1)))
    else:
        scores, det_labels = F.softmax(cls_scores, dim=-1)[..., :-1].max(-1)
        scores, bbox_index = scores.topk(max_per_img, dim=1)
        bbox_preds = torch.gather(
            bbox_preds, 1,
            bbox_index.unsqueeze(-1).expand(-1, -1, bbox_preds.size(-1)))
        det_labels = torch.gather(det_labels, 1, bbox_index)

    det_bboxes = bbox_cxcywh_to_xyxy(bbox_preds)
    det_bboxes.clamp_(min=0., max=1.)
    shape_scale = img_shape.flip(1).repeat(1, 2).unsqueeze(1)
    det_bboxes = det_bboxes * shape_scale
    det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(-1)), -1)

    return det_bboxes, det_labels


@FUNCTION_REWRITER.register_rewriter(func_name='mmcv.ops.batched_nms')
def batched_nms__tensorrt(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    keep = torchvision.ops.nms(
        boxes, scores, iou_threshold=nms_cfg.iou_threshold)
    dets = torch.cat([boxes[keep], scores[keep, None]], dim=1)
    return dets, keep
