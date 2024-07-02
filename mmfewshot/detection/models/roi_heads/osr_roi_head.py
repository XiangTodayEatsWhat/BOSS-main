
# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.core import bbox2roi

from mmdet.models.roi_heads import StandardRoIHead
from mmdet.models.builder import HEADS
import torch

@HEADS.register_module()
class OSRRoIHead(StandardRoIHead):
    
    def __init__(self,
                 sep_cfg=list(),
                 **kwargs):
        self.sep_cfg = sep_cfg
        super(StandardRoIHead, self).__init__(**kwargs)
        
    def reset_input(self, x, img_metas):
        # The feature map produced by FPN is doubled, only the second half will be fed to RPN during training.
        # Select the second half of the feature map.
        if 'neck' in self.sep_cfg:
            feature_size = len(x[0])
            num_imgs = len(img_metas)
            x = tuple([ft[(feature_size // num_imgs - 1) * num_imgs:] for ft in x])
        return x
    
    def reset_test_input(self, proposals, img_metas):
        rois = bbox2roi(proposals)
        # RPN separation will result in twice the size of N.
        if 'neck' in self.sep_cfg or 'rpn_head' in self.sep_cfg:
            num_img = len(img_metas)
            # FPN without separation need to be correct.
            if 'neck' not in self.sep_cfg:
                rois[:, 0] %= num_img
            proposals = torch.cat(proposals).reshape(num_img, -1, 5)
            num_rois_per_img = tuple(len(p) for p in proposals)
            num_bbox_per_img = num_rois_per_img
            num_score_per_img = tuple(len(p) // 2 for p in proposals)
        
        elif 'roi_head' in self.sep_cfg:
            num_rois_per_img = tuple(len(p) for p in proposals)
            num_bbox_per_img = tuple(len(p) * 2 for p in proposals)
            num_score_per_img = num_rois_per_img
        else:
            num_rois_per_img = tuple(len(p) for p in proposals)
            num_bbox_per_img = num_rois_per_img
            num_score_per_img = num_rois_per_img
        return num_rois_per_img, num_bbox_per_img, num_score_per_img, rois, proposals
    
    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        x = self.reset_input(x, img_metas)
        
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, 
                                                    sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, 
                                                    sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses
    
    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        
        num_rois_per_img, num_bbox_per_img, num_score_per_img, rois, proposals = self.reset_test_input(proposals, img_metas)
        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0, ), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros(
                    (0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size

        bbox_results = self._bbox_forward(x, rois)
                               
        
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        rois = rois.split(num_rois_per_img, 0)
        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_bbox_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_bbox_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)
            
        cls_score = cls_score.split(num_score_per_img, 0)


        # apply bbox post-processing to each image individually
        det_bboxes = []      
        det_labels = []
        for i in range(len(img_metas)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0, ), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.bbox_head.fc_cls.out_features))

            else:
                det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels