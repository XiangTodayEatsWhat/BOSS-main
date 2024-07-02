# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Tuple
import torch
from mmdet.models import RPNHead
import torch
import torch.nn as nn
from torch.nn import functional as F
from mmdet.models.builder import HEADS
from mmfewshot.detection.models.loha.loha import LohaModule

@HEADS.register_module()
class SepRPNHead(RPNHead):
    """SepRPNHead

    Args:
        sep_cfg (list): Config of LohaModule.
    """
    def __init__(self,
                 in_channels,
                 init_cfg=dict(type='Normal', layer='Conv2d', std=0.01),
                 num_convs=1,
                 sep_cfg=list(),
                 **kwargs):
        self.sep_cfg = sep_cfg
        self.num_convs = num_convs
        super(RPNHead, self).__init__(
            1, in_channels, init_cfg=init_cfg, **kwargs)
        self.reset_layers()
        
    def reset_layers(self):
        if 'rpn_head' in self.sep_cfg:
            self.rpn_cls = LohaModule('rpn_head.rpn_cls', self.rpn_cls, lora_dim=64)
            self.rpn_conv = LohaModule('rpn_head.rpn_conv', self.rpn_conv, lora_dim=64)
            self.rpn_reg = LohaModule('rpn_head.rpn_reg', self.rpn_reg, lora_dim=64)     
               
    def reset_input(self, x):
        if 'neck' in self.sep_cfg:
            src_x, lora_x = x.split(len(x) // 2)
        else:
            src_x, lora_x = x, x
        return src_x, lora_x
    
    def reset_test_output(self, outs, img_metas):
        if 'rpn_head' in self.sep_cfg or 'neck' in self.sep_cfg:
            src_outs = (outs[0], outs[2])
            lora_outs = (outs[1], outs[3])
            src_proposal_list = self.get_bboxes(*src_outs, img_metas=img_metas)
            lora_proposal_list = self.get_bboxes(*lora_outs, img_metas=img_metas)
            
            dst_num = self.test_cfg['max_per_img']
            c = src_proposal_list[0].shape[-1]
            
            # padding for the proposal list
            for i in range(len(img_metas)):
                src_pre_num = len(src_proposal_list[i])
                lora_pre_num = len(lora_proposal_list[i])
                
                if src_pre_num < dst_num:
                    src_proposal_list[i] = torch.cat([src_proposal_list[i], 
                            torch.zeros((dst_num - src_pre_num), c).to(src_proposal_list[i].device)])
                    
                if lora_pre_num < dst_num:
                    lora_proposal_list[i] = torch.cat([lora_proposal_list[i], 
                            torch.zeros((dst_num - lora_pre_num), c).to(lora_proposal_list[i].device)])
            
            proposal_list = src_proposal_list + lora_proposal_list
            return proposal_list    
        
        return self.get_bboxes(*outs, img_metas=img_metas)
        
    def forward_single(self, x):
        """Forward feature map of a single scale level."""
        # if with lora_rpn
        if 'rpn_head' in self.sep_cfg:
            src_x, lora_x = self.reset_input(x)
            src_x, lora_x = self.rpn_conv(src_x, lora_x)
        
            src_x = F.relu(src_x, inplace=True)
            lora_x = F.relu(lora_x, inplace=True)
            
            src_rpn_cls_score, lora_rpn_cls_score = self.rpn_cls(src_x, lora_x)
            src_rpn_bbox_pred, lora_rpn_bbox_pred = self.rpn_reg(src_x, lora_x)
        
            return src_rpn_cls_score, lora_rpn_cls_score, src_rpn_bbox_pred, lora_rpn_bbox_pred
        
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        
        # both not existing
        if 'neck' not in self.sep_cfg:
            return rpn_cls_score, rpn_bbox_pred
        
        # not with lora_rpn, but with lora_fpn
        src_rpn_cls_score, lora_rpn_cls_score = rpn_cls_score.split(len(x) // 2)
        src_rpn_bbox_pred, lora_rpn_bbox_pred = rpn_bbox_pred.split(len(x) // 2)
        return src_rpn_cls_score, lora_rpn_cls_score, src_rpn_bbox_pred, lora_rpn_bbox_pred
       
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if 'rpn_head' in self.sep_cfg or 'neck' in self.sep_cfg:
            # only use lora outs for training
            outs = (outs[1], outs[3])

        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list
        
    def simple_test_rpn(self, x, img_metas):
        rpn_outs = self(x)
        
        return self.reset_test_output(rpn_outs, img_metas) 
        