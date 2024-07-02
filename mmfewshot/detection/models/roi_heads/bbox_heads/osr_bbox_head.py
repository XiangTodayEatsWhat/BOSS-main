import torch
import torch.nn as nn
from torch.nn import functional as F
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads import ConvFCBBoxHead

from mmcv.runner import force_fp32
from mmdet.models.losses import accuracy

from mmdet.core import multiclass_nms
import random
from mmfewshot.detection.models.loha.loha import LohaModule

@HEADS.register_module()
class OSRBBoxHead(ConvFCBBoxHead):
    """OSRBBoxHead for BOSS

    Args:
        base_group (int): Orthogonal subspace dimension of base classes.
        novel_group (int): Orthogonal subspace dimension of novel classes.
        sep_cfg (list): Config of LohaModule.
    """

    def __init__(self, 
                 fc_out_channels=1024,
                 n_base=15, 
                 n_novel=0,
                 base_group=1,
                 novel_group=5,
                 fewshot=True,
                 sep_cfg=list(),
                 *args,
                 **kwargs) -> None:
        super(OSRBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            with_cls=False,
            *args,
            **kwargs)
        
        self.fewshot = fewshot
        self.n_base = n_base
        self.n_novel = n_novel
        self.sep_cfg = sep_cfg
        self.base_group = base_group
        self.novel_group = novel_group
        
        self.classifier = nn.Sequential(
            nn.Linear(fc_out_channels, fc_out_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(fc_out_channels, fc_out_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(fc_out_channels, 1, bias=False)
        )
        if self.fewshot:
            self.base_emb = nn.Parameter(torch.zeros(n_base * max(self.base_group, 1), fc_out_channels), requires_grad=False)
            self.novel_emb = nn.Parameter(torch.zeros(n_novel * max(self.novel_group, 1), fc_out_channels), requires_grad=True)
            self.classifier_novel = nn.Sequential(
                nn.Linear(fc_out_channels, fc_out_channels, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(fc_out_channels, fc_out_channels, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(fc_out_channels, 1, bias=False)
            )
            if self.novel_group == 0:
                nn.init.xavier_normal_(self.novel_emb)
            else:
                nn.init.orthogonal_(self.novel_emb)
            self.init_cls_n()
            self.ft_freeze()
        else:
            if self.base_group == 0:
                self.base_emb = nn.Parameter(torch.zeros(n_base, fc_out_channels), requires_grad=True)
                nn.init.xavier_normal_(self.base_emb)
            else:
                self.base_emb = nn.Parameter(torch.zeros(n_base * self.base_group, fc_out_channels), requires_grad=True)
                nn.init.orthogonal_(self.base_emb)
            self.novel_emb = None
        if 'roi_head' in self.sep_cfg:
            self.shared_fcs = nn.ModuleList([
                LohaModule('roi_head.bbox_head.shared_fcs.{}'.format(fc_id), fc) for fc_id, fc in enumerate(self.shared_fcs)
            ])
        
    def init_cls_n(self):
        for param_q, param_k in zip(self.classifier.parameters(), self.classifier_novel.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            
    def ft_freeze(self):
        for param in self.classifier.parameters():
            param.requires_grad = False
    
    def orthogonal_decompose(self, feats, bases_b, bases_n=None):
        '''
            feats: [BxCxN]
            bases_b: [1xKxC]
            bases_n: [1xKxC]
            ---
            out_fg:   [BxKxCxN]
            out_bg:   [Bx1xCxN]
        '''
        q = feats.to(torch.float) # [BxCxN]
        s1 = F.normalize(bases_b.to(torch.float), p=2, dim=-1) # [1xKxC]
        proj1 = torch.matmul(s1, q) # [BxKxN]
        out_fg_b = proj1.unsqueeze(2) * s1.unsqueeze(-1) # [BxKxCxN]
        out_fg_b = out_fg_b.reshape(out_fg_b.shape[0], self.n_base, max(self.base_group, 1), -1, 1).sum(2)
        out_bg = q - out_fg_b.sum(1) # [BxCxN]
        
        if bases_n is not None:
            # Divide it into two halves, the first half only counts base, and obtain the foreground score and background score
            # The second half is considered both base and novel
            _, q_novel = self.reset_osr_input(q)
            out_fg_src, _ = self.reset_osr_input(out_fg_b)
            out_base_bg, out_novel_bg = self.reset_osr_input(out_bg)
            
            s2 = F.normalize(bases_n, p=2, dim=-1) 
            proj2 = torch.matmul(s2, q_novel) # [BxKxN]
            out_fg_n = proj2.unsqueeze(2) * s2.unsqueeze(-1) # [BxKxCxN]
            out_fg_n = out_fg_n.reshape(out_fg_n.shape[0], self.n_novel, max(self.novel_group, 1), -1, 1).sum(2)
            out_bg = out_novel_bg - out_fg_n.sum(1)# [BxCxN]
            return out_fg_src, out_base_bg, out_fg_n, out_bg.unsqueeze(1)
        else:
            out_fg = out_fg_b
            return out_fg, out_bg.unsqueeze(1)
    
    
    def reset_input(self, x):
        if not self.training and ('neck' in self.sep_cfg or 'rpn_head' in self.sep_cfg):
            return x, x[len(x) // 2:]
        return x, x
    
    def reset_output(self, x):
        x_cls = x
        x_reg = x
        
        if 'roi_head' in self.sep_cfg:
            if not self.training:
                if 'neck' in self.sep_cfg or 'rpn_head' in self.sep_cfg:
                    x_reg = torch.cat([x[:len(x) // 3], x[len(x) // 3 * 2:]])
            else:
                x_reg = x[len(x) // 2:]
        elif self.fewshot:
            x_cls = torch.cat([x, x])
        
        return x_cls, x_reg
            
    
    def forward(self, x):
        '''
            x       : [BxCxHxW]
        '''
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)
            if 'roi_head' in self.sep_cfg:
                src_x, lora_x = self.reset_input(x)
                for fc in self.shared_fcs:
                    src_x, lora_x = fc(src_x, lora_x)
                    src_x = self.relu(src_x)
                    lora_x = self.relu(lora_x)
                x = torch.cat([src_x, lora_x])
            else:
                for fc in self.shared_fcs:
                    x = self.relu(fc(x))
        # separate branches
        x_cls, x_reg = self.reset_output(x)

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        if self.fewshot:
            cls_score = self.forward_all(x_cls)
        else:
            cls_score = self.forward_base(x_cls)
        
        return cls_score, bbox_pred
    
    def reset_osr_input(self, x):
        if not self.training and ('neck' in self.sep_cfg or 'rpn_head' in self.sep_cfg):
            return x.split(len(x) // 3 * 2)
        return x.split(len(x) // 2)
    
    def reset_osr_output(self, x):
        if not self.training and ('neck' in self.sep_cfg or 'rpn_head' in self.sep_cfg):
            guide_x = x[len(x) // 2:]
            base_ans = guide_x.argmax(-1)
            self.novel_id = base_ans == self.n_base
            return x[:len(x) // 2]
        
        base_ans = x.argmax(-1)
        self.novel_id = base_ans == self.n_base
        return x
    
    def forward_all(self, x_cls):
        '''
            x_cls       : [BxCxHxW]
        '''
        
        B, C = x_cls.shape
        base_emb = self.base_emb.unsqueeze(0) # [1xKbasexC]
        novel_emb = self.novel_emb.unsqueeze(0) # [1xKnovelxC]

        out_fg_base, out_src_bg, out_fg_novel, feats_bg = self.orthogonal_decompose(x_cls.unsqueeze(-1), base_emb, novel_emb)

         
        out_fg_base = out_fg_base.contiguous().view(-1, C) # [(BxKb)xCxhxw]
        preds_base = self.classifier(out_fg_base) # [(BxKb)x1xhxw]
        preds_base = preds_base.view(-1, self.n_base) # [BxKbxhxw]

        feats_novel = torch.cat([out_fg_novel, feats_bg], dim=1) # [Bx(1+Kn)xCxN]
        
        feats_src_bg = out_src_bg.view(-1, C)
        preds_src_bg = self.classifier(feats_src_bg)
           
        feats_novel = feats_novel.contiguous().view(-1, C) # [(Bx(1+Kn))xCxhxw]
        
        preds_novel = self.classifier_novel(feats_novel) # [(Bx(1+Kn))x1xhxw]
        preds_novel = preds_novel.view(-1, self.n_novel + 1) # [Bx(1+Kn)xhxw]
        
        preds_base_bg = torch.cat([preds_base, preds_src_bg], dim=1)
        preds_base_bg = self.reset_osr_output(preds_base_bg)
        preds = torch.cat([preds_base_bg, preds_novel], dim=1)
        
        return preds

    def forward_base(self, x_cls):
        '''
            x_cls       : [BxCxHxW]
        '''

        B, C = x_cls.shape
        cls_emb = self.base_emb.unsqueeze(0) # [1xKbasexC]

        n_class = 1 + self.n_base
        x_cls = x_cls.unsqueeze(-1)
        feats_fg, feats_bg = self.orthogonal_decompose(x_cls, cls_emb)

        feats_all = torch.cat([feats_fg, feats_bg], dim=1) # [Bx(1+K)xCxN]
        feats_all = feats_all.contiguous().view(B * n_class, C) # [(Bx(1+K))xCxhxw]

        preds = self.classifier(feats_all) # [(Bx(1+K))x1xhxw]
        preds = preds.view(B, n_class) # [Bx(1+K)xhxw]
        
        return preds
    
    def get_orth_loss(self, proto_sim):
        '''
            protos:   : [K1xK2] K1 <= K2
        '''
        eye_sim = torch.triu(torch.ones_like(proto_sim), diagonal=1)
        loss_orth = torch.abs(proto_sim[eye_sim == 1]).mean()
        return loss_orth

    
    def seploss(self, cls_score, labels):
        sample_multi_bce_weights = torch.full_like(cls_score, 10)
        sample_multi_bce_weights[labels == self.n_novel, -1] = 1
        sample_multi_bce_weights[labels == self.n_novel, :-1] = 2
        sample_multi_bce_weights[labels != self.n_novel, :-1] = 20
        loss_cls_sep = nn.BCELoss(weight=sample_multi_bce_weights)
        return loss_cls_sep
    
    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        
        B = cls_score.shape[0]
        if cls_score is not None:
            if self.fewshot:
                # The indexes classified as background in stage 1 and not labeled as base are taken out for finetuning
                stage2_ind = self.novel_id * (labels >= self.n_base)
                
                sample_cls_score = cls_score[stage2_ind, self.n_base + 1:]
                sample_labels = labels[stage2_ind] - self.n_base
                
                loss_cls_sep = self.seploss(sample_cls_score, sample_labels)
                
            else:
                avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            
            if cls_score.numel() > 0:
                if self.fewshot:
                    loss_cls_ = loss_cls_sep(
                        torch.sigmoid(sample_cls_score),
                        F.one_hot(sample_labels, self.n_novel + 1).float(),
                        )
                else:
                    loss_cls_ = self.loss_cls(
                        cls_score,
                        labels,
                        label_weights,
                        avg_factor=avg_factor,
                        reduction_override=reduction_override
                        )
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    if self.fewshot:
                        losses['acc'] = accuracy(torch.cat((cls_score[:, :self.n_base], cls_score[:, self.n_base + 1:]), dim=1), labels)
                    else:
                        losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
                
        if self.base_group != 0:
            novel_emb = self.novel_emb
            base_emb = self.base_emb
            C = base_emb.shape[-1]
            if self.novel_emb is not None:
                novel_emb = F.normalize(novel_emb.to(torch.float), p=2, dim=-1) # [1xKnovelxC]
                novel_emb = novel_emb.reshape(-1, C) # [KnxC]
                all_emb = torch.cat([novel_emb, F.normalize(base_emb.to(torch.float).squeeze(0), p=2, dim=-1)], dim=0) # [((Kn+Kb)xC]
                proto_sim = torch.matmul(novel_emb, all_emb.t()) # [Knx(Kn+Kb)]
            else:
                base_emb = F.normalize(base_emb, p=2, dim=-1).squeeze(0) # [KbasexC]
                proto_sim = torch.matmul(base_emb, base_emb.t()) # [KbasexKbase]
                
            losses['loss_osr'] = self.get_orth_loss(proto_sim) 
        return losses

    
    def get_base_bboxes(self,
                        rois,
                        cls_score,
                        bboxes,
                        cfg=None):
        if self.custom_cls_channels:
            base_scores = self.loss_cls.get_activation(cls_score[:, :self.n_base + 1])
        else:
            base_scores = F.softmax(cls_score[:, :self.n_base + 1], dim=-1)
            
            base_bboxes = bboxes[:, :self.n_base * 4]
            base_bboxes_remain_id = (rois[:, 1:] != 0).sum(-1) != 0
                
            base_bboxes = base_bboxes[base_bboxes_remain_id]
            base_scores = base_scores[base_bboxes_remain_id]
        
        base_det_bboxes, base_det_labels = multiclass_nms(base_bboxes, 
                                                        base_scores,
                                                        cfg.score_thr, cfg.nms,
                                                        cfg.max_per_img)
        return base_det_bboxes, base_det_labels
    
    def get_novel_bboxes(self,
                        rois,
                        cls_score,
                        bboxes,
                        cfg=None):
        if self.custom_cls_channels:
            novel_scores = self.loss_cls.get_activation(cls_score[self.novel_id, self.n_base + 1:])
        else:
            novel_scores = F.softmax(cls_score[self.novel_id, self.n_base + 1:], dim=-1)
            
            novel_bboxes = bboxes[self.novel_id, self.n_base * 4:]
            novel_bboxes_remain_id = (rois[self.novel_id, 1:] != 0).sum(-1) != 0
                
            novel_bboxes = novel_bboxes[novel_bboxes_remain_id]
            novel_scores = novel_scores[novel_bboxes_remain_id]
                
        novel_det_bboxes, novel_det_labels = multiclass_nms(novel_bboxes, 
                                                              novel_scores,
                                                              cfg.novel_score_thr, cfg.nms,
                                                              cfg.novel_max_per_img)
        return novel_det_bboxes, novel_det_labels
            
    def reset_test_input(self, rois, bbox_pred):
        if 'neck' not in self.sep_cfg and 'rpn_head' not in self.sep_cfg:
            rois = rois.repeat(2, 1)
            if 'roi_head' not in self.sep_cfg:
                bbox_pred = bbox_pred.repeat(2, 1)
        return rois, bbox_pred
    
    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            rois (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_shape (Sequence[int], optional): Maximum bounds for boxes,
                specifies (H, W, C) or (H, W).
            scale_factor (ndarray): Scale factor of the
               image arrange as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[Tensor, Tensor]:
                First tensor is `det_bboxes`, has the shape
                (num_boxes, 5) and last
                dimension 5 represent (tl_x, tl_y, br_x, br_y, score).
                Second tensor is the labels with shape (num_boxes, ).
        """     
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        rois, bbox_pred = self.reset_test_input(rois, bbox_pred)
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                bboxes.size()[0], -1)
        B = len(rois)    
        det_bboxes, det_labels = self.get_base_bboxes(rois[:B // 2], cls_score, bboxes[:B // 2], cfg)
        

        if self.fewshot:
            novel_det_bboxes, novel_det_labels = self.get_novel_bboxes(rois[B // 2:], cls_score, bboxes[B // 2:], cfg)
            det_bboxes = torch.cat([det_bboxes, novel_det_bboxes])
            det_labels = torch.cat([det_labels, novel_det_labels + self.n_base])
            
        return det_bboxes, det_labels