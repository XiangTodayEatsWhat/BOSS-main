_base_ = ['../_base_/models/faster_rcnn_r50_caffe_fpn.py']
model = dict(
    type='BOSS',
    frozen_parameters=[
        'backbone', 
        'neck', 
        # 'rpn_head', 
        'roi_head.bbox_head.shared_fcs',
    ],
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101, frozen_stages=4),
    roi_head=dict(
        type='OSRRoIHead',
        bbox_head=dict(type='OSRBBoxHead', 
                       num_classes=20,
                       base_group=1,
                       novel_group=1,
                       fewshot=True,
                       n_novel=5,
                       init_cfg=[
                            dict(
                                type='Caffe2Xavier',
                                override=dict(type='Caffe2Xavier', name='shared_fcs')),
                            dict(
                                type='Normal',
                                override=dict(type='Normal', name='fc_reg', std=0.001))
                            ]
                        )),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            novel_score_thr=0.008,
            novel_max_per_img=200,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.3),
            max_per_img=1000))
    )