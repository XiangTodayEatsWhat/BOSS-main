_base_ = [
    '../../../_base_/datasets/fine_tune_based/base_nwpu.py',
    '../../../_base_/schedules/schedule.py',
    '../../../_base_/models/faster_rcnn_r50_caffe_fpn.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotNWPUDataset
data = dict(
    train=dict(classes='BASE_CLASSES_SPLIT1'),
    val=dict(classes='BASE_CLASSES_SPLIT1'),
    test=dict(classes='BASE_CLASSES_SPLIT1'))
lr_config = dict(warmup_iters=100, step=[6000, 8000])
runner = dict(max_iters=9000)
# model settings
model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101),
    rpn_head=dict(
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='OSRRoIHead',
        bbox_head=dict(type='OSRBBoxHead', 
                       n_base=7,
                       base_group=10,
                       num_classes=7, 
                       fewshot=False,
                       init_cfg=[
                            dict(
                                type='Caffe2Xavier',
                                override=dict(type='Caffe2Xavier', name='shared_fcs')),
                            dict(
                                type='Normal',
                                override=dict(type='Normal', name='fc_reg', std=0.001))
                            ]
                        )))
# using regular sampler can get a better base model
use_infinite_sampler = False
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
work_dir = 'work_dirs/nwpu/split1/boss_r101_nwpu-split1_base-training'