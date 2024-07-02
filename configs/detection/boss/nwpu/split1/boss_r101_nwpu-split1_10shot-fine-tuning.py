_base_ = [
    '../../../_base_/datasets/fine_tune_based/few_shot_nwpu.py',
    '../../../_base_/schedules/schedule.py', '../../boss_r101.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotNWPUDataset
data = dict(
    train=dict(
        type='FewShotNWPUDataset',
        num_novel_shots=10,
        num_base_shots=10,
        classes='ALL_CLASSES_SPLIT1'),
    val=dict(classes='ALL_CLASSES_SPLIT1'),
    test=dict(classes='ALL_CLASSES_SPLIT1'))
evaluation = dict(
    interval=2000,
    class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
checkpoint_config = dict(interval=2000)

optimizer = dict(lr=0.0025)
lr_config = dict(
    warmup_iters=1000, step=[
        6000
    ])
runner = dict(max_iters=10000)

# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head_boss.py
# please refer to configs/detection/boss/README.md for more details.
load_from = ('work_dirs/nwpu/split1/boss_r101_nwpu-split1_base-training/base_model_random_init_bbox_head.pth')
work_dir = 'work_dirs/nwpu/split1/boss_r101_nwpu-split1_10shot-fine-tuning'

# Only the following separated configurations are supported

# ['neck', 'rpn_head', 'roi_head'] 
# ['neck', 'roi_head'] 
# ['rpn_head', 'roi_head'] 
# ['roi_head'] 
# []

sep_cfg = ['neck', 'rpn_head', 'roi_head']

model = dict(
    type='BOSS',
    frozen_parameters=[
        'backbone', 
        'neck', 
        'rpn_head', 
        'roi_head.bbox_head.shared_fcs',
    ],
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101, frozen_stages=4),
    neck=dict(
        type='SepFPN',
        sep_cfg=sep_cfg
    ),
    rpn_head=dict(
        type='SepRPNHead',
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=5.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        sep_cfg=sep_cfg
    ),
    roi_head=dict(
        type='OSRRoIHead',
        sep_cfg=sep_cfg,
        bbox_head=dict(type='OSRBBoxHead', 
                       num_classes=10,
                       base_group=10,
                       novel_group=5,
                       fewshot=True,
                       sep_cfg=sep_cfg,
                       n_base=7,
                       n_novel=3,
                       init_cfg=[
                            dict(
                                type='Caffe2Xavier',
                                override=dict(type='Caffe2Xavier', name='shared_fcs')),
                            dict(
                                type='Normal',
                                override=dict(type='Normal', name='fc_reg', std=0.001))
                            ],
                       ))
    )