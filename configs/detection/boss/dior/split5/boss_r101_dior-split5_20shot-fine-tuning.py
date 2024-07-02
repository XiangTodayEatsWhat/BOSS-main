_base_ = [
    '../../../_base_/datasets/fine_tune_based/few_shot_dior.py',
    '../../../_base_/schedules/schedule.py', '../../boss_r101.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotDIORDataset
data = dict(
    train=dict(
        type='FewShotDIORDataset',
        num_novel_shots=20,
        num_base_shots=20,
        classes='ALL_CLASSES_SPLIT5'),
    val=dict(classes='ALL_CLASSES_SPLIT5'),
    test=dict(classes='ALL_CLASSES_SPLIT5'))
evaluation = dict(
    interval=2000,
    class_splits=['BASE_CLASSES_SPLIT5', 'NOVEL_CLASSES_SPLIT5'])
checkpoint_config = dict(interval=2000)

optimizer = dict(lr=0.01)
lr_config = dict(
    warmup_iters=1000, step=[
        6000, 8000
    ])
runner = dict(max_iters=10000)

# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head_boss.py
# please refer to configs/detection/boss/README.md for more details.
load_from = ('work_dirs/dior/split5/boss_r101_dior-split5_base-training/base_model_random_init_bbox_head.pth')
work_dir = 'work_dirs/dior/split5/boss_r101_dior-split5_20shot-fine-tuning'

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
                       num_classes=20,
                       base_group=10,
                       novel_group=5,
                       fewshot=True,
                       sep_cfg=sep_cfg,
                       n_novel=5,
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