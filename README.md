# BOSS-main


This repository is an official implementation of 

### Balanced Orthogonal Subspace Separation Detector for Few-Shot Object Detection in Aerial Imagery
Hongxiang Jiang, Qixiong Wang, Jiaqi Feng, Guangyun Zhang, Jihao Yin

Transactions on Geoscience and Remote Sensing (TGRS) 2024

<p align="center">
<img src=resources/framework.png> 

## Few-Shot DIOR Results and Models

|   Model  | Style |  Base mAP  | Novel mAP |  Config  |  Download  |
| :------: | :---: |  :-------: | :-------: | :------: | :--------: |
|   BOSS (split F)   |  Pretrain  | 73.6 |        | [boss_r101_dior-split1_base-training](configs/detection/boss/dior/split1/boss_r101_dior-split1_base-training.py)   |   [download](https://drive.google.com/drive/folders/1izNGDPZX3vMk0VI9Adnh3o87zKpM-Tw_?usp=drive_link)  |   
| BOSS (split F)   |  5-shot  | 74.1 |    35.1    | [boss_r101_dior-split1_5shot-fine-tuning](configs/detection/boss/dior/split1/boss_r101_dior-split1_5shot-fine-tuning.py)   |   [download](https://drive.google.com/drive/folders/1izNGDPZX3vMk0VI9Adnh3o87zKpM-Tw_?usp=drive_link)   |                    
|   BOSS (split F)   |  10-shot  | 74.5 |    42.0    | [boss_r101_dior-split1_10shot-fine-tuning](configs/detection/boss/dior/split1/boss_r101_dior-split1_10shot-fine-tuning.py)   |  [download](https://drive.google.com/drive/folders/1izNGDPZX3vMk0VI9Adnh3o87zKpM-Tw_?usp=drive_link)    |      
|   BOSS (split F)   |  20-shot  | 73.9 |    46.6    | [boss_r101_dior-split1_20shot-fine-tuning](configs/detection/boss/dior/split1/boss_r101_dior-split1_20shot-fine-tuning.py)   |   [download](https://drive.google.com/drive/folders/1izNGDPZX3vMk0VI9Adnh3o87zKpM-Tw_?usp=drive_link)   |      

## Installation
Install CUDA 11.1 and Python 3.8 with Anaconda firstly, and clone the repository locally:
```shell
conda create -n boss python=3.8
git clone https://github.com/XiangTodayEatsWhat/Inference-InfoGAN.git
```
Then, install PyTorch and torchvision:
```shell
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```
Our BOSS depends on mmfewshot 0.1.0, mmcv-full 1.5.0 and mmdet 2.25.0. 
Please refer to [install.md](/docs/en/install.md) for installation of MMFewShot and build our BOSS, or as follows:
```
pip install mmcv-full==1.5.0
pip install mmdet==2.25.0
pip install mmfewshot
pip install mmcls
```

Other requirements:
```
yapf==0.40.1
numpy==1.23.2
```

## Datasets Preparation


### DIOR dataset
Download [DIOR](https://drive.google.com/drive/folders/1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC) and create the file structure as follows:
```
└── data
    └── DIOR
        └── Annotations
        	└── Horizontal Bounding Boxes
        		└── 00001.xml
        		└── 00002.xml
				└── ...
        └── JPEGImages
        	└── 00001.jpg
        	└── 00002.jpg
			└── ...
    	└── Main
        	└── trainval.txt
        	└── test.txt
    └── other dataset (NWPU...)
└── BOSS-main
```

### NWPU dataset
Download the [NWPU](https://drive.google.com/file/d/1I4uqKKwK2r94k1NLXOpnzuP-mlO73N23/view?usp=drive_link) dataset split in our experiments.
## How to reproduce BOSS 
Following the original implementation, it consists of 3 steps:

- **Step1: Base Pretraining**

  - use all the images and annotations of base classes to train a base model.

- **Step2: Add LoRA to the base model**:

  - create a new bbox head for novel classes finetuning using provided script.
  - add lora to base model (optional roi, rpn and neck).

- **Step3: Novel finetuning**:

  - use the base model from step2 as model initialization and further finetune with few shot datasets.


### Base Pretraining
Train BOSS with single/multiple gpu:
```shell
bash tools/detection/dist_train.sh ${config} ${gpu-num}
```


### Add LoRA to base model
```shell
python tools/detection/misc/initialize_bbox_head_boss.py --src1 ${checkpoint_path} --method random_init --save-dir ${save_dir} --param-name roi_head.bbox_head.fc_reg

## ${checkpoint_path} is the model obtained from base pretraining
## To reproduce the results of NWPU dataset, need to add --nwpu
## The lora modulation part can be adjusted by --lora-name
```


### Novel Finetuning
Finetune BOSS with single/multiple gpus:
```shell
bash tools/detection/dist_train.sh ${config} ${gpu-num} --seed ${seed}
```

## Evaluation
Evaluate BOSS with single/multiple gpu:
```shell
bash tools/detection/dist_test.sh ${config} ${checkpoint_path} ${gpu-num} --eval mAP
```


## Citation
