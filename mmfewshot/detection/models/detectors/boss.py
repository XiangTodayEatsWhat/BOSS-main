# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector
import numpy as np

@DETECTORS.register_module()
class BOSS(TwoStageDetector):
    """Implementation of `BOSS`"""