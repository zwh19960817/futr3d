from typing import Any, Dict
import numpy as np
from numpy import random
import mmcv
import cv2
import torch
from PIL import Image
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import RandomFlip
from mmcv.utils import build_from_cfg
from mmdet3d.core.points.lidar_points import LiDARPoints


@PIPELINES.register_module()
class PointsDimFilter(object):
    """
     提取需要的点维度
    """

    def __init__(self, use_dim=[0, 1, 2]):
        self.use_dim = use_dim

    def __call__(self, results):
        # 注意camera2lidar外参之后也要改动
        points = results['points'].tensor.numpy()
        results['points'] = LiDARPoints(points[:, self.use_dim],
                                        points_dim=len(self.use_dim))
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(use_dim={self.use_dim})'
        return repr_str
