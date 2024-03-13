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
class NormalizeGround(object):
    """
    把地面调到0,不同数据可能不一样
    """

    def __init__(self, offset_z=0.0):
        self.offset_z = offset_z

    def __call__(self, results):
        # 注意camera2lidar外参之后也要改动
        points = results['points'].tensor.numpy()
        points[:, 2] += self.offset_z
        points = LiDARPoints(points,
                             points_dim=points.shape[-1])
        results['points'] = points
        if 'gt_bboxes_3d' in results:
            gt_bboxes = results['gt_bboxes_3d'].tensor
            gt_bboxes[:, 2] += torch.Tensor([self.offset_z])
            results['gt_bboxes_3d'].tensor = gt_bboxes
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(offset_z={self.offset_z})'
        return repr_str
