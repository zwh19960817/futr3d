import mmcv
import torch
import math
import time
import os
import copy
import numpy as np
from torchvision import utils as vutils
from mmdet3d.datasets.builder import PIPELINES
from nuscenes.utils.data_classes import RadarPointCloud
from mmdet3d.core.points.lidar_points import LiDARPoints
from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from mmdet3d.core.bbox import box_np_ops
from mmdet3d.datasets.pipelines import data_augment_utils
from mmdet3d.datasets.builder import OBJECTSAMPLERS
from mmdet3d.datasets.pipelines.dbsampler import BatchSampler
# from plugin.fudet.core.points import RadarPoints
from plugin.futr3d.core.points import RadarPoints

# def reduce_LiDAR_beams(pts, reduce_beams_to=32, chosen_beam_id=13):
#     #print(pts.size())
#     if isinstance(pts, np.ndarray):
#         pts = torch.from_numpy(pts)
#     radius = torch.sqrt(pts[:, 0].pow(2) + pts[:, 1].pow(2) + pts[:, 2].pow(2))
#     sine_theta = pts[:, 2] / radius
#     # [-pi/2, pi/2]
#     theta = torch.asin(sine_theta)
#     phi = torch.atan2(pts[:, 1], pts[:, 0])
#
#     top_ang = 0.1862
#     down_ang = -0.5353
#
#     beam_range = torch.zeros(32)
#     beam_range[0] = top_ang
#     beam_range[31] = down_ang
#
#     for i in range(1, 31):
#         beam_range[i] = beam_range[i-1] - 0.023275
#     # beam_range = [1, 0.18, 0.15, 0.13, 0.11, 0.085, 0.065, 0.03, 0.01, -0.01, -0.03, -0.055, -0.08, -0.105, -0.13, -0.155, -0.18, -0.205, -0.228, -0.251, -0.275,
#     #                -0.295, -0.32, -0.34, -0.36, -0.38, -0.40, -0.425, -0.45, -0.47, -0.49, -0.52, -0.54]
#
#     num_pts, _ = pts.size()
#     mask = torch.zeros(num_pts)
#     if reduce_beams_to == 16:
#         for id in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]:
#             beam_mask = (theta < (beam_range[id-1]-0.012)) * (theta > (beam_range[id]-0.012))
#             mask = mask + beam_mask
#         mask = mask.bool()
#     elif reduce_beams_to == 4:
#         for id in chosen_beam_id:
#             beam_mask = (theta < (beam_range[id-1]-0.012)) * (theta > (beam_range[id]-0.012))
#             mask = mask + beam_mask
#         mask = mask.bool()
#     # pick the 14th beam
#     elif reduce_beams_to == 1:
#         mask = (theta <(beam_range[chosen_beam_id[0]-1]-0.012)) * (theta > (beam_range[chosen_beam_id[0]]-0.012))
#
#     points = pts[mask]
#
#     return points
        


