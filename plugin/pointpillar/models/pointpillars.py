import torch
from mmdet3d.models.builder import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from .pillarnet import PillarNet
from mmdet3d.core import (Box3DMode, Coord3DMode)
from mmcv.parallel import DataContainer as DC
from os import path as osp
# from mmdet3d.core import show_result
from plugin.visualizer import show_result


@DETECTORS.register_module()
class PointPillars(MVXTwoStageDetector):
    def __init__(self,
                 use_lidar=True,
                 use_camera=False,
                 use_radar=False,
                 use_grid_mask=False,
                 hand=None,
                 pts_backbone=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(PointPillars, self).__init__(pts_backbone=pts_backbone,
                                           pts_neck=pts_neck,
                                           pts_bbox_head=pts_bbox_head,
                                           train_cfg=train_cfg,
                                           test_cfg=test_cfg,
                                           pretrained=pretrained,
                                           init_cfg=init_cfg)

        self.use_lidar = use_lidar
        self.use_camera = use_camera
        self.use_radar = use_radar
        self.use_grid_mask = use_grid_mask
        self.hand = PillarNet(**hand)

    # # 用于输出两个onnx
    # def forward(self, input: torch.Tensor, part: str = 'rpn_backbone'):
    #     if part == 'vfe_model':
    #         return self.hand.pillarencoder.pfn_layer(input)
    #     elif part == 'rpn_backbone':
    #         return self.pts_bbox_head(self.pts_neck(self.pts_backbone(input)))
    #     else:
    #         raise TypeError('{} should be in [vfe_model, rpn_backbone]')

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        x = self.hand(pts)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def show_results(self, data, result, out_dir, show=False):
        """Results visualization.

        Args:
            data (dict): Input points and the information of the sample.
            result (dict): Prediction results.
            out_dir (str): Output directory of visualization result.
        """
        for batch_id in range(len(result)):
            if isinstance(data['points'][0], DC):
                points = data['points'][0]._data[0][batch_id].numpy()
            elif mmcv.is_list_of(data['points'][0], torch.Tensor):
                points = data['points'][0][batch_id]
            else:
                ValueError(f"Unsupported data type {type(data['points'][0])} "
                           f'for visualization!')
            if isinstance(data['img_metas'][0], DC):
                pts_filename = data['img_metas'][0]._data[0][batch_id][
                    'pts_filename']
                box_mode_3d = data['img_metas'][0]._data[0][batch_id][
                    'box_mode_3d']
            elif mmcv.is_list_of(data['img_metas'][0], dict):
                pts_filename = data['img_metas'][0][batch_id]['pts_filename']
                box_mode_3d = data['img_metas'][0][batch_id]['box_mode_3d']
            else:
                ValueError(
                    f"Unsupported data type {type(data['img_metas'][0])} "
                    f'for visualization!')
            file_name = osp.split(pts_filename)[-1].split('.')[0]

            assert out_dir is not None, 'Expect out_dir, got none.'
            inds = result[batch_id]['pts_bbox']['scores_3d'] > 0.1
            pred_bboxes = result[batch_id]['pts_bbox']['boxes_3d'][inds]
            pred_labels = result[batch_id]['pts_bbox']['labels_3d'][inds]

            # for now we convert points and bbox into depth mode
            if (box_mode_3d == Box3DMode.CAM) or (box_mode_3d
                                                  == Box3DMode.LIDAR):
                points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                                   Coord3DMode.DEPTH)
                pred_bboxes = Box3DMode.convert(pred_bboxes, box_mode_3d,
                                                Box3DMode.DEPTH)
                if box_mode_3d == Box3DMode.LIDAR and pred_bboxes.tensor.shape[1] == 9:  # 注意速度也要变换
                    pred_bboxes.tensor = pred_bboxes.tensor[:, [0, 1, 2, 3, 4, 5, 6, 8, 7]]
                    pred_bboxes.tensor[:, 7:8] = -pred_bboxes.tensor[:, 7:8]
            elif box_mode_3d != Box3DMode.DEPTH:
                ValueError(
                    f'Unsupported box_mode_3d {box_mode_3d} for conversion!')

            pred_bboxes = pred_bboxes.tensor.cpu().numpy()
            show_result(points, None, pred_bboxes, out_dir, file_name, show, show_yaw=True, pred_labels=pred_labels,
                        show_speed=True)
