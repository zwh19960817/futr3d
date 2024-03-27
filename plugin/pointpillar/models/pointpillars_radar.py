from mmdet3d.models import builder
from mmdet3d.models.builder import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from .pillarnet import PillarNet
from mmdet3d.core import (Box3DMode, Coord3DMode)
from mmcv.parallel import DataContainer as DC
from os import path as osp
# from mmdet3d.core import show_result
from plugin.visualizer import show_result


@DETECTORS.register_module()
class PointPillars_Radar(MVXTwoStageDetector):
    def __init__(self,
                 use_lidar=False,
                 use_camera=False,
                 use_radar=True,
                 hand=None,
                 pts_backbone=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 radar_pts_backbone=None,
                 radar_pts_neck=None,
                 radar_pts_bbox_head=None,
                 img_backbone=None,
                 img_neck=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(PointPillars_Radar, self).__init__(pts_backbone=pts_backbone,
                                           pts_neck=pts_neck,
                                           pts_bbox_head=pts_bbox_head,
                                           train_cfg=train_cfg,
                                           test_cfg=test_cfg,
                                           pretrained=pretrained,
                                           init_cfg=init_cfg)

        self.use_lidar = use_lidar
        self.use_camera = use_camera
        self.use_radar = use_radar
        self.hand = PillarNet(**hand)
        self.radar_hand = PillarNet(**hand)
        if radar_pts_backbone:
            self.radar_pts_backbone = builder.build_backbone(radar_pts_backbone)
        if radar_pts_neck is not None:
            self.radar_pts_neck = builder.build_neck(radar_pts_neck)
        if radar_pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            radar_pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            radar_pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.radar_pts_bbox_head = builder.build_head(radar_pts_bbox_head)


        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)
        if img_rpn_head is not None:
            self.img_rpn_head = builder.build_head(img_rpn_head)
        if img_roi_head is not None:
            self.img_roi_head = builder.build_head(img_roi_head)

    @property
    def with_radar_pts_neck(self):
        """bool: Whether the detector has a neck in 3D detector branch."""
        return hasattr(self, 'radar_pts_neck') and self.radar_pts_neck is not None

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img)
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    # def extract_radar_pts_feat(self, radar_pts):
    #     """Extract features of points."""
    #     x = self.radar_hand(radar_pts)
    #     x = self.radar_pts_backbone(x)
    #     if self.with_radar_pts_neck:
    #         x = self.radar_pts_neck(x)
    #     return x

    def extract_pts_feat(self, radar_pts, img_feats, img_metas):
        """Extract features of points."""
        x = self.radar_hand(radar_pts)
        x = self.radar_pts_backbone(x)
        if self.with_radar_pts_neck:
            x = self.radar_pts_neck(x)
        return x

    def extract_feat(self, radar_pts, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        radar_pts_feats = self.extract_pts_feat(radar_pts, img_feats, img_metas)
        return (img_feats, radar_pts_feats)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      radar=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor, optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """

        # radar_pts_feats = self.extract_radar_pts_feat(radar_pts=radar)

        img_feats, radar_pts_feats = self.extract_feat(radar_pts=radar,
                                                       img=img,
                                                       img_metas=img_metas)
        losses = dict()

        # losses_pts = self.forward_pts_train(pts_feats, radar_pts_feats, gt_bboxes_3d,
        #                                     gt_labels_3d, gt_bboxes_ignore)
        if radar_pts_feats:
            losses_pts = self.forward_pts_train(radar_pts_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)

        return losses

        # losses_pts = self.forward_pts_train(radar_pts_feats, gt_bboxes_3d,
        #                                     gt_labels_3d, gt_bboxes_ignore)
        # losses.update(losses_pts)

        # return losses
    def forward_pts_train(self,
                          radar_pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.radar_pts_bbox_head(radar_pts_feats)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.radar_pts_bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

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
