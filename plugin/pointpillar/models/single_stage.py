import torch.nn as nn
from mmdet3d.models.builder import DETECTORS, build_backbone, build_neck, build_head
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from .pillarnet import PillarNet
from mmdet.core import multi_apply


@DETECTORS.register_module()
class PointPillar(MVXTwoStageDetector):
    def __init__(self,
                 use_lidar=True,
                 use_camera=False,
                 use_radar=False,
                 use_grid_mask=False,
                 point_cloud_range=None,
                 hand=None,
                 backbone=None,
                 neck=None,
                 pts_bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(PointPillar, self).__init__(init_cfg=init_cfg)

        self.use_lidar = use_lidar
        self.use_camera = use_camera
        self.use_radar = use_radar
        self.use_grid_mask = use_grid_mask
        hand["point_cloud_range"] = point_cloud_range
        self.hand = PillarNet(**hand)
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)

        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = build_head(pts_bbox_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def preprocess(self, lidar_pts_list: list, radar_pts_list: list):
        """Directly extract features from the backbone+neck."""
        if self.hand.pt_type == 'lidar':
            listed_pts = lidar_pts_list
        elif self.hand.pt_type == 'radar':
            listed_pts = radar_pts_list
        batched_tensor = self.hand(listed_pts)
        return batched_tensor

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.pts_bbox_head(x)
        return outs

    def extract_img_feat(self, img, img_metas):
        pass

    def extract_pts_feat(self, pts):
        x = self.hand(pts)
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_radar_feat(self, radar, img_metas):
        pass

    def extract_feat(self, points, img=None, radar=None, img_metas=None):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas) if self.use_camera else None
        pts_feats = self.extract_pts_feat(points) if self.use_lidar else None
        radar_feats = self.extract_radar_feat(radar, img_metas) if self.use_radar else None
        return (img_feats, pts_feats, radar_feats)

    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          radar_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.pts_bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

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
        img_feats, pts_feats, radar_feats = self.extract_feat(
            points=points, img=img, radar=radar, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(pts_feats, img_feats, radar_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, pts_feats, radar_feats = self.extract_feat(
            points, img=img, radar=None, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                pts_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list

    def extract_feats(self, points, img_metas, imgs=None):
        """Extract point and image features of multiple samples."""
        if imgs is None:
            imgs = [None] * len(img_metas)
        img_feats, pts_feats, radar_feats = multi_apply(self.extract_feat, points, imgs,
                                           None, img_metas)
        return img_feats, pts_feats
