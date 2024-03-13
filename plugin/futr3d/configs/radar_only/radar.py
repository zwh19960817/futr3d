_base_ = [
    '../../../../configs/_base_/datasets/nus-3d.py',
    '../../../../configs/_base_/schedules/cyclic_20e.py',
    '../../../../configs/_base_/default_runtime.py'
]

plugin = 'plugin/futr3d'

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
radar_voxel_size = [0.8, 0.8, 8]
# x y z rcs vx_comp vy_comp x_rms y_rms vx_rms vy_rms
radar_use_dims = [0, 1, 2, 8, 9, 18]

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

input_modality = dict(
    use_lidar=False,
    use_camera=False,
    use_radar=True,
    use_map=False,
    use_external=False)

model = dict(
    type='FUTR3D',
    use_lidar=False,
    use_camera=False,
    use_radar=True,
    use_grid_mask=True,
    radar_voxel_layer=dict(
        max_num_points=10,
        voxel_size=radar_voxel_size,
        max_voxels=(90000, 120000),
        point_cloud_range=point_cloud_range),
    radar_voxel_encoder=dict(
        type='RadarFeatureNet',
        in_channels=6,
        feat_channels=[32, 64],
        with_distance=False,
        point_cloud_range=point_cloud_range,
        voxel_size=radar_voxel_size,
        norm_cfg=dict(
            type='BN1d',
            eps=1.0e-3,
            momentum=0.01)
    ),
    radar_middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        output_shape=[128, 128],
    ),
    pts_bbox_head=dict(
        type='FUTR3DHead',
        use_dab=True,
        anchor_size=3,
        num_query=900,
        num_classes=10,
        in_channels=256,
        pc_range=point_cloud_range,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        transformer=dict(
            type='FUTR3DTransformer',
            use_dab=True,
            decoder=dict(
                type='FUTR3DTransformerDecoder',
                num_layers=6,
                use_dab=True,
                anchor_size=3,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='FUTR3DAttention',
                            use_lidar=False,
                            use_camera=False,
                            use_radar=True,
                            pc_range=point_cloud_range,
                            embed_dims=256)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0)),
    # training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),  # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range))),
    test_cfg=dict(pts=dict(
        max_num=300,
        score_threshold=0,
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    )))

dataset_type = 'NuScenesDataset'
# data_root = '/mnt/data/adt_dataset/nuscenes/'
data_root = '/media/zwh/ZWH4T/ZWH/Dataset3d/nuscenes/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=3,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(
        type='LoadRadarPointsMultiSweeps',
        load_dim=18,
        sweeps_num=10,
        use_dim=radar_use_dims,
        max_num=1200, ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d','points', 'radar'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=3,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(
        type='LoadRadarPointsMultiSweeps',
        load_dim=18,
        sweeps_num=10,
        use_dim=radar_use_dims,
        max_num=1200, ),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'radar'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        ann_file=data_root + 'nuscenes_infos_val.pkl'),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        ann_file=data_root + 'nuscenes_infos_train.pkl'))
# For nuScenes dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 24. Please change the interval accordingly if you do not
# use a default schedule.
evaluation = dict(interval=30)

find_unused_parameters=True

custom_hooks = [dict(type='FadeOjectSampleHook', num_last_epochs=5)]

checkpoint_config = dict(interval=10, max_keep_ckpts=5)

log_config = dict(
    interval=4)
runner = dict(type='EpochBasedRunner', max_epochs=400)
workflow = [('train', 5),('val', 1)]

# resume_from = 'work_dirs/lidar_0075v_900q_1b_server/latest.pth'

