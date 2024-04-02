plugin = 'plugin/pointpillar'

# key Parameters      >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
batch_size = 32
num_works = 8
max_epochs = 42

# 数据集相关 >>>
# For nuscenes >>
dataset_type = 'NuScenesDataset'
# data_root = '/mnt/data/adt_dataset/nuscenes/' # mini in server
data_root = '/mnt/data/adt_dataset/OpenDataLab___nuScenes/nuscenes/' # full in server
DataBaseSampler = 'DataBaseSamplerNuscenes'
ann_file_prefix = data_root + 'nuscenes_'  # nuscenes
load_dim = 5  # xyzit
offset_z = 1.86  # 平移地面至z_ground=0
# For nuscenes <<

# # For CYW >>
# dataset_type = 'CYWDataset'
# data_root = '/media/zwh/ZWH4T/ZWH/Dataset3d/final/dataset_18xx_final_test'
# # data_root = '/mnt/data/adt_dataset/dataset_18xx_final_test/'
# DataBaseSampler = 'DataBaseSamplerf'
# ann_file_prefix = 'data_base/'  # cyw
# load_dim = 3  # xyz
# offset_z = 0.0  # 平移地面至z_ground=0   1.86 For Nusencens; 0.0 For CYW
# # For CYW <<
# 数据集相关 <<<

# 速度相关配置 >>>
with_velocity = False
bbox_code_size = 7 + (2 if with_velocity else 0)
bbox_code_weight = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
custom_values = []  # anchor的附加值 如x/y速度
# 速度相关配置 <<<
resume_from = 'work_dirs/pointpillars_lidar_only/latest.pth'
# key Parameters      <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# Configs of Datasets >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
point_cloud_range = [-50, -50, -3, 50, 50, 5]
voxel_size = [0.25, 0.25, point_cloud_range[5] - point_cloud_range[2]]

# x y z
# radar_use_dims = [0, 1, 2]

# x y z rcs vx vy
radar_use_dims = [0, 1, 2, 5, 8, 9]

# For nuScenes we usually do 10-class detection
# class_names = [
#     'car', 'pedestrian'
# ]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=False,
    use_camera=False,
    use_radar=True,
    use_map=False,
    use_external=False)

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=load_dim,
        use_dim=3,
        file_client_args=file_client_args),
    # dict(
    #     type='LoadRadarPointsMultiSweeps',
    #     load_dim=18,
    #     sweeps_num=10,
    #     use_dim=radar_use_dims,
    #     max_num=1200, ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='PointsDimFilter', use_dim=[0, 1, 2]),
    dict(type='NormalizeGround', offset_z=offset_z),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925 * 2, 0.3925 * 2],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[1.5, 1.5, 0.5]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    # dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'radar'])
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=load_dim,
        use_dim=3,
        file_client_args=file_client_args),
    # dict(
    #     type='LoadRadarPointsMultiSweeps',
    #     load_dim=18,
    #     sweeps_num=10,
    #     use_dim=radar_use_dims,
    #     max_num=1200, ),
    dict(type='PointsDimFilter', use_dim=[0, 1, 2]),
    dict(type='NormalizeGround', offset_z=offset_z),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            # dict(type='Collect3D', keys=['points', 'radar'])
            dict(type='Collect3D', keys=['points'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=load_dim,
        use_dim=3,
        file_client_args=file_client_args),
    # dict(
    #     type='LoadRadarPointsMultiSweeps',
    #     load_dim=18,
    #     sweeps_num=10,
    #     use_dim=radar_use_dims,
    #     max_num=1200, ),
    dict(type='PointsDimFilter', use_dim=[0, 1, 2]),
    dict(type='NormalizeGround', offset_z=offset_z),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    # dict(type='Collect3D', keys=['points', 'radar'])
    dict(type='Collect3D', keys=['points'])
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=num_works,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_prefix + 'infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        with_velocity=with_velocity,
        modality=input_modality,
        test_mode=False,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_prefix + 'infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        with_velocity=with_velocity,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_prefix + 'infos_train.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        with_velocity=with_velocity,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))
# Configs of Datasets <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# Configs of Model    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
model = dict(
    type='PointPillars_Lidar',
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    hand=dict(
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_num_points=32,
        max_voxels=[16000, 40000],
        out_channel=64,
        pt_type='lidar'),
    radar_pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    radar_pts_neck=dict(
        type='FPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU'),
        in_channels=[64, 128, 256],
        out_channels=256,
        start_level=0,
        num_outs=3),
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=len(class_names),
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [point_cloud_range[0], point_cloud_range[1], 0.0,
                 point_cloud_range[3], point_cloud_range[4], 0.0]],
            scales=[1, 2, 4],
            sizes=[
                [4.60718145, 1.95017717, 1.72270761],  # car
                [6.73778078, 2.4560939, 2.73004906],  # truck
                [1.68452161, 0.60058911, 1.27192197],  # bicycle
                [0.7256437, 0.66344886, 1.75748069],  # pedestrian
            ],
            custom_values=custom_values,
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=-0.7854,  # -pi / 4
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=bbox_code_size),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            allowed_border=0,
            code_weight=bbox_code_weight,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=1000,
            nms_thr=0.2,
            score_thr=0.1,
            min_bbox_size=0,
            max_num=500)))
# Configs of Model    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# Configs of Schedules >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
workflow = [('train', 5), ('val', 1)]
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.01)
# max_norm=10 is better for SECOND
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[20, 23])
momentum_config = None
# Configs of Schedules <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# Configs of Others    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
evaluation = dict(interval=21, pipeline=eval_pipeline)
checkpoint_config = dict(interval=1, max_keep_ckpts=5)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
# Configs of Others    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
