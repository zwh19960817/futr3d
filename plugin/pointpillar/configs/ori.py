_base_ = [
    '../../../configs/_base_/datasets/nus-3d.py',
    '../../../configs/_base_/schedules/cyclic_20e.py',
    '../../../configs/_base_/default_runtime.py'
]

plugin = 'plugin/pointpillar'

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
voxel_size = [0.2, 0.2, 1]
point_cloud_range = [-24, -24, -10, 72, 24, 3]

model = dict(
    type='PointPillar',
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_grid_mask=False,
    point_cloud_range=point_cloud_range,
    hand=dict(
        voxel_size=voxel_size,
        max_num_points=32,
        max_voxels=[16000, 40000],
        out_channel=64,
        pt_type='lidar'),
    backbone=dict(
        type='PPBackbone',
        in_channel=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5]
    ),
    neck=dict(
        type='PPNeck',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    bbox_head=dict(
        type="SSD3dHEAD",
        in_channel=384,
        n_anchors=6,
        n_classes=len(class_names),
        anchor_cfg=dict(
            levels_cz=[0.6, 1.0, 0.7],
            levels_size=[[0.8, 0.6, 1.70], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],
            rotations=[0, 1.57]
        )
    )
)

dataset_type = 'NuScenesDataset'
data_root = '/mnt/data/adt_dataset/nuscenes/'
# data_root = '/media/zwh/ZWH4T/ZWH/Dataset3d/nuscenes/'
file_client_args = dict(backend='disk')

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    # points_loader=dict(
    #     type='LoadReducedPointsFromFile',
    #     coord_type='LIDAR',
    #     load_dim=5,
    #     use_dim=[0, 1, 2, 3, 4],
    #     reduce_beams_to=1,
    #     chosen_beam_id=[9],
    #     file_client_args=file_client_args))
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=3,
    #     use_dim=[0, 1, 2, 3, 4],
    #     file_client_args=file_client_args,
    #     pad_empty_sweeps=True,
    #     remove_close=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(type='NormalizeGround', offset_z=1.86),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925 * 2, 0.3925 * 2],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.5, 0.5, 0.5]),
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
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
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
            dict(type='Collect3D', keys=['points'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            test_mode=False,
            use_valid_flag=True,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR')),
    val=dict(pipeline=test_pipeline, classes=class_names, ann_file=data_root + 'nuscenes_infos_val.pkl'),
    test=dict(pipeline=test_pipeline, classes=class_names, ann_file=data_root + 'nuscenes_infos_val.pkl'))
# For nuScenes dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 24. Please change the interval accordingly if you do not
# use a default schedule.
evaluation = dict(interval=1)

find_unused_parameters = True

# custom_hooks = [dict(type='FadeOjectSampleHook', num_last_epochs=5)]

checkpoint_config = dict(interval=10, max_keep_ckpts=5)

log_config = dict(
    interval=2)
runner = dict(type='EpochBasedRunner', max_epochs=300)
workflow = [('train', 1), ('val', 1)]

# resume_from = 'work_dirs/lidar_0075v_900q_1b_server/latest.pth'
