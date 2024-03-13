import torch
from torch import nn
from mmcv.ops import Voxelization
import torch.nn.functional as F
from mmcv.runner import force_fp32


class PillarNet(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels, out_channel, pt_type='lidar'):
        super().__init__()
        self.pt_type = pt_type
        if self.pt_type == 'lidar':
            in_channel = 9
            self.num_features = 3  # use [xyz] for lidar
        elif self.pt_type == 'radar':
            in_channel = 10
            self.num_features = 3  # use [xyz] for radar
        self.voxelizer = Voxelizer(voxel_size, point_cloud_range, max_num_points, max_voxels)
        self.pillarencoder = PillarEncoder(voxel_size, point_cloud_range, in_channel, out_channel)

    def forward(self, batched_pts):
        pillars, npoints_per_pillar, coors_batch = self.voxelizer(batched_pts)
        batched_canvas = self.pillarencoder(pillars[:, :, 0:self.num_features], npoints_per_pillar, coors_batch)
        return batched_canvas


class Voxelizer(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super().__init__()
        self.voxel_layer = Voxelization(voxel_size=voxel_size,
                                        point_cloud_range=point_cloud_range,
                                        max_num_points=max_num_points,
                                        max_voxels=max_voxels)

    @torch.no_grad()
    @force_fp32()
    def forward(self, batched_pts):
        pillars, coors, npoints_per_pillar = [], [], []
        for i, pts in enumerate(batched_pts):
            pts[0, 0] = 0.75
            pts[0, 1] = 0.25
            pts[0, 2] = 2
            # 输入点云[n,c],输出 [有效pillar [p,m,c], 有效pillar在map中的坐标[p,3], 有效pillar实际点数[p,](实际数量<=m)]
            voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(pts)
            # voxels_out: (max_voxel, num_points, c), coors_out: (max_voxel, 3)
            # num_points_per_voxel_out: (max_voxel, )
            pillars.append(voxels_out)
            coors.append(coors_out.long())
            npoints_per_pillar.append(num_points_per_voxel_out)

        pillars = torch.cat(pillars, dim=0)  # (p1 + p2 + ... + pb, num_points, c) 合并batch,为什么???
        npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0)  # (p1 + p2 + ... + pb, )
        coors_batch = []
        for i, cur_coors in enumerate(coors):  # 这一步把batch的id,记在了pillar坐标的第0维
            coors_batch.append(F.pad(cur_coors, (1, 0), value=i))
        coors_batch = torch.cat(coors_batch, dim=0)  # (p1 + p2 + ... + pb, 1 + 3)

        return pillars, npoints_per_pillar, coors_batch


class PFNLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv = nn.Conv1d(in_channel, out_channel, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)

    def forward(self, features):
        features = F.relu(self.bn(self.conv(features)))  # (p1 + p2 + ... + pb, out_channels, num_points)
        pooling_features = torch.max(features, dim=-1)[0]  # (p1 + p2 + ... + pb, out_channels)
        return pooling_features


class PillarEncoder(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, in_channel, out_channel):
        super().__init__()
        self.in_channel = in_channel  # 每个点8/10个特征
        self.out_channel = out_channel
        self.vx, self.vy, self.vz = voxel_size[0], voxel_size[1], voxel_size[2]
        self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        self.z_offset = voxel_size[2] / 2 + point_cloud_range[2]
        self.x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_l = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])

        self.pfn_layer = PFNLayer(self.in_channel, out_channel)

    def forward(self, pillars, npoints_per_pillar, coors_batch):
        '''
        pillars: (p1 + p2 + ... + pb, num_points, c), c = 3
        npoints_per_pillar: (p1 + p2 + ... + pb, )
        coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        return:  (bs, out_channel, y_l, x_l)
        '''
        device = pillars.device
        # bzyx转bxyz
        # coors_batch = coors_batch[:, [0, 3, 2, 1]]
        # 1. calculate offset to the points center (in each pillar)  点位置(x,y,z)-pillar中心位置
        points_mean = (
                pillars[:, :, :3].sum(dim=1, keepdim=True) /
                npoints_per_pillar.type_as(pillars).view(-1, 1, 1))
        offset_pt_center = pillars[:, :, :3] - points_mean  # (p1 + p2 + ... + pb, num_points, 3)

        # 2. calculate offset to the pillar center  点云位置(x,y) - pillar栅格位置
        x_offset_pi_center = pillars[:, :, 0:1] - (
                coors_batch[:, None, 3:4] * self.vx + self.x_offset)  # (p1 + p2 + ... + pb, num_points, 1)
        y_offset_pi_center = pillars[:, :, 1:2] - (
                coors_batch[:, None, 2:3] * self.vy + self.y_offset)  # (p1 + p2 + ... + pb, num_points, 1)
        z_offset_pi_center = pillars[:, :, 2:3] - (
                coors_batch[:, None, 1:2] * self.vz + self.z_offset)  # (p1 + p2 + ... + pb, num_points, 1)

        # 3. encoder
        # [x,y,z,adx,ady,adz,cdx,cdy,cdy]
        features = torch.cat([pillars[:, :, :],
                              offset_pt_center,
                              x_offset_pi_center,
                              y_offset_pi_center,
                              z_offset_pi_center],
                             dim=-1)  # (p1 + p2 + ... + pb, num_points, 3+3+2)

        # 前两个维度换成 [x,y,z,adx,ady,adz,cdx,cdy] -> [cdx,cdy,z,adx,ady,adz,cdx,cdy]
        features[:, :, 0:1] = x_offset_pi_center  # tmp
        features[:, :, 1:2] = y_offset_pi_center  # tmp
        # In consitent with mmdet3d.
        # The reason can be referenced to https://github.com/open-mmlab/mmdetection3d/issues/1150

        # 4. find mask for (0, 0, 0) and update the encoded features  就是只保留每个pillar中有点的，无点的用0补充
        # a very beautiful implementation
        voxel_ids = torch.arange(0, pillars.size(1)).to(device)  # (num_points, )
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :]  # (num_points, p1 + p2 + ... + pb)
        mask = mask.permute(1, 0).contiguous()  # (p1 + p2 + ... + pb, num_points)标记pillar中32点是否存在
        features *= mask[:, :, None]

        # 5. embedding  所谓pointnet pillar数作为了batch维度，所以可变
        features = features.permute(0, 2, 1).contiguous()  # (p1 + p2 + ... + pb, 9, num_points)
        pooling_features = self.pfn_layer(features)

        # 6. pillar scatter
        batched_canvas = []
        bs = coors_batch[-1, 0] + 1
        for i in range(bs):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]  # 取出i号batch的pillar坐标
            cur_features = pooling_features[cur_coors_idx]  # 取出i号batch的pillar特征

            canvas = torch.zeros((self.x_l, self.y_l, self.out_channel), dtype=torch.float32, device=device)
            canvas[cur_coors[:, 3], cur_coors[:, 2]] = cur_features  # 按照索引在map中存pillar特征
            canvas = canvas.permute(2, 1, 0).contiguous()  # 此时是cyx
            # canvas = canvas.permute(2, 0, 1).contiguous()
            batched_canvas.append(canvas)
        batched_canvas = torch.stack(batched_canvas, dim=0)  # (bs, in_channel, self.y_l, self.x_l) #append是list，合并batch
        return batched_canvas


