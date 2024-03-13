import torch
import torch.nn as nn
from mmdet3d.models.builder import NECKS
from mmcv.runner import BaseModule, auto_fp16


@NECKS.register_module()
class PPNeck(BaseModule):
    def __init__(self,
                 in_channels,
                 upsample_strides,
                 out_channels,
                 init_cfg=None):
        super(PPNeck, self).__init__(init_cfg=init_cfg)
        assert len(in_channels) == len(upsample_strides)
        assert len(upsample_strides) == len(out_channels)

        self.decoder_blocks = nn.ModuleList()
        for i in range(len(in_channels)):
            decoder_block = []
            decoder_block.append(nn.ConvTranspose2d(in_channels[i],
                                                    out_channels[i],
                                                    upsample_strides[i],
                                                    stride=upsample_strides[i],
                                                    bias=False))
            decoder_block.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            decoder_block.append(nn.ReLU(inplace=True))

            self.decoder_blocks.append(nn.Sequential(*decoder_block))

        if init_cfg is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='ConvTranspose2d'),
            ]

    @auto_fp16()
    def forward(self, x):
        '''
        x: [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        return: (bs, 384, 248, 216)
        '''
        outs = []
        for i in range(len(self.decoder_blocks)):
            xi = self.decoder_blocks[i](x[i])  # (bs, 128, 248, 216)
            outs.append(xi)
        out = torch.cat(outs, dim=1)
        return [out]
