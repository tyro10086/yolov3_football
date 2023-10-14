from collections import OrderedDict

import torch
import torch.nn as nn

from nets.darknet import darknet53


# DBL块
def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

# concat之后 5层DBL加1层DBL加一层1x1卷积
# 前5层DBL输出的通道数是filters_list[0]
def make_last_layers(filters_list, in_filters, out_filter):
    return nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
    )

class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, pretrained = False):
        super(YoloBody, self).__init__()
        #   生成darknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        self.backbone = darknet53()
        if pretrained:
            self.backbone.load_state_dict(torch.load("model_data/darknet53_backbone_weights.pth"))

        out_filters = [64, 128, 256, 512, 1024]

        # darknet53最后一层
        self.last_layer0 = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))

        # concat前的上采样与卷积
        self.last_layer1_conv = conv2d(512, 256, 1)
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # darknet53倒数第二层
        self.last_layer1 = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))

        # concat前的上采样与卷积
        self.last_layer2_conv = conv2d(256, 128, 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # darknet53倒数第三层
        self.last_layer2 = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))

    def forward(self, x):
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256；26,26,512；13,13,1024
        x2, x1, x0 = self.backbone(x)

        #   第一个特征层
        # 5个DBL层以后的输出 13,13,512
        out0_branch = self.last_layer0[:5](x0)
        # lastlayer走完以后的输出 13,13,255
        out0 = self.last_layer0[5:](out0_branch)

        # 卷积与上采样，准备与上一层的输出concat
        x1_in = self.last_layer1_conv(out0_branch) # 13,13,256
        x1_in = self.last_layer1_upsample(x1_in) # 26,26,256

        # concat 26,26,256 + 26,26,512 -> 26,26,768
        x1_in = torch.cat([x1_in, x1], 1)

        #   第二个特征层
        # 5个DBL层以后的输出 26,26,256
        out1_branch = self.last_layer1[:5](x1_in)
        # lastlayer走完以后的输出 26,26,255
        out1 = self.last_layer1[5:](out1_branch)

        # 卷积与上采样，准备与上一层的输出concat
        # 26,26,256 -> 26,26,128 -> 52,52,128
        x2_in = self.last_layer2_conv(out1_branch) # 26,26,128
        x2_in = self.last_layer2_upsample(x2_in) # 52,52,128

        # concat 52,52,128 + 52,52,256 -> 52,52,384
        x2_in = torch.cat([x2_in, x2], 1)

        #   第三个特征层
        # lastlayer走完以后的输出 52,52,255
        out2 = self.last_layer2(x2_in)
        return out0, out1, out2
