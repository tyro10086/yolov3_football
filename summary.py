import torch
from thop import clever_format, profile
from torchsummary import summary

from nets.yolo import YoloBody

#   训练好网络后，用来获取网络信息：包括网络结构、参数量、浮点运算次数
if __name__ == "__main__":
    input_shape = [416, 416]
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes = 80

    #   看网络结构
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = YoloBody(anchors_mask, num_classes).to(device)
    summary(m, (3, input_shape[0], input_shape[1]))

    #   评估计算浮点运算次数和参数数量
    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params = profile(m.to(device), (dummy_input,), verbose=False)
    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
