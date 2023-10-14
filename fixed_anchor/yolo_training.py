import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn

class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, cuda, anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]):
        super(YOLOLoss, self).__init__()
        #   13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
        #   26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
        #   52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
        self.anchors = anchors

        self.giou = True
        #   不同尺寸对应的置信度损失的权重
        self.balance = [0.4, 1.0, 4]
        #   三部分损失的权重
        self.box_ratio = 0.05
        self.obj_ratio = 5 * (input_shape[0] * input_shape[1]) / (416 ** 2)
        self.cls_ratio = 1 * (num_classes / 80)

        self.ignore_threshold = 0.5
        self.cuda = cuda

    # 对张量的clip操作
    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def MSELoss(self, pred, target):
        return torch.pow(pred - target, 2)

    def BCELoss(self, pred, target):
        epsilon = 1e-7
        pred = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output

    def box_giou(self, b1, b2):
        """
        输入为：
        b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        返回为：
        giou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
        """
        # 求出预测框左上角右下角坐标
        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2:4]
        b1_wh_half = b1_wh / 2.
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half
        # 求出真实框左上角右下角坐标
        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2:4]
        b2_wh_half = b2_wh / 2.
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half

        # 求真实框和预测框所有的iou
        intersect_mins = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))  # 重叠部分的宽高
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        union_area = b1_area + b2_area - intersect_area
        iou = intersect_area / union_area

        # 外接矩形的长 宽 面积
        subscribe_mins = torch.min(b1_mins, b2_mins)
        subscribe_maxes = torch.max(b1_maxes, b2_maxes)
        subscribe_wh = torch.max(subscribe_maxes - subscribe_mins, torch.zeros_like(intersect_maxes))
        subscribe_area = subscribe_wh[..., 0] * subscribe_wh[..., 1]

        giou = iou - (subscribe_area - union_area) / subscribe_area

        return giou

    def forward(self, l, input, targets=None):
        """
        #   l代表的是，当前输入进来的有效特征层，是第几个有效特征层 不同层有不同的尺寸
        #   输入的input表示如何修正xywh 减少先验框的偏差 一共有三种shape
        #   batch_size, 255, 13, 13
        #   batch_size, 255, 26, 26
        #   batch_size, 255, 52, 52
        #   targets代表的是真实框。
        """

        # 获得图片数量，特征层的高和宽
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)

        #   计算步长 每一个特征点对应原来的图片上多少个像素点
        #   如果特征层为13x13的话，一个特征点就对应原来的图片上的32个像素点
        #   如果特征层为26x26的话，一个特征点就对应原来的图片上的16个像素点
        #   如果特征层为52x52的话，一个特征点就对应原来的图片上的8个像素点
        #   stride_h = stride_w = 32、16、8
        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w

        #   此时获得的scaled_anchors以网格为单位 9x2尺寸
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

        #   输入resize一下 255变成 3x85 代表3种尺寸和85个bbox_attrs
        #   batch_size, 3, 13, 13, 85
        #   batch_size, 3, 26, 26, 85
        #   batch_size, 3, 52, 52, 85
        prediction = input.view(bs, len(self.anchors_mask[l]), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4,
                                                                                                    2).contiguous()

        #   先验框的中心位置的调整参数
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])

        #   先验框的宽高调整参数
        w = prediction[..., 2]
        h = prediction[..., 3]

        #   获得置信度，是否有物体
        conf = torch.sigmoid(prediction[..., 4])

        #   种类置信度
        pred_cls = torch.sigmoid(prediction[..., 5:])

        #   y_true先验框的各属性值 batch_size, 3, 13, 13, 5 + num_classes
        #   noobj_mask表示负样本，这一步先排除包含物体的先验框 batch_size, 3, 13, 13
        #   box_loss_scale设归一化后的宽高之积 对小目标进行增强 对大目标进行减弱 batch_size, 3, 13, 13
        y_true, noobj_mask, box_loss_scale = self.get_target(l, targets, scaled_anchors, in_h, in_w)

        #   pred_boxes将预测结果进行解码 batch_size, 3, in_h, in_w, 4
        #   noobj_mask再排除重合程度过大的先验框，因为这些特征点预测比较准确，作为负样本不合适
        noobj_mask, pred_boxes = self.get_ignore(l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask)

        if self.cuda:
            y_true = y_true.type_as(x)
            noobj_mask = noobj_mask.type_as(x)
            box_loss_scale = box_loss_scale.type_as(x)

        #   box_loss_scale是真实框宽高的乘积，宽高均在0-1之间，因此乘积也在0-1之间。
        #   2-宽高的乘积代表真实框越大，比重越小，小框的比重更大。
        box_loss_scale = 2 - box_loss_scale

        loss = 0
        #   哪些先验框包含物体 batch_size, 3, 13, 即正样本
        obj_mask = y_true[..., 4] == 1
        n = torch.sum(obj_mask)

        if n != 0:
            #   预测框x, y, w, h的损失
            #   只考虑正样本，即目标存在的框
            if self.giou:
                #   计算预测结果和真实结果的giou，pred_boxes和y_true都以网格为单位
                giou = self.box_giou(pred_boxes, y_true[..., :4]).type_as(x)
                loss_loc = torch.mean((1 - giou)[obj_mask])
            else:
                #   计算预测结果和真实结果的Loss，xy用BCE，wh用MSE
                #   xywh和y_true都进行了sigmoid或者类似于归一化的操作，因此最后乘0.1系数而不是开根号
                #   需要根据目标大小调整权重
                loss_x = torch.mean(self.BCELoss(x[obj_mask], y_true[..., 0][obj_mask]) * box_loss_scale[obj_mask])
                loss_y = torch.mean(self.BCELoss(y[obj_mask], y_true[..., 1][obj_mask]) * box_loss_scale[obj_mask])
                loss_w = torch.mean(self.MSELoss(w[obj_mask], y_true[..., 2][obj_mask]) * box_loss_scale[obj_mask])
                loss_h = torch.mean(self.MSELoss(h[obj_mask], y_true[..., 3][obj_mask]) * box_loss_scale[obj_mask])
                loss_loc = (loss_x + loss_y + loss_h + loss_w) * 0.1

            #   种类置信度损失，计算预测结果和真实结果的BCELoss
            #   只考虑正样本，即目标存在的框
            loss_cls = torch.mean(self.BCELoss(pred_cls[obj_mask], y_true[..., 5:][obj_mask]))
            loss += loss_loc * self.box_ratio + loss_cls * self.cls_ratio

        #   是否有物体的置信度损失，计算预测结果和真实结果的BCELoss
        #   正负样本都有，即目标存在的框，和目标不存在且重合程度小于阈值的框
        loss_conf = torch.mean(self.BCELoss(conf, obj_mask.type_as(conf))[noobj_mask.bool() | obj_mask])
        loss += loss_conf * self.balance[l] * self.obj_ratio
        return loss

    def calculate_iou(self, _box_a, _box_b):
        """
        #   _box_a真实框 尺寸为[num_true_box,4]
        #   _box_b预测框 尺寸为[9, 4]
        """
        #   计算真实框和预测框的左上角和右下角
        b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
        b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
        b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
        b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2

        #   将真实框和预测框都转化成左上角右下角的形式
        box_a = torch.zeros_like(_box_a)
        box_b = torch.zeros_like(_box_b)
        box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
        box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2

        #   真实框和先验框的数量
        A = box_a.size(0)
        B = box_b.size(0)

        #   计算交的面积
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)  # 负数要变成0
        inter = inter[:, :, 0] * inter[:, :, 1]

        #   计算预测框和真实框各自的面积
        area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]

        #   计算并的面积
        union = area_a + area_b - inter
        return inter / union  # [A,B]

    #   利用真实框进行标注
    def get_target(self, l, targets, anchors, in_h, in_w):
        """
        #   l代表的是，当前输入进来的有效特征层，是第几个有效特征层
        #   targets代表的是真实框。尺寸batch_size,num_true_box,5 第五维是种类序号
        #   anchors代表先验框的大小，以网格为单位9x2
        #   in_h和in_w是网格尺寸
        """
        #   同一批有多少张图片
        bs = len(targets)
        #   用于选取哪些先验框不包含物体 batch_size, 3, 13, 13
        noobj_mask = torch.ones(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)
        #   设置目标权重 然后对小目标进行增强对大目标进行减弱 batch_size, 3, 13, 13
        box_loss_scale = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)
        #   先验框的各属性值 batch_size, 3, 13, 13, 5 + num_classes
        y_true = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, self.bbox_attrs, requires_grad=False)

        for b in range(bs):
            if len(targets[b]) == 0:
                continue
            batch_target = torch.zeros_like(targets[b])

            #  之前归一化的数据重新变成以网格为单位
            batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * in_w
            batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * in_h
            batch_target[:, 4] = targets[b][:, 4]
            batch_target = batch_target.cpu()

            #   看在原点重合的情况下，真实框和9种标准框的哪种最重合
            #   也就是哪种形状最贴合，因此坐标都是0
            #   gt_box和anchor_shapes尺寸分别是[num_true_box, 4]和[9, 4]
            gt_box = torch.FloatTensor(torch.cat((torch.zeros(batch_target.size(0), 2), batch_target[:, 2:4]), 1))
            anchor_shapes = torch.FloatTensor(
                torch.cat((torch.zeros(len(anchors), 2), torch.FloatTensor(anchors)), 1))

            #   返回值尺寸[num_true_box, 9] 表示每个真实框和每个先验框的交并比
            #   然后找每个真实框最大的重合度的先验框的序号
            best_ns = torch.argmax(self.calculate_iou(gt_box, anchor_shapes), dim=-1)

            for t, best_n in enumerate(best_ns):
                #	判断最重合的先验框是不是属于当前层的尺寸的先验框 如果是就取出
                if best_n not in self.anchors_mask[l]:
                    continue
                k = self.anchors_mask[l].index(best_n)

                #   获得真实框属于哪个网格点 取出真实框的种类的序号
                i = torch.floor(batch_target[t, 0]).long()
                j = torch.floor(batch_target[t, 1]).long()
                c = batch_target[t, 4].long()

                #   这些真实框有目标，对应的先验框页可以看作有目标，因此置False
                noobj_mask[b, k, j, i] = 0

                if not self.giou:
                    #   对先验框的85个属性进行标注
                    #   如果不用giou，xy坐标只看对于网格原点的偏移量，wh为真实框除以先验框
                    y_true[b, k, j, i, 0] = batch_target[t, 0] - i.float()
                    y_true[b, k, j, i, 1] = batch_target[t, 1] - j.float()
                    y_true[b, k, j, i, 2] = math.log(batch_target[t, 2] / anchors[best_n][0])
                    y_true[b, k, j, i, 3] = math.log(batch_target[t, 3] / anchors[best_n][1])
                    y_true[b, k, j, i, 4] = 1  # 因为是真实框，置信度为1
                    y_true[b, k, j, i, c + 5] = 1
                else:
                    #   如果用giou，不需要额外操作
                    y_true[b, k, j, i, 0] = batch_target[t, 0]
                    y_true[b, k, j, i, 1] = batch_target[t, 1]
                    y_true[b, k, j, i, 2] = batch_target[t, 2]
                    y_true[b, k, j, i, 3] = batch_target[t, 3]
                    y_true[b, k, j, i, 4] = 1
                    y_true[b, k, j, i, c + 5] = 1

                #   用于获得xywh的比例
                #   大目标loss权重小，小目标loss权重大
                box_loss_scale[b, k, j, i] = batch_target[t, 2] * batch_target[t, 3] / in_w / in_h
        return y_true, noobj_mask, box_loss_scale

    def get_ignore(self, l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask):
        """
                #   l代表的是，当前输入进来的有效特征层，是第几个有效特征层 不同层有不同的尺寸
                #   xywh尺寸为  batch_size, 3, in_h, in_w
                #   scaled_anchors以网格为单位 9x2尺寸
                #   targets代表的是真实框。尺寸batch_size, num_true_box, 4
        """
        #   计算一共有多少张图片
        bs = len(targets)

        #   生成网格，先验框中心为网格左上角，尺寸和x一样，值是[0,...in_h-1]的堆叠
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(x.shape).type_as(x)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(y.shape).type_as(x)

        # 生成先验框的宽高 宽和高的值分别是是l层三种scaled_anchors的尺寸的堆叠
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
        anchor_w = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([0])).type_as(x)
        anchor_h = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([1])).type_as(x)
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)

        #   计算调整后的先验框中心与宽高
        pred_boxes_x = torch.unsqueeze(x + grid_x, -1)
        pred_boxes_y = torch.unsqueeze(y + grid_y, -1)
        pred_boxes_w = torch.unsqueeze(torch.exp(w) * anchor_w, -1)
        pred_boxes_h = torch.unsqueeze(torch.exp(h) * anchor_h, -1)
        pred_boxes = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim=-1)

        for b in range(bs):
            #   取出一个batch 尺寸变为num_anchors, 4
            pred_boxes_for_ignore = pred_boxes[b].view(-1, 4)

            if len(targets[b]) > 0:
                #  真实框之前归一化的数据重新变成以网格为单位
                batch_target = torch.zeros_like(targets[b])
                batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * in_w
                batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * in_h
                batch_target = batch_target[:, :4].type_as(x)

                #   计算每个真实框和每个先验框的交并比，尺寸num_true_box, num_anchors
                anch_ious = self.calculate_iou(batch_target, pred_boxes_for_ignore)

                #   每个先验框对应真实框的最大重合度 尺寸3, in_h, in_w
                anch_ious_max, _ = torch.max(anch_ious, dim=0)
                anch_ious_max = anch_ious_max.view(pred_boxes[b].size()[:3])

                #   找出所有交并比大于阈值的先验框
                #   这些先验框属于预测比较准确的，不计入损失
                noobj_mask[b][anch_ious_max > self.ignore_threshold] = 0
        return noobj_mask, pred_boxes

    #   权值初始化
    def weights_init(net, init_type='normal', init_gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and classname.find('Conv') != -1:
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, init_gain)  # 正态
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)  # 正态且输入和输出的方差
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # 正态且一层有一半的神经元
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)  # 正态且正交
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            elif classname.find('BatchNorm2d') != -1:
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

        print('initialize network with %s type' % init_type)
        net.apply(init_func)

#   权值初始化
def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain) #正态
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain) #正态且输入和输出的方差
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') #正态且一层有一半的神经元
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain) #正态且正交
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


#   动态调整学习率
def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.05, step_num=10):
    #   余弦退火学习率调度
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0 + math.cos(
                math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    #   step学习率调度
    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

#   在每个epoch结束后重新设置学习率及优化器
def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr