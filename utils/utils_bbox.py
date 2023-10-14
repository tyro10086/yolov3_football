import torch
import torch.nn as nn
from torchvision.ops import nms
import numpy as np


class DecodeBox():
    def __init__(self, anchors, num_classes, input_shape, anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]):
        super(DecodeBox, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        #   416x416
        self.input_shape = input_shape
        #   anchors_mask是authors数组的下标
        #   13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
        #   26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
        #   52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
        self.anchors_mask = anchors_mask

    #   回归修正先验框，尽可能贴近ground truth
    def decode_box(self, inputs):
        outputs = []
        for i, input in enumerate(inputs):
            #   input是原始图片经过神经网络之后的输出，归一化过
            #   表示先验框的预测结果：坐标、是否包含物体，种类。 一共有三种shape
            #   batch_size, 255, 13, 13
            #   batch_size, 255, 26, 26
            #   batch_size, 255, 52, 52
            batch_size = input.size(0)
            input_height = input.size(2)
            input_width = input.size(3)

            # 计算步长 每一个特征点对应原来的图片上多少个像素点
            # stride_h = stride_w = 32、16、8
            stride_h = self.input_shape[0] / input_height
            stride_w = self.input_shape[1] / input_width

            #   wh的单位由像素变成网格 3个二元组
            scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in
                              self.anchors[self.anchors_mask[i]]]

            #   输入resize一下 255变成 3x85 代表3种尺寸和85个bbox_attrs
            #   batch_size, 3, 13, 13, 85
            #   batch_size, 3, 26, 26, 85
            #   batch_size, 3, 52, 52, 85
            prediction = input.view(batch_size, len(self.anchors_mask[i]),
                                    self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

            #   先验框的中心位置的调整参数(需要如何修正来让候选框与真实框贴合)
            x = torch.sigmoid(prediction[..., 0])
            y = torch.sigmoid(prediction[..., 1])

            #   先验框的宽高调整参数
            w = prediction[..., 2]
            h = prediction[..., 3]

            #   获得置信度，是否有物体
            conf = torch.sigmoid(prediction[..., 4])

            #   种类置信度
            pred_cls = torch.sigmoid(prediction[..., 5:])

            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

            #   生成网格，网格左上角当作先验框中心，记录其坐标为grid_x grid_y
            #   把[0,1,...,w-1/h-1]重复h/w x 尺寸变为batch_size x 3次
            #   得到batch_size, 3, h/w 个[0,1,...,w-1/h-1]
            grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(y.shape).type(FloatTensor)

            #   按照网格格式生成先验框的宽高
            #   从scaled_anchors分别选取第一维的第1 2位置的数 得到1x3数组
            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            #   重复多次 尺寸变为batch_size,3,w,h
            anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

            #   按照公式，由先验框得到预测框
            pred_boxes = FloatTensor(prediction[..., :4].shape)
            pred_boxes[..., 0] = x.data + grid_x
            pred_boxes[..., 1] = y.data + grid_y
            pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
            pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

            #   输出为修正后的预测框
            #   将输出结果归一化成小数的形式并重新拼接
            #   输出尺寸batch_size, anchor_num, 85
            _scale = torch.Tensor([input_width, input_height, input_width, input_height]).type(FloatTensor)
            output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale,
                                conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
            outputs.append(output.data)
        return outputs

    #   预测框参数的单位由网格变成像素
    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        #   预处理，以确保输入图像与训练网络的期望尺寸相匹配
        if letterbox_image:
            new_shape = np.round(image_shape * np.min(input_shape / image_shape))
            offset = (input_shape - new_shape) / 2. / input_shape
            scale = input_shape / new_shape

            box_yx = (box_yx - offset) * scale
            box_hw *= scale

        #   求左下角和右上角的坐标 拼接 并放缩成像素单位
        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]],
                               axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    #   非极大值抑制
    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5,
                            nms_thres=0.4):
        """
        #   prediction表示预测框，尺寸batch_size, num_anchors, 85
        #   返回的output是非极大值抑制后的预测框，即对于同一个物体，只保留置信度最大的框
        #   尺寸batch_size, num_objects, 7
        #   去掉了所有置信度小于conf_thres的框，以及重合度大于nms_thres的框对中置信度较小的框
        """
        #   将预测结果的格式转换成左上角右下角x1y1x2y2的格式。
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            #   对每个框种类预测部分取max。
            #   class_conf  [num_anchors, 1]    种类置信度
            #   class_pred  [num_anchors, 1]    索引 即种类
            class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

            #   根据阈值制造掩码 bool作用 num_anchors尺寸
            conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

            #   根据掩码筛选图片、置信度、种类
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]

            if not image_pred.size(0):
                continue

            #   detections尺寸[num_anchors, 7] 7的内容为：x1, y1, x2, y2, 物体置信度, 种类置信度, 种类
            detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

            #   获得预测结果中包含的所有种类
            unique_labels = detections[:, -1].cpu().unique()

            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()

            for c in unique_labels:
                #   只保留预测结果为c的行
                detections_class = detections[detections[:, -1] == c]

                #   实际置信度 = 物体置信度 * 种类置信度
                #   非极大值抑制 返回的keep表示要保留的框的索引
                keep = nms(
                    detections_class[:, :4],
                    detections_class[:, 4] * detections_class[:, 5],
                    nms_thres
                )
                max_detections = detections_class[keep]

                # 对于种类c 将要保留的框的属性拼接并存入output
                output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

            #   之前都是对框(网格)的操作 xy的值以网格为单位 值[0,1]需要变换成以像素为单位
            if output[i] is not None:
                output[i] = output[i].cpu().numpy()
                box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4] = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
        return output
