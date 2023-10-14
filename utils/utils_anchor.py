from __future__ import division, print_function

import numpy as np
import random
import math

#   计算框和簇的IOU
def cas_iou(box, cluster):
    x = np.minimum(cluster[:, 0], box[0])
    y = np.minimum(cluster[:, 1], box[1])

    intersection = x * y
    area1 = box[0] * box[1]

    area2 = cluster[:,0] * cluster[:,1]
    iou = intersection / (area1 + area2 - intersection)

    return iou

def kmeans(box, k):
    #   获取框个数
    row = box.shape[0]
    #   存储框和每个簇的距离
    distance = np.empty((row, k))
    #   存储上次结果
    last_clu = np.zeros((row, ))

    np.random.seed()
    #   随机选择k个中心
    cluster = box[np.random.choice(row, k, replace = False)]

    iter = 0
    while True:
        #   计算每个框和每个簇的中心的IOU
        for i in range(row):
            distance[i] = 1 - cas_iou(box[i], cluster)

        #   获得每个框距离哪个簇最近 尺寸(row,)
        near = np.argmin(distance, axis=1)
        #   两次划分结果一致，则结束循环
        if (last_clu == near).all():
            break

        #   把簇中心变为中位数
        for j in range(k):
            clu = box[near == j]
            if len(clu) != 0:
                cluster[j] = np.median(clu,axis=0)

        last_clu = near
        iter += 1

    #   返回簇中心和聚类结果
    return cluster, near

def autoanchor(labels, input_shape, anchors_num=9):
    #   变成数组，获取wh维度
    labels = np.array(labels, dtype=np.float32)
    boxes = labels[:, 2:4]

    #反归一化，之后四舍五入
    boxes[:, 0] = boxes[:, 0] * input_shape[0]
    boxes[:, 1] = boxes[:, 1] * input_shape[1]
    boxes[:, 0] = list(map(round, boxes[:, 0]))
    boxes[:, 1] = list(map(round, boxes[:, 1]))

    #   kmeans聚类找合适的尺寸
    anchors, ave_iou = kmeans(boxes, anchors_num)
    anchors = np.array(anchors, dtype=np.int)
    anchors = list(anchors)
    #   按照面积升序排序
    anchors.sort(key=lambda x: x[0] * x[1], reverse=False)

    return anchors

