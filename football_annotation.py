#   用于处理COCO数据集，根据json文件生成txt文件用于训练
import json
import os
from collections import defaultdict
import numpy as np
from PIL import Image


#   指向了football训练集与验证集图片的路径
train_datasets_path = "football_dataset/images/train"
val_datasets_path = "football_dataset/images/valid"

#   指向了football训练集与验证集标签的路径
train_labels_path = "football_dataset/labels/train"
val_labels_path = "football_dataset/labels/valid"


#   生成的txt文件路径
train_output_path = "football_train.txt"
val_output_path = "football_valid.txt"


#   在进行训练之前，用这部分数据预处理下载的annotations，从中选出有用信息
#   把有用信息写入新文件，用新文件训练
if __name__ == "__main__":
    annotations = []

    #   遍历所有txt文件
    for filename in os.listdir(train_labels_path):
        #   获取内容
        path = os.path.join(train_labels_path, filename)
        f = open(path, 'r')
        lines = f.readlines()
        f.close()

        #   内容变成
        lines = np.array([np.array(list(map(float, line.split(' ')))) for line in lines])

        #   xywh变成x1y1x2y2
        lines[...,1:3] = lines[...,1:3] - lines[...,3:5] / 2
        lines[...,3:5] += lines[...,1:3]

        #   获取的名字、种类、真实框存入dict
        #   每个dict都是一个框的标注
        name = filename.split('.')[0]
        for ls in lines:
            cat_id = ls[0]
            bbox = ls[1:5]
            dic = dict()
            dic["name"] = name
            dic["category_id"] = cat_id
            dic["bbox"] = bbox
        annotations.append(dic)

    name_box_id = defaultdict(list)
    id_name = dict()

    #   取出标注
    for ant in annotations:
        name = ant['name']
        name = os.path.join(train_datasets_path, name)
        name = name + '.jpg'
        cat = int(ant['category_id'])
        name_box_id[name].append([ant['bbox'], cat])

    #   遍历键值，也就是每个框，将坐标与路径写入文件
    f = open(train_output_path, 'w')
    for key in name_box_id.keys():
        f.write(key)
        image = Image.open(key)
        iw, ih = image.size
        box_infos = name_box_id[key]
        for info in box_infos:
            #   真实框数据需要反归一化
            x_min = int(info[0][0] * iw)
            y_min = int(info[0][1] * ih)
            x_max = int(info[0][2] * iw)
            y_max = int(info[0][3] * ih)

            box_info = " %d,%d,%d,%d,%d" % (
                x_min, y_min, x_max, y_max, info[1])
            f.write(box_info)
        f.write('\n')
    f.close()

    #   对验证集再来一遍
    annotations = []

    for filename in os.listdir(val_labels_path):
        path = os.path.join(val_labels_path, filename)
        f = open(path, 'r')
        lines = f.readlines()
        f.close()

        lines = np.array([np.array(list(map(float, line.split(' ')))) for line in lines])

        #   xywh变成x1y1x2y2
        lines[...,1:3] = lines[...,1:3] - lines[...,3:5] / 2
        lines[...,3:5] += lines[...,1:3]

        name = filename.split('.')[0]
        for ls in lines:
            cat_id = ls[0]
            bbox = ls[1:5]

            dic = dict()
            dic["name"] = name
            dic["category_id"] = cat_id
            dic["bbox"] = bbox
        annotations.append(dic)

    name_box_id = defaultdict(list)
    id_name = dict()

    for ant in annotations:
        name = ant['name']
        name = os.path.join(val_datasets_path, name)
        name = name + '.jpg'
        cat = int(ant['category_id'])
        name_box_id[name].append([ant['bbox'], cat])

    #   遍历键值，也就是每张图片，将坐标与路径写入文件
    f = open(val_output_path, 'w')
    for key in name_box_id.keys():
        f.write(key)
        image = Image.open(key)
        iw, ih = image.size
        box_infos = name_box_id[key]
        for info in box_infos:
            x_min = int(info[0][0] * iw)
            y_min = int(info[0][1] * ih)
            x_max = int(info[0][2] * iw)
            y_max = int(info[0][3] * ih)

            box_info = " %d,%d,%d,%d,%d" % (
                x_min, y_min, x_max, y_max, info[1])
            f.write(box_info)
        f.write('\n')
    f.close()
