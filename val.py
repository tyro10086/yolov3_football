#   输出图片，测试预处理是否正确
import colorsys
import json
import os
from PIL import ImageDraw, ImageFont
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image, show_config)


#   生成的txt文件路径
train_output_path = "football_train.txt"
val_output_path = "football_valid.txt"

file_pathname = "football_dataset/labels/train"

def mark(image, box, cls, input_shape, class_names, letterbox_image = True):
    #   图像转换成RGB形式
    image = cvtColor(image)
    #   给图像增加灰条，实现不失真的resize
    #image_data = resize_image(image, (input_shape[1], input_shape[0]), letterbox_image)
    #   把通道的维度放前面，然后在第一维添加一个维度用于batch_size，且batch_size=1
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image, dtype='float32')), (2, 0, 1)), 0)

    #   设置字体与边框厚度
    font = ImageFont.truetype(font='model_data/simhei.ttf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = int(max((image.size[0] + image.size[1]) // np.mean(input_shape), 1))
    for i, c in list(enumerate(cls)):
        c = int(c)
        #   把框可视化
        predicted_class = class_names[c]
        score = 1
        left, bottom, right, top = box[i]

        top = max(0, np.floor(top).astype('int32'))
        left = max(0, np.floor(left).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom).astype('int32'))
        right = min(image.size[0], np.floor(right).astype('int32'))

        #   label是文字信息 包括类别和置信度
        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)
        label = label.encode('utf-8')

        #   label放到合适的位置
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        #   画多个重合的框，以达到足够的厚度
        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
        del draw

    return image

def compute_overlap(annot_a, annot_b):
        if annot_a is None:
            return False
        left_max = max(annot_a[0], annot_b[0])
        top_max = max(annot_a[1], annot_b[1])
        right_min = min(annot_a[2], annot_b[2])
        bottom_min = min(annot_a[3], annot_b[3])
        inter = max(0, (right_min - left_max)) * max(0, (bottom_min - top_max))
        if inter != 0:
            return True
        else:
            return False


def donot_overlap(new_annot, annots):
    for annot in annots:
        if compute_overlap(new_annot, annot):
            return False
    return True


def create_annot(h, w, obj):
    annot = obj.copy()
    annot_w = obj[2] - obj[0]
    annot_h = obj[3] - obj[1]

    annot[0] = np.random.randint(0, w - annot_w)
    annot[1] = np.random.randint(0, h - annot_h)
    annot[2] = annot[0] + annot_w
    annot[3] = annot[1] + annot_h
    return annot

def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def small_obj_aug( w, h, img_data, obj):
    annots = [obj]
    x_min, y_min, x_max, y_max = map(int, obj[0:4])
    newobj_num = np.random.randint(0, 5) + 1

    for i in range(newobj_num):
        while True:
            annot = create_annot(h, w, obj)
            if donot_overlap(annot, annots):
                break

        x1, y1, x2, y2 = map(int, annot[0:4])

        img_data[y1:y2, x1:x2] = img_data[y_min:y_max, x_min:x_max]
        annots.append(annot)

    annots = np.array(annots, dtype=np.float32)
    return img_data, annots



#   在进行训练之前，用这部分数据预处理下载的annotations，从中选出有用信息
#   把有用信息写入新文件，用新文件训练
if __name__ == "__main__":
    dir_save_path = "img/test/"
    class_names = ["baseball", "football"]
    input_shape = [416, 416]
    with open(val_output_path) as f:
        lines = f.readlines()
    val_nums = len(lines)
    hsv_tuples = [(x / 2, 1., 1.) for x in range(2)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    for i, annotation_line in enumerate(lines):
        line = annotation_line.split()
        #   读取图像并转换成RGB图像
        image = Image.open(line[0])
        image = cvtColor(image)
        iw, ih = image.size
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        image = np.array(image, np.uint8)
        labels = []

        for annot in box:
            if annot[2] - annot[0] <= 32 and annot[3] - annot[1] <= 32:
                image, label = small_obj_aug(iw, ih, image, annot)
                for ele in label:
                    labels.append(ele)
            else:
                labels.append(annot)
        labels = np.array(labels)
        image = Image.fromarray(image)

        cls = labels[:, 4]
        box = labels[:, 0:4]
        r_image = mark(image, box, cls, image.size, class_names)
        if not os.path.exists(dir_save_path):
            os.makedirs(dir_save_path)
        name = os.path.join(dir_save_path, str(i)+'.png')
        r_image.save(name, quality=95, subsampling=0)
