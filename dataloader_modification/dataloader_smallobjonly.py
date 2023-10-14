import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train):
        super(YoloDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.length = len(self.annotation_lines)
        self.train = train

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        #   训练时进行数据的随机增强，验证时不进行数据的随机增强
        line = self.annotation_lines[index].split()
        #   读取图像并转换成RGB图像
        image = Image.open(line[0])
        image = cvtColor(image)
        iw, ih = image.size
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        image = np.array(image, np.uint8)
        labels = []

        for annot in box:
            if annot[2] - annot[0] <= 32 and annot[3] - annot[1] <= 32:
                image, label = self.small_obj_aug(iw, ih, image, annot)
                for ele in label:
                    labels.append(ele)
            else:
                labels.append(annot)
        labels = np.array(labels)
        image_data, box_data = self.get_random_data(image, labels, self.input_shape[0:2], random=self.train)
        image_data = np.transpose(preprocess_input(np.array(image_data, dtype=np.float32)), (2, 0, 1))
        box_data = np.array(box_data, dtype=np.float32)

        if len(box) != 0:
            #   归一化
            box_data[:, [0, 2]] = box_data[:, [0, 2]] / self.input_shape[1]
            box_data[:, [1, 3]] = box_data[:, [1, 3]] / self.input_shape[0]

            #   左下角、右上角坐标转化成中心点坐标和宽高
            box_data[:, 2:4] = box_data[:, 2:4] - box_data[:, 0:2]
            box_data[:, 0:2] = box_data[:, 0:2] + box_data[:, 2:4] / 2
        return image_data, box_data

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, box, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        """
        jitter: 用于扭曲长宽比
        hue sat val: 用于色域变换
        """
        #   获得图像的高宽与目标高宽
        image = Image.fromarray(image)
        iw, ih = image.size
        h, w = input_shape

        #   不随机增强
        if not random:
            #   纵横比不变的前提下进行放缩
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            #   记录要填充的宽高
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            #   将图像多余的部分加上灰条，完成resize
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            #   resize后真实框也要调整中心点坐标和宽高
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

            #   返回值image_data描述图片，box描述真实框
            return image_data, box

        #   进行随机增强
        #   对图像进行缩放并且进行长和宽的扭曲
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter) # 新的宽高比
        scale = self.rand(.25, 2) # 随机放缩
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        #   将图像多余的部分加上灰条
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #   随机翻转图像
        flip = self.rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        #   对图像进行色域变换
        image_data = np.array(image, np.uint8)
        #   计算色域变换的参数
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #   将图像转到HSV上
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        #   应用变换
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        #   resize后真实框也要调整中心点坐标和宽高
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        #   返回值image_data描述增强后的图片，box描述真实框
        return image_data, box


    def small_obj_aug(self, w, h, img_data, obj):
        annots = [obj]
        x_min, y_min, x_max, y_max = map(int, obj[0:4])
        newobj_num = np.random.randint(0, 5) + 1

        for i in range (newobj_num):
            while True:
                annot = create_annot(h, w, obj)
                if donot_overlap(annot, annots):
                    break

            x1, y1, x2, y2 = map(int, annot[0:4])

            img_data[y1:y2,x1:x2] = img_data[y_min:y_max,x_min:x_max]
            annots.append(annot)

        annots = np.array(annots, dtype=np.float32)
        return img_data, annots


#   DataLoader中collate_fn使用，如何将多个样本组合成一个batch
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    # 相当于在最前面添加了一个batch_size维度
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images, bboxes


#   判断两个目标是否重叠
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


#   判断新生成的小目标是否和其它目标都不重叠
def donot_overlap(new_annot, annots):
        for annot in annots:
            if compute_overlap(new_annot, annot):
                return False
        return True


#   随机生成相同尺寸的目标的位置
def create_annot(h, w, obj):
    annot = obj.copy()
    annot_w = obj[2] - obj[0]
    annot_h = obj[3] - obj[1]

    annot[0] = np.random.randint(0, w - annot_w)
    annot[1] = np.random.randint(0, h - annot_h)
    annot[2] = annot[0] + annot_w
    annot[3] = annot[1] + annot_h
    return annot
