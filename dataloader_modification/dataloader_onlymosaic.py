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
        #   获取图片、真实框，并把图片的通道放在第一维
        #   训练时进行数据的随机增强，验证时不进行数据的随机增强
        img, box = self.load_mosaic(index)

        image_data, box_data = self.get_random_data(img, box, self.input_shape[0:2],random=self.train)
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

    # 使用mosaic来实现数据增强
    def load_mosaic(self, index, img_shape=[736, 736]):
        """
        将四张图片拼接在一张马赛克图像中
        :param self:
        :param index: 需要获取的图像索引
        :return: img4: mosaic和仿射增强后的一张图片
                 labels4: img4对应的target
        """
        #   随机初始化拼接图像的中心点坐标  [s*0.5, s*1.5]之间随机取2个数作为拼接图像的中心坐标
        h, w = img_shape
        xc = int((self.rand() + 0.5) * w)
        yc = int((self.rand() + 0.5) * h)

        #   用于存放拼接图像（4张图拼成一张）的label信息
        #   增强后图片长宽变成原来两倍
        img4 = np.full((w * 2, h * 2, 3), 114, dtype=np.uint8)  # only_yolov3 image with 4 tiles
        labels4 = []

        # 从dataset中随机寻找三张图像进行拼接
        indices = [index] + [np.random.randint(0, self.length - 1) for _ in range(3)]

        # 遍历四张图像进行拼接 4张不同大小的图像 => 1张[1472, 1472, 3]的图像
        for i, index in enumerate(indices):
            line = self.annotation_lines[index].split()
            #   读取图像并转换成RGB图像
            image = Image.open(line[0])
            image = cvtColor(image)
            #   获得图像的高宽与目标高宽
            iw, ih = image.size
            image = np.array(image, np.float32)

            box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
            box = np.array(box, dtype=np.float32)


            #   确定四张图片分别截取哪部分
            #   x1y1x2y2是左上角和右下角的坐标
            #   a是增强后图片的坐标，b是原始图片的坐标
            if i == 0:
                #   左上角
                x1a, y1a, x2a, y2a = max(xc - iw, 0), max(yc - ih, 0), xc, yc
                #   右下角
                x1b, y1b, x2b, y2b = iw - (x2a - x1a), ih - (y2a - y1a), iw, ih
            elif i == 1:
                #   右上角
                x1a, y1a, x2a, y2a = xc, max(yc - ih, 0), min(xc + iw, w * 2), yc
                #   左下角
                x1b, y1b, x2b, y2b = 0, ih - (y2a - y1a), min(iw, x2a - x1a), ih
            elif i == 2:
                #   左下角
                x1a, y1a, x2a, y2a = max(xc - iw, 0), yc, xc, min(h * 2, yc + ih)
                #   右上角
                x1b, y1b, x2b, y2b = iw - (x2a - x1a), 0, max(xc, iw), min(y2a - y1a, ih)
            elif i == 3:
                #   右下角
                x1a, y1a, x2a, y2a = xc, yc, min(xc + iw, w * 2), min(h * 2, yc + ih)
                #   左上角
                x1b, y1b, x2b, y2b = 0, 0, min(iw, x2a - x1a), min(y2a - y1a, ih)

            #   将截取的图像区域填充到马赛克图像的相应位置   img4[h, w, c]
            #   将图像img的【(x1b,y1b)左上角 (x2b,y2b)右下角】区域截取出来填充到马赛克图像的【(x1a,y1a)左上角 (x2a,y2a)右下角】区域
            img4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]

            # 以左上角为基准点计算pad，即马赛克处理后图像相比之前偏移了多少
            padw = x1a - x1b
            padh = y1a - y1b

            #   计算锚框数据在马赛克图像中位置
            #   把xywh变成左下角右上角坐标并加上偏移量
            labels = box.copy()
            if box.size > 0:  # Normalized xywh to pixel xyxy format
                # w * (x[:, 1] - x[:, 3] / 2): 将相对原图片(375, 500, 3)的label映射到load_image函数resize后的图片(552, 736, 3)上
                # w * (x[:, 1] - x[:, 3] / 2) + padw: 将相对resize后图片(552, 736, 3)的label映射到相对img4的图片(1472, 1472, 3)上
                labels[:, 0] = box[:, 0] + padw
                labels[:, 1] = box[:, 1] + padh
                labels[:, 2] = box[:, 2] + padw
                labels[:, 3] = box[:, 3] + padh
            labels4.append(labels)


        # Concat/clip labels4 把labels4（[(2, 5), (1, 5), (3, 5), (1, 5)] => (7, 5)）压缩到一起
        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            # label[:, 1:]中的所有元素的值（位置信息）必须在[0, 2*s]之间,小于0就令其等于0,大于2*s就等于2*s
            labels4[:, [0, 2]] = np.clip(labels4[:, [0, 2]], 0, 2 * w)
            labels4[:, [1, 3]] = np.clip(labels4[:, [1, 3]], 0, 2 * h)

        plt.figure(figsize=(20, 16))
        img4 = img4[:, :, ::-1]  # BGR -> RGB
        plt.subplot(1, 2, 1)
        plt.imshow(img4)
        plt.title('仿射变换前 shape={}'.format(img4.shape), fontsize=25)
        plt.close()

        return img4, labels4

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


# DataLoader中collate_fn使用，如何将多个样本组合成一个batch
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
