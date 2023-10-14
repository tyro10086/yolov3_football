import numpy as np
from PIL import Image


#   将图像转换成RGB图像
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

#   对输入图像进行resize
def resize_image(image, size, letterbox_image):
    iw, ih = image.size
    w, h = size
    #   letterbox保持图像内容的纵横比不变，通过填充背景使其达到指定的尺寸。
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128)) # 灰色图片，然后粘贴到原图片上
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2)) # 以左上角为原点
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


#   文件操作获得待识别的类的名称
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


#   文件操作获得先验框尺寸
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)


#   获取优化器当前学习率
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#   图片的RGB值归一化
def preprocess_input(image):
    image /= 255.0
    return image


def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)
