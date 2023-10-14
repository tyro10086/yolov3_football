import json
import os

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from utils.utils import cvtColor, preprocess_input, resize_image
from utils.utils_yolo import YOLO

#   map_mode用于指定该文件运行时计算的内容
#   map_mode为0代表整个map计算流程，包括获得预测结果、计算map。
#   map_mode为1代表仅仅获得预测结果。
#   map_mode为2代表仅仅获得计算map。
map_mode = 0

#   指向了验证集标签与图片路径
cocoGt_path = '/football_valid.txt'
dataset_img_path = 'football_dataset/images/valid'

#   结果输出的文件夹，默认为map_out
temp_save_path = 'map_out/eval'


class mAP_YOLO(YOLO):
    #   检测图片
    def detect_image(self, image_id, image, results):
        #   计算输入图片的高和宽
        image_shape = np.array(np.shape(image)[0:2])
        #   图像转换成RGB形式
        image = cvtColor(image)
        #   给图像增加灰条，实现不失真的resize
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #   把通道的维度放前面，然后在第一维添加一个维度用于batch_size，且batch_size=1
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #   将图像输入网络当中进行预测，并解码将先验框变成候选框
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #   将预测框宽高拼接，然后进行非极大抑制
            outputs = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)

            if outputs[0] is None:
                return results

            #   处理非极大值抑制筛选过后的框，提取种类、置信度、坐标
            #   由于batch_size等于1，result尺寸1, objects_num, 7，只需提取result[0]
            top_label = np.array(outputs[0][:, 6], dtype='int32')
            top_conf = outputs[0][:, 4] * outputs[0][:, 5]
            top_boxes = outputs[0][:, :4]

        #   以dict形式存储图片上物体的种类、置信度、坐标，图片的id，作为返回值
        for i, c in enumerate(top_label):
            result = {}
            top, left, bottom, right = top_boxes[i]

            result["image_id"] = int(image_id)
            result["category_id"] = clsid2catid[c]
            result["bbox"] = [float(left), float(top), float(right - left), float(bottom - top)]
            result["score"] = float(top_conf[i])
            results.append(result)
        return results

#   训练好模型以后，验证其性能时使用
if __name__ == "__main__":
    if not os.path.exists(temp_save_path):
        os.makedirs(temp_save_path)

    #   获取真实框、所有图像id和目标类别
    cocoGt = COCO(cocoGt_path)
    ids = list(cocoGt.imgToAnns.keys())
    clsid2catid = cocoGt.getCatIds()

    #   获得预测结果
    if map_mode == 0 or map_mode == 1:
        yolo = mAP_YOLO(confidence=0.001, nms_iou=0.65)

        with open(os.path.join(temp_save_path, 'eval_results.json'), "w") as f:
            #   检测验证集中所有图片，获取图片上物体的种类、置信度、坐标，图片的id
            results = []
            for image_id in tqdm(ids):
                image_path = os.path.join(dataset_img_path, cocoGt.loadImgs(image_id)[0]['file_name'])
                image = Image.open(image_path)
                results = yolo.detect_image(image_id, image, results)

            #   将这些信息变成JSON格式并写入
            json.dump(results, f)

    #   计算mAP
    if map_mode == 0 or map_mode == 2:
        #   加载先前写入的，模型的检测结果
        cocoDt = cocoGt.loadRes(os.path.join(temp_save_path, 'eval_results.json'))
        #   创建评估类对象
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        #   进行评估
        cocoEval.evaluate()
        #   累计求平均
        cocoEval.accumulate()
        #   出评估结果的总结信息
        cocoEval.summarize()
        print("Get map done.")
