# YOLO3-football
参考自：https://github.com/bubbliiiing/yolo3-pytorch

详细注释源代码
# 数据集
使用数据集 Detection_FootballvsCricketBall
https://www.kaggle.com/datasets/mlwhiz/detection-footballvscricketball

# 改进
主要针对原有模型对小尺寸目标识别能力差的问题

调整损失函数，增强小尺寸目标影响力

马赛克数据增强

随机复制小尺寸目标，增加样本量
# 性能
详见文件夹 mosaic smallobj raw


