import os

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from utils.utils import get_lr


def fit_one_epoch(model_train, model, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step,
                  epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    """
    model_train: 训练模式的模型
    model: 模型
    yolo_loss: 损失函数类的实例
    loss_history: 损失记录及画图的实例
    eval_callback: 评估用实例
    optimizer: 优化器
    epoch: 训练到了第几个世代
    epoch_step: 训练时一个世代有多少步
    epoch_step_val: 验证时一个世代有多少步
    gen: 训练数据集
    gen_val: 验证数据集
    Epoch: 模型总共训练的epoch
    save_period: 多少个epoch保存一次权值
    save_dir: 日志存储路径
    """
    loss = 0
    val_loss = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()

    #   遍历训练集中所有batch数据
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        #   image是图像数据，targets是真实框数据，都有个batch_size维度
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]

        #   清零梯度
        optimizer.zero_grad()
        if not fp16:
            #   前向传播
            outputs = model_train(images)

            #   3种尺寸的输出都要计算损失，并求和
            loss_value_all = 0
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all += loss_item
            loss_value = loss_value_all

            #   反向传播
            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #   前向传播
                outputs = model_train(images)

                #   3种尺寸的输出都要计算损失，并求和
                loss_value_all = 0
                for l in range(len(outputs)):
                    loss_item = yolo_loss(l, outputs[l], targets)
                    loss_value_all += loss_item
                loss_value = loss_value_all

            #   反向传播
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()

        #   loss这个世代总损失
        loss += loss_value.item()

        if local_rank == 0:
            pbar.set_postfix(**{'loss': loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    #   进入评估模式
    model_train.eval()

    #   遍历验证集中所有batch数据
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break

        #   image是图像数据，targets是真实框数据，都有个batch_size维度
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]

            #   清零梯度
            optimizer.zero_grad()
            #   前向传播
            outputs = model_train(images)

            #   计算损失
            loss_value_all = 0
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all += loss_item
            loss_value = loss_value_all

        val_loss += loss_value.item()

        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    #   此时训练完1个epoch
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')

        #   计算每步平均损失，存入loss_history并打印
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))

        #   如果到了save_period，或者是最后一个周期，保存权值
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (
            epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        #   验证损失小于历史最小验证损失，表明这是目前为止最好的模型
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        #   保存最新权值
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))