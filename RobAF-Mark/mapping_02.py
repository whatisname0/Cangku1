import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
from calculate_NC import *
import random
import csv
from find_point import *
from PIL import Image
from figure_feature import *
from MyNetwork.stegastamp_1 import *
import torch.nn.functional as F
import math
#from MyNetwork.stegastamp_2 import *

def mapping(host_imageName, modelName, watermark_idx, lr, epoch, gauss_rate, image_idx, attacks, xishu = 0.02):

    # ********************************************读入和预处理部分********************************************

    # 获取秘密信息01串
    with open("watermark.csv", "r") as f:
        reader = csv.reader(f)
        strings = list(reader)
        watermark= strings[watermark_idx][0]    # 根据watermark_idx取出相应的水印
        if watermark_idx == 192 or watermark_idx == 224:
            watermark2 = strings[watermark_idx + 1][0]
            watermark = watermark + watermark2
        if watermark_idx == 256:
            watermark2 = strings[watermark_idx + 1][0]
            watermark3 = strings[watermark_idx + 2][0]
            watermark = watermark + watermark2 + watermark3

    watermark_size = round(math.sqrt(len(watermark)))
    #print(watermark_size)
    watermark = torch.Tensor([int(c) for c in watermark]).reshape(watermark_size, watermark_size)
    watermark = arnold_transform(watermark, 1, 1, 10, watermark_size)

    # ********************************************特征提取部分********************************************

    #img_feature = get_feature(host_imageName, image_idx, attacks, "mapping")
    #img_feature = img_feature.unsqueeze(0)
    # ********************************************神经网络训练部分********************************************
    net = train_network_stega(image_idx, watermark, watermark_size, lr, epoch, gauss_rate, xishu)
    #net = train_network(img_feature, watermark, 2, lr, epoch, gauss_rate)

    # ********************************************保存神经网络模型********************************************
    #torch.save(net, modelName)
    torch.save(net.state_dict(), modelName)


if __name__== "__main__" :
    host_imageName = "./image/ori_image/kodim23.png"
    modelName = "./model/mapping_net_test.pth"

    mapping(host_imageName, modelName, 0)