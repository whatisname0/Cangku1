import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from calculate_NC import *
import time
from find_point import *
import csv
from figure_feature import *
import torch.nn.functional as F

from MyNetwork.stegastamp_1 import *
#from MyNetwork.stegastamp_2 import *

# 提取水印，watermark_idx为-1时表示没有对应的水印，不为-1时会将提取出的水印与原水印比较来衡量模型效果
def extracting(host_imageName, modelName, watermark_idx, image_idx):

    # ********************************************读入和预处理部分********************************************

    if watermark_idx != -1:
        with open("watermark.csv", "r") as f:
            reader = csv.reader(f)
            strings = list(reader)
            watermark= strings[watermark_idx][0]
            if watermark_idx == 192 or watermark_idx == 224:
                watermark2 = strings[watermark_idx + 1][0]
                watermark = watermark + watermark2
            if watermark_idx == 256:
                watermark2 = strings[watermark_idx + 1][0]
                watermark3 = strings[watermark_idx + 2][0]
                watermark = watermark + watermark2 + watermark3
        watermark_size = round(math.sqrt(len(watermark)))
        watermark = torch.Tensor([int(c) for c in watermark]).reshape((watermark_size, watermark_size))
        #print(watermark)
        #print(watermark)
    else:
        pass

    #********************************************特征提取部分********************************************

    img_feature = get_feature(host_imageName, image_idx, None, "extracting")
    img_feature = img_feature.unsqueeze(0)

    #********************************************使用神经网络从特征提取水印部分********************************************
    #net = torch.load(modelName)
    #net = Network(2)
    channels = (watermark_size // 32) * (watermark_size // 32)
    net = StegaStampNetwork_Eecoder(channels)
    net.load_state_dict(torch.load(modelName))
    tic = time.time()
    watermark_extracted = net(img_feature)
    toc = time.time()
    result = watermark_extracted.reshape(watermark_size, watermark_size)
    result = arnold_inverse_transform(result, 1, 1, 10, watermark_size)

    
    result = torch.round(result.squeeze())
    NC_value = computeNC(watermark, result).item()

    # 画出提取出的水印
    #plt.imshow(result, cmap='gray')
    #plt.show()

    #print(result, watermark)

    if watermark_idx != -1:
        
        
        print(f"NC值: {NC_value}; 提取时间: {toc - tic}")
        return NC_value, result

        result_ = (watermark == result).sum()
        ratio = result_ / (watermark.size(0) * watermark.size(1))
        print("准确的比率为：", ratio)
        return ratio
    else:
        pass


if __name__== "__main__" :
    host_imageName = "./image/ori_image/kodim23.png"
    modelName = "./model/mapping_net_test.pth"

    extracting(host_imageName, modelName, 0)