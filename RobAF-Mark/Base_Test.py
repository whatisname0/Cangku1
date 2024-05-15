# 高斯的影响测试
import numpy as np
import torch
import cv2 as cv
import csv
from superpoint import *
from utils import *
from argparse_opt import arg
from mapping_02 import *
from extracting_01 import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import time
import os

res_table = [[None for _ in range(220)] for _ in range(220)] 

for t in tqdm(range(1, 13)):
    i = t
    image_idx = i
    watermark_idx = 3
    lr = 0.002
    gauss_rate = 0.1

    epochs = 15000
    host_imageName = "./image/ori_image/kodim" + str(image_idx) + ".png"
    version = f"Kodak24_test_w3{i}"
    modelName = "./model/mapping_net_xiaorongtest_" + version + ".pth"

    rr1 = 0.05
    rr2 = 5
    attacks1 = ["pepper/pepper", "guassian/gb","rescaling/rescaling", "rescaling/rescaling2", "speckle/speckle","jepg/jepg","rotate/rotate"]
    attacks2 = ["bilateral_filtering/bilateral","gaussian_filtering/gaussian", 
                "median_filtering/median","mean_filtering/mean", "box_filtering/boxf","wiener/wiener"]
    attacks3 = [ "histogram/histogram","sharpen/sharpen"]
    for j in range(len(attacks1)):
        if j == 2 or j == 3:
            attacks1[j] = attacks1[j] + f"_{1-rr1*5}_{image_idx}"
        elif j == 5:
            attacks1[j] = attacks1[j] + f"_{60}_{image_idx}"
        elif j == 6:
            attacks1[j] = attacks1[j] + f"_{10}_{image_idx}"
        else:
            attacks1[j] = attacks1[j] + f"_{rr1}_{image_idx}"
    for j in range(len(attacks2)):
        attacks2[j] = attacks2[j] + f"_{rr2}_{image_idx}"
    for j in range(len(attacks3)):
        attacks3[j] = attacks3[j] + f"_{image_idx}"

    mapping(host_imageName, modelName, watermark_idx, lr, epochs, gauss_rate, image_idx, attacks1 + attacks2 + attacks3, xishu = 0.06)

    print("attack")
    rr1 = 0.05
    x1 = 5
    attacks1 = ["pepper/pepper", "guassian/gb","rescaling/rescaling", "rescaling/rescaling2", 
                "speckle/speckle", "jepg/jepg","rotate/rotate","crop/crop"]
    cnt = 0
    for j in range(len(attacks1)):
        if j == 2 or j == 3:
            attacks1[j] = attacks1[j] + f"_{1-rr1*5}_{image_idx}"
        elif j == 5:
            attacks1[j] = attacks1[j] + f"_{100-x1*10}_{image_idx}"
        elif j == 6:
            attacks1[j] = attacks1[j] + f"_{x1}_{image_idx}"
        elif j == 7:
            attacks1[j] = attacks1[j] + f"_{4*rr1}_{image_idx}"
        else:
            attacks1[j] = attacks1[j] + f"_{rr1}_{image_idx}"
    attacks = attacks1
    cntttt = 0
    for attack_path in attacks:
        attack_path2 = "./image/attacked_image2/" + attack_path + '.jpg'
        res,p = extracting(attack_path2, modelName, watermark_idx, image_idx)
        if cntttt == 0 and res <= 0.75:
            mapping(host_imageName, modelName, watermark_idx, lr, epochs, gauss_rate, image_idx, attacks1 + attacks2 + attacks3, xishu = 0.03)
            cntttt += 1
        res_table[i][cnt] = res
        cnt += 1

    print("filter")
    rr2 = 5
    attacks2 = ["bilateral_filtering/bilateral","gaussian_filtering/gaussian", 
                "median_filtering/median", "mean_filtering/mean","box_filtering/boxf","wiener/wiener"]
    for j in range(len(attacks2)):
        attacks2[j] = attacks2[j] + f"_{rr2}_{image_idx}"
    attacks = attacks2
    cnt = 20
    for attack_path in attacks:
        attack_path2 = "./image/attacked_image2/" + attack_path + '.jpg'
        res,p = extracting(attack_path2, modelName, watermark_idx, image_idx)
        res_table[i][cnt] = res
        cnt += 1

    print("other")
    cnt = 30
    for k in range(13, 25):
        jiayang = "./image/ori_image/kodim" + str(k) + ".png"
        res,p = extracting(jiayang, modelName, watermark_idx, image_idx)
        res_table[i][cnt] = res
        cnt += 1
    time.sleep(1)

save_path = "./test_result/" + version + ".csv"
with open(save_path, "w", encoding="utf-8") as file:
    writer = csv.writer(file)
    for i in range(len(res_table)):
        writer.writerow(res_table[i])