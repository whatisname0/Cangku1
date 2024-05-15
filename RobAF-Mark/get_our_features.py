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

res = [[] for _ in range(100)]

for i in range(1, 13):
    image_idx = i
    host_imageName = "./image/ori_image/kodim" + str(image_idx) + ".png"

    attacks1 = ["pepper/pepper", "guassian/gb","rescaling/rescaling", "rescaling/rescaling2", "speckle/speckle","jepg/jepg","rotate/rotate"]
    attacks2 = ["bilateral_filtering/bilateral","gaussian_filtering/gaussian", 
                "median_filtering/median","mean_filtering/mean", "box_filtering/boxf","wiener/wiener"]
    attacks3 = [ "histogram/histogram","sharpen/sharpen"]
    rr1 = 0.05
    rr2 = 5
    for j in range(len(attacks1)):
        if j == 2 or j == 3:
            attacks1[j] = attacks1[j] + f"_{1-rr1*5}_{i}"
        elif j == 5:
            attacks1[j] = attacks1[j] + f"_{60}_{i}"
        elif j == 6:
            attacks1[j] = attacks1[j] + f"_{6}_{i}"
        else:
            attacks1[j] = attacks1[j] + f"_{rr1}_{i}"
    for j in range(len(attacks2)):
        attacks2[j] = attacks2[j] + f"_{rr2}_{i}"
    for j in range(len(attacks3)):
        attacks3[j] = attacks3[j] + f"_{i}"
    attacks = attacks1 + attacks2 + attacks3

    img_feature1 = get_feature(host_imageName, image_idx, attacks, "mapping")
    res[i].append(img_feature1)




    attacks1 = ["pepper/pepper", "guassian/gb","rescaling/rescaling", "rescaling/rescaling2", "speckle/speckle","jepg/jepg","rotate/rotate"]
    attacks2 = ["bilateral_filtering/bilateral","gaussian_filtering/gaussian", 
                "median_filtering/median","mean_filtering/mean", "box_filtering/boxf","wiener/wiener"]
    attacks3 = [ "histogram/histogram","sharpen/sharpen"]
    rr1 = 0.01
    rr2 = 3
    for j in range(len(attacks1)):
        if j == 2 or j == 3:
            attacks1[j] = attacks1[j] + f"_{1-rr1*5}_{i}"
        elif j == 5:
            attacks1[j] = attacks1[j] + f"_{80}_{i}"
        elif j == 6:
            attacks1[j] = attacks1[j] + f"_{6}_{i}"
        else:
            attacks1[j] = attacks1[j] + f"_{rr1}_{i}"
    for j in range(len(attacks2)):
        attacks2[j] = attacks2[j] + f"_{rr2}_{i}"
    for j in range(len(attacks3)):
        attacks3[j] = attacks3[j] + f"_{i}"
    attacks = attacks1 + attacks2 + attacks3
    
    cnt = 0
    for attack_path in attacks:
        cnt += 1
        #if cnt == 4:
        #    continue
        print(attack_path)
        host_imageName = "./image/attacked_image2/" + attack_path + '.jpg'
        img_feature = get_feature(host_imageName, image_idx, attacks, "extracting")
        res[i].append(img_feature)
    #print(len(res[i]))
    #print(res[i][0].shape)

torch.save(res, './feature/our_features.pt')