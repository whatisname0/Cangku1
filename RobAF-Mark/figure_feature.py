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
from torchvision import models
from global_feature import *
import torch.nn.functional as F

class CustomResNet(torch.nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()
        # resnet50预训练模型
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = torch.nn.Linear(2048, 32*32)
    
    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, 1, 32, 32)
        return x
    
class CustomResNet2(torch.nn.Module):
    def __init__(self):
        super(CustomResNet2, self).__init__()
        # resnet50预训练模型
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = torch.nn.Linear(2048, 3*32*32)
    
    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, 3, 32, 32)
        return x
    
def freeze_bn(module):  
    classname = module.__class__.__name__  
    if classname.find('BatchNorm') != -1:  
        module.eval()  

# 思路：对每一个3*32*32的小框分别提取特征（先resize成224*224，resnet50里面用的下采样太多了，32*32会报错），然后并到一起。
def get_feature(host_imageName, image_idx, attacks, mode):
    torch.manual_seed(42)  
    #net_1 = CustomResNet()

    host_image = cv2.imread(host_imageName)
    host_image_gray = cv2.imread(host_imageName, 0)
    host_image_ori = cv2.imread(host_imageName)
    #host_image_ori = np.stack([host_image_gray]*3, axis=-1)
    host_image = cv2.cvtColor(host_image, cv2.COLOR_BGR2RGB)

    transform_host_image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # 标准化，使用ImageNet的均值和标准差
                            std=[0.229, 0.224, 0.225])
    ])

    host_image = transform_host_image(host_image)

    if mode == "mapping":
        block_cord = find_robust_point_mapping(host_image_ori, host_image_gray, host_imageName, image_idx, attacks)
    elif mode == "extracting":
        block_cord = find_robust_point_extracting(host_imageName, image_idx)
    else:
        print("无法识别的mode")
    # print(type(block_cord))
        
    # print(block_cord)

    resize_transform = transforms.Resize((224, 224))
    
    torch.manual_seed(42) 
    custom_resnet = CustomResNet()  
    custom_resnet2 = CustomResNet2()
    

    custom_resnet.eval()


    blocks = [] 
    for block in block_cord: 
        
        torch.manual_seed(42) 
        y1, y2, x1, x2 = block 

        # 从3*32*32 resize成 3*224*224
        #print(host_image.shape)
        image_partial = host_image[:, y1:y2, x1:x2]
        try:
            image_partial = resize_transform(image_partial)
        except:
            image_partial = host_image[:, 0:32, 0:32]
            image_partial = resize_transform(image_partial)
            
        image_partial = image_partial.unsqueeze(0)
        
        with torch.no_grad():  
            feature = custom_resnet(image_partial)
        
        feature = feature.squeeze(0)

        blocks.append(feature)

    #host_image = host_image.unsqueeze(0)
    #with torch.no_grad():
    #    feature_global = custom_resnet(host_image)
    #feature_global = feature_global.squeeze(0)
    # 按照第0维拼接三个tensor
    #img_feature = torch.stack((blocks[0], blocks[1], blocks[2],feature_global), dim=0)
    img_feature = torch.stack((blocks[0], blocks[1], blocks[2]), dim=0)
    # print(type(img_feature))

    img_feature = img_feature.view(3,32,32)
    #img_feature = img_feature.view(4, 32, 32)

    #host_image = host_image.unsqueeze(0)
    #with torch.no_grad():
    #    img_feature_global = custom_resnet2(host_image)
    #img_feature_global = img_feature_global.squeeze(0)

    ###上面的部分是局部区域特征提取
    ###下面的部分是图像整体特征提取
    
    #resize_transform2 = transforms.Compose([
    #transforms.ToTensor(),
    #transforms.Resize((512, 512))
    #])
    #image_ori_512 = resize_transform2(host_image_ori)
    img_g = cv2.imread(host_imageName)
    img_g = cv2.resize(img_g,(512,512))
    image_g = cv2.cvtColor(img_g,cv2.COLOR_RGB2GRAY)/255
    img_feature_global = extracting_global(image_g)

    #img_feature_global = transform_host_image(img_feature_global)
    img_feature_global = img_feature_global / np.max(img_feature_global)  
  
    img_feature_global = torch.from_numpy(img_feature_global).float()  

    #print(img_feature,type(img_feature))
    #print(img_feature_global, img_feature_global.shape,type(img_feature_global))
    
    img_feature_res = torch.cat((img_feature,img_feature_global), dim = 0)
    #print(img_feature_res.shape)
    #img_feature_res = img_feature_res.unsqueeze(0)
    #img_feature_res = F.interpolate(img_feature_res, scale_factor=(0.5, 0.5), mode='bilinear', align_corners=False)
    #img_feature_res = F.interpolate(img_feature_res, size=(6, 40, 40), mode='nearest')
    #img_feature_res = img_feature_res.squeeze()
    #print(img_feature_res.shape)
    
    #return img_feature
    #return img_feature_global
    return img_feature_res