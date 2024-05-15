import sys
import os
import utils
import torch
import numpy as np
from torch import nn
from kornia import color
import torch.nn.functional as F
from torchvision import transforms
from calculate_NC import *

import torch
import numpy
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from arnold import *
import random

import torch.nn as nn
#from options import HiDDenConfiguration

class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, channels_in, channels_out, stride=1):

        super(ConvBNRelu, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)

class Discriminator(nn.Module):
    """
    Discriminator network. Receives an image and has to figure out whether it has a watermark inserted into it, or not.
    """
    def __init__(self):
        super(Discriminator, self).__init__()

        layers = [ConvBNRelu(1, 64)]
        for _ in range(3-1):
            layers.append(ConvBNRelu(64, 64))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.before_linear = nn.Sequential(*layers)
        self.linear = nn.Linear(64, 1)

    def forward(self, image):
        X = self.before_linear(image)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        X.squeeze_(3).squeeze_(2)
        X = self.linear(X)
        X = torch.sigmoid(X)
        return X


class Dense(nn.Module):
    def __init__(self, in_features, out_features, activation='relu', kernel_initializer='he_normal'):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.kernel_initializer = kernel_initializer

        self.linear = nn.Linear(in_features, out_features)
        # initialization
        if kernel_initializer == 'he_normal':
            nn.init.kaiming_normal_(self.linear.weight)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        outputs = self.linear(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
        return outputs


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', strides=1):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, strides, int((kernel_size - 1) / 2))
        # default: using he_normal as the kernel initializer
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
            else:
                raise NotImplementedError
        return outputs

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)

class SpatialTransformerNetwork(nn.Module):
    def __init__(self):
        super(SpatialTransformerNetwork, self).__init__()
        self.localization = nn.Sequential(
            Conv2D(3, 32, 3, strides=2, activation='relu'),
            Conv2D(32, 64, 3, strides=2, activation='relu'),
            Conv2D(64, 128, 3, strides=2, activation='relu'),
            Flatten(),
            Dense(2048, 128, activation='relu'),
            nn.Linear(128, 6)
        )
        self.localization[-1].weight.data.fill_(0)
        self.localization[-1].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    def forward(self, image):
        theta = self.localization(image)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, image.size(), align_corners=False)
        transformed_image = F.grid_sample(image, grid, align_corners=False)
        return transformed_image


class StegaStampNetwork_Decoder(nn.Module):
    def __init__(self, secret_size=1024):
        super(StegaStampNetwork_Decoder, self).__init__()
        self.secret_size = secret_size
        self.stn = SpatialTransformerNetwork()
        self.decoder = nn.Sequential(
            Conv2D(3, 32, 3, strides=1, activation='relu'),
            Conv2D(32, 32, 3, activation='relu'),
            Conv2D(32, 64, 3, strides=2, activation='relu'),
            Conv2D(64, 64, 3, activation='relu'),
            Conv2D(64, 64, 3, strides=2, activation='relu'),
            Conv2D(64, 128, 3, strides=2, activation='relu'),
            Conv2D(128, 128, 3, strides=2, activation='relu'),
            Flatten(),
            Dense(512, 512, activation='relu'),
            Dense(512, 1024, activation=None))

    def forward(self, image):
        transformed_image = self.stn(image)
        return torch.sigmoid(self.decoder(transformed_image))
    
class StegaStampNetwork_Eecoder(nn.Module):
    def __init__(self,cs):
        super(StegaStampNetwork_Eecoder, self).__init__()
        self.secret_dense = Dense(1024, 1600, activation='relu', kernel_initializer='he_normal')
        
        self.conv1 = Conv2D(6, 32, 3, activation='relu')
        self.conv2 = Conv2D(32, 32, 3, activation='relu', strides=2)
        self.up9 = Conv2D(32, 32, 3, activation='relu')
        self.conv9 = Conv2D(70, 32, 3, activation='relu')
        self.residual = Conv2D(32, cs, 1, activation=None) 

    def forward(self, image):
        conv1 = self.conv1(image)
        conv2 = self.conv2(conv1)
        up9 = self.up9(nn.Upsample(scale_factor=(2, 2))(conv2))
        merge9 = torch.cat([conv1, up9, image], dim=1)
        conv9 = self.conv9(merge9)
        residual = self.residual(conv9)
        return torch.sigmoid(residual)

def plot_loss(record):
    plot_epoch = range(1, len(record) + 1)
    plt.plot(plot_epoch, record, color='blue', label='loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')

    plt.legend()
    plt.show()

def train_network_stega(image_idx, watermark, watermark_size, lr, epoch, gauss_rate, xishu):
    channels = (watermark_size // 32) * (watermark_size // 32)
    net = StegaStampNetwork_Eecoder(cs = channels)
    net = net.cuda()
    optimizer = optim.Adam(net.parameters(), lr)
    criterion = torch.nn.BCELoss()

    discriminator = Discriminator()
    discriminator = discriminator.cuda()
    optimizer_discrim = torch.optim.Adam(discriminator.parameters())

    cnt = 0
    record = []
    last = 0
    earlystopnum = 0
    our_feature = torch.load('./feature/our_features.pt')
    interf_feature = torch.load('./feature/other_features.pt')

    while True:
        cnt += 1

        random_img = random.randint(0, 15)
        signal = our_feature[image_idx][random_img]
        signal = signal.unsqueeze(0)

        traindata = signal + gauss_rate * torch.randn(1, 6, 32, 32)

        traindata = traindata.cuda()
        result = net(traindata)

        result = result.view(1,1,watermark_size,watermark_size)        

        random_interf = random.randint(1001,1088)
        random_img2 = random.randint(0, 15)
        fake_result = net(interf_feature[random_interf][random_img2].unsqueeze(0).cuda())
        fake_result = fake_result.view(1,1,watermark_size,watermark_size)
        
        optimizer_discrim.zero_grad()
        d_real = discriminator(result.detach().cuda())
        d_real_loss = criterion(d_real, torch.ones(1,1).cuda())
        d_real_loss = d_real_loss.cuda()
        d_real_loss.backward()

        d_fake = discriminator(fake_result.detach().cuda())
        d_fake_loss = criterion(d_fake, torch.zeros(1,1).cuda())
        d_fake_loss = d_fake_loss.cuda()
        d_fake_loss.backward()
        optimizer_discrim.step()

        optimizer.zero_grad() 

        d_fake_2 = discriminator(fake_result.cuda())
        g_loss_adv = criterion(d_fake_2, torch.ones(1,1).cuda())

        result = result.squeeze()
        
        loss = criterion(result.cuda(), watermark.cuda()) - xishu*g_loss_adv

        loss = loss.cuda()

        record.append(loss.item())

        loss.backward() 
        optimizer.step() 

        NC_value = computeNC(watermark.cuda(), result.cuda()).item()

        result = torch.round(result.squeeze())

        if abs(loss.item() - last) <= 0.001:
            earlystopnum += 1
        last = loss.item()

        # if cnt % 1000 == 0:
        #     result_ = (watermark.cuda() == result.cuda()).sum()
        #     ratio = result_ / (watermark.size(0) * watermark.size(1))
        #     #print("准确的比率为：", ratio)
        #     print("epochs:", cnt, "loss:", loss.item(), "准确率:", ratio.item(), "NC值:", NC_value)

        if cnt >= epoch:
            print("共", cnt, "个epoch，最终loss为", loss)
            return net
