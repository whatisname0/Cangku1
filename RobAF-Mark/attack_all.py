# 生成对比实验的攻击后图像

from scipy.signal import wiener
from skimage.util import random_noise
# from sklearn import metrics as mr
import cv2
import copy
import numpy as np
from utils import *
import random

def attack(watermarked_image,mode,rr):  #mode:jpegCompression,salt_pepper,Guass,speckle,bifilter,Gaussian_Blur,medianBlur,mean_filter,sharpen,crop,flip,zoom缩放
    w,h = watermarked_image.shape[0],watermarked_image.shape[1]
    if mode == 'Gaussian Filter':
        ksize = (rr, rr)  
        sigmaX = 0  
        blurred_image = cv2.GaussianBlur(watermarked_image, ksize, sigmaX)
        return blurred_image
    
    if mode == 'rescaling':
        rescale_factor = rr
        rescaled_img = np.clip(watermarked_image * rescale_factor, 0, 255).astype(np.uint8)
        return rescaled_img
    
    if mode == 'rescaling2':
        watermarked_image = cv2.resize(cv2.resize(watermarked_image,(int(rr*h),int(rr*w))),(h,w))
        return watermarked_image

    if mode =='jepg':
        # n=input("n of jpegcompress is(20):")
        #cv2.imwrite("/work/wzz/match/attack/jepg.jpg",watermarked_image,[cv2.IMWRITE_JPEG_QUALITY,10])  #95 
        return watermarked_image

    if mode =='salt_pepper':
        percentage = rr
        num=int(percentage*watermarked_image.shape[0]*watermarked_image.shape[1])#  椒盐噪声点数量
        random.randint(0, watermarked_image.shape[0])
        img2=watermarked_image.copy()
        for i in range(num):
            X=random.randint(0,img2.shape[0]-1)#从0到图像长度之间的一个随机整数,因为是闭区间所以-1
            Y=random.randint(0,img2.shape[1]-1)
            if random.randint(0,1) ==0: #黑白色概率55开
                img2[X,Y] = (255,255,255)#白色
            else:
                img2[X,Y] =(0,0,0)#黑色
        #cv2.imwrite('/work/wzz/match/attack/salt_pepper{}.jpg'.format(percentage), img2)
        return img2
   
    if mode =='speckle':
        #i=input("the var of speckle noise is(0.01): ")   #先放在这里，后面需要调整！
        #i=int(i)
        var = rr  # 0.005
        #img2 = random_noise(watermarked_image, mode='speckle', var=0.01)
        img2 = random_noise(watermarked_image, mode='speckle',var = var)
        #cv2.imwrite('/work/wzz/match/attack/speckle{}.jpg'.format(var), img2*255)
        return img2
    
    if mode == 'bifilter':
        d = rr # 3
        img2 = cv2.bilateralFilter(src=watermarked_image, d=d, sigmaColor=100, sigmaSpace=15)
        #cv2.imwrite('/work/wzz/match/attack/bifilterd={}.jpg'.format(d), img2)
        return img2

    if mode == 'Gaussian_Blur':
        image = np.array(watermarked_image / 255, dtype=float)
        xita = rr  #(0.002) 效果是比较好的
        noise = np.random.normal(0, xita, image.shape)  # np.random.normal(mu, sigma, 1000)20
        gauss_noise = image + noise
        if gauss_noise.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        gauss_noise = np.clip(gauss_noise, low_clip, 1.0)
        gauss_noise = np.uint8(gauss_noise * 255)
        #cv2.imwrite('/work/wzz/match/attack/Gaussian_Blur{}.jpg'.format(xita), gauss_noise)
        return gauss_noise

    if mode == 'mean_shift':
        img2 = cv2.pyrMeanShiftFiltering(src=watermarked_image, sp=15, sr=20)
        #cv2.imwrite('/work/wzz/match/attack/mean_shift.jpg', img2)
        return img2

    if mode == 'median_filter':
        idx = rr  #3
        img2 = cv2.medianBlur(watermarked_image, idx)   # 3是不是好一些
        #cv2.imwrite('/work/wzz/match/attack/mean_filter{}.jpg'.format(idx), img2)
        return img2
    
    if mode == 'mean_filter':
        idx = (rr,rr)  #3
        img2 = cv2.blur(watermarked_image, idx)   # 3是不是好一些
        #cv2.imwrite('/work/wzz/match/attack/mean_filter{}.jpg'.format(idx), img2)
        return img2
    
    if mode == 'sharpen':
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
        img2 = cv2.filter2D(watermarked_image, -1, kernel=kernel)
        #cv2.imwrite('/work/wzz/match/attack/sharpen.jpg', img2)
        return img2

    if mode =='crop':
        img = copy.deepcopy(watermarked_image)

        c_size = rr

        height,width,_ = img.shape
        d = int(min(height,width)*c_size)
        x1 = np.random.randint(0, img.shape[1] - d)
        y1 = np.random.randint(0, img.shape[0] - d)
        # print(x,y)
        x2 = x1+d
        y2 = y1+d
        # x,y =502,205
        img[x1:x2,y1:y2,:]=0
        return img

    if mode =='flip':
        img2 = cv2.flip(watermarked_image, -1)
        #cv2.imwrite('/work/wzz/match/attack/flip.jpg', img2)
        return img2

    if mode =='zoom':
        # i=input('缩放系数为(0.5)：') #这里需要设置系数 ---0.1等
        i = rr
        i=float(i)
        img2 = cv2.resize(cv2.resize(watermarked_image, (int(i * h), int(i * w))), (h, w))
        #cv2.imwrite('/work/wzz/match/attack/zoom_{}.jpg'.format(i), img2)
        return img2

    if mode =='gamma':
        # gamma=input("gamma系数为(2.0)：")
        gamma = rr #0.9
        gamma=float(gamma)
        scale = np.max(watermarked_image) - np.min(watermarked_image)
        dst = ((watermarked_image.astype(np.float32) / scale) ** gamma) * scale
        #cv2.imwrite('/work/wzz/match/attack/gamma{}.jpg'.format(gamma), dst)
        return dst

    if mode =='histogram':
        src = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2GRAY)
        dst = cv2.equalizeHist(src)
        #img2 = img2.astype(np.float32) / 255
        #cv2.imwrite('/work/wzz/match/attack/histogram.jpg', dst)
        return dst

    if mode =='Box_filters':
        kernal = rr
        img2 = cv2.boxFilter(watermarked_image, -1, (kernal, kernal), normalize=1)
        #cv2.imwrite('/work/wzz/match/attack/Box_filters{}.jpg'.format(kernal), img2)
        return img2

    if mode =='wiener':
        img2 = wiener(watermarked_image, rr)
        #cv2.imwrite('/work/wzz/match/attack/wiener.jpg', img2)
        return img2
    #else:
    #    print('无此攻击变换类型！')
    if mode =='rotate':
        
        height, width = watermarked_image.shape[:2]    # 输入(H,W,C)，取 H，W 的zhi
        # center = (width / 2, height / 2)   # 绕图片中心进行旋转
        center = (height / 2, width / 2)   # 绕图片中心进行旋转
        angle = rr  # 旋转方向取（-180，180）中的随机整数值，负为逆时针，正为顺势针
        scale = 1                       # 将图像缩放为80%
    
        # 获得旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, scale)
        # 进行仿射变换，边界填充为255，即白色，默认为黑色 # (height, width) ---真的是因为这个！
        image_rotation = cv2.warpAffine(src=watermarked_image, M=M, dsize=(width, height), borderValue=(255, 255, 255))
        return image_rotation
    else:
        print('无此攻击变换类型！')


def generate_attack(img_ori,i):
     #--------------奇怪组------------#
     pass
    # img1 = attack(img_ori, 'jepg')
    # cv2.imwrite('./image/attacked_image2/jepg/jpeg{}.jpg'.format(i), img1,[cv2.IMWRITE_JPEG_QUALITY,95])
    # img2 = attack(img_ori, 'zoom')
    # cv2.imwrite('./image/attacked_image2/zoom/zoom{}.jpg'.format(i), img2)
    # img3 = attack(img_ori, 'sharpen')
    # cv2.imwrite('./image/attacked_image2/sharpen/sharpen{}.jpg'.format(i), img3)
    
    # #--------------滤波组------------#
    
    # img4 = attack(img_ori, 'mean_filter')
    # cv2.imwrite('./image/attacked_image2/mean_filtering/mean_f{}.jpg'.format(i), img4)
    # img5 = attack(img_ori, 'Box_filters')  #噪声而不是滤波
    # cv2.imwrite('./image/attacked_image2/box_filtering/box_f{}.jpg'.format(i), img5)
    # img6 = attack(img_ori, 'bifilter')
    # cv2.imwrite('./image/attacked_image2/bilateral_filtering/bilateral_f{}.jpg'.format(i), img6)
    
    
    # #--------------噪声组----------------#

    # img7 = attack(img_ori, 'speckle')
    # cv2.imwrite('./image/attacked_image2/speckle/speckle{}.jpg'.format(i), img7*255)
    # img8 = attack(img_ori, 'Gaussian_Blur')  #噪声而不是滤波
    # cv2.imwrite('./image/attacked_image2/guassian/guassian{}.jpg'.format(i), img8)
    # img9 = attack(img_ori, 'gamma')
    # cv2.imwrite('./image/attacked_image2/gamma/gamma{}.jpg'.format(i), img9)
    # img10 = attack(img_ori, 'flip')
    # cv2.imwrite('./image/attacked_image2/flip/flip{}.jpg'.format(i), img10)
    # img11 = attack(img_ori, 'crop')
    # cv2.imwrite('./image/attacked_image2/crop/crop{}.jpg'.format(i), img11)

if __name__ == '__main__':

    x2 = [3,5,7,9,11,13]
    for x1 in range(1, 11):
        rr1 = x1 / 100
        if x1 <= 6:
            rr2 = x2[x1-1]
        for i in range(1, 25):
            imgPath = './image/ori_image/kodim{}.png'.format(i)
            img_ori = cv2.imread(imgPath)

            #img1 = attack(img_ori, 'jepg')
            #cv2.imwrite('./image/attacked_image2/ori/ori{}.jpg'.format(i),img1)  #95 
            #img2 = attack(img_ori, 'salt_pepper_05')
            #cv2.imwrite('./image/attacked_image2/pepper/pepper_05{}.jpg'.format(i), img2)
            
            #img4 = attack(img_ori, 'Gaussian_Blur005')
            #cv2.imwrite('./image/attacked_image2/guassian/gb_005{}.jpg'.format(i), img4)
            
            
            
            img3 = attack(img_ori, 'salt_pepper', rr1)
            cv2.imwrite(f'./image/attacked_image2/pepper/pepper_{rr1}_{i}.jpg', img3)

            img5 = attack(img_ori, 'Gaussian_Blur',rr1)
            cv2.imwrite(f'./image/attacked_image2/guassian/gb_{rr1}_{i}.jpg', img5)

            img6 = attack(img_ori, 'bifilter',rr2)
            cv2.imwrite(f'./image/attacked_image2/bilateral_filtering/bilateral_{rr2}_{i}.jpg', img6)

            img7 = attack(img_ori, 'Gaussian Filter',rr2)
            cv2.imwrite(f'./image/attacked_image2/gaussian_filtering/gaussian_{rr2}_{i}.jpg', img7)

            img11 = attack(img_ori, 'median_filter',rr2)
            cv2.imwrite(f'./image/attacked_image2/median_filtering/median_{rr2}_{i}.jpg', img11)
            
            rr3 = 1 - rr1*5
            img14 = attack(img_ori, 'rescaling',rr3)
            cv2.imwrite(f'./image/attacked_image2/rescaling/rescaling_{rr3}_{i}.jpg', img14)
            
            img144= attack(img_ori, 'rescaling2',1.25)
            cv2.imwrite(f'./image/attacked_image2/rescaling/rescaling2_{1.25}_{i}.jpg', img144)

            img22 = attack(img_ori, 'rescaling2',rr3)
            cv2.imwrite(f'./image/attacked_image2/rescaling/rescaling2_{rr3}_{i}.jpg', img22)
            
            img19 = attack(img_ori, 'histogram',0)
            cv2.imwrite(f'./image/attacked_image2/histogram/histogram_{i}.jpg', img19)

            img12 = attack(img_ori, 'sharpen',0)
            cv2.imwrite(f'./image/attacked_image2/sharpen/sharpen_{i}.jpg', img12)

            img16 = attack(img_ori, 'Box_filters',rr2)  #噪声而不是滤波
            cv2.imwrite(f'./image/attacked_image2/box_filtering/boxf_{rr2}_{i}.jpg', img16)

            img17 = attack(img_ori, 'wiener',rr2)  #噪声而不是滤波
            cv2.imwrite(f'./image/attacked_image2/wiener/wiener_{rr2}_{i}.jpg', img17)

            img20 = attack(img_ori, 'speckle',rr1)
            cv2.imwrite(f'./image/attacked_image2/speckle/speckle_{rr1}_{i}.jpg', img20*255)

            img13 = attack(img_ori, 'jepg', 0) 
            cv2.imwrite(f'./image/attacked_image2/jepg/jepg_{100-x1*10}_{i}.jpg',img13,[cv2.IMWRITE_JPEG_QUALITY,100-x1*10])  #95 

            img102 = attack(img_ori, 'mean_filter',rr2)
            cv2.imwrite(f'./image/attacked_image2/mean_filtering/mean_{rr2}_{i}.jpg', img102)
            
            img103 = attack(img_ori, 'crop',rr1*5)
            cv2.imwrite(f'./image/attacked_image2/crop/crop_{rr1*5}_{i}.jpg', img103)

            #print(rr3)
            img101 = attack(img_ori, 'rotate',x1)
            cv2.imwrite(f'./image/attacked_image2/rotate/rotate_{x1}_{i}.jpg', img101)
            
            #img13 = attack(img_ori, 'jepg', 0) 
            #cv2.imwrite(f'./image/attacked_image2/jepg/jepg_{x1*5}_{i}.jpg',img13,[cv2.IMWRITE_JPEG_QUALITY,x1*5])