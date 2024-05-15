import cv2
from pywt import swt2,dwt2
import numpy as np
from scipy.signal import wiener
import matplotlib.pyplot as plt
from scipy import linalg

def extract_fun(mat,m,n):
    signal = np.zeros((3,m,n));n = 3
    huatu = np.zeros((512,512))
    for i in range(int(mat.shape[0]/16)):
        for j in range(int(mat.shape[0]/16)):
            matrix = (mat[i*16:(i+1)*16,j*16:(j+1)*16])#对于低频进行8*8分块
            matrix_dct = cv2.dct(matrix)        #对小块进行dct变换
            for ii in range(16):
                for jj in range(16):
                    huatu[i*16+ii][j*16+jj] = matrix_dct[ii][jj]
            #print(f"maxtrix = {matrix}")
            A1 = matrix_dct[4:8,:4]             #挑选的三个矩阵块
            A2 = matrix_dct[:4,4:8]
            A3 = matrix_dct[:4,:4]
            #print(A1)
            u1,s1,v1 = np.linalg.svd(A1)      #分别对三个矩阵块进行svd分解
            u2,s2,v2 = np.linalg.svd(A2)
            u3,s3,v3 = np.linalg.svd(A3)
            signal[0,i,j] = s1[0]            #保存三个矩阵对应奇异值最鲁棒的第一个分量
            signal[1,i,j] = s2[0]
            signal[2,i,j] = s3[0]
    #cv2.imwrite('./image/paper use/dct.jpg', huatu*255)
    return signal
#####################################################################
def extracting_global(watermarked_image):

    m,n = 32,32
    signal = np.zeros((3,m,n))
    watermarked_image = cv2.medianBlur(watermarked_image.astype(np.float32),3)#中值滤波（5*5）
    watermarked_image = cv2.medianBlur(watermarked_image.astype(np.float32),5)#中值滤波（5*5）
    #watermarked_image = cv2.medianBlur(watermarked_image.astype(np.float32),7)#中值滤波（5*5）
    #watermarked_image = cv2.medianBlur(watermarked_image.astype(np.float32),7)#中值滤波（5*5）
    watermarked_image = cv2.medianBlur(watermarked_image.astype(np.float32),5)#中值滤波（5*5）
    watermarked_image = cv2.medianBlur(watermarked_image.astype(np.float32),3)#中值滤波（5*5）
    #cv2.imwrite('./image/paper use/median.jpg', watermarked_image*255)
    watermarked_image = wiener(watermarked_image,5)
    #cv2.imwrite('./image/paper use/wiener.jpg', watermarked_image*255)
    watermarked_image = cv2.GaussianBlur(watermarked_image.astype(np.float32),(3,3),1.5)
    #cv2.imwrite('./image/paper use/gaussian.jpg', watermarked_image*255)
    #coeffs = swt2(watermarked_image,'haar',1)#dwt分解
    #cA1,(cH,cV,cD) = coeffs[0]

    cA = watermarked_image

    ########################################
    
    
    signal= extract_fun(cA,m,n)
    f1 = (signal[0] - np.min(signal[0])) / (np.max(signal[0]) - np.min(signal[0]))  
    f2 = (signal[1] - np.min(signal[1])) / (np.max(signal[1]) - np.min(signal[1]))  
    f3 = (signal[2] - np.min(signal[2])) / (np.max(signal[2]) - np.min(signal[2]))  
    #cv2.imwrite('./image/paper use/final1.jpg', f1*255)
    #cv2.imwrite('./image/paper use/final2.jpg', f2*255)
    #cv2.imwrite('./image/paper use/final3.jpg', f3*255)
    #cv2.imwrite('./image/paper use/final1.jpg', signal[0]*255)
    #cv2.imwrite('./image/paper use/final2.jpg', signal[1]*255)
    #cv2.imwrite('./image/paper use/final3.jpg', signal[2]*255)
    #print(signal[2])

    return signal

if __name__== "__main__" :
    img_g = cv2.imread('./image/ori_image/Kodim25.jpg')
    #cv2.imwrite('./image/paper use/yuan.jpg', img_g)
    img_g = cv2.resize(img_g,(512,512))
    image_g = cv2.cvtColor(img_g,cv2.COLOR_RGB2GRAY)/255
    img_feature_global = extracting_global(image_g)