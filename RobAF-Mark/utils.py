
from ast import Name
import math
import cv2
import numpy as np
import copy
import shutil
import os



def distance(pointA:list,pointB:list): # 这里面的输入是一个(x,y),(x,y)
    result = math.sqrt(
    math.pow(pointA[0] - pointB[0],2) +
    math.pow(pointA[1] -pointB[1],2)
    )
    return result

def imageAlignmentSimple(img1,img2):
    '''
    重新匹配 ori_h,ori_w, pho, kp1, kp2, goodMatches
    '''
    orb = cv2.ORB_create()

    # 寻找关键点
    kp1 = orb.detect(img1)
    kp2 = orb.detect(img2)
    """
    kp包含多个点(元组的形式),每个点为一个字典(可以用.方法来调用)
    每个点包括angle,class_id,octave,pt:(x,y)位置参数,response(响应值),size(点的大小)
    """
    # kp_temp = []

    # print(type(kp1))# tuple-->元组
    # print(kp1[0].response) # 0.003020771313458681 
    # kp_temp = copy.copy(kp1) # deepcopy不行
    # kp_temp = list(kp_temp)
    # # del kp_temp[-1] #确实是删去了最后一个,证明是有用的
    # print(kp_temp[0].pt)    # 输出元组
    
    # kp_temp = tuple(kp_temp)
    
    # 计算描述符
    kp1, des1 = orb.compute(img1, kp1)
    # kp1, des1 = orb.compute(img1, kp1)
    kp2, des2 = orb.compute(img2, kp2)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # 对描述子进行匹配
    matches = bf.match(des1, des2)

    # 计算最大距离和最小距离
    min_distance = matches[0].distance
    max_distance = matches[0].distance
    for x in matches:
        if x.distance < min_distance:
            min_distance = x.distance
        if x.distance > max_distance:
            max_distance = x.distance

    # 筛选匹配点
    '''
        当描述子之间的距离大于两倍的最小距离时,认为匹配有误。
        但有时候最小距离会非常小,所以设置一个经验值30作为下限。
    '''
    goodMatches = []
    for x in matches:
        if x.distance <= max(2 * min_distance, 30):
            goodMatches.append(x)
    

    ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
    ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
    if len(goodMatches) > 4:
        ransacReprojThreshold = 10
        """
        在kp1是list的情况下:ptsA = (500,1,2):[[[381.,397.]]]
        kp1是元组的情况下:ptsA = (69,1,2)
        
        同时ptsB = (69,1,2)
        """
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold)
        if H is not None:
            imgOut = cv2.warpPerspective(img2, H, (img1.shape[1],img1.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            imgOut = np.zeros((1,1,3))
    else:
        imgOut = np.zeros((1,1,3))
        
    # pic_matches = cv2.drawMatches(ori, kp1, pho, kp2, goodMatches, pho, flags=2)
    return imgOut

def select_kp(kp,std_distance):
    """
    需要先计算 :
    1.符合距离在x之内的点有哪些点
    2.比较responce的值,然后如果是本区域里面的店里面最大的,则keep下来  
    3.kp包含多个点(元组的形式),每个点为一个字典(可以用.方法来调用)
    4.每个点包括angle,class_id,octave,pt:(x,y)位置参数,response(响应值),size(点的大小)
    """
    temp_kp = [] #用于放置符合条件的kp,之后要转化成为tuple
    # std_distance = 30*math.sqrt(2)              
    # 40---有效率是50%
    for i in range(len(kp)):
        ismax = True
        for j in range(len(kp)): #range(1,11)是0-10,右边不包括   ！这里有问题！
            
            pt_distance = distance(kp[i].pt,kp[j].pt)
            x = kp[i].response
            y = kp[j].response
            if pt_distance < std_distance and kp[i].response < kp[j].response: #两个点之间的距离小于标准距离()
                ismax = False
                break
            
        if ismax:
            temp_kp.append(kp[i])  
            # print('原特征点第{}个点的坐标是:'.format(i),(kp[i].pt[0],kp[i].pt[1]))       
    return temp_kp

def select0(kp,R):
    temp_kp = [[] for i in range(3)] #用于放置符合条件的kp,之后要转化成为tuple
    
    # std_distance = 30*math.sqrt(2)              
    # 40---有效率是50%
    std_distance = R 
    num = len(kp[0])
    flag = [1]*num
    for i in range(num):  # 点的个数
        ismax = True
        for j in range(i+1,num): #range(1,11)是0-10,右边不包括   ！这里有问题！
            p1 = (kp[0][i],kp[1][i])
            p2 = (kp[0][j],kp[1][j])
            
            pt_distance = distance(p1,p2)
            v1 = kp[2][i]
            v2 = kp[2][j]
            if pt_distance < std_distance and v1 < v2: #两个点之间的距离小于标准距离()
                flag[i] = 0
            if pt_distance < std_distance and v1 > v2: 
                flag[j] = 0
            
    for k in range(num):
        if flag[k] == 1:
            temp_kp[0].append(kp[0][k])  
            temp_kp[1].append(kp[1][k])  
            temp_kp[2].append(kp[2][k])  
            # print('原特征点第{}个点的坐标是:'.format(i),(kp[i].pt[0],kp[i].pt[1]))       
    return temp_kp
    
    






def RemoveDir(filepath):
    '''
    如果文件夹不存在就创建,如果文件存在就清空！
    
    '''
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)

def compare(kp1:tuple,kp2:tuple,opt):  
    
    """
    比较两个图的特征点的相似程度
    由于经历了纠正的过程所以会有一些冗余,设置为 δ
    经过肉眼观察,x,y一般不会超过1(作为同一个点的话)
    kp1是ori的点,kp2是纠正后的点
    """
    common_kp = []
    delta = opt.delta
    for i in range(len(kp1)):
        for j in range(len(kp2)):
            pt_distance = distance(kp1[i].pt,kp2[j].pt)  #函数之间是可以直接使用的原来
            if pt_distance < delta:  #这个算是还行的误差
                common_kp.append(kp1[i])
                
    ratio = (len(common_kp)/len(kp1))
    return ratio,common_kp

def kp2rectangle(img,kp,size_block,name:str):
    h,w = img.shape[0],img.shape[1]
    cord = find_cordination(kp,size_block,h,w)
    for s in range(len(cord)):
        img = cv2.putText(img, '{}'.format(s), (int(cord[s][2]),int(cord[s][0])-5), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)# cv2.putText(img, str,origin,font,size,color,thickness)
        img = cv2.rectangle(img, (int(cord[s][2]),int(cord[s][0])), (int(cord[s][3]),int(cord[s][1])), (0, 255, 0), 1)
        img = cv2.rectangle(img, (int(cord[s][2]), int(cord[s][0])), (int(cord[s][3]), int(cord[s][1])), (0, 255, 255), 1)
    cv2.imwrite('/work/wzz/match/rectangle/rectangle{}.jpg'.format(Name),img)
    

def get_common_kp(img_ori,img_pho,opt): #输入是imread进来的矩阵
    """
    1.输入标准图和raw图片---获得ori_img和corrected_img
    2.ORB特征提取 kp_ori和kp_pho
    3.利用select_kp获得select_ori_kp和select_pho_kp
    
    """
    orb = cv2.ORB_create()
    kp = orb.detect(img_ori)
    
    corrected_img =  imageAlignmentSimple(img_ori,img_pho)  # 怎么会出问题？？
    kp_pho = orb.detect(corrected_img)
    
    R = opt.R
    kp_selected = select_kp(kp,R)
    kp_pho_selected = select_kp(kp_pho,R)
    #这里我是先求圆域内的最大特征点,再求common_pt所以其实不影响
    temp1 = cv2.drawKeypoints(corrected_img, keypoints=kp_pho_selected, outImage=None)
    temp2 = cv2.drawKeypoints(img_ori, keypoints=kp_selected, outImage=None)
    temp3 = cv2.drawKeypoints(corrected_img, keypoints=kp_pho, outImage=None)
    temp4 = cv2.drawKeypoints(img_ori, keypoints=kp, outImage=None)
    cv2.imwrite('/work/wzz/match/feature/corrected_uninserted_pho.jpg',temp1)
    cv2.imwrite('/work/wzz/match/feature/ori{}.jpg'.format(round(opt.R,3)),temp2)
    cv2.imwrite('/work/wzz/match/feature/corrected_uninserted_pho_all.jpg',temp3)
    cv2.imwrite('/work/wzz/match/feature/ori_all.jpg',temp4)
    
    print('原图筛选后的特征点数量为:{}'.format(len(kp_selected)))
    print('拍照后筛选特征点数量为:{}'.format(len(kp_pho_selected)))
    
    _,common_kp_selected = compare(kp_selected,kp_pho_selected,opt)  #  改动！不要
    temp3 = cv2.drawKeypoints(img_ori, keypoints=common_kp_selected, outImage=None)
    cv2.imwrite('/work/wzz/match/feature/ori_common.jpg',temp3)
    print('原图和拍照纠正图的共同特征点个数为{},有效率为{}'.format(len(common_kp_selected),_))
    # print('共同特征点数量为:{}'.format(len(common_kp_selected)))
    return _,common_kp_selected
    
def find_cordination(common_kp,size_block,h,w):
    block_cord = []
    for i in range(len(common_kp)):
        y = common_kp[i].pt[0]
        x = common_kp[i].pt[1]
        start_x = int(x - 0.5*size_block)
        end_x = int(x + 0.5*size_block)
        start_y = int(y - 0.5*size_block)
        end_y = int(y + 0.5*size_block)
        if end_x < h and start_x < w:
            block_cord.append([start_x,end_x,start_y,end_y])
            # print('有效嵌入点的中心坐标为:',(round(x,3),round(y,3)))  ---有框图的情况下可以说是没啥用
    return block_cord

def find_cordination_new(common_kp,size_block,h,w):
    cnt = 0     # 有效嵌入点计数器
    block_cord = []
    for i in range(len(common_kp[0])):
        y = common_kp[0][i]
        x = common_kp[1][i]
        start_x = int(x - 0.5*size_block)
        end_x = int(x + 0.5*size_block)
        start_y = int(y - 0.5*size_block)
        end_y = int(y + 0.5*size_block)
        if end_x < h and start_x < w:
            block_cord.append([start_x,end_x,start_y,end_y])
            cnt += 1
            # if cnt >= 3:
            #     break
            # print('有效嵌入点的中心坐标为:',(round(x,3),round(y,3)))  ---有框图的情况下可以说是没啥用
    return block_cord
 
def if_in(pt,cord): 
    #用来决定是否这个点在cord里面
    #pt = [x1,y1,x2,y2]:传统的点坐标
    flag = False
    for i in range(len(cord)):
        distance  = math.sqrt(
    math.pow(pt[0] - cord[i][0],2) +
    math.pow(pt[1] - cord[i][1],2) +
    math.pow(pt[2] - cord[i][2],2) +
    math.pow(pt[3] - cord[i][3],2)
    )
        if distance < 12:
            flag = True
            break

    return flag

if __name__ == '__main__':
    #用来测试
    num = 0
    for m in range(1,15):
        img_ori = cv2.imread('/work/wzz/match/demo_pts_img/ori{}.jpg'.format(m))
        # img_pho = cv.imread('/work/wzz/match/wm_img/corrected_A.png')
        # img_pho = cv.imread('/work/wzz/match/corrected_ori.png')
        orb = cv2.ORB_create()

        R = 40*math.sqrt(2) # 这个时候按理来说有16个可以48
        
        # 寻找关键点
        kp = orb.detect(img_ori)
        kp_part = kp[:] # 用于调节参与选择的特征点的数量
        print('原图原始点的数量:',len(kp_part)) #
        kp_selected = select_kp(kp_part,R)  #第二项是一个位置参数
        print('原图筛选后的点的数量:',len(kp_selected)) #74
        print('\n')
        num += len(kp_selected)

        
        
        for x in range(len(kp_selected)):
            point_size = 1
            point_color = (0, 0, 255)  # BGR
            thickness = 2 # https://blog.csdn.net/hzblucky1314/article/details/123896460
            pts = (int(kp_selected[x].pt[0]),int(kp_selected[x].pt[1]))
            cv2.circle(img_ori, pts, 5, point_color, thickness)
            # cv2.imwrite('/work/wzz/match/points/real_pts{}.jpg'.format(),img_ori)
        cv2.imwrite('/work/wzz/match/ORB_img/ORB_img{}.jpg'.format(m),img_ori)
    print(num/14)

    
    

    
    

