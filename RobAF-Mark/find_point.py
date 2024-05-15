import numpy as np
import torch
import cv2 as cv
import csv
from superpoint import *
from utils import *
from argparse_opt import arg
import matplotlib.pyplot as plt  
import matplotlib.patches as patches

# 画一组矩形
def plot_point(img_ori, block_cord, color):
	img_RGB = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)

	for box in block_cord:
		y_start, y_end, x_start, x_end = box
		# 根据描述创建矩形对象  
		rect = patches.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start, fill=False, color = color)
		# 在图像上添加矩形  
		plt.imshow(img_RGB)
		plt.gca().add_artist(rect)
	
	plt.show()

# 同时画两组矩形
def plot_point_2(img_ori, block_cord1, block_cord2, color1, color2):
	img_RGB = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)  

	for box in block_cord1:  
		y_start, y_end, x_start, x_end = box  
		rect = patches.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start, fill=False, color = color1)  
		plt.imshow(img_RGB)
		plt.gca().add_artist(rect)
	
	for box in block_cord2:  
		y_start, y_end, x_start, x_end = box  
		rect = patches.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start, fill=False, color = color2)  
		plt.imshow(img_RGB)
		plt.gca().add_artist(rect)
	
	plt.show()

# 改框的大小在agr()里面
def find_block(img_ori, grayim, if_plot):
	opt = arg()

	fe = SuperPointFrontend(weights_path=opt.weights_path,
					nms_dist=opt.nms_dist,
					conf_thresh=opt.conf_thresh,
					nn_thresh=opt.nn_thresh,
					cuda=opt.cuda)

	# imgPath = 'D:\Code!\WaterMark\my_code1\image\cover_image\patrick.png'

	#img_ori = cv.imread(imgPath)
	b,g,r = cv.split(img_ori)
	size_block = opt.size_block
	h,w = b.shape

	#grayim = cv.imread(imgPath, 0)
	interp = cv.INTER_AREA
	grayim = cv.resize(grayim, (opt.W, opt.H), interpolation=interp)
	grayim = (grayim.astype('float32') / 255.)
	pts, desc, heatmap = fe.run(grayim)
	real_pts = (pts[0]*(w/160.),pts[1]*(h/120.),pts[2]) # real_pts -->  (3,num)

	# print(len(real_pts[0]))

	select_real_pts = select0(real_pts,opt.R)
	# print('筛选前：',len(real_pts[0]))
	# print('筛选后：',len(select_real_pts[0]))

	# print(select_real_pts)

	# 在函数内控制选择的点的个数
	# 在opt中控制框的大小
	block_cord = find_cordination_new(select_real_pts,size_block,h,w)
	# print("排除边界后有效的水印嵌入块个数为：",len(block_cord))

	# print(block_cord)

	if if_plot == 1:	# 画出全部框
		plot_point(img_ori, block_cord, "black")

	if if_plot == 3:	# 画出前三个框
		plot_point(img_ori, block_cord[0:3], 'black')

	return block_cord

# 用来计算两个矩形重叠面积
def get_overlap_area(rect1, rect2):
	# 如果两个矩形不重叠，返回0
	if rect1[0] >= rect2[1] or rect1[1] <= rect2[0] or rect1[2] >= rect2[3] or rect1[3] <= rect2[2]:
		return 0
	y_len = min(rect1[1], rect2[1]) - max(rect1[0], rect2[0])
	x_len = min(rect1[3], rect2[3]) - max(rect1[2], rect2[2])
	return y_len * x_len

# 用来计算两组矩形重叠面积
def find_max_overlap(L1, L2, mode):
	result = []
	# 遍历L1中的每个矩形，对L1中每个矩形都找到L2中重叠面积最大的矩形
	for rect1 in L1:
		max_area = 0
		max_rect = None
		for rect2 in L2:
			# 计算当前两个矩形的重叠面积
			area = get_overlap_area(rect1, rect2)
			# 更新最大重叠面积和对应的矩形
			if area > max_area:
				max_area = area
				max_rect = rect2
		if mode == "mapping":
			result.append(max_area)
		elif mode == "extracting":
			result.append(max_rect)
	return result

# 如果要改框的个数，记得改这里
def find_robust_point_mapping(img_ori, img_gray, imgPath, image_idx, attacks):
	#img_ori = cv.imread(imgPath)
	block_cord_ori = find_block(img_ori, img_gray, 0)
	
	cnt = 0
	block_table = []
	for attack in attacks:
		cnt += 1
		attack_path = "./image/attacked_image2/" + attack + '.jpg'
		host_image_gray = cv2.imread(attack_path, 0)
		host_image_ori = cv2.imread(attack_path)
		#host_image_ori = np.stack([host_image_gray]*3, axis=-1)  
		block_cord_attacked = find_block(host_image_ori, host_image_gray, 0)
		max_overleap = find_max_overlap(block_cord_ori, block_cord_attacked, "mapping")
		block_table.append(max_overleap)

		# 画图
		if cnt < 0:
			plot_point_2(img_ori, block_cord_ori, block_cord_attacked, "blue", "green")

	# print(block_table)
	save_path = "./block_table/" + "table_of_" + str(image_idx) + ".csv"
	with open(save_path, 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerows(block_table)
	
	sum_overleap = [0 for _ in range(len(block_table[0]))]
	for i in range(len(block_table)):
		for j in range(len(block_table[0])):
			sum_overleap[j] += block_table[i][j]
	# print(sum_overleap)
	
	
	min_overleap = [99999 for _ in range(len(block_table[0]))]
	for i in range(len(block_table)):
		for j in range(len(block_table[0])):
			min_overleap[j] = min(block_table[i][j], min_overleap[j])

	zonghe_overleap = [0 for _ in range(len(block_table[0]))]
	for i in range(len(block_table[0])):
		zonghe_overleap[i] = min_overleap[i]*len(block_table) + sum_overleap[i]
	#print(min_overleap, len(block_table), sum_overleap)
	
	# print(sum_overleap)
	# 取总重叠面积最大的三个点(取最小重叠面积最大的点可能会更好？)
	# 取最小重叠面积最大的三个点（取总重叠面积最大效果不好）

	# 使用enumerate()函数获取每个元素及其索引
	#enum_sum = list(enumerate(max_overleap))
	#enum_sum = list(enumerate(min_overleap))
	enum_sum = list(enumerate(zonghe_overleap))

	sorted_enum_sum = sorted(enum_sum, key=lambda x: x[1], reverse=True)
	
	block_cord_final = []
	for i, (index, value) in enumerate(sorted_enum_sum):  
		#print(index)
		#print(value)
		#if i == 20 or i == 21 or i == 19:
		block_cord_final.append(block_cord_ori[index])
		if i >= 2:	# 只要前三个
			break
	# print(block_cord_final)
	
	#画图
	#plot_point(img_ori, block_cord_final,  "blue")
		
	save_path = "./block_cord/" + "block_of_" + str(image_idx) + ".csv"
	with open(save_path, 'w', newline='') as file:  
		writer = csv.writer(file)
		writer.writerows(block_cord_final)
		
	return block_cord_final

def distance(rect1, rect2):
	y1 = (rect1[0] + rect1[1]) / 2
	x1 = (rect1[2] + rect1[3]) / 2
	y2 = (rect2[0] + rect2[1]) / 2
	x2 = (rect2[2] + rect2[3]) / 2
	return ((y1 - y2) ** 2 + (x1 - x2) ** 2) ** 0.5

def find_min_distance(block_cord1, block_cord2):
	result = []
	for rect1 in block_cord1:
		min_dist = float('inf')
		closest_rect = None
		for rect2 in block_cord2:
			dist = distance(rect1, rect2)
			if dist < min_dist:
				min_dist = dist
				closest_rect = rect2
		result.append(closest_rect)
	return result

def find_robust_point_extracting(imgPath, image_idx):
	# img_now = cv.imread(imgPath)
	# block_cord_now = find_block(imgPath, 0)

	read_path = "./block_cord/" + "block_of_" + str(image_idx) + ".csv"
	with open(read_path, 'r') as file:  
		reader = csv.reader(file)  
		block_cord_ori = [row for row in reader]
	block_cord_ori = [[int(x) for x in sublist] for sublist in block_cord_ori]
	#print(block_cord_ori)
	
	# 这句是找重合面积最大的框，有可能出现没有重叠的框的情况。
	# block_cord = find_max_overlap(block_cord_ori, block_cord_now, "extracting")

	# 这句是找距离最小的对应框
	# block_cord = find_min_distance(block_cord_ori, block_cord_now)

	# 这句是直接用记录的框
	block_cord = block_cord_ori

	

	return block_cord

if __name__== "__main__" :
	imgPath = "./image/ori_image/kodim23.png"
	block_cord = find_robust_point_mapping(imgPath, 23)
	print(block_cord)

	img_ori = cv.imread(imgPath)
	plot_point(img_ori, block_cord, 'black')

	#imgPath = './image/attacked_image/jepg/jepg23.jpg'
	#imgPath = './image/attacked_image/zoom/ori23.jpg'
	#block_cord = find_robust_point_extracting(imgPath, 23)
	#print(block_cord)