import numpy as np
import csv
import torch

# 定义arnold变换函数
def arnold_transform(wm, a, b, n,watermark_size):
    # wm: 输入的图像，必须是32*32的numpy数组
    # a, b: 变换的参数，必须是整数
    # n: 变换的次数，必须是正整数
    # 返回值: 变换后的图像，也是32*32的numpy数组
    N = watermark_size # 图像的大小
    #print(wm, type(wm))
    new_wm = np.zeros_like(wm) # 创建一个空的数组
    for i in range(n): # 重复n次变换
        for x in range(N): # 遍历每个像素的横坐标
            for y in range(N): # 遍历每个像素的纵坐标
                # 根据arnold变换公式计算新的坐标
                new_x = (x + b * y) % N
                new_y = (a * x + (a * b + 1) * y) % N
                # 将原像素的值赋给新的位置
                new_wm[new_x, new_y] = wm[x, y]
        # 更新wm为变换后的图像
        wm = new_wm.copy()
    # 返回最终的变换结果
    new_wm = torch.tensor(new_wm, dtype=torch.float32)
    return new_wm

# 定义arnold逆变换函数
def arnold_inverse_transform(wm, a, b, n, watermark_size):
    # wm: 输入的图像，必须是32*32的numpy数组
    # a, b: 变换的参数，必须是整数
    # n: 变换的次数，必须是正整数
    # 返回值: 变换后的图像，也是32*32的numpy数组
    N = watermark_size # 图像的大小
    #print(wm, type(wm))
    wm = wm.detach()
    new_wm = np.zeros_like(wm) # 创建一个空的数组
    for i in range(n): # 重复n次变换
        for x in range(N): # 遍历每个像素的横坐标
            for y in range(N): # 遍历每个像素的纵坐标
                # 根据arnold逆变换公式计算新的坐标
                new_x = ((a * b + 1) * x - b * y) % N
                new_y = (-a * x + y) % N
                # 将原像素的值赋给新的位置
                new_wm[new_x, new_y] = wm[x, y]
        # 更新wm为变换后的图像
        wm = new_wm.copy()
    # 返回最终的变换结果
    new_wm = torch.tensor(new_wm, dtype=torch.float32)
    return new_wm

if __name__ == "__main__":
    # 测试代码
    # 创建一个32*32的随机数组，模拟一个灰度图像
    with open("wm.csv", "r") as f:
        reader = csv.reader(f)
        strings = list(reader)
        wm= strings[0][0]    # 根据wm_idx取出相应的水印

    wm = torch.Tensor([int(c) for c in wm]).reshape((32, 32))
    # 打印原始图像
    print("原始图像:")
    print(wm)
    # 对图像进行arnold变换，参数为a=1, b=1, n=10
    encoded_wm = arnold_transform(wm, 1, 1, 10)
    # 打印变换后的图像
    print("变换后的图像:")
    print(encoded_wm)
    # 对变换后的图像进行arnold逆变换，参数与变换相同
    decoded_wm = arnold_inverse_transform(encoded_wm, 1, 1, 10)
    # 打印逆变换后的图像
    print("逆变换后的图像:")
    print(decoded_wm)
    # 检查原始图像和逆变换后的图像是否相同
    print("原始图像和逆变换后的图像是否相同:", np.array_equal(wm, decoded_wm))
