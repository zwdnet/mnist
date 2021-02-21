# coding:utf-8
# 测试遍历目录下文件


import os
from skimage import io,data,transform
import numpy as np
import matplotlib.pyplot as plt
import run


# 将图片文件转换为MNIST数据
@run.change_dir
def changeData():
    path = "./mynum/num/"
    dir_or_files = os.listdir(path)
    files = []
    for dir_file in os.listdir(path):
        files.append(dir_file)
        
    MNIST_SIZE = 28
    data = []
    for file in files:
        # 处理图片
        # 读入图片并变成灰色
        img = io.imread(path+files[0], as_gray=True)
        # 缩小到28*28
        translated_img = transform.resize(img, (MNIST_SIZE, MNIST_SIZE))
        # 变成1*784的一维数组
        flatten_img = np.reshape(translated_img, 784)
        # 1代表黑，0代表白
        result = np.array([1 - flatten_img])
        data.append(result)
    data = np.array(data)
    # print(data.shape)


if __name__ == "__main__":
    changeData()
    