import cv2
import os
import tensorflow as tf
from PIL import Image
import numpy as np


def has_white_area(image_path):
    # 加载图像
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 将图像进行二值化处理
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 计算二值化图像中白色像素的比例
    white_pixels_ratio = cv2.countNonZero(threshold) / (threshold.shape[0] * threshold.shape[1])

    # 根据白色像素的比例判断是否存在白色区域
    if white_pixels_ratio > 0.01:  # 根据具体情况调整阈值
        return 1
    else:
        return 0

# 训练集图像文件夹路径
image_folder_train = './train'

# 打开txt文件，准备写入结果
with open('train_labels.txt', 'w') as f:
    # 遍历图像文件夹中的图像文件并按顺序判断
    for filename in sorted(os.listdir(image_folder_train)):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = os.path.join(image_folder_train, filename)
            label = has_white_area(image_path)
            f.write(f'{filename}\t{label}\n')

# 验证集图像文件夹路径
image_folder_validation = './validation'

# 打开txt文件，准备写入结果
with open('validation_labels.txt', 'w') as f:
    # 遍历图像文件夹中的图像文件并按顺序判断
    for filename in sorted(os.listdir(image_folder_validation)):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = os.path.join(image_folder_validation, filename)
            label = has_white_area(image_path)
            f.write(f'{filename}\t{label}\n')

# 图像文件夹路径
image_folder_test = './test'

# 打开txt文件，准备写入结果
with open('test_labels.txt', 'w') as f:
    # 遍历图像文件夹中的图像文件并按顺序判断
    for filename in sorted(os.listdir(image_folder_test)):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = os.path.join(image_folder_test, filename)
            label = has_white_area(image_path)
            f.write(f'{filename}\t{label}\n')

print("图像识别完成")

# 根据所给教程修改源代码如下
train_path = './train/'
train_txt = 'train_labels.txt'
x_train_savepath = './train/x_train.npy'
y_train_savepath = './train/y_train.npy'

test_path = './test/'
test_txt = 'test_labels.txt'
x_test_savepath = './test/x_test.npy'
y_test_savepath = './test/y_test.npy'

validation_path = './validation/'
validation_txt = 'validation_labels.txt'
x_validation_savepath = './validation/x_validation.npy'
y_validation_savepath = './validation/y_validation.npy'


def generateds(path, txt):
    f = open(txt, 'r')  # 以只读形式打开txt文件
    contents = f.readlines()  # 读取文件中所有行
    f.close()  # 关闭txt文件
    x, y_ = [], []  # 建立空列表
    for content in contents:  # 逐行取出
        value = content.split()  # 以空格分开，图片路径为value[0] , 标签为value[1] , 存入列表
        img_path = path + value[0]  # 拼出图片路径和文件名
        img = Image.open(img_path)  # 读入图片
        img = np.array(img.convert('L'))  # 图片变为8位宽灰度值的np.array格式
        img = img / 255.  # 数据归一化 （实现预处理）
        x.append(img)  # 归一化后的数据，贴到列表x
        y_.append(value[1])  # 标签贴到列表y_
        print('loading : ' + content)  # 打印状态提示

    x = np.array(x)  # 变为np.array格式
    y_ = np.array(y_)  # 变为np.array格式
    y_ = y_.astype(np.int64)  # 变为64位整型
    return x, y_  # 返回输入特征x，返回标签y_


if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath) and os.path.exists(
        x_test_savepath) and os.path.exists(y_test_savepath) and os.path.exists(
        x_validation_savepath) and os.path.exists(y_validation_savepath):
    print('-------------Load Datasets-----------------')
    x_train_save = np.load(x_train_savepath)
    y_train = np.load(y_train_savepath)
    x_test_save = np.load(x_test_savepath)
    y_test = np.load(y_test_savepath)
    x_validation_save = np.load(x_validation_savepath)
    y_validation = np.load(y_validation_savepath)
    x_train = np.reshape(x_train_save, (len(x_train_save), 28, 28))
    x_test = np.reshape(x_test_save, (len(x_test_save), 28, 28))
    x_validation = np.reshape(x_validation_save, (len(x_validation_save), 28, 28))
else:
    print('-------------Generate Datasets-----------------')
    x_train, y_train = generateds(train_path, train_txt)
    x_test, y_test = generateds(test_path, test_txt)
    x_validation, y_validation = generateds(validation_path, validation_txt)
    print('-------------Save Datasets-----------------')
    x_train_save = np.reshape(x_train, (len(x_train), -1))
    x_test_save = np.reshape(x_test, (len(x_test), -1))
    x_validation_save = np.reshape(x_validation, (len(x_validation), -1))
    np.save(x_train_savepath, x_train_save)
    np.save(y_train_savepath, y_train)
    np.save(x_test_savepath, x_test_save)
    np.save(y_test_savepath, y_test)
    np.save(x_validation_savepath,x_validation_save)
    np.save(y_validation_savepath,y_validation)
