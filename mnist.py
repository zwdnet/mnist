# coding:utf-8
# MNIST手写数字识别


import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import os
import run
import copy
import tqdm
import matplotlib.pyplot as plt


from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from skimage import io,data,transform


# 公共模块
# 下载并加载数据
@run.change_dir
def loadData(batch_size = 64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_train = MNIST(os.getcwd(), train=True, download=True,  transform=transform)
    mnist_test = MNIST(os.getcwd(), train=False, download=True,  transform=transform)
    
    mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])
    
    # 创建 DataLoader
    mnist_train = DataLoader(mnist_train, batch_size)
    mnist_val = DataLoader(mnist_val, batch_size)
    mnist_test = DataLoader(mnist_test, batch_size)
    
    return mnist_train, mnist_val, mnist_test
    
    
# 将DataLoader数据转换为numpy数组
def Loader2numpy(Loader):
    X = []
    Y = []
    for x, y in Loader:
        for i in x:
            i = i.view(1, -1).detach().numpy()
            # print("测试", i.shape)
            X.append(i[0])
        for j in y.detach().numpy():
            Y.append(j)
    return np.array(X), np.array(Y)
    
    
# 计算预测准确率
def accuracy(y_pred, y_true):
    acc = (y_pred == y_true).sum()/y_true.shape[0]
    return acc
    
    
# 保存算法准确率数据
@run.change_dir
def saveResult(accuracy):
    accuracy.to_csv("./output/results.csv")
    
    
# 进行研究
@run.change_dir
def Research():
    torch.manual_seed(666)
    batch_size = 500
    print("准备数据")
    mnist_train, mnist_val, mnist_test = loadData(batch_size)
    X_train, Y_train = Loader2numpy(mnist_train)
    X_val, Y_val = Loader2numpy(mnist_val)
    X_test, Y_test = Loader2numpy(mnist_test)
    # print("合并前", X_train.shape, X_val.shape)
    # 合并训练集和验证集
    X_train = np.concatenate((X_train, X_val), axis = 0)
    Y_train = np.concatenate((Y_train, Y_val), axis = 0)
    # print("合并后", X_train.shape, X_val.shape)
    
    # 模型预测准确率
    accuracy_results = pd.DataFrame()
    
    # 算法1:瞎猜
    print("算法1:瞎猜")
    acc = Random_Model(X_train.copy(), Y_train.copy(), X_test.copy(), Y_test.copy())
    accuracy_results["随机算法"] = [acc]
    
    # 算法2:逻辑回归
    print("算法2:逻辑回归")
    acc = LogisticRegression_Model(X_train.copy(), Y_train.copy(), X_test.copy(), Y_test.copy())
    accuracy_results["逻辑回归算法"] = [acc]
    
    # 算法3:朴素贝叶斯
    print("算法3:朴素贝叶斯")
    acc = Bayes_Model(X_train.copy(), Y_train.copy(), X_test.copy(), Y_test.copy())
    accuracy_results["朴素贝叶斯算法"] = [acc]
    
    # 算法4:支持向量机
    print("算法4:支持向量机")
    acc = SVM_Model(X_train.copy(), Y_train.copy(), X_test.copy(), Y_test.copy())
    accuracy_results["支持向量机算法"] = [acc]
    
    # 算法5:KNN
    print("算法5:KNN算法")
    acc = KNN_Model(X_train.copy(), Y_train.copy(), X_test.copy(), Y_test.copy())
    accuracy_results["支持向量机算法"] = [acc]
    
    # 算法6:随机森林
    print("算法6:随机森林算法")
    acc = RF_Model(X_train.copy(), Y_train.copy(), X_test.copy(), Y_test.copy())
    accuracy_results["随机森林算法"] = [acc]
    
    # 算法7:一般神经网络
    print("算法7:一般神经网络算法")
    acc = NN_Model(copy.deepcopy(mnist_train), copy.deepcopy(mnist_val), copy.deepcopy(mnist_test), batch_size = batch_size)
    accuracy_results["一般神经网络算法"] = [acc]

    # 算法8:卷积神经网络
    print("算法8:卷积神经网络算法")
    acc = CONV_Model(copy.deepcopy(mnist_train), copy.deepcopy(mnist_val), copy.deepcopy(mnist_test), batch_size = batch_size)
    accuracy_results["卷积神经网络算法"] = [acc]
    
    # 记录各个算法的准确率
    saveResult(accuracy_results)
    
    print("研究结果")
    print(accuracy_results)
    
    
# 具体算法

# 算法1:随机算法
@run.timethis
def Random_Model(X_train, Y_train, X_test, Y_test):
    y_pred = np.random.randint(low = 0, high = 10, size = Y_test.shape[0])
    acc = accuracy(y_pred, Y_test)
    print("随机算法预测准确率:{}".format(acc))
    return acc


# 算法2:逻辑回归
@run.timethis
def LogisticRegression_Model(X_train, Y_train, X_test, Y_test):
    lor = LogisticRegression(C=100,multi_class='ovr')
    # 训练模型
    lor.fit(X_train,Y_train)
    # score = lor.score(X_std_test, Y_test)
    y_pred = lor.predict(X_test)
    acc = accuracy(y_pred, Y_test)
    print("逻辑回归算法预测准确率:{}".format(acc))
    return acc
    
    
# 算法3:朴素贝叶斯
@run.timethis
def Bayes_Model(X_train, Y_train, X_test, Y_test):
    clf = GaussianNB()
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy(y_pred, Y_test)
    print("朴素贝叶斯算法预测准确率:{}".format(acc))
    return acc
    
    
# 算法4:支持向量机
@run.timethis
def SVM_Model(X_train, Y_train, X_test, Y_test):
    model = SVC()
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    acc = accuracy(y_pred, Y_test)
    print("支持向量机算法预测准确率:{}".format(acc))
    return acc
    
    
# 算法5:KNN
@run.timethis
def KNN_Model(X_train, Y_train, X_test, Y_test):
    model = KNeighborsClassifier()
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    acc = accuracy(y_pred, Y_test)
    print("KNN算法预测准确率:{}".format(acc))
    return acc
    
    
# 算法6:随机森林
@run.timethis
def RF_Model(X_train, Y_train, X_test, Y_test):
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    acc = accuracy(y_pred, Y_test)
    print("随机森林算法预测准确率:{}".format(acc))
    return acc
    

# 算法7:一般神经网络
class fc_net(nn.Module):
    def __init__(self, batch_size = 64):
        super(fc_net, self).__init__()
        self.layer_1 = nn.Linear(28*28, 200)
        self.layer_2 = nn.Linear(200, 100)
        self.layer_3 = nn.Linear(100, 20)
        self.layer_4 = nn.Linear(20, 10)
        self.batch_size = batch_size
        
    def forward(self, x):
        x = self.layer_1(x)
        nn.ReLU()
        x = self.layer_2(x)
        nn.ReLU()
        x = self.layer_3(x)
        nn.ReLU()
        x = self.layer_4(x)

        return x
    

@run.change_dir
@run.timethis
def NN_Model(mnist_train, mnist_val, mnist_test, batch_size = 64, lr = 0.001):
    epochs = 20
    net = fc_net(batch_size)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters(), lr = lr, weight_decay=0.0)
    # print(batch_size)
    
    for epoch in tqdm.tqdm(range(epochs)):
        # 训练过程
        train_loss = []
        accuracy = 0.0
        train_accuracy = []
        for x, y in mnist_train:
            #print(x.shape)
            x = x.view(batch_size, -1)
            y_pred = net.forward(x)
            loss = criterion(y_pred, y)
            train_loss.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
            # 计算预测准确率
            with torch.no_grad():
                y_pred = torch.max(y_pred.data, 1).indices
                accuracy += (y_pred == y).sum().item()
        train_accuracy.append(accuracy/(len(mnist_train)*batch_size))
        mean_train_loss = torch.mean(torch.tensor(train_loss))
        # 验证过程
        with torch.no_grad():
            val_loss = []
            accuracy = 0.0
            val_accuracy = []
            for x, y in mnist_val:
                x = x.view(batch_size, -1)
                y_pred = net.forward(x)
                # y_pred = torch.max(y_pred.data, 1).indices
                loss = criterion(y_pred, y)
                val_loss.append(loss.item())
                # 计算预测准确率
                y_pred = torch.max(y_pred.data, 1).indices
                accuracy += (y_pred == y).sum().item()
            val_accuracy.append(accuracy/(len(mnist_val)*batch_size))
            mean_val_loss = torch.mean(torch.tensor(val_loss))
        print("第{}次迭代，训练集平均损失{}，预测准确率{}，验证集平均损失{}，预测准确率{}".format(epoch, mean_train_loss, train_accuracy[-1], mean_val_loss, val_accuracy[-1]))
    # 画损失值曲线和正确率曲线
    plt.figure()
    plt.plot(train_loss)
    plt.savefig("./output/NN_train_loss.png")
    plt.close()
    plt.figure()
    plt.plot(val_loss)
    plt.savefig("./output/NN_val_loss.png")
    plt.figure()
    plt.plot(train_accuracy)
    plt.savefig("./output/NN_train_accuracy.png")
    plt.close()
    plt.figure()
    plt.plot(val_accuracy)
    plt.savefig("./output/NN_val_accuracy.png")
        
    # 用测试数据测试
    test_accuracy = 0
    for x, y in mnist_test:
        x = x.view(batch_size, -1)
        y_pred = net.forward(x)
        y_pred = torch.max(y_pred.data, 1).indices
        test_accuracy += (y_pred == y).sum().item()
        # print(test_accuracy, len(mnist_test)*batch_size)
    accuracy = test_accuracy/(len(mnist_test)*batch_size)
    print("一般神经网络算法预测准确率:{}".format(accuracy))
    return accuracy
    
    
# 算法8:卷积神经网络
# 需要将数据转换成二维图片形式
class conv_net(nn.Module):
    def __init__(self):
        super(conv_net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size = 3),
            nn.BatchNorm2d(25),
            nn.ReLU(inplace = True)
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size = 3),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace = True)
        )
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(50*5*5, 1024),
            nn.ReLU(inplace = True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace = True),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    
@run.change_dir
@run.timethis
def CONV_Model(mnist_train, mnist_val, mnist_test, batch_size = 64, lr = 0.001):
    epochs = 20
    net = conv_net()
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters(), lr = lr, weight_decay=0.0)
    # print(batch_size)
    
    net.train()
    for epoch in tqdm.tqdm(range(epochs)):
        # 训练过程
        train_loss = []
        accuracy = 0.0
        train_accuracy = []
        for x, y in mnist_train:
            y_pred = net.forward(x)
            loss = criterion(y_pred, y)
            train_loss.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
            # 计算预测准确率
            with torch.no_grad():
                y_pred = torch.max(y_pred.data, 1).indices
                accuracy += (y_pred == y).sum().item()
        train_accuracy.append(accuracy/(len(mnist_train)*batch_size))
        mean_train_loss = torch.mean(torch.tensor(train_loss))
        # 验证过程
        with torch.no_grad():
            val_loss = []
            accuracy = 0.0
            val_accuracy = []
            for x, y in mnist_val:
                # x = x.view(batch_size, -1)
                y_pred = net.forward(x)
                # y_pred = torch.max(y_pred.data, 1).indices
                loss = criterion(y_pred, y)
                val_loss.append(loss.item())
                # 计算预测准确率
                y_pred = torch.max(y_pred.data, 1).indices
                accuracy += (y_pred == y).sum().item()
            val_accuracy.append(accuracy/(len(mnist_val)*batch_size))
            mean_val_loss = torch.mean(torch.tensor(val_loss))
        print("第{}次迭代，训练集平均损失{}，预测准确率{}，验证集平均损失{}，预测准确率{}".format(epoch, mean_train_loss, train_accuracy[-1], mean_val_loss, val_accuracy[-1]))
    # 画损失值曲线和正确率曲线
    plt.figure()
    plt.plot(train_loss)
    plt.savefig("./output/CONV_train_loss.png")
    plt.close()
    plt.figure()
    plt.plot(val_loss)
    plt.savefig("./output/CONV_val_loss.png")
    plt.figure()
    plt.plot(train_accuracy)
    plt.savefig("./output/CONV_train_accuracy.png")
    plt.close()
    plt.figure()
    plt.plot(val_accuracy)
    plt.savefig("./output/CONV_val_accuracy.png")
        
    # 用测试数据测试
    net.eval()
    test_accuracy = 0
    for x, y in mnist_test:
        # x = x.view(batch_size, -1)
        y_pred = net.forward(x)
        y_pred = torch.max(y_pred.data, 1).indices
        test_accuracy += (y_pred == y).sum().item()
        # print(test_accuracy, len(mnist_test)*batch_size)
    accuracy = test_accuracy/(len(mnist_test)*batch_size)
    print("卷积神经网络算法预测准确率:{}".format(accuracy))
    return accuracy
    
    
# 实际运用算法来识别自己的手写数据

# 将图片文件转换为MNIST数据
@run.change_dir
def changeData(path = "./mynum/num/"):
    # path = "./mynum/num/"
    dir_or_files = os.listdir(path)
    files = []
    for dir_file in os.listdir(path):
        files.append(dir_file)
        
    MNIST_SIZE = 28
    datas = []
    labels = []
    for file in files:
        # 处理图片
        # print(path+files[0])
        # 读入图片并变成灰色
        img = io.imread(path+file, as_gray=True)
        # 缩小到28*28
        translated_img = transform.resize(img, (MNIST_SIZE, MNIST_SIZE))
        # 变成1*784的一维数组
        flatten_img = np.reshape(translated_img, 784)
        # 1代表黑，0代表白
        result = np.array([1 - flatten_img])
        # 提取标签
        labels.append(int(file[-5]))
        datas.append(result)
        # print(file)
    datas = np.array(datas)
    labels = np.array(labels)
    return datas, labels
    
    
# 将数据画图看看
@run.change_dir
def drawData(data):
    plt.figure()
    plt.imshow(data.reshape(28, 28))
    plt.savefig("./output/num.png")
    plt.close()
    
    
# 训练要用的模型并保存，一般只运行一次
def doTraining():
    pass
    
    
# 实际运用模型来识别手写数据
def work():
    pass
    

if __name__ == "__main__":
    Research()
#    testdatas, testlabels = changeData()
#    print(testdatas.shape)
#    # print(testdatas)
#    drawData(testdatas[20])
#    print(testlabels[20])
    
    