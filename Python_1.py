# 编写python程序
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
# 生成随机数据
def generate_data(num_samples, num_features, noise_std):
    # 生成随机特征
    features = np.random.randn(num_samples, num_features)
    # 生成随机权重
    weights = np.random.randn(num_features) 
    # 生成随机噪声
    noise = np.random.randn(num_samples) * noise_std
    # 计算标签
    labels = np.dot(features, weights) + noise*0.01
    return features, labels, weights, noise_std
# 定义线性回归模型
def linear_regression(features, labels, learning_rate, num_epochs):
    # 初始化权重和偏置
    num_samples, num_features = features.shape
    weights = np.random.randn(num_features) * 0.01
    bias = 0    # 初始化偏置为0
    losses = []  # 新增：用于记录每个epoch的loss
    
    for epoch in range(num_epochs):
        # 计算预测值
        predictions = np.dot(features, weights) + bias
        # 计算损失
        loss = np.mean((predictions - labels) ** 2)
        # 计算梯度
        gradients = np.dot(features.T, (predictions - labels)) / num_samples
        # 更新权重和偏置
        weights -= learning_rate * gradients
        bias -= learning_rate * np.mean(predictions - labels)
        # 打印损失
        losses.append(loss)  # 记录当前loss
        if epoch % 100 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss}")
    
    return weights, bias, losses  # 修改：返回losses记录
# 定义预测函数
def predict(features, weights, bias):
    return np.dot(features, weights) + bias
# 定义主函数
def main():
    # 生成随机数据 num_samples 样本数 num_features 特征数 noise_std 噪声标准差
    num_samples = 10000
    num_features = 2
    noise_std = 0.01
    features, labels, weights, noise_std = generate_data(num_samples, num_features, noise_std)
    # 训练模型 learning_rate 学习率 num_epochs 迭代次数
    learning_rate = 0.01
    num_epochs = 300
    # 修改：接收返回的losses
    weights, bias, losses = linear_regression(features, labels, learning_rate, num_epochs)
    # 打印权重和偏置
    print("Weights:", weights)
    print("Bias:", bias)
    # 绘制真实数据和预测数据的散点图
    plt.scatter(features[:, 0], labels, label="True Data")
    plt.scatter(features[:, 0], predict(features, weights, bias), label="Predicted Data")
    plt.legend()    # 显示图例
    plt.show()      # 显示图像    可以保存图像 plt.savefig("linear_regression.png") 保存图像为png格式
    # 添加打印第一行数据的代码
    print("第一行特征数据:", features[0])
    print("第一个标签:", labels[0])
    
    # 新增：绘制loss下降曲线
    plt.figure()
    plt.plot(range(num_epochs), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()
    # 新增：打印最后一个epoch的loss
    print("最后一个epoch的loss:", losses[-1])


if __name__ == "__main__":
    main()    # 调用主函数    可以保存图像 plt.savefig("linear_regression.png") 保存图像为png格式   
    # print(f"True Weights: {features[:, 0]}")        
#x修改相关文件增加一段打程序
    print("hello world")