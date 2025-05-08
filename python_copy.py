# 编写python程序
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
# 生成随机数据

def generate_random_data(true_w, true_b, num_samples):
    # 生成随机特征
    features = torch.randn(num_samples, len(true_w))
    # 生成随机噪声
    noise = torch.randn(num_samples) * 0.01
    # 计算标签
    labels = torch.mv(features, true_w) + true_b + noise
    return features, labels
# 同步数据
def sync_data(num_samples, true_w, true_b,bacth_size=10):
    # 生成随机特征和标签
    features, labels = generate_random_data(true_w, true_b, num_samples)
    # 修改：调整labels的维度为[num_samples, 1]
    labels = labels.reshape(-1, 1)
    # 将特征和标签堆叠在一起
    indices = list(range(num_samples))
    random.shuffle(indices)  # 随机打乱索引
    for i in range(0, num_samples, bacth_size):
        batch_indices = torch.tensor(indices[i: min(i + bacth_size, num_samples)])
        yield features[batch_indices], labels[batch_indices]
# 定义线性回归模型
def linear_regression(features,labels,learning_rate, num_epochs):
    # 初始化权重和偏置
    num_samples, num_features = features.shape
    weights = torch.randn(num_features,requires_grad=True) * 0.01
    bias = torch.zeros(1, requires_grad=True)    # 初始化偏置为0
    losses = []  # 新增：用于记录每个epoch的loss
    for epoch in range(num_epochs):
        # 遍历数据
        for batch_features, batch_labels in sync_data(num_samples, weights, bias):
            # 计算预测值
            predictions = torch.mv(batch_features, weights) + bias
            # 计算损失
        # 计算预测值
        predictions = torch.mv(features, weights) + bias
        # 计算损失
        loss = torch.mean((predictions - labels) ** 2)
        # 计算梯度
        gradients = torch.mv(features.t(), (predictions - labels)) / num_samples
        # 更新权重和偏置
        weights -= learning_rate * gradients
        bias -= learning_rate * torch.mean(predictions - labels)
        # 打印损失
        losses.append(loss.item())  # 记录当前loss
        if epoch % 100 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss}") 
    return weights, bias, losses  # 修改：返回losses记录
# 定义预测函数
def predict(features, weights, bias):
    return torch.mv(features, weights) + bias
# 定义主函数
def main():
    # 生成随机数据 num_samples 样本数 num_features 特征数 noise_std 噪声标准差
    num_samples = 10000
    num_features = 2
    noise_std = 0.01
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    
    features, labels = sync_data(num_samples, true_w, true_b)
    # 训练模型 learning_rate 学习率 num_epochs 迭代次数
    learning_rate = 0.01
    num_epochs = 300
    # 修改：接收返回的losses
    weights, bias, losses = linear_regression(features, labels, learning_rate, num_epochs)
    # 打印权重和偏置
    print("Weights:", weights)
    print("Bias:", bias)
    # 绘制真实数据和预测数据的散点图
    plt.scatter(features[:, 1], labels, label="True Data")
    plt.scatter(features[:, 0], predict(features, weights, bias), label="Predicted Data")
    plt.legend()
    plt.show()
    # 绘制loss曲线
    plt.plot(range(num_epochs), losses)  # 绘制loss曲线
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.show()
if __name__ == "__main__":
    main(
        
    )
