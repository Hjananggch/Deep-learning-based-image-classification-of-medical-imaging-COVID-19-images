from torchvision import transforms
import os
from torchvision import datasets
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models
import torch.optim as optim
import torch.nn as nn
import torch
from torch.optim import lr_scheduler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

# 训练集图像预处理：缩放裁剪、图像增强、转 Tensor、归一化
train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ])

# 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                     ])


def train_one_batch(images, labels):
    # 获得一个 batch 的数据和标注
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)  # 输入模型，执行前向预测
    loss = criterion(outputs, labels)  # 计算当前 batch 中，每个样本的平均交叉熵损失函数值

    # 优化更新权重
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 获取当前 batch 的标签类别和预测类别
    _, preds = torch.max(outputs, 1)  # 获得当前 batch 所有图像的预测类别
    preds = preds.cpu().numpy()
    loss = loss.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    # 计算分类评估指标
    train_loss = loss
    train_accuracy = accuracy_score(labels, preds)
    train_precision = precision_score(labels, preds, average='macro')
    print(f"train_loss:{train_loss}", f"train_acc:{train_accuracy}", f"train_pre:{train_precision}")



def evaluate_testset():
    loss_list = []
    labels_list = []
    preds_list = []

    with torch.no_grad():
        for images, labels in test_loader:  # 生成一个 batch 的数据和标注
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)  # 输入模型，执行前向预测

            # 获取整个测试集的标签类别和预测类别
            _, preds = torch.max(outputs, 1)  # 获得当前 batch 所有图像的预测类别
            preds = preds.cpu().numpy()
            loss = criterion(outputs, labels)  # 由 logit，计算当前 batch 中，每个样本的平均交叉熵损失函数值
            loss = loss.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            loss_list.append(loss)
            labels_list.extend(labels)
            preds_list.extend(preds)

    # 计算分类评估指标
    test_loss = np.mean(loss_list)
    test_accuracy = accuracy_score(labels_list, preds_list)
    test_precision = precision_score(labels_list, preds_list, average='macro')
    test_recall = recall_score(labels_list, preds_list, average='macro')
    print(f"test_loss:{test_loss}", f"test_acc:{test_accuracy}",
          f"test_pre:{test_precision}",f"test_recall:{test_recall}")



if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 数据集文件夹路径
    dataset_dir = r""
    train_path = os.path.join(dataset_dir, 'train')
    test_path = os.path.join(dataset_dir, 'test')

    # 载入训练集
    train_dataset = datasets.ImageFolder(train_path, train_transform)
    # 载入测试集
    test_dataset = datasets.ImageFolder(test_path, test_transform)

    # 各类别名称
    class_names = train_dataset.classes
    n_class = len(class_names)

    # 映射关系：类别 到 索引号
    train_dataset.class_to_idx
    # 映射关系：索引号 到 类别
    idx_to_labels = {y: x for x, y in train_dataset.class_to_idx.items()}

    # 保存为本地的 npy 文件
    np.save('idx_to_labels.npy', idx_to_labels)
    np.save('labels_to_idx.npy', train_dataset.class_to_idx)

    BATCH_SIZE = 32
    # 训练集的数据加载器
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=4
                              )

    # 测试集的数据加载器
    test_loader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=4
                             )

    model = models.resnet18(pretrained=True)  # 载入预训练模型

    # 修改全连接层，使得全连接层的输出与当前数据集类别数对应
    # 新建的层默认 requires_grad=True
    model.fc = nn.Linear(model.fc.in_features, n_class)
    # 只微调训练最后一层全连接层的参数，其它层冻结
    optimizer = optim.Adam(model.fc.parameters())

    model = model.to(device)

    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    # 学习率降低策略
    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 训练轮次 Epoch
    EPOCHS = 100

    best_test_accuracy = 0

    for epoch in range(1, EPOCHS + 1):

        print(f'Epoch {epoch}/{EPOCHS}')

        ## 训练阶段
        model.train()
        for images, labels in train_loader:
            train_one_batch(images, labels)

        lr_scheduler.step()
        ## 测试阶段
        model.eval()
        evaluate_testset()