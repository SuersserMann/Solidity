import re

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from pyevmasm import disassemble_hex
import sys
import os
import datetime
import copy
# from pytorchtools import EarlyStopping
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
# import subprocess
# import webbrowser
import pandas as pd
import datasets
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings

# 忽略所有的警告
warnings.filterwarnings("ignore")
print(torch.__version__)
# 使用cuda
device_ids = [0, 1]
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 如果是cuda则调用cuda反之则cpu
print('device=', device)

def truncate_list(lst, length):
    new_lst = []
    for item in lst:
        if len(item) <= length:
            new_lst.append(item)
        else:
            for i in range(0, len(item), length):
                new_lst.append(item[i:i + length])
    return new_lst


def calculate_f1(precision, recall):
    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1



# fine tune向后传播而不修改之前的参数

# 定义了下游任务模型，包括一个全连接层和forward方法。
class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.pretrained = AutoModel.from_pretrained("microsoft/codebert-base")
        # 将预训练模型移动到GPU设备上（如果需要）
        self.pretrained.to(device)
        # 冻结预训练模型的参数
        for param in self.pretrained.parameters():
            param.requires_grad_(False)
        # 定义一个全连接层，输入维度为768，输出维度为6
        # self.fc = torch.nn.Linear(768, 6)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(768, 100),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(100, 3))
    def forward(self, input_ids, attention_mask):
        # 将输入传入预训练模型，并记录计算图以计算梯度

        out = self.pretrained(input_ids=input_ids,
                              attention_mask=attention_mask,
                              )
        # 只保留预训练模型输出的最后一个隐藏状态，并通过全连接层进行分类
        out = self.fc(out.last_hidden_state[:, 0])

        return out

pretrained = AutoModel.from_pretrained("microsoft/codebert-base")
# 实例化下游任务模型并将其移动到 GPU 上 (如果可用)
model = Model()
model = nn.DataParallel(model, device_ids=device_ids)
model.to(device)

# 从 Excel 文件中读取数据
data = pd.read_excel('345.xlsx')

# 将数据存储到 DataFrame 中，并选择需要的列
df = pd.DataFrame(data, columns=['slither', 'source_code'])

# 将 DataFrame 转换为 Dataset 对象
all_dataset = datasets.Dataset.from_pandas(df)
all_dataset = [[all_dataset['slither'][i], all_dataset['source_code'][i]] for i in range(len(all_dataset))]
train_ratio = 0.0  # 训练集比例
val_ratio = 0.0  # 验证集比例
test_ratio = 1.0  # 测试集比例
random.shuffle(all_dataset)

# 计算训练集、验证集和测试集的数量
train_size = int(len(all_dataset) * train_ratio)
val_size = int(len(all_dataset) * val_ratio)
test_size = len(all_dataset) - train_size - val_size

# 划分数据集
train_dataset = all_dataset[:train_size]
val_dataset = all_dataset[train_size:train_size + val_size]
test_dataset = all_dataset[-test_size:]

len(test_dataset), test_dataset[0]

# 加载字典和分词工具
token = AutoTokenizer.from_pretrained("microsoft/codebert-base")

def delete_comment(java_code):
    # 用正则表达式匹配 Java 代码中的注释
    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//.*?$|\n)"
    regex = re.compile(pattern, re.MULTILINE | re.DOTALL)

    def replace(match):
        if match.group(2) is not None:
            # 返回匹配结果中的第二个分组，即注释
            return ""
        else:
            # 返回匹配结果中的第一个分组，即字符串
            return match.group(1)

    # 用空字符串替换掉 Java 代码中的注释
    return regex.sub(replace, java_code)


def collate_fn(data):
    source_codes = [delete_comment(i[1]) for i in data]
    # bytecodes = [bytecode_to_opcodes(i[1]) for i in data]
    labels = [i[0] for i in data]
    labels = [[label] for label in labels]

    # amount = 0
    # cutted_list = []
    # cut_labels = []
    # for i, cut_bytecode in enumerate(bytecodes):
    #     new_labels = []
    #
    #     new_labels.append(cut_bytecode)
    #     cutted = truncate_list(new_labels, 2048)
    #     for gg in cutted:
    #         cutted_list.append(gg)
    #
    #     for dd in range(len(cutted)):
    #         cut_labels.insert(i + amount, labels[i])
    #     amount += len(cutted)
    # labels = cut_labels
    # bytecodes = cutted_list
    amount = 0
    cutted_list = []
    cut_labels = []
    for i, cut_sourcecode in enumerate(source_codes):
        new_labels = []

        new_labels.append(cut_sourcecode)
        cutted = truncate_list(new_labels, 2048)
        for gg in cutted:
            cutted_list.append(gg)

        for dd in range(len(cutted)):
            cut_labels.insert(i + amount, labels[i])
        amount += len(cutted)
    labels = cut_labels
    source_codes = cutted_list
    # 编码
    data = token.batch_encode_plus(
        source_codes,
        # bytecodes,
        padding='max_length',
        truncation=True,
        max_length=510,
        return_tensors='pt',
        return_length=True)

    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)

    labels_tensor = torch.zeros(len(labels), 3).to(device)
    for i, label in enumerate(labels):
        labels_tensor[i][label] = 1

    return input_ids, attention_mask, labels_tensor


test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=32,  # 是每个批轮的大小，也就是每轮处理的样本数量。
                                          collate_fn=collate_fn,  # 是一个函数，用于对每个批轮中的样本进行编码和处理。
                                          shuffle=False,  # 是一个布尔值，表示是否对数据进行随机重排。
                                          drop_last=False)  # 是一个布尔值，表示是否在最后一个批轮中舍弃不足一个批轮大小的数据

criterion = torch.nn.BCEWithLogitsLoss()

model.load_state_dict(torch.load('best_model_8.pth', map_location=torch.device('cpu')))
model.eval()
test_loss = 0
test_f1 = 0
test_acc = 0
test_recall = 0
test_count = 0
with torch.no_grad():
    for i, (input_ids, attention_mask, labels) in enumerate(test_loader):
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(out, labels.float())  # 计算损失
        out = torch.sigmoid(out)  # 将预测值转化为概率
        out = torch.where(out > 0.5, torch.ones_like(out), torch.zeros_like(out))  # 找到概率大于0.5的位置，并将其设为1，否则设为0
        predicted_labels = []
        true_labels = []

        f2 = 0
        f2_precision = 0
        f2_recall = 0
        f2_accuracy = 0
        for j in range(len(out)):
            predicted_label = torch.where(out[j] == 1)[0].tolist()  # 将位置索引转换为标签
            predicted_labels.append(predicted_label)
            true_label = torch.where(labels[j] == 1)[0].tolist()
            true_labels.append(true_label)

            # 计算F1分数
            predicted_set = set(predicted_label)
            true_set = set(true_label)
            all_set = true_set.union(predicted_set)

            TP = len(predicted_set.intersection(true_set))
            FP = len(predicted_set - true_set)
            FN = len(true_set - predicted_set)
            TN = len(all_set - predicted_set - true_set)

            precision = TP / (TP + FP) if TP + FP else 0
            recall = TP / (TP + FN) if TP + FN else 0
            accuracy = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN else 0

            f1 = calculate_f1(precision, recall)
            f2 = f1 + f2
            # f2_precision = precision + f2_precision
            f2_recall = recall + f2_recall
            f2_accuracy = accuracy + f2_accuracy

        average_test_f1 = f2 / len(out)
        # f2_precision = f2_precision / len(out)
        f2_recall = f2_recall / len(out)
        f2_accuracy = f2_accuracy / len(out)

        test_f1 += average_test_f1
        test_acc += f2_accuracy
        test_recall += f2_recall
        test_count += 1

        print(f"predicted_labels：{predicted_labels}", '\n', f"true_labels：{true_labels}")
        print(f"第{i + 1}轮测试, loss：{loss.item()}, 第{i + 1}轮测试集F1准确率为:{average_test_f1},第{i + 1}轮测试集accuracy:{f2_accuracy},第{i + 1}轮测试集recall:{f2_recall}")

    test_loss /= test_count
    test_f1 /= test_count
    test_acc /= test_count
    test_recall /= test_count

print(f"------------总测试集F1为{test_f1},总accuracy为{test_acc}，总recall为{test_recall}------------")
