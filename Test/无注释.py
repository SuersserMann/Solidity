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
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.filename = filename
        self.log = open(self.filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


now = datetime.datetime.now()
filename = "log_{:%Y-%m-%d_%H-%M-%S}.txt".format(now)
path = os.path.abspath(os.path.dirname(__file__))
log_dir_path = os.path.join(path, "log")
log_file_path = os.path.join(log_dir_path, filename)

if not os.path.exists(log_dir_path):
    os.makedirs(log_dir_path)

sys.stdout = Logger(log_file_path)


def truncate_list(lst, length):
    new_lst = []
    for item in lst:
        if len(item) <= length:
            new_lst.append(item)
        else:
            for i in range(0, len(item), length):
                new_lst.append(item[i:i + length])
    return new_lst


def bytecode_to_opcodes(bytecode):
    bytecode = bytecode.replace("0x", "")
    disassembled = disassemble_hex(bytecode).replace("\n", " ")
    return disassembled


def calculate_f1(precision, recall):
    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1


print(torch.__version__)
device_ids = [0, 1]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device=', device)

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.pretrained = AutoModel.from_pretrained("microsoft/codebert-base")
        self.pretrained.to(device)
        for param in self.pretrained.parameters():
            param.requires_grad_(False)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(768, 100),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(100, 3))

    def forward(self, input_ids, attention_mask):


        out = self.pretrained(input_ids=input_ids,
                              attention_mask=attention_mask,
                              )

        out = self.fc(out.last_hidden_state[:, 0])

        return out

model = Model()
model = nn.DataParallel(model, device_ids=device_ids)
model.to(device)


import pandas as pd
import datasets
import random


data = pd.read_excel('123.xlsx')


df = pd.DataFrame(data, columns=['slither', 'source_code'])


all_dataset = datasets.Dataset.from_pandas(df)
all_dataset = [[all_dataset['slither'][i], all_dataset['source_code'][i]] for i in range(len(all_dataset))]
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
random.shuffle(all_dataset)


train_size = int(len(all_dataset) * train_ratio)
val_size = int(len(all_dataset) * val_ratio)
test_size = len(all_dataset) - train_size - val_size


train_dataset = all_dataset[:train_size]
val_dataset = all_dataset[train_size:train_size + val_size]
test_dataset = all_dataset[-test_size:]

len(train_dataset), train_dataset[0]


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
        return_tensors='pt',  # 返回pytorch模型
        return_length=True)

    # 对于 `data` 字典中的每个键值对：
    # `input_ids`: 编码后的数字表示。
    # `attention_mask`: 表示哪些位置是有效的，哪些位置是补零的（0/1）。
    # `token_type_ids`: BERT 能够区分两个句子的方法（第一句话000..第二句话111...）。
    # `length`: 编码后序列的长度

    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    # token_type_ids = data['token_type_ids'].to(device)

    # 将 labels 转换为多标签格式
    labels_tensor = torch.zeros(len(labels), 3).to(device)
    for i, label in enumerate(labels):
        labels_tensor[i][label] = 1

    return input_ids, attention_mask, labels_tensor



train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=32,
                                           collate_fn=collate_fn,
                                           shuffle=True,
                                           drop_last=False)

test_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                          batch_size=32,
                                          collate_fn=collate_fn,
                                          shuffle=False,
                                          drop_last=False)

val_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                         batch_size=32,
                                         collate_fn=collate_fn,
                                         shuffle=False,
                                         drop_last=False)

def train_model(learning_rate, num_epochs):
    writer = SummaryWriter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # 使用传入的学习率
    criterion = torch.nn.BCEWithLogitsLoss()
    best_val_f1 = 0
    best_model_state = None

    try:
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            train_f1 = 0
            train_acc = 0
            train_recall = 0
            for i, (input_ids, attention_mask, labels) in enumerate(train_loader):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(out, labels.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                out = torch.sigmoid(out)
                out = torch.where(out > 0.5, torch.ones_like(out), torch.zeros_like(out))
                predicted_labels = []
                true_labels = []
                f2 = 0
                f2_precision = 0
                f2_recall = 0
                for j in range(len(out)):
                    predicted_label = torch.where(out[j] == 1)[0].tolist()
                    predicted_labels.append(predicted_label)
                    true_label = torch.where(labels[j] == 1)[0].tolist()
                    true_labels.append(true_label)
                    predicted_set = set(predicted_label)
                    true_set = set(true_label)

                    TP = len(predicted_set.intersection(true_set))
                    FP = len(predicted_set - true_set)
                    FN = len(true_set - predicted_set)
                    precision = TP / (TP + FP) if TP + FP else 0
                    recall = TP / (TP + FN) if TP + FN else 0
                    f1 = calculate_f1(precision, recall)

                    f2 = f1 + f2
                    f2_precision = precision + f2_precision
                    f2_recall = recall + f2_recall
                f2 = f2 / len(out)
                f2_precision = f2_precision / len(out)
                f2_recall = f2_recall / len(out)

                train_loss += loss.item()
                train_f1 += f2
                train_acc += f2_precision
                train_recall += f2_recall

                print(f"predicted_labels：{predicted_labels}", '\n', f"true_labels：{true_labels}")
                print(f"第{epoch + 1}周期：第{i + 1}轮训练, loss：{loss.item()}, 第{i + 1}轮训练集F1准确率为:{f2},第{i + 1}轮训练集accuracy:{f2_precision},第{i + 1}轮训练集recall:{f2_recall}")
            train_loss /= len(train_loader)
            train_f1 /= len(train_loader)
            train_acc /= len(train_loader)
            train_recall /= len(train_loader)

            writer.add_scalar('Train Loss', train_loss, epoch)  # 记录训练损失
            writer.add_scalar('Train F1', train_f1, epoch)  # 记录训练F1得分
            writer.add_scalar('Train Accuracy', train_acc, epoch)  # 记录训练准确度
            writer.add_scalar('Train Recall', train_recall, epoch)  # 记录训练召回率

            model.eval()
            val_loss = 0
            val_f1 = 0
            val_acc = 0
            val_recall = 0
            with torch.no_grad():
                for i, (input_ids, attention_mask, labels) in enumerate(val_loader):
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(out, labels.float())
                    out = torch.sigmoid(out)
                    out = torch.where(out > 0.5, torch.ones_like(out), torch.zeros_like(out))
                    predicted_labels = []
                    true_labels = []

                    f2 = 0
                    f2_precision = 0
                    f2_recall = 0
                    for j in range(len(out)):
                        predicted_label = torch.where(out[j] == 1)[0].tolist()
                        predicted_labels.append(predicted_label)
                        true_label = torch.where(labels[j] == 1)[0].tolist()
                        true_labels.append(true_label)

                        predicted_set = set(predicted_label)
                        true_set = set(true_label)

                        TP = len(predicted_set.intersection(true_set))
                        FP = len(predicted_set - true_set)
                        FN = len(true_set - predicted_set)
                        precision = TP / (TP + FP) if TP + FP else 0
                        recall = TP / (TP + FN) if TP + FN else 0
                        f1 = calculate_f1(precision, recall)
                        f2 = f1 + f2
                        f2_precision = precision + f2_precision
                        f2_recall = recall + f2_recall
                    average_val_f1 = f2 / len(out)
                    f2_precision = f2_precision / len(out)
                    f2_recall = f2_recall / len(out)

                    val_loss += loss.item()
                    val_f1 += average_val_f1
                    val_acc += f2_precision
                    val_recall += f2_recall
                    print(f"predicted_labels：{predicted_labels}", '\n', f"true_labels：{true_labels}")
                    print(f"第{epoch + 1}周期：第{i + 1}轮验证, loss：{loss.item()}, 第{i + 1}轮验证集F1准确率为:{average_val_f1},第{i + 1}轮验证集accuracy:{f2_precision},第{i + 1}轮验证集recall:{f2_recall}")
                val_loss /= len(val_loader)
                val_f1 /= len(val_loader)
                val_acc /= len(val_loader)
                val_recall /= len(val_loader)

                writer.add_scalar('Val Loss', val_loss, epoch)  # 记录验证损失
                writer.add_scalar('Val F1', val_f1, epoch)  # 记录验证F1得分
                writer.add_scalar('Val Accuracy', val_acc, epoch)  # 记录验证准确度
                writer.add_scalar('Val Recall', val_recall, epoch)  # 记录验证召回率

            if average_val_f1 > best_val_f1:
                best_val_f1 = average_val_f1
                best_model_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_model_state)
        model.eval()

        with torch.no_grad():
            for i, (input_ids, attention_mask, labels) in enumerate(test_loader):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(out, labels.float())
                out = torch.sigmoid(out)
                out = torch.where(out > 0.5, torch.ones_like(out), torch.zeros_like(out))
                predicted_labels = []
                true_labels = []
                f2 = 0
                f2_precision = 0
                f2_recall = 0
                for j in range(len(out)):
                    predicted_label = torch.where(out[j] == 1)[0].tolist()
                    predicted_labels.append(predicted_label)
                    true_label = torch.where(labels[j] == 1)[0].tolist()
                    true_labels.append(true_label)
                    predicted_set = set(predicted_label)
                    true_set = set(true_label)
                    TP = len(predicted_set.intersection(true_set))
                    FP = len(predicted_set - true_set)
                    FN = len(true_set - predicted_set)
                    precision = TP / (TP + FP) if TP + FP else 0
                    recall = TP / (TP + FN) if TP + FN else 0
                    f1 = calculate_f1(precision, recall)
                    f2 = f1 + f2
                    f2_precision = precision + f2_precision
                    f2_recall = recall + f2_recall
                average_test_f1 = f2 / len(out)
                f2_precision = f2_precision / len(out)
                f2_recall = f2_recall / len(out)
                print(f"predicted_labels：{predicted_labels}", '\n', f"true_labels：{true_labels}")
                print(f"第{i + 1}轮测试, loss：{loss.item()}, 第{i + 1}轮测试集F1准确率为:{average_test_f1},第{i + 1}轮测试集accuracy:{f2_precision},第{i + 1}轮测试集recall:{f2_recall}")
        print(f"测试集 F1 分数：{average_test_f1}")
        return average_test_f1, best_model_state
    except KeyboardInterrupt:
        print('手动终止训练')
        model_save_path = "../Test/model_interrupted_1.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"当前模型已保存到：{model_save_path}")
    finally:
        writer.close()

learning_rate = 3e-5
num_epochs = 1000

test_f1, model = train_model(learning_rate, num_epochs)

model_save_path = "../Test/best_model_9.pth"
torch.save(model, model_save_path)
print(f"使用指定的超参数训练的模型已保存到：{model_save_path}")
print(f"测试集 F1 分数：{test_f1}")
