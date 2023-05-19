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
import subprocess
import webbrowser
import pandas as pd
import datasets
import random

import warnings

# 忽略所有的警告
warnings.filterwarnings("ignore")
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
# 使用cuda
device_ids = [0, 1]
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 如果是cuda则调用cuda反之则cpu
print('device=', device)


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
            torch.nn.Linear(100, 4))  # 添加多层神经网络

    def forward(self, input_ids, attention_mask):
        # 将输入传入预训练模型，并记录计算图以计算梯度

        out = self.pretrained(input_ids=input_ids,
                              attention_mask=attention_mask,
                              )
        # 只保留预训练模型输出的最后一个隐藏状态，并通过全连接层进行分类
        out = self.fc(out.last_hidden_state[:, 0])

        return out


# 实例化下游任务模型并将其移动到 GPU 上 (如果可用)
model = Model()
model = nn.DataParallel(model, device_ids=device_ids)
model.to(device)

# 从 Excel 文件中读取数据
data = pd.read_csv('456.csv', encoding='GBK')

# 将数据存储到 DataFrame 中，并选择需要的列
df = pd.DataFrame(data, columns=['slither', 'source_code'])

# 将 DataFrame 转换为 Dataset 对象
all_dataset = datasets.Dataset.from_pandas(df)

all_dataset = list(zip(all_dataset['slither'], all_dataset['source_code']))
train_ratio = 0.8  # 训练集比例
val_ratio = 0.1  # 验证集比例
test_ratio = 0.1  # 测试集比例
random.shuffle(all_dataset)

# 计算训练集、验证集和测试集的数量
train_size = int(len(all_dataset) * train_ratio)
val_size = int(len(all_dataset) * val_ratio)
test_size = len(all_dataset) - train_size - val_size

# 划分数据集
train_dataset = all_dataset[:train_size]
val_dataset = all_dataset[train_size:train_size + val_size]
test_dataset = all_dataset[-test_size:]

len(train_dataset), train_dataset[0]

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
    labels_tensor = torch.zeros(len(labels), 4).to(device)
    for i, label in enumerate(labels):
        labels_tensor[i][label] = 1

    return input_ids, attention_mask, labels_tensor


# 数据加载器，数据少的话，具体原因不清楚，batchsize不能太大，明白了，数据太少了，刚才的数据被drop_last丢掉了
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=256,  # 是每个批轮的大小，也就是每轮处理的样本数量。
                                           collate_fn=collate_fn,  # 是一个函数，用于对每个批轮中的样本进行编码和处理。
                                           shuffle=True,  # 是一个布尔值，表示是否对数据进行随机重排。
                                           drop_last=False)  # 是一个布尔值，表示是否在最后一个批轮中舍弃不足一个批轮大小的数据

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=256,  # 是每个批轮的大小，也就是每轮处理的样本数量。
                                         collate_fn=collate_fn,  # 是一个函数，用于对每个批轮中的样本进行编码和处理。
                                         shuffle=False,  # 是一个布尔值，表示是否对数据进行随机重排。
                                         drop_last=False)  # 是一个布尔值，表示是否在最后一个批轮中舍弃不足一个批轮大小的数据

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=512,  # 是每个批轮的大小，也就是每轮处理的样本数量。
                                          collate_fn=collate_fn,  # 是一个函数，用于对每个批轮中的样本进行编码和处理。
                                          shuffle=False,  # 是一个布尔值，表示是否对数据进行随机重排。
                                          drop_last=False)  # 是一个布尔值，表示是否在最后一个批轮中舍弃不足一个批轮大小的数据


def reset_model_parameters(model):
    if hasattr(model, 'module'):
        model = model.module
    for parameter in model.parameters():
        parameter.data.normal_(mean=0.0, std=0.02)


from sklearn.metrics import confusion_matrix


def calculate_confusion_matrix(predicted_values, true_values):
    predicted_labels = np.concatenate(predicted_values)  # 将多个样本的预测类别列表合并为一个一维数组
    true_labels = np.concatenate(true_values)  # 将多个样本的真实类别列表合并为一个一维数组

    cm = confusion_matrix(true_labels, predicted_labels)
    num_classes = len(cm)

    fn = [sum(cm[i]) - cm[i, i] for i in range(num_classes)]
    fp = [sum(cm[:, i]) - cm[i, i] for i in range(num_classes)]
    tn = [sum(sum(cm)) - sum(cm[i]) - sum(cm[:, i]) + cm[i, i] for i in range(num_classes)]
    tp = [cm[i, i] for i in range(num_classes)]

    return fn, fp, tn, tp

def calculate_accuracy(fn, fp, tn, tp):
    total_samples = sum(fn) + sum(fp) + sum(tn) + sum(tp)
    correct_predictions = sum(tp) + sum(tn)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    return accuracy


def calculate_precision(fn, fp, tn, tp):
    num_classes = len(fn)
    precision_scores = []

    for i in range(num_classes):
        precision = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0
        precision_scores.append(precision)

    return precision_scores


def calculate_recall(fn, fp, tn, tp):
    num_classes = len(fn)
    recall_scores = []

    for i in range(num_classes):
        recall = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0
        recall_scores.append(recall)

    return recall_scores


def calculate_f1_score(fn, fp, tp):
    num_classes = len(fn)
    f1_scores = []

    for i in range(num_classes):
        precision = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0
        recall = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1_score)

    return f1_scores


def train_model(learning_rate, num_epochs):
    # reset_model_parameters(model.fc)
    # 训练

    writer = SummaryWriter()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # 使用传入的学习率
    criterion = torch.nn.BCEWithLogitsLoss()

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    # patience = 20  # 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
    # early_stopping = EarlyStopping(patience, verbose=True)

    best_val_f1 = 0  # 初始化最佳验证集 F1 分数
    best_model_state = None  # 保存最佳模型参数

    # # 启动 TensorBoard
    # tb_process = subprocess.Popen(['tensorboard', '--logdir', 'runs/'])
    #
    # # 打开 TensorBoard 网页
    # webbrowser.open_new_tab('http://localhost:6006/')

    patience = 10
    counter = 0
    labels_num = 4
    try:
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            train_count = 0
            train_fn = [0] * 4
            train_fp = [0] * 4
            train_tn = [0] * 4
            train_tp = [0] * 4

            for i, (input_ids, attention_mask, labels) in enumerate(train_loader):
                # 遍历数据集，并将数据转移到GPU上
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                # 进行前向传播，得到预测值out
                loss = criterion(out, labels.float())  # 计算损失
                loss.backward()  # 反向传播，计算梯度
                optimizer.step()  # 更新参数
                optimizer.zero_grad()  # 梯度清零，防止梯度累积

                out = torch.sigmoid(out)  # 将预测值转化为概率

                max_value = out.max(dim=1, keepdim=True).values

                # 将最大值位置设置为0.6，其他位置设置为0.1
                out = torch.where(out == max_value, torch.tensor(0.6).to(out.device), torch.tensor(0.1).to(out.device))

                out = torch.where(out > 0.5, torch.ones_like(out), torch.zeros_like(out))  # 找到概率大于0.5的位置，并将其设为1，否则设为0
                predicted_labels = []
                true_labels = []

                for j in range(len(out)):
                    predicted_label = torch.where(out[j] == 1)[0].tolist()  # 将位置索引转换为标签
                    predicted_labels.append(predicted_label)
                    true_label = torch.where(labels[j] == 1)[0].tolist()
                    true_labels.append(true_label)

                fn, fp, tn, tp = calculate_confusion_matrix(predicted_labels, true_labels)
                accuracy = calculate_accuracy(fn, fp, tn, tp)
                precision = calculate_precision(fn, fp, tn, tp)
                recall = calculate_recall(fn, fp, tn, tp)
                f1 = calculate_f1_score(fn, fp, tp)

                train_fn = [x + y for x, y in zip(train_fn, fn)]
                train_fp = [x + y for x, y in zip(train_fp, fp)]
                train_tn = [x + y for x, y in zip(train_tn, tn)]
                train_tp = [x + y for x, y in zip(train_tp, tp)]

                train_loss += loss.item()

                train_count += 1

                #print(f"predicted_labels：{predicted_labels}", '\n', f"true_labels：{true_labels}")
                print(
                    f"第{epoch + 1}周期：第{i + 1}轮训练, loss：{loss.item()}, 第{i + 1}轮训练集F1准确率为:{f1},第{i + 1}轮训练集accuracy:{accuracy},第{i + 1}轮训练集precision:{precision},第{i + 1}轮训练集recall:{recall}")

            train_loss /= train_count
            train_f1 = calculate_f1_score(train_fn, train_fp, train_tp)
            train_acc = calculate_accuracy(train_fn, train_fp, train_tn, train_tp)
            train_precision = calculate_precision(train_fn, train_fp, train_tn, train_tp)
            train_recall = calculate_recall(train_fn, train_fp, train_tn, train_tp)
            print(
                f"------------总训练集单个标签loss为{train_loss},总训练集F1为{train_f1},总accuracy为{train_acc}，总precision为{train_precision},总recall为{train_recall}------------")
            print(
                f"------------总训练集loss为{train_loss},总F1为{sum(train_f1) / labels_num},总accuracy为{train_acc}，总precision为{sum(train_precision) / labels_num},总recall为{sum(train_recall) / labels_num}------------")
            print(f"------------总fn为{train_fn},总fp为{train_fp}，总tn为{train_tn}，总tp为{train_tp}")

            writer.add_scalar('Train Loss', train_loss, epoch)  # 记录训练损失
            writer.add_scalar('Train F1', sum(train_f1) / labels_num, epoch)  # 记录训练F1得分
            writer.add_scalar('Train Accuracy', train_acc, epoch)  # 记录训练准确度
            writer.add_scalar('Train Precision', sum(train_precision) / labels_num, epoch)  # 记录训练精准度
            writer.add_scalar('Train Recall', sum(train_recall) / labels_num, epoch)  # 记录训练召回率

            # 验证
            model.eval()
            val_loss = 0
            val_count = 0
            train_fn = [0] * 4
            train_fp = [0] * 4
            train_tn = [0] * 4
            train_tp = [0] * 4
            with torch.no_grad():
                for i, (input_ids, attention_mask, labels) in enumerate(val_loader):
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(out, labels.float())  # 计算损失
                    out = torch.sigmoid(out)  # 将预测值转化为概率

                    max_value = out.max(dim=1, keepdim=True).values

                    # 将最大值位置设置为0.6，其他位置设置为0.1
                    out = torch.where(out == max_value, torch.tensor(0.6).to(out.device),
                                      torch.tensor(0.1).to(out.device))

                    out = torch.where(out > 0.5, torch.ones_like(out),
                                      torch.zeros_like(out))  # 找到概率大于0.5的位置，并将其设为1，否则设为0
                    predicted_labels = []
                    true_labels = []

                    for j in range(len(out)):
                        predicted_label = torch.where(out[j] == 1)[0].tolist()  # 将位置索引转换为标签
                        predicted_labels.append(predicted_label)
                        true_label = torch.where(labels[j] == 1)[0].tolist()
                        true_labels.append(true_label)

                    fn, fp, tn, tp = calculate_confusion_matrix(predicted_labels, true_labels)
                    accuracy = calculate_accuracy(fn, fp, tn, tp)
                    precision = calculate_precision(fn, fp, tn, tp)
                    recall = calculate_recall(fn, fp, tn, tp)
                    f1 = calculate_f1_score(fn, fp, tp)

                    train_fn = [x + y for x, y in zip(train_fn, fn)]
                    train_fp = [x + y for x, y in zip(train_fp, fp)]
                    train_tn = [x + y for x, y in zip(train_tn, tn)]
                    train_tp = [x + y for x, y in zip(train_tp, tp)]

                    val_loss += loss.item()
                    val_count += 1

                    #print(f"predicted_labels：{predicted_labels}", '\n', f"true_labels：{true_labels}")
                    print(
                        f"第{epoch + 1}周期：第{i + 1}轮验证, loss：{loss.item()}, 第{i + 1}轮验证集F1准确率为:{f1},第{i + 1}轮验证集accuracy:{accuracy},第{i + 1}轮验证集recall:{recall}")

                val_loss /= val_count
                val_f1 = calculate_f1_score(train_fn, train_fp, train_tp)
                val_acc = calculate_accuracy(train_fn, train_fp, train_tn, train_tp)
                val_precision = calculate_precision(train_fn, train_fp, train_tn, train_tp)
                val_recall = calculate_recall(train_fn, train_fp, train_tn, train_tp)
                val_compare_f1=sum(val_f1) / labels_num
                print(
                    f"------------总验证集loss为{val_loss},总F1为{val_f1},总accuracy为{val_acc}，总precision为{val_precision}，总recall为{val_recall}------------")
                print(
                    f"------------总测试集loss为{val_loss},总F1为{sum(val_f1) / labels_num},总accuracy为{val_acc}，总precision为{sum(val_precision) / labels_num},总recall为{sum(val_recall) / labels_num}------------")
                print(f"------------总fn为{train_fn},总fp为{train_fp}，总tn为{train_tn}，总tp为{train_tp}")

                writer.add_scalar('Train Loss', val_loss, epoch)  # 记录训练损失
                writer.add_scalar('Train F1', sum(val_f1) / labels_num, epoch)  # 记录训练F1得分
                writer.add_scalar('Train Accuracy', val_acc, epoch)  # 记录训练准确度
                writer.add_scalar('Train Precision', sum(val_precision) / labels_num, epoch)  # 记录训练精准度
                writer.add_scalar('Train Recall', sum(val_recall) / labels_num, epoch)

            if val_compare_f1 > best_val_f1:
                best_val_f1 = val_compare_f1
                best_model_state = copy.deepcopy(model.state_dict())
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print("Early stopping!")
                # 保存当前模型
                torch.save(model.state_dict(), "model_early_1.pt")
                break
            lr_scheduler.step()
            print(f"学习率为{lr_scheduler.get_last_lr()}")
        # 加载具有最佳验证集性能的模型参数
        model.load_state_dict(best_model_state)
        # 测试
        model.eval()
        test_loss = 0
        test_count = 0

        train_fn = [0] * 4
        train_fp = [0] * 4
        train_tn = [0] * 4
        train_tp = [0] * 4

        with torch.no_grad():
            for i, (input_ids, attention_mask, labels) in enumerate(test_loader):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(out, labels.float())  # 计算损失
                out = torch.sigmoid(out)  # 将预测值转化为概率

                max_value = out.max(dim=1, keepdim=True).values

                # 将最大值位置设置为0.6，其他位置设置为0.1
                out = torch.where(out == max_value, torch.tensor(0.6).to(out.device), torch.tensor(0.1).to(out.device))

                out = torch.where(out > 0.5, torch.ones_like(out), torch.zeros_like(out))  # 找到概率大于0.5的位置，并将其设为1，否则设为0
                predicted_labels = []
                true_labels = []

                for j in range(len(out)):
                    predicted_label = torch.where(out[j] == 1)[0].tolist()  # 将位置索引转换为标签
                    predicted_labels.append(predicted_label)
                    true_label = torch.where(labels[j] == 1)[0].tolist()
                    true_labels.append(true_label)

                fn, fp, tn, tp = calculate_confusion_matrix(predicted_labels, true_labels)
                accuracy = calculate_accuracy(fn, fp, tn, tp)
                precision = calculate_precision(fn, fp, tn, tp)
                recall = calculate_recall(fn, fp, tn, tp)
                f1 = calculate_f1_score(fn, fp, tp)

                train_fn = [x + y for x, y in zip(train_fn, fn)]
                train_fp = [x + y for x, y in zip(train_fp, fp)]
                train_tn = [x + y for x, y in zip(train_tn, tn)]
                train_tp = [x + y for x, y in zip(train_tp, tp)]

                test_loss += loss.item()
                test_count += 1

                print(
                    f"：第{i + 1}轮验测试, loss：{loss.item()}, 第{i + 1}轮测试集F1准确率为:{f1},第{i + 1}轮测试集accuracy:{accuracy},第{i + 1}轮测试集precision:{precision},第{i + 1}轮验证集recall:{recall}")
            test_loss /= test_count
            test_f1 = calculate_f1_score(train_fn, train_fp, train_tp)
            test_acc = calculate_accuracy(train_fn, train_fp, train_tn, train_tp)
            test_precision = calculate_precision(train_fn, train_fp, train_tn, train_tp)
            test_recall = calculate_recall(train_fn, train_fp, train_tn, train_tp)

        print(
            f"------------总测试集单个标签loss为{test_loss},总F1为{test_f1},总accuracy为{test_acc}，总precision为{test_precision},总recall为{test_recall}------------")
        print(
            f"------------总测试集loss为{test_loss},总F1为{sum(test_f1) / labels_num},总accuracy为{test_acc}，总precision为{sum(test_precision) / labels_num},总recall为{sum(test_recall) / labels_num}------------")
        print(f"------------总fn为{train_fn},总fp为{train_fp}，总tn为{train_tn}，总tp为{train_tp}")
        return test_f1, best_model_state

    except KeyboardInterrupt:
        # 捕捉用户手动终止训练的异常
        print('手动终止训练')
        model_save_path = "model_interrupted_2.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"当前模型已保存到：{model_save_path}")


# 定义一个超参数空间，用于搜佳超参数
learning_rate = 1e-3
num_epochs = 2

# 使用指定的超参数训练模型
test_f1, model = train_model(learning_rate, num_epochs)

# 保存训练好的模型
model_save_path = "best_model_05_19_01.pth"
torch.save(model, model_save_path)
print(f"使用指定的超参数训练的模型已保存到：{model_save_path}")
print(f"测试集 F1 分数：{test_f1}")
