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
# import pandas as pd
# import datasets
# import random
import json
import torch.utils.data as data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings


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
            torch.nn.Linear(768, 720),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(720, 673)
        )  # 添加多层神经网络

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

with open('frame_info.json', 'r', encoding='utf-8') as f:
    frame_info = json.load(f)

frame2idx = {}
for z, frame_idx in enumerate(frame_info):
    frame2idx[frame_idx['frame_name']] = z


class Dataset(data.Dataset):
    def __init__(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        sentence_id = item['sentence_id']
        cfn_spans = item['cfn_spans']
        frame = item['frame']
        target = item['target']
        text = item['text']
        word = item['word']
        frame_index = frame2idx[frame]

        return sentence_id, cfn_spans, frame, target, text, word, frame_index


train_dataset = Dataset('cfn-train.json')
val_dataset = Dataset('cfn-dev.json')
test_dataset = Dataset('cfn-test-A.json')
frame_dataset = Dataset('frame_info.json')
# 加载字典和分词工具
token = AutoTokenizer.from_pretrained("bert-base-chinese")


def find_indices(cfn_spans_start, word_start):
    matches = []
    for i, num in enumerate(word_start):
        if num in cfn_spans_start:
            matches.append(i)
    return matches


fe_name_count = []
for entry in frame_dataset:
    fe_name = [elem['fe_name'] for elem in entry[1]]
    for z in fe_name:
        if z not in fe_name_count:
            fe_name_count.append(z)


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


def get_indices(input_list, fe_name_count):
    index_list = []
    for item in input_list:
        if item in fe_name_count:
            index = fe_name_count.index(item)
            index_list.append(index)

        else:
            print("get_indices数据出错")
    return index_list


def collate_fn(data):
    sentence_ids = []
    cfn_spanss = []
    frames = []
    targets = []
    texts = []
    words = []
    frame_indexs = []
    result = []
    labels = []
    for i in data:

        sentence_ids.append(i[0])
        cfn_spanss_one = i[1]
        cfn_spanss.append(cfn_spanss_one)
        frames.append(i[2])

        targets_one = i[3]
        targets.append(targets_one)

        texts_one = i[4]
        texts.append(texts_one)
        words_one = i[5]
        words.append(words_one)
        frame_indexs.append(i[6])

        new_text = []
        for word_info in words_one:
            word_text = texts_one[word_info['start']:word_info['end'] + 1]
            word_pos = word_info['pos']
            word_str = f"{word_text} ({word_pos})"
            new_text.append(word_str)

        # 将每个单词及其词性标注组成的字符串都括起来
        word_start = [elem['start'] for elem in words_one]
        word_end = [elem['end'] for elem in words_one]
        target_start = [targets_one['start']]
        target_end = [targets_one['end']]
        cfn_spans_start = [elem['start'] for elem in cfn_spanss_one]
        cfn_spans_end = [elem['end'] for elem in cfn_spanss_one]
        cfn_spans_fe_name = [elem['fe_name'] for elem in cfn_spanss_one]
        target_pos = [targets_one['pos']]
        label_result = get_indices(cfn_spans_fe_name, fe_name_count)

        cfn_spans_find_start = find_indices(cfn_spans_start, word_start)
        cfn_spans_find_end = find_indices(cfn_spans_end, word_end)
        target_find_start = find_indices(target_start, word_start)
        target_find_end = find_indices(target_end, word_end)
        z = 0
        for ics in cfn_spans_find_start:
            new_text[ics] = f"({cfn_spans_fe_name[z]}:{new_text[ics]}"
            z += 1
        z = 0
        for ice in cfn_spans_find_end:
            new_text[ice] = f"{new_text[ice]})"
            z += 1
        z = 0
        for its in target_find_start:
            new_text[its] = f"({target_pos[z]}:{new_text[its]}"
            z += 1
        z = 0
        for ite in target_find_end:
            new_text[ite] = f"{new_text[ite]})"
            z += 1

        str_my_list = ''.join(new_text)
        str_my_list = str_my_list.replace('"', '').replace(',', '').replace("'", "").replace(" ", "")
        result.append(str_my_list)
        labels.appebd(label_result)
    # 编码
    data = token.batch_encode_plus(
        # sentence_ids,
        # cfn_spanss,
        # frames,
        # targets,
        # texts,
        # words,
        result,
        padding='max_length',
        truncation=True,
        max_length=510,
        return_tensors='pt',  # 返回pytorch模型
        return_length=True)

    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    labels_tensor = torch.zeros(len(labels), 673).to(device)
    for i, label in enumerate(labels):
        labels_tensor[i][label] = 1

    return input_ids, attention_mask, labels_tensor


# 数据加载器，数据少的话，具体原因不清楚，batchsize不能太大，明白了，数据太少了，刚才的数据被drop_last丢掉了
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=512,  # 是每个批轮的大小，也就是每轮处理的样本数量。
                                           collate_fn=collate_fn,  # 是一个函数，用于对每个批轮中的样本进行编码和处理。
                                           shuffle=True,  # 是一个布尔值，表示是否对数据进行随机重排。
                                           drop_last=False)  # 是一个布尔值，表示是否在最后一个批轮中舍弃不足一个批轮大小的数据

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=512,  # 是每个批轮的大小，也就是每轮处理的样本数量。
                                         collate_fn=collate_fn,  # 是一个函数，用于对每个批轮中的样本进行编码和处理。
                                         shuffle=False,  # 是一个布尔值，表示是否对数据进行随机重排。
                                         drop_last=False)  # 是一个布尔值，表示是否在最后一个批轮中舍弃不足一个批轮大小的数据

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=128,  # 是每个批轮的大小，也就是每轮处理的样本数量。
                                          collate_fn=collate_fn,  # 是一个函数，用于对每个批轮中的样本进行编码和处理。
                                          shuffle=False,  # 是一个布尔值，表示是否对数据进行随机重排。
                                          drop_last=False)  # 是一个布尔值，表示是否在最后一个批轮中舍弃不足一个批轮大小的数据


def reset_model_parameters(model):
    if hasattr(model, 'module'):
        model = model.module
    for parameter in model.parameters():
        parameter.data.normal_(mean=0.0, std=0.02)


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
    try:
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            train_f1 = 0
            train_acc = 0
            train_recall = 0
            train_count = 0
            for i, (input_ids, attention_mask, labels) in enumerate(train_loader):
                # 遍历数据集，并将数据转移到GPU上
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                # 进行前向传播，得到预测值out
                loss = criterion(out, labels.float())  # 计算损失
                loss.backward()  # 反向传播，计算梯度
                optimizer.step()  # 更新参数
                optimizer.zero_grad()  # 梯度清零，防止梯度累积

                out = torch.sigmoid(out)  # 将预测值转化为概率
                out = torch.where(out > 0.5, torch.ones_like(out), torch.zeros_like(out))  # 找到概率大于0.5的位置，并将其设为1，否则设为0
                predicted_labels = []
                true_labels = []

                f2 = 0
                # f2_precision = 0
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
                f2 = f2 / len(out)
                # f2_precision = f2_precision / len(out)
                f2_recall = f2_recall / len(out)
                f2_accuracy = f2_accuracy / len(out)

                train_loss += loss.item()
                train_f1 += f2
                train_acc += f2_accuracy
                train_recall += f2_recall
                train_count += 1
                print(f"predicted_labels：{predicted_labels}", '\n', f"true_labels：{true_labels}")
                print(
                    f"第{epoch + 1}周期：第{i + 1}轮训练, loss：{loss.item()}, 第{i + 1}轮训练集F1准确率为:{f2},第{i + 1}轮训练集accuracy:{f2_accuracy},第{i + 1}轮训练集recall:{f2_recall}")

            train_loss /= train_count
            train_f1 /= train_count
            train_acc /= train_count
            train_recall /= train_count

            print(f"------------总训练集F1为{train_f1},总accuracy为{train_acc}，总recall为{train_recall}------------")

            writer.add_scalar('Train Loss', train_loss, epoch)  # 记录训练损失
            writer.add_scalar('Train F1', train_f1, epoch)  # 记录训练F1得分
            writer.add_scalar('Train Accuracy', train_acc, epoch)  # 记录训练准确度
            writer.add_scalar('Train Recall', train_recall, epoch)  # 记录训练召回率

            # 验证
            model.eval()
            val_loss = 0
            val_f1 = 0
            val_acc = 0
            val_recall = 0
            val_count = 0
            with torch.no_grad():
                for i, (input_ids, attention_mask, labels) in enumerate(val_loader):
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(out, labels.float())  # 计算损失
                    out = torch.sigmoid(out)  # 将预测值转化为概率
                    out = torch.where(out > 0.5, torch.ones_like(out),
                                      torch.zeros_like(out))  # 找到概率大于0.5的位置，并将其设为1，否则设为0
                    predicted_labels = []
                    true_labels = []

                    # early_stopping(loss, model)
                    # # 若满足 early stopping 要求
                    # if early_stopping.early_stop:
                    #     print("Early stopping")
                    #     # 结束模型训练
                    #     break

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

                    average_val_f1 = f2 / len(out)
                    # f2_precision = f2_precision / len(out)
                    f2_recall = f2_recall / len(out)
                    f2_accuracy = f2_accuracy / len(out)

                    val_loss += loss.item()
                    val_f1 += average_val_f1
                    val_acc += f2_accuracy
                    val_recall += f2_recall
                    val_count += 1

                    print(f"predicted_labels：{predicted_labels}", '\n', f"true_labels：{true_labels}")
                    print(
                        f"第{epoch + 1}周期：第{i + 1}轮验证, loss：{loss.item()}, 第{i + 1}轮验证集F1准确率为:{average_val_f1},第{i + 1}轮验证集accuracy:{f2_accuracy},第{i + 1}轮验证集recall:{f2_recall}")

                val_loss /= val_count
                val_f1 /= val_count
                val_acc /= val_count
                val_recall /= val_count

                print(f"------------总验证集F1为{val_f1},总accuracy为{val_acc}，总recall为{val_recall}------------")

                writer.add_scalar('Val Loss', val_loss, epoch)  # 记录验证损失
                writer.add_scalar('Val F1', val_f1, epoch)  # 记录验证F1得分
                writer.add_scalar('Val Accuracy', val_acc, epoch)  # 记录验证准确度
                writer.add_scalar('Val Recall', val_recall, epoch)  # 记录验证召回率

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
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

        return best_val_f1, best_model_state

    except KeyboardInterrupt:
        # 捕捉用户手动终止训练的异常
        print('手动终止训练')
        model_save_path = "../Test/model_interrupted_1.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"当前模型已保存到：{model_save_path}")


# 定义一个超参数空间，用于搜佳超参数
learning_rate = 1e-4
num_epochs = 500

# 使用指定的超参数训练模型
test_f1, model = train_model(learning_rate, num_epochs)

# 保存训练好的模型
model_save_path = "../Test/best_model_9.pth"
torch.save(model, model_save_path)
print(f"使用指定的超参数训练的模型已保存到：{model_save_path}")
print(f"测试集 F1 分数：{test_f1}")
