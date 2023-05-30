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

# 忽略所有的警告
warnings.filterwarnings("ignore")


# class Logger(object):
#     def __init__(self, filename):
#         self.terminal = sys.stdout
#         self.filename = filename
#         self.log = open(self.filename, "a")
#
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#
#     def flush(self):
#         pass
#
#
# now = datetime.datetime.now()
# filename = "log_{:%Y-%m-%d_%H-%M-%S}.txt".format(now)
# path = os.path.abspath(os.path.dirname(__file__))
# log_dir_path = os.path.join(path, "log")
# log_file_path = os.path.join(log_dir_path, filename)
#
# if not os.path.exists(log_dir_path):
#     os.makedirs(log_dir_path)
#
# sys.stdout = Logger(log_file_path)


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
        self.pretrained = AutoModel.from_pretrained("bert-base-chinese")
        self.pretrained.to(device)
        for param in self.pretrained.parameters():
            param.requires_grad_(False)
        self.gru = torch.nn.GRU(768, 768, num_layers=2, batch_first=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(512, 695)
        )

    def forward(self, input_ids, attention_mask):
        out = self.pretrained(input_ids=input_ids, attention_mask=attention_mask)
        out = out.last_hidden_state[:, 0]  # Only keep the last hidden state
        out, _ = self.gru(out.unsqueeze(0))
        out = self.fc(out.squeeze(0))
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

# 加载字典和分词工具
token = AutoTokenizer.from_pretrained("bert-base-chinese")


def find_indices(cfn_spans_start, word_start):
    matches = []
    for i, num in enumerate(word_start):
        if num in cfn_spans_start:
            matches.append(i)
    return matches

def get_value_by_pos(pos):
    mapping = {
        '': 100,
        'e': 101,
        'r': 102,
        'i': 103,
        'd': 104,
        'u': 105,
        'nh': 106,
        'ws': 107,
        'v': 108,
        'ni': 109,
        'm': 110,
        'k': 111,
        'b': 112,
        'c': 113,
        'nd': 114,
        'n': 115,
        'a': 116,
        'wp': 117,
        'o': 118,
        'nt': 119,
        'h': 120,
        'nl': 121,
        'p': 122,
        'q': 123,
        'j': 124,
        'nz': 125,
        'ns': 126,
        '无': 127,
        'e无': 128,
        'r无': 129,
        'i无': 130,
        'd无': 131,
        'u无': 132,
        'nh无': 133,
        'ws无': 132,
        'v无': 135,
        'ni无': 136,
        'm无': 137,
        'k无': 138,
        'b无': 139,
        'c无': 140,
        'nd无': 141,
        'n无': 142,
        'a无': 143,
        'wp无': 144,
        'o无': 145,
        'nt无': 146,
        'h无': 147,
        'nl无': 148,
        'p无': 149,
        'q无': 150,
        'j无': 151,
        'nz无': 152,
        'ns无': 153
    }
    return mapping.get(pos)

def collate_fn(data):
    sentence_ids = []
    cfn_spanss = []
    frames = []
    targets = []
    texts = []
    words = []
    frame_indexs = []
    result = []
    characters_list = []
    # labels = []
    result_list = []
    c_t = 0
    c_e = 0
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

        target_start = targets_one['start']
        target_end = targets_one['end']

        list1 = []
        for z in range(len(words_one)):
            if words_one[z]['start'] == target_start:
                for g in range(target_start, target_end + 1):
                    list1.append(texts_one[g])
            else:
                list1.append(words_one[z]['pos'])

        list2 = [get_value_by_pos(pos) for pos in list1]
        string_list = [str(num) for num in list2]

        for i, item in enumerate(words_one):
            if item["start"] == target_start:
                c_t = i
        for i, item in enumerate(words_one):
            if item["end"] == target_end:
                c_e = i
        a=0

        for zg in range(c_t, c_e + 1 + (target_end - target_start)):
            string_list[zg] = texts_one[target_start + a]
            a += 1
        result_list.append(string_list)

        # for g in range(target_start, target_end + 1):
        #     string_list[g] = texts_one[g]

        # for u in range(target_end - target_start + 1):
        #     for g in range(target_start, target_end + 1):
        #         string_list[g] = texts_one[g]
        #         # string_list[g] = "154"

        new_text = []

        # characters = [char for char in texts_one]
        #
        # characters_list.append(characters)
        len_text = len(string_list)

        cfn_spans_start = [elem['start'] for elem in cfn_spanss_one]
        cfn_spans_end = [elem['end'] for elem in cfn_spanss_one]
        # cfn_spans_combined = [[start, end] for start, end in zip(target_start, target_end)]
        # target_combined = [target_start[0], target_end[0]]

        label = [0] * len_text

        # if target_start is not None:
        #     label[target_start[0]] = 3
        #
        # if target_start is not None and target_end is not None:
        #     for x in range(target_start[0] + 1, target_end[0] + 1):
        #         label[x] = 4

        index_start = []
        index_end = []
        for jz in range(len(cfn_spans_start)):
            c_start = cfn_spans_start[jz]
            c_end = cfn_spans_end[jz]
            c_len = target_end - target_start + 1
            if c_start < target_start:
                for i, item in enumerate(words_one):
                    if item["start"] == c_start:
                        index_start.append(i)
                        break
                for i, item in enumerate(words_one):
                    if item["end"] == c_end:
                        index_end.append(i)
                        break
            else:
                for i, item in enumerate(words_one):
                    if item["start"] == c_start:
                        index_start.append(i + c_len - 1)
                        break
                for i, item in enumerate(words_one):
                    if item["end"] == c_end:
                        index_end.append(i + c_len - 1)
                        break

        # for jx in range(len(index_start)):
        #     if index_start is not None:
        #         label[index_start[jx]] = 1
        #
        #     if index_start is not None and index_end is not None:
        #         for x in range(index_start[jx] + 1, index_end[jx] + 1):
        #             label[x] = 2
        #
        # labels.append(label)

    # 编码
    data = token.batch_encode_plus(
        # sentence_ids,
        # cfn_spanss,
        # frames,
        # targets,
        # texts,
        # words,
        # result,
        result_list,

        padding=True,
        truncation=True,
        # max_length=512,
        return_tensors='pt',  # 返回pytorch模型
        is_split_into_words=True,
        return_length=True)

    lens = data['input_ids'].shape[1]

    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    labels = torch.LongTensor(frame_indexs).to(device)

    return input_ids, attention_mask, labels



# batchsize不能太大，明白了，数据太少了，刚才的数据被drop_last丢掉了
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=32,
                                           collate_fn=collate_fn,
                                           shuffle=True,
                                           drop_last=False)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=64,
                                         collate_fn=collate_fn,
                                         shuffle=False,
                                         drop_last=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=640,
                                          collate_fn=collate_fn,
                                          shuffle=False,
                                          drop_last=False)


def train_model(learning_rate, num_epochs):
    writer = SummaryWriter()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)  # 使用传入的学习率
    criterion = torch.nn.CrossEntropyLoss()

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    best_val_f1 = 0
    best_accuracy = 0 # 初始化最佳验证集 F1 分数
    best_model_state = None  # 保存最佳模型参数

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
                loss = criterion(out, labels)  # 计算损失
                loss.backward()  # 反向传播，计算梯度
                optimizer.step()  # 更新参数
                optimizer.zero_grad()  # 梯度清零，防止梯度累积

                out = out.argmax(dim=1)

                predicted_labels = []
                true_labels = []

                for j in range(len(labels)):
                    predicted_label = out[j].tolist()  # 将位置索引转换为标签
                    predicted_labels.append(predicted_label)
                    true_label = labels[j].tolist()
                    true_labels.append(true_label)

                # 计算准确率
                accuracy = accuracy_score(true_labels, predicted_labels)
                # 计算精确率
                # precision = precision_score(true_labels, predicted_labels, average='macro')
                # 计算召回率
                recall = recall_score(true_labels, predicted_labels, average='macro')
                # 计算F1分数
                f1 = f1_score(true_labels, predicted_labels, average='macro')

                train_loss += loss.item()
                train_f1 += f1
                train_acc += accuracy
                train_recall += recall
                train_count += 1
                # print(f"predicted_labels：{predicted_labels}", '\n', f"true_labels：{true_labels}")
                # print(
                #   f"第{epoch + 1}周期：第{i + 1}轮训练, loss：{loss.item()}, 第{i + 1}轮训练集F1准确率为:{f1},第{i + 1}轮训练集accuracy:{accuracy},第{i + 1}轮训练集recall:{recall}")

            train_loss /= train_count
            train_f1 /= train_count
            train_acc /= train_count
            train_recall /= train_count

            print(
                f"----------第{epoch + 1}周期,loss为{train_loss},总训练集F1为{train_f1},总accuracy为{train_acc}，总recall为{train_recall}------------")

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
                    loss = criterion(out, labels)  # 计算损失
                    out = out.argmax(dim=1)
                    predicted_labels = []
                    true_labels = []

                    for j in range(len(labels)):
                        predicted_label = out[j].tolist()  # 将位置索引转换为标签
                        predicted_labels.append(predicted_label)
                        true_label = labels[j].tolist()
                        true_labels.append(true_label)

                    # 计算准确率
                    accuracy = accuracy_score(true_labels, predicted_labels)
                    # 计算精确率
                    # precision = precision_score(true_labels, predicted_labels, average='macro')
                    # 计算召回率
                    recall = recall_score(true_labels, predicted_labels, average='macro')
                    # 计算F1分数
                    f1 = f1_score(true_labels, predicted_labels, average='macro')

                    val_loss += loss.item()
                    val_f1 += f1
                    val_acc += accuracy
                    val_recall += recall
                    val_count += 1

                    # print(f"predicted_labels：{predicted_labels}", '\n', f"true_labels：{true_labels}")
                    # print(
                    #    f"第{epoch + 1}周期：第{i + 1}轮验证, loss：{loss.item()}, 第{i + 1}轮验证集F1准确率为:{f1},第{i + 1}轮验证集accuracy:{accuracy},第{i + 1}轮验证集recall:{recall}")

                val_loss /= val_count
                val_f1 /= val_count
                val_acc /= val_count
                val_recall /= val_count

                print(
                    f"------------第{epoch + 1}周期,loss为{val_loss}，总验证集F1为{val_f1},总accuracy为{val_acc}，总recall为{val_recall}------------")


            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_model_state = model.state_dict()
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print("Early stopping!")
                # 保存当前模型
                torch.save(best_model_state, "task1_26_model_early_1.pt")
                break
            lr_scheduler.step()
            print(f"学习率为{lr_scheduler.get_last_lr()}")
        # 加载具有最佳验证集性能的模型参数
        model.load_state_dict(best_model_state)

        print(f"验证集 F1 分数：{best_val_f1}")
        return best_accuracy, best_model_state

    except KeyboardInterrupt:
        # 捕捉用户手动终止训练的异常
        print('手动终止训练')
        model_save_path = "c_model_interrupted_1.pt"
        torch.save(model.state_dict(), model_save_path)
        print(f"当前模型已保存到：{model_save_path}")


# 定义一个超参数空间，用于搜佳超参数
learning_rate = 0.001
num_epochs = 200

# 使用指定的超参数训练模型
test_f1, model = train_model(learning_rate, num_epochs)

# 保存训练好的模型
model_save_path = "task1_26_best_model_1.pt"
torch.save(model, model_save_path)
print(f"使用指定的超参数训练的模型已保存到：{model_save_path}")
print(test_f1)
