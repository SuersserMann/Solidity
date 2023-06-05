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
            torch.nn.Dropout(p=0),
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


def idx2frame(index):
    frame = next((frame for frame, idx in frame2idx.items() if idx == index), None)
    return frame


class Dataset(data.Dataset):
    def __init__(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        sentence_id = item['sentence_id']
        # cfn_spans = item['cfn_spans']
        # frame = item['frame']
        target = item['target']
        text = item['text']
        word = item['word']
        # frame_index = frame2idx[frame]

        return sentence_id, target, text, word


train_dataset = Dataset('cfn-train.json')
val_dataset = Dataset('cfn-dev.json')
test_dataset = Dataset('cfn-test-B.json')

# 加载字典和分词工具
token = AutoTokenizer.from_pretrained("bert-base-chinese")


def find_indices(cfn_spans_start, word_start):
    matches = []
    for i, num in enumerate(word_start):
        if num in cfn_spans_start:
            matches.append(i)
    return matches


def collate_fn(data):
    sentence_ids = []
    cfn_spanss = []
    frames = []
    targets = []
    texts = []
    words = []
    frame_indexs = []
    result = []
    for i in data:
        sentence_ids.append(i[0])
        # cfn_spanss_one = i[1]
        # cfn_spanss.append(cfn_spanss_one)
        # frames.append(i[2])

        targets_one = i[1]
        targets.append(targets_one)

        texts_one = i[2]
        texts.append(texts_one)
        words_one = i[3]
        words.append(words_one)
        # frame_indexs.append(i[6])

        new_text = []

        word_text = texts_one[targets_one['start']:targets_one['end'] + 1]
        word_pos = targets_one['pos']
        word_str = f"{word_text}{word_pos}"
        new_text.append(word_str)

        str_my_list = ''.join(new_text)
        str_my_list = str_my_list.replace('"', '').replace(',', '').replace("'", "").replace(" ", "")
        result.append(str_my_list)

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
    # labels = torch.LongTensor(frame_indexs).to(device)

    return input_ids, attention_mask,sentence_ids


# batchsize不能太大，明白了，数据太少了，刚才的数据被drop_last丢掉了


val_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                         batch_size=256,
                                         collate_fn=collate_fn,
                                         shuffle=False,
                                         drop_last=False)

model.load_state_dict(torch.load('c_model_early_2.pt'))
# 测试
model.eval()
criterion = torch.nn.CrossEntropyLoss()
val_loss = 0
val_f1 = 0
val_acc = 0
val_recall = 0
val_count = 0
list3 = []
with torch.no_grad():
    aa_list = []
    bb_list = []
    for i, (input_ids, attention_mask, sentence_ids) in enumerate(val_loader):
        out = model(input_ids=input_ids, attention_mask=attention_mask)

        out = out.argmax(dim=1)
        predicted_labels = []
        true_labels = []

        for j in range(len(out)):
            predicted_label = out[j].tolist()  # 将位置索引转换为标签
            predicted_labels.append(predicted_label)
            sentence_id = sentence_ids[j]
            a_list = idx2frame(predicted_label)

            new_list = [sentence_id]+[{a_list}]
            print(new_list)
            bb_list.append(new_list)
        json_data = json.dumps(bb_list,ensure_ascii=False)
        with open('B_task1_test.json', 'w', encoding='UTF-8') as f:
            f.write(json_data)
