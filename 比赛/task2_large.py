import re
import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

import copy

import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter

import json
import torch.utils.data as data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
import torch.nn.functional as F

warnings.filterwarnings("ignore")


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
        self.pretrained = AutoModel.from_pretrained("xlm-roberta-large")
        self.pretrained.to(device)
        for param in self.pretrained.parameters():
            param.requires_grad_(False)
        self.lstm = nn.LSTM(1024, 512, num_layers=2, batch_first=True, bidirectional=True)
        self.rfc = nn.Linear(1024, 4)

    def forward(self, input_ids, attention_mask):
        out = self.pretrained(input_ids=input_ids, attention_mask=attention_mask)
        out = out.last_hidden_state  # Keep the last hidden state for all positions
        out, _ = self.lstm(out)
        out = self.rfc(out)
        # out = F.softmax(out, dim=-1)
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
test_dataset = Dataset('cfn-test-B.json')

# 加载字典和分词工具
# token = AutoTokenizer.from_pretrained("bert-base-chinese")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")


def find_indices(cfn_spans_start, word_start):
    matches = []
    for i, num in enumerate(word_start):
        if num in cfn_spans_start:
            matches.append(i)
    return matches


def reshape_and_remove_pad(outs, labels, attention_mask):
    outs = outs[attention_mask == 1]

    # Reshape 'labels' tensor based on attention_mask
    labels = labels[attention_mask == 1]

    return outs, labels


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
    labels = []
    result_list=[]
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

        list1 = []
        for z in range(len(words_one)):
            if words_one[z]['end'] - words_one[z]['start'] == 0:
                list1.append(words_one[z]['pos'])
            else:
                for j in range(words_one[z]['end'] - words_one[z]['start'] + 1):
                    if j == 0:
                        list1.append(words_one[z]['pos'])
                    else:
                        list1.append(f"{words_one[z]['pos']}无")

        # print(list1)
        list2 = [get_value_by_pos(pos) for pos in list1]
        string_list = [str(num) for num in list2]

        target_start = targets_one['start']
        target_end = targets_one['end']
        for u in range(target_end - target_start + 1):
            for g in range(target_start, target_end + 1):
                string_list[g] = texts_one[g]
                # string_list[g] = "154"

        for u in range(len(words_one)):
            if words_one[u]['pos'] == 'wp':
                for x in range(words_one[u]['start'], words_one[u]['end'] + 1):
                    string_list[x] = texts_one[x]

        result_list.append(string_list)
        new_text = []

        characters = [char for char in texts_one]

        characters_list.append(characters)
        len_text = len(texts_one)

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
        for jx in range(len(cfn_spans_start)):
            if cfn_spans_start is not None:
                label[cfn_spans_start[jx]] = 1

            if cfn_spans_start is not None and cfn_spans_end is not None:
                for x in range(cfn_spans_start[jx] + 1, cfn_spans_end[jx] + 1):
                    label[x] = 2

        labels.append(label)

    # 编码
    data = tokenizer.batch_encode_plus(
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

    for i in range(len(labels)):
        labels[i] = [3] + labels[i]
        labels[i] += [3] * lens
        labels[i] = labels[i][:lens]

    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    labels = torch.LongTensor(labels).to(device)

    return input_ids, attention_mask, labels


# batchsize不能太大，明白了，数据太少了，刚才的数据被drop_last丢掉了
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=128,
                                           collate_fn=collate_fn,
                                           shuffle=True,
                                           drop_last=False)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=256,
                                         collate_fn=collate_fn,
                                         shuffle=True,
                                         drop_last=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=640,
                                          collate_fn=collate_fn,
                                          shuffle=False,
                                          drop_last=False)


def train_model(learning_rate, num_epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # 使用传入的学习率
    criterion = torch.nn.CrossEntropyLoss()

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    best_val_f1 = 0  # 初始化最佳验证集 F1 分数
    best_model_state = None  # 保存最佳模型参数
    best_val_loss = 1
    patience = 5
    counter = 0

    try:
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            train_f1 = 0
            train_acc = 0
            train_precision = 0
            train_recall = 0
            train_count = 0
            for i, (input_ids, attention_mask, labels) in enumerate(train_loader):
                # 遍历数据集，并将数据转移到GPU上
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                # 进行前向传播，得到预测值out
                out_z, labels_z = reshape_and_remove_pad(out, labels, attention_mask)

                loss = criterion(out_z, labels_z)  # 计算损失

                loss.backward()  # 反向传播，计算梯度
                optimizer.step()  # 更新参数
                optimizer.zero_grad()  # 梯度清零，防止梯度累积

                out = torch.argmax(out, dim=2)

                predicted_labels = []
                true_labels = []
                list5 = []
                list6 = []
                for j in range(len(labels)):
                    predicted_label = out[j].tolist()

                    true_label = labels[j].tolist()
                    t_first_index = true_label.index(3)
                    t_second_index = true_label.index(3, t_first_index + 1)
                    t_modified_label = true_label[t_first_index:t_second_index + 1]
                    true_labels.append(t_modified_label)

                    modified_label = predicted_label[t_first_index:t_second_index + 1]
                    predicted_labels.append(modified_label)

                y_true = [label for sublist in true_labels for label in sublist]
                y_pred = [label for sublist in predicted_labels for label in sublist]

                # 计算准确率
                # accuracy = accuracy_score(true_labels, predicted_labels)
                # 计算精确率
                precision = precision_score(y_true, y_pred, average='macro')
                # 计算召回率
                recall = recall_score(y_true, y_pred, average='macro')
                # 计算F1分数
                f1 = f1_score(y_true, y_pred, average='macro')

                train_loss += loss.item()
                train_f1 += f1
                train_precision += precision
                train_recall += recall
                train_count += 1
                # print(f"predicted_labels：{predicted_labels}", '\n', f"true_labels：{true_labels}")
                print(
                    f"第{epoch + 1}周期：第{i + 1}轮训练, loss：{loss.item()}, 第{i + 1}轮训练集F1准确率为:{f1},第{i + 1}轮训练集precision:{precision},第{i + 1}轮训练集recall:{recall}")

            train_loss /= train_count
            train_f1 /= train_count
            train_precision /= train_count
            train_recall /= train_count

            print(
                f"----------第{epoch + 1}周期,loss为{train_loss},总训练集F1为{train_f1},总precision为{train_precision}，总recall为{train_recall}------------")

            # 验证
            model.eval()
            val_loss = 0
            val_f1 = 0
            val_acc = 0
            val_recall = 0
            val_count = 0
            val_precision = 0
            with torch.no_grad():
                for i, (input_ids, attention_mask, labels) in enumerate(val_loader):
                    # 遍历数据集，并将数据转移到GPU上
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                    # 进行前向传播，得到预测值out
                    out_z, labels_z = reshape_and_remove_pad(out, labels, attention_mask)

                    loss = criterion(out_z, labels_z)  # 计算损失

                    out = torch.argmax(out, dim=2)

                    predicted_labels = []
                    true_labels = []
                    list5 = []
                    list6 = []
                    for j in range(len(labels)):
                        predicted_label = out[j].tolist()

                        true_label = labels[j].tolist()
                        t_first_index = true_label.index(3)
                        t_second_index = true_label.index(3, t_first_index + 1)
                        t_modified_label = true_label[t_first_index:t_second_index + 1]
                        true_labels.append(t_modified_label)

                        modified_label = predicted_label[t_first_index:t_second_index + 1]
                        predicted_labels.append(modified_label)


                    y_true = [label for sublist in true_labels for label in sublist]
                    y_pred = [label for sublist in predicted_labels for label in sublist]

                    # 计算准确率
                    # accuracy = accuracy_score(true_labels, predicted_labels)
                    # 计算精确率
                    precision = precision_score(y_true, y_pred, average='macro')
                    # 计算召回率
                    recall = recall_score(y_true, y_pred, average='macro')
                    # 计算F1分数
                    f1 = f1_score(y_true, y_pred, average='macro')

                    val_loss += loss.item()
                    val_f1 += f1
                    val_precision += precision
                    val_recall += recall
                    val_count += 1
                    # print(f"predicted_labels：{predicted_labels}", '\n', f"true_labels：{true_labels}")
                    print(
                        f"第{epoch + 1}周期：第{i + 1}轮验证, loss：{loss.item()}, 第{i + 1}轮验证集F1准确率为:{f1},第{i + 1}轮验证集precision:{precision},第{i + 1}轮训练集recall:{recall}")

                val_loss /= val_count
                val_f1 /= val_count
                val_precision /= val_count
                val_recall /= val_count

                print(
                    f"----------第{epoch + 1}周期,loss为{val_loss},总验证集F1为{val_f1},总precision为{val_precision}，总recall为{val_recall}------------")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = copy.deepcopy(model.state_dict())
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print("Early stopping!")
                # 保存当前模型
                torch.save(best_model_state, "task2_model_early_6.pt")
                break
            lr_scheduler.step()
            print(f"学习率为{lr_scheduler.get_last_lr()}")

        print(f"验证集 F1 分数：{val_f1}")
        return best_val_f1, best_model_state

    except KeyboardInterrupt:
        # 捕捉用户手动终止训练的异常
        print('手动终止训练')
        model_save_path = "c_model_interrupted_2.pth"
        torch.save(best_model_state, model_save_path)
        print(f"当前模型已保存到：{model_save_path}")


# 定义一个超参数空间，用于搜佳超参数
learning_rate = 0.001
num_epochs = 25

# 使用指定的超参数训练模型
test_f1, model_x = train_model(learning_rate, num_epochs)

# 保存训练好的模型
model_save_path = "task2_best_model_14.pth"
torch.save(model_x, model_save_path)
print(f"使用指定的超参数训练的模型已保存到：{model_save_path}")
print(test_f1)