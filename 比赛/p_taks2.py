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


def get_value(label):
    cfn_spans_start = []
    cfn_spans_end = []
    start = None
    for i, value in enumerate(label):
        if value == 4:
            start = i
        elif value == 6 and start is not None:
            cfn_spans_start.append(start)
            cfn_spans_end.append(i)
            start = None
    return cfn_spans_start, cfn_spans_end


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
        self.lstm = nn.LSTM(768, 384, num_layers=2, batch_first=True, bidirectional=True)
        self.rfc = nn.Linear(768, 8)

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
        # cfn_spans = item['cfn_spans']
        # frame = item['frame']
        target = item['target']
        text = item['text']
        word = item['word']
        # frame_index = frame2idx[frame]

        return sentence_id, target, text, word


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


def reshape_and_remove_pad(outs, labels, attention_mask):
    outs = outs[attention_mask == 1]

    # Reshape 'labels' tensor based on attention_mask
    labels = labels[attention_mask == 1]

    return outs, labels


# 获取正确数量和总数
def get_correct_and_total_count(labels, outs):
    # [b*lens, 8] -> [b*lens]
    outs = outs.argmax(dim=1)
    correct = (outs == labels).sum().item()
    total = len(labels)

    # 计算除了0以外元素的正确率,因为0太多了,包括的话,正确率很容易虚高
    select = labels != 0
    outs = outs[select]
    labels = labels[select]
    correct_content = (outs == labels).sum().item()
    total_content = len(labels)

    return correct, total, correct_content, total_content


def collate_fn(data):
    sentence_ids = []
    # cfn_spanss = []
    # frames = []
    targets = []
    texts = []
    words = []
    # frame_indexs = []
    result = []
    characters_list = []
    # labels = []
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

        characters = [char for char in texts_one]
        characters_list.append(characters)

        # target_start = [targets_one['start']]
        # target_end = [targets_one['end']]
        # cfn_spans_start = [elem['start'] for elem in cfn_spanss_one]
        # cfn_spans_end = [elem['end'] for elem in cfn_spanss_one]
        # cfn_spans_combined = [[start, end] for start, end in zip(target_start, target_end)]
        # target_combined = [target_start[0], target_end[0]]

        # label = [0] * len_text

        # if target_start is not None:
        # label[target_start[0]] = 1

        # if target_start[0] != target_end[0] & target_end[0] is not None:
        # label[target_end[0]] = 3

        # if target_start is not None and target_end is not None:
        # for x in range(target_start[0] + 1, target_end[0]):
        # label[x] = 2

        # for jx in range(len(cfn_spans_start)):
        # if cfn_spans_start is not None:
        # label[cfn_spans_start[jx]] = 4

        # if cfn_spans_start[jx] != cfn_spans_end[jx] & cfn_spans_end[jx] is not None:
        # label[cfn_spans_end[jx]] = 6

        # if cfn_spans_start is not None and cfn_spans_end is not None:
        # for x in range(cfn_spans_start[jx] + 1, cfn_spans_end[jx]):
        # label[x] = 5

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
        characters_list,

        padding=True,
        truncation=True,
        # max_length=512,
        return_tensors='pt',  # 返回pytorch模型
        is_split_into_words=True,
        return_length=True)

    # lens = data['input_ids'].shape[1]

    # for i in range(len(labels)):
    # labels[i] = [7] + labels[i]
    # labels[i] += [7] * lens
    # labels[i] = labels[i][:lens]

    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    # labels = torch.LongTensor(labels).to(device)

    return input_ids, attention_mask


# batchsize不能太大，明白了，数据太少了，刚才的数据被drop_last丢掉了

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=640,
                                          collate_fn=collate_fn,
                                          shuffle=False,
                                          drop_last=False)

model.load_state_dict(torch.load('task2_best_model_2.pth'))
# 验证
model.eval()

with torch.no_grad():
    aa_list = []
    bb_list = []

    for i, (input_ids, attention_mask) in enumerate(test_loader):

        # 遍历数据集，并将数据转移到GPU上
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        # 进行前向传播，得到预测值out
        out = torch.argmax(out, dim=2)

        predicted_labels = []
        # true_labels = []

        for j in range(len(out)):
            predicted_label = out[j].tolist()
            predicted_labels.append(predicted_label)
            a_list, b_list = get_value(predicted_label)
            aa_list.append(a_list)
            bb_list.append(b_list)

print(aa_list)
print(bb_list)
print(predicted_labels)
