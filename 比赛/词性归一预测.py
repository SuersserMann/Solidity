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


def find_occurrences(lst):
    first_index = lst.index(2)  # 找到第一次出现数字3的位置
    second_index = lst.index(2, first_index + 1)  # 从第一次出现的位置之后找到第二次出现数字3的位置
    new_list = lst[first_index + 1:second_index]  # 根据位置切片生成新的列表
    return new_list


# def extract_list(lst):
#     # 找到第一个3的索引
#     first_index = lst.index(3)
#     # 找到第二个3的索引
#     second_index = lst.index(3, first_index + 1)
#     # 切片提取列表
#     result = lst[:second_index + 1]
#     return result


# def get_value(label):
#     intervals = []
#     three_index = extract_list(label)
#     start_index = None
#     for i, tag in enumerate(three_index):
#         if tag == 1:  # 开始区间
#             if i + 1 < len(three_index) and three_index[i + 1] == 2:  # 开头是1且后面是2
#                 start_index = i
#         elif tag != 2:  # 非中间或结束区间
#             if start_index is not None:
#                 intervals.append([start_index - 1, i - 1])
#                 start_index = None
#
#     # 处理最后一个区间
#     if start_index is not None and three_index[start_index] == 1:
#         intervals.append([start_index - 1, len(three_index) - 1])
#     return intervals


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


def find_label_ranges(t_index):
    ranges = []
    start_index = None
    # t_index = extract_list(lst)
    for i in range(len(t_index)):
        if t_index[i] == 1 or t_index[i] == 2:
            if start_index is None:
                start_index = i
        else:
            if start_index is not None:
                ranges.append([start_index, i - 1])
                start_index = None

    if start_index is not None:
        ranges.append([start_index, len(t_index) - 1])

    return ranges


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
        self.rfc = nn.Linear(768, 3)

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
        # frame = item['frame']
        target = item['target']
        text = item['text']
        word = item['word']
        # frame_index = frame2idx[frame]

        return sentence_id, cfn_spans, target, text, word


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
    # frames = []
    targets = []
    texts = []
    words = []
    # frame_indexs = []
    result = []
    characters_list = []
    # labels = []
    result_list = []
    vv_list = []
    c_t = 0
    c_e = 0
    for i in data:
        sentence_ids.append(i[0])
        cfn_spanss_one = i[1]
        cfn_spanss.append(cfn_spanss_one)
        # frames.append(i[2])

        targets_one = i[2]
        targets.append(targets_one)

        texts_one = i[3]
        texts.append(texts_one)
        words_one = i[4]
        words.append(words_one)
        vv = len(texts_one)
        vv_list.append(vv)
        # frame_indexs.append(i[6])

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
        a = 0

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
    #
    # for i in range(len(labels)):
    #     labels[i] = [3] + labels[i]
    #     labels[i] += [3] * lens
    #     labels[i] = labels[i][:lens]

    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    # labels = torch.LongTensor(labels).to(device)

    return input_ids, attention_mask, sentence_ids, cfn_spanss, vv_list, words, texts,targets


# batchsize不能太大，明白了，数据太少了，刚才的数据被drop_last丢掉了

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=64,
                                          collate_fn=collate_fn,
                                          shuffle=False,
                                          drop_last=False)

model.load_state_dict(torch.load('task2_best_model_119.pth'))
# 验证
model.eval()

with torch.no_grad():
    aa_list = []
    bb_list = []

    for i, (input_ids, attention_mask, sentence_ids, cfn_spanss, vv_list, words, texts,targets) in enumerate(test_loader):

        # 遍历数据集，并将数据转移到GPU上
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        # 进行前向传播，得到预测值out
        out = torch.argmax(out, dim=2)

        predicted_labels = []
        # true_labels = []

        for j in range(len(out)):
            sentence_id = sentence_ids[j]
            cfn_spans = cfn_spanss[j]
            vv = vv_list[j]
            word = words[j]
            target = targets[j]
            t_start = target['start']
            t_end = target['end']
            t_len = target['end'] - target['start'] + 1
            predicted_label = out[j].tolist()
            predicted_label_3 = find_occurrences(predicted_label)
            predicted_label_new = []
            for jj, item in enumerate(word):
                if item['start'] < t_start:
                    for zz in range(item['end'] - item['start'] + 1):
                        predicted_label_new.append(predicted_label_3[jj])
                if item['start'] == t_start:
                    for zz in range(t_len):
                        predicted_label_new.append(0)
                if item['start'] > t_start:
                    for zz in range(item['end'] - item['start'] + 1):
                        predicted_label_new.append(predicted_label_3[jj+t_len-1])

            # predicted_labels.append(predicted_label_new)
            target = targets[j]
            for p in range(target['end'] - target['start'] + 1):
                predicted_label_new[target['start'] + p] = 4

            a_list = find_label_ranges(predicted_label_new)
            for z in range(len(a_list)):
                new_list = [sentence_id] + a_list[z]
                print(new_list)
                bb_list.append(new_list)
    json_data = json.dumps(bb_list)
    with open('output.json', 'w', encoding='UTF-8') as f:
        f.write(json_data)
