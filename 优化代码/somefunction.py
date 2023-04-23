import torch
from transformers import BertTokenizer
import numpy as np
# 加载字典和分词工具
token = BertTokenizer.from_pretrained('bert-base-uncased')

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 如果是cuda则调用cuda反之则cpu

def collate_fn(data):
    source_codes = [i[0] for i in data]
    bytecodes = [i[1] for i in data]
    labels = [i[2] for i in data]

    # 编码
    data = token.batch_encode_plus(
        source_codes,
        bytecodes,
        padding='max_length',
        truncation=True,
        max_length=500,
        return_tensors='pt',  # 返回pytorch模型
        return_length=True)

    # 对于 `data` 字典中的每个键值对：
    # `input_ids`: 编码后的数字表示。
    # `attention_mask`: 表示哪些位置是有效的，哪些位置是补零的（0/1）。
    # `token_type_ids`: BERT 能够区分两个句子的方法（第一句话000..第二句话111...）。
    # `length`: 编码后序列的长度

    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    token_type_ids = data['token_type_ids'].to(device)

    # 将 labels 转换为多标签格式
    labels_tensor = torch.zeros(len(labels), 6).to(device)
    for i, label in enumerate(labels):
        labels_tensor[i][label] = 1

    return input_ids, attention_mask, token_type_ids, labels_tensor

def calculate_f1(precision, recall):
    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1
