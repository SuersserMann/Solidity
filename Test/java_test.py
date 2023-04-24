import re

import torch
from datasets import load_dataset
from transformers import BertTokenizer
import numpy as np
from sklearn.metrics import f1_score
# 从transformers调用现有的model
from transformers import BertModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
def calculate_f1(precision, recall):
    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1

print(torch.__version__)

# 使用cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 如果是cuda则调用cuda反之则cpu
print('device=', device)



# 加载预训练模型，使用预训练模型可以加快训练的速度
pretrained = AutoModelForMaskedLM.from_pretrained("neulab/codebert-java")
# 需要移动到cuda上
pretrained.to(device)

# pretrained 模型中所有参数的 requires_grad 属性设置为 False，这意味着这些参数在训练过程中将不会被更新，其值将保持不变
for param in pretrained.parameters():
    param.requires_grad_(False)
# fine tune向后传播而不修改之前的参数

# 定义了下游任务模型，包括一个全连接层和forward方法。
class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # 定义一个全连接层，输入维度为768，输出维度为6
        self.fc = torch.nn.Linear(768, 6)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 将输入传入预训练模型，并记录计算图以计算梯度
        out = pretrained(input_ids=input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids)
        # 只保留预训练模型输出的最后一个隐藏状态，并通过全连接层进行分类
        out = self.fc(out.last_hidden_state[:, 0])

        return out


# 实例化下游任务模型并将其移动到 GPU 上 (如果可用)
model = Model()
model.to(device)


# 定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.dataset = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split=split,
                                    verification_mode='no_checks')
        # 从huggingface导入数据集

    def __len__(self):
        return len(self.dataset)
        # 计算数据集长度，方便后面进行一个批量的操作

    def __getitem__(self, i):
        source_code = self.dataset[i]['source_code']
        bytecode = self.dataset[i]['bytecode']
        label = self.dataset[i]['slither']
        # 从数据集遍历数据，比如第一批是16个，那么第二批就可以从17-32
        return source_code, bytecode, label


train_dataset = Dataset('train')
# 只截取训练集
len(train_dataset), train_dataset[0]
val_dataset = Dataset('validation')
test_dataset = Dataset('test')
# 加载字典和分词工具
token = AutoTokenizer.from_pretrained("neulab/codebert-java")

def delete_comment(java_code):
    # 用正则表达式匹配 Java 代码中的注释
    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//.*?$)"
    regex = re.compile(pattern, re.MULTILINE|re.DOTALL)
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
    source_codes = [delete_comment(i[0]) for i in data]
    bytecodes = [i[1] for i in data]
    labels = [i[2] for i in data]

    # 编码
    data = token.batch_encode_plus(
        source_codes,
        bytecodes,
        padding='max_length',
        truncation=True,
        max_length=512,
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


# 数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=32,  # 是每个批次的大小，也就是每次处理的样本数量。
                                           collate_fn=collate_fn,  # 是一个函数，用于对每个批次中的样本进行编码和处理。
                                           shuffle=True,  # 是一个布尔值，表示是否对数据进行随机重排。
                                           drop_last=True)  # 是一个布尔值，表示是否在最后一个批次中舍弃不足一个批次大小的数据

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=32,  # 是每个批次的大小，也就是每次处理的样本数量。
                                          collate_fn=collate_fn,  # 是一个函数，用于对每个批次中的样本进行编码和处理。
                                          shuffle=False,  # 是一个布尔值，表示是否对数据进行随机重排。
                                          drop_last=True)  # 是一个布尔值，表示是否在最后一个批次中舍弃不足一个批次大小的数据

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=32,  # 是每个批次的大小，也就是每次处理的样本数量。
                                         collate_fn=collate_fn,  # 是一个函数，用于对每个批次中的样本进行编码和处理。
                                         shuffle=False,  # 是一个布尔值，表示是否对数据进行随机重排。
                                         drop_last=True)  # 是一个布尔值，表示是否在最后一个批次中舍弃不足一个批次大小的数据


def train_model(learning_rate, num_epochs):

    # 训练
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # 使用传入的学习率
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_f1 = 0  # 初始化最佳验证集 F1 分数
    best_model_state = None  # 保存最佳模型参数

    for epoch in range(num_epochs):
        model.train()
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
            # 遍历数据集，并将数据转移到GPU上
            out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            # 进行前向传播，得到预测值out
            loss = criterion(out, labels.float())  # 计算损失
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新参数
            optimizer.zero_grad()  # 梯度清零，防止梯度累积

            out = torch.sigmoid(out)  # 将预测值转化为概率
            out = torch.where(out > 0.5, torch.ones_like(out), torch.zeros_like(out))  # 找到概率大于0.5的位置，并将其设为1，否则设为0
            predicted_labels = []
            True_labels = []
            f2 = 0
            for j in range(len(out)):
                predicted_label = torch.where(out[j] == 1)[0].tolist()  # 将位置索引转换为标签
                predicted_labels.append(predicted_label)
                True_label = train_dataset[i * 5 + j][2]
                True_label = sorted(True_label)
                True_labels.append(True_label)

                # 计算F1分数
                predicted_set = set(predicted_label)
                true_set = set(True_label)

                TP = len(predicted_set.intersection(true_set))
                FP = len(predicted_set - true_set)
                FN = len(true_set - predicted_set)
                precision = TP / (TP + FP) if TP + FP else 0
                recall = TP / (TP + FN) if TP + FN else 0
                f1 = calculate_f1(precision, recall)

                f2 = f1 + f2
            f2 = f2 / train_loader.batch_size

            if i % 10 == 0:
                print(f"predicted_labels：{predicted_labels}", '\n', f"True_labels：{True_labels}")
                print(f"第{i}轮训练, loss：{loss.item()}, 第{i}次训练集F1准确率为:{f2}")
        # 验证
        model.eval()

        with torch.no_grad():
            for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(val_loader):
                out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                # 进行前向传播，得到预测值out
                loss = criterion(out, labels.float())  # 计算损失
                loss.backward()  # 反向传播，计算梯度
                optimizer.step()  # 更新参数
                optimizer.zero_grad()  # 梯度清零，防止梯度累积

                out = torch.sigmoid(out)  # 将预测值转化为概率
                out = torch.where(out > 0.5, torch.ones_like(out), torch.zeros_like(out))  # 找到概率大于0.5的位置，并将其设为1，否则设为0
                predicted_labels = []
                True_labels = []

                for j in range(len(out)):
                    predicted_label = torch.where(out[j] == 1)[0].tolist()  # 将位置索引转换为标签
                    predicted_labels.append(predicted_label)
                    True_label = train_dataset[i * 5 + j][2]
                    True_label = sorted(True_label)
                    True_labels.append(True_label)

                    # 计算F1分数
                    predicted_set = set(predicted_label)
                    true_set = set(True_label)

                    TP = len(predicted_set.intersection(true_set))
                    FP = len(predicted_set - true_set)
                    FN = len(true_set - predicted_set)
                    precision = TP / (TP + FP) if TP + FP else 0
                    recall = TP / (TP + FN) if TP + FN else 0
                    f1 = calculate_f1(precision, recall)

                    f2 = f1 + f2
                average_val_f1 = f2 / val_loader.batch_size if val_loader.batch_size else 0
                if i % 10 == 0:
                    print(f"predicted_labels：{predicted_labels}", '\n', f"True_labels：{True_labels}")
                    print(f"第{i}轮训练, loss：{loss.item()}, 第{i}次验证集F1准确率为:{average_val_f1}")
        # 如果当前模型在验证集上的性能更好，则保存模型参数
        if average_val_f1 > best_val_f1:
            best_val_f1 = average_val_f1
            best_model_state = model.state_dict()

    # 加载具有最佳验证集性能的模型参数
    model.load_state_dict(best_model_state)

    # 测试
    model.eval()

    with torch.no_grad():
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(test_loader):

            out = torch.sigmoid(out)  # 将预测值转化为概率
            out = torch.where(out > 0.5, torch.ones_like(out), torch.zeros_like(out))  # 找到概率大于0.5的位置，并将其设为1，否则设为0
            predicted_labels = []
            True_labels = []

            for j in range(len(out)):
                predicted_label = torch.where(out[j] == 1)[0].tolist()  # 将位置索引转换为标签
                predicted_labels.append(predicted_label)
                True_label = train_dataset[i * 5 + j][2]
                True_label = sorted(True_label)
                True_labels.append(True_label)

                # 计算F1分数
                predicted_set = set(predicted_label)
                true_set = set(True_label)

                TP = len(predicted_set.intersection(true_set))
                FP = len(predicted_set - true_set)
                FN = len(true_set - predicted_set)
                precision = TP / (TP + FP) if TP + FP else 0
                recall = TP / (TP + FN) if TP + FN else 0
                f1 = calculate_f1(precision, recall)

                f2 = f1 + f2
            average_test_f1 = f2 / test_loader.batch_size if test_loader.batch_size else 0

            if i % 10 == 0:
                print(f"predicted_labels：{predicted_labels}", '\n', f"True_labels：{True_labels}")
                print(f"第{i}轮训练, loss：{loss.item()}, 第{i}次测试集F1准确率为:{average_test_f1}")
    print(f"测试集 F1 分数：{average_test_f1}")

    return average_test_f1


# 定义一个超参数空间，用于搜索最佳超参数
learning_rates = [1e-5, 3e-5, 1e-4]
num_epochs_list = [1, 2, 3]

best_hyperparams = None
best_test_f1 = 0

# 使用网格搜索遍历所有可能的超参数组合
for lr in learning_rates:
    for num_epochs in num_epochs_list:
        print(f"正在训练模型，学习率：{lr}，训练周期：{num_epochs}")
        test_f1 = train_model(lr, num_epochs)

        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_hyperparams = {'learning_rate': lr, 'num_epochs': num_epochs}

print(f"最佳超参数：{best_hyperparams}，测试集 F1 分数：{best_test_f1}")


def train_and_save_best_model(best_hyperparams, save_path):
    best_lr = best_hyperparams['learning_rate']
    best_num_epochs = best_hyperparams['num_epochs']

    # 使用找到的最佳超参数重新训练模型
    model = train_model(best_lr, best_num_epochs)

    # 保存训练好的模型
    torch.save(model.state_dict(), save_path)
    print(f"使用最佳超参数训练的模型已保存到：{save_path}")


model_save_path = "../Nlp/best_model.pth"
train_and_save_best_model(best_hyperparams, model_save_path)