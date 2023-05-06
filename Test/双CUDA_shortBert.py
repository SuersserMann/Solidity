import re
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
    disassembled = disassemble_hex(bytecode)
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
            torch.nn.Linear(100, 3))  # 添加多层神经网络

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
model = nn.DataParallel(model,device_ids=device_ids)
model.to(device)

# 定义数据集
# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, split):
#         self.dataset = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split=split,
#                                     verification_mode='no_checks')
#         # 从huggingface导入数据集
#
#     def __len__(self):
#         return len(self.dataset)
#         # 计算数据集长度，方便后面进行一个批量的操作
#
#     def __getitem__(self, i):
#         source_code = self.dataset[i]['source_code']
#         bytecode = self.dataset[i]['bytecode']
#         label = self.dataset[i]['slither']
#         # 从数据集遍历数据，比如第一批是16个，那么第二批就可以从17-32
#         return source_code, bytecode, label


# train_dataset = Dataset('train')
# val_dataset = Dataset('validation')
# test_dataset = Dataset('test')
import pandas as pd
import datasets
import random

# 从 Excel 文件中读取数据
data = pd.read_excel('123.xlsx')

# 将数据存储到 DataFrame 中，并选择需要的列
df = pd.DataFrame(data, columns=['slither', 'source_code'])

# 将 DataFrame 转换为 Dataset 对象
all_dataset = datasets.Dataset.from_pandas(df)
all_dataset = [[all_dataset['slither'][i], all_dataset['source_code'][i]] for i in range(len(all_dataset))]
all_dataset = all_dataset*20
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
    labels_tensor = torch.zeros(len(labels), 3).to(device)
    for i, label in enumerate(labels):
        labels_tensor[i][label] = 1

    return input_ids, attention_mask, labels_tensor


# 数据加载器，数据少的话，具体原因不清楚，batchsize不能太大，明白了，数据太少了，刚才的数据被drop_last丢掉了
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=80,  # 是每个批轮的大小，也就是每轮处理的样本数量。
                                           collate_fn=collate_fn,  # 是一个函数，用于对每个批轮中的样本进行编码和处理。
                                           shuffle=True,  # 是一个布尔值，表示是否对数据进行随机重排。
                                           drop_last=False)  # 是一个布尔值，表示是否在最后一个批轮中舍弃不足一个批轮大小的数据

test_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                          batch_size=80,  # 是每个批轮的大小，也就是每轮处理的样本数量。
                                          collate_fn=collate_fn,  # 是一个函数，用于对每个批轮中的样本进行编码和处理。
                                          shuffle=False,  # 是一个布尔值，表示是否对数据进行随机重排。
                                          drop_last=False)  # 是一个布尔值，表示是否在最后一个批轮中舍弃不足一个批轮大小的数据

val_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                         batch_size=80,  # 是每个批轮的大小，也就是每轮处理的样本数量。
                                         collate_fn=collate_fn,  # 是一个函数，用于对每个批轮中的样本进行编码和处理。
                                         shuffle=False,  # 是一个布尔值，表示是否对数据进行随机重排。
                                         drop_last=False)  # 是一个布尔值，表示是否在最后一个批轮中舍弃不足一个批轮大小的数据


def reset_model_parameters(model):
    for parameter in model.parameters():
        parameter.data.normal_(mean=0.0, std=0.02)


def train_model(learning_rate, num_epochs):
    reset_model_parameters(model)
    # 训练
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # 使用传入的学习率
    criterion = torch.nn.BCEWithLogitsLoss()

    # patience = 20  # 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
    # early_stopping = EarlyStopping(patience, verbose=True)

    best_val_f1 = 0  # 初始化最佳验证集 F1 分数
    best_model_state = None  # 保存最佳模型参数

    for epoch in range(num_epochs):
        model.train()
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
            f2_precision = 0
            f2_recall = 0
            for j in range(len(out)):
                predicted_label = torch.where(out[j] == 1)[0].tolist()  # 将位置索引转换为标签
                predicted_labels.append(predicted_label)
                true_label = torch.where(labels[j] == 1)[0].tolist()
                true_labels.append(true_label)

                # 计算F1分数
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
            print(f"predicted_labels：{predicted_labels}", '\n', f"true_labels：{true_labels}")
            print(f"第{i + 1}轮训练, loss：{loss.item()}, 第{i + 1}轮训练集F1准确率为:{f2},第{i + 1}轮训练集accuracy:{f2_precision},第{i + 1}轮训练集recall:{f2_recall}")
        # 验证
        model.eval()

        with torch.no_grad():
            for i, (input_ids, attention_mask, labels) in enumerate(val_loader):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(out, labels.float())  # 计算损失
                out = torch.sigmoid(out)  # 将预测值转化为概率
                out = torch.where(out > 0.5, torch.ones_like(out), torch.zeros_like(out))  # 找到概率大于0.5的位置，并将其设为1，否则设为0
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
                for j in range(len(out)):
                    predicted_label = torch.where(out[j] == 1)[0].tolist()  # 将位置索引转换为标签
                    predicted_labels.append(predicted_label)
                    true_label = torch.where(labels[j] == 1)[0].tolist()
                    true_labels.append(true_label)

                    # 计算F1分数
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

                print(f"predicted_labels：{predicted_labels}", '\n', f"true_labels：{true_labels}")
                print(f"第{i + 1}轮验证, loss：{loss.item()}, 第{i + 1}轮验证集F1准确率为:{average_val_f1},第{i + 1}轮验证集accuracy:{f2_precision},第{i + 1}轮验证集recall:{f2_recall}")

        if average_val_f1 > best_val_f1:
            best_val_f1 = average_val_f1
            best_model_state = copy.deepcopy(model.state_dict())

    # 加载具有最佳验证集性能的模型参数
    model.load_state_dict(best_model_state)
    # 测试
    model.eval()

    with torch.no_grad():
        for i, (input_ids, attention_mask, labels) in enumerate(test_loader):
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(out, labels.float())  # 计算损失
            out = torch.sigmoid(out)  # 将预测值转化为概率
            out = torch.where(out > 0.5, torch.ones_like(out), torch.zeros_like(out))  # 找到概率大于0.5的位置，并将其设为1，否则设为0
            predicted_labels = []
            true_labels = []

            f2 = 0
            f2_precision = 0
            f2_recall = 0
            for j in range(len(out)):
                predicted_label = torch.where(out[j] == 1)[0].tolist()  # 将位置索引转换为标签
                predicted_labels.append(predicted_label)
                true_label = torch.where(labels[j] == 1)[0].tolist()
                true_labels.append(true_label)

                # 计算F1分数
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


# 定义一个超参数空间，用于搜索最佳超参数
learning_rates = [3e-5, 1e-4, 3e-4]
num_epochs_list = [100, 300, 500]

best_hyperparams = None
best_test_f1 = 0

# 使用网格搜索遍历所有可能的超参数组合
for lr in learning_rates:
    for num_epochs in num_epochs_list:
        print(f"正在训练模型，学习率：{lr}，训练周期：{num_epochs}")

        test_f1, _ = train_model(lr, num_epochs)

        if test_f1 >= best_test_f1:
            best_test_f1 = test_f1
            best_hyperparams = {'learning_rate': lr, 'num_epochs': num_epochs}

print(f"最佳超参数：{best_hyperparams}，测试集 F1 分数：{best_test_f1}")


def train_and_save_best_model(best_hyperparams, save_path):
    best_lr = best_hyperparams['learning_rate']
    best_num_epochs = best_hyperparams['num_epochs']

    # 使用找到的最佳超参数重新训练模型
    _, model = train_model(best_lr, best_num_epochs)

    # 保存训练好的模型
    torch.save(model, save_path)
    print(f"使用最佳超参数训练的模型已保存到：{save_path}")


model_save_path = "../Test/best_model_7.pth"
train_and_save_best_model(best_hyperparams, model_save_path)
