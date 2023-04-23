import torch
from Model_test import Model
from Dataset_class import Dataset
from somefunction import calculate_f1
from loader_test import train_loader,test_loader,val_loader

print(torch.__version__)
# 使用cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 如果是cuda则调用cuda反之则cpu
print('device=', device)

# 实例化下游任务模型并将其移动到 GPU 上 (如果可用)
model = Model()
model.to(device)

# 定义数据集

train_dataset = Dataset('train')
# 只截取训练集
len(train_dataset), train_dataset[0]
val_dataset = Dataset('validation')
test_dataset = Dataset('test')


# 训练
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # 定义优化器，AdamW是一种优化算法，model.parameters()返回模型中所有参数的迭代器
criterion = torch.nn.BCEWithLogitsLoss()  # 定义损失函数，交叉熵损失用于多分类任务

model.train()  # 将模型设置为训练模式

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
    print(f"predicted_labels：{predicted_labels}", '\n', f"True_labels：{True_labels}")
    print(f"第{i}轮训练, loss：{loss.item()}, 第{i}次F1准确率为:{f2}")
    # accumulate f1

    if i == 100:  # 训练500个batch后停止
        break

# 保存模型
model_save_path = "../Nlp/model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"模型已保存到：{model_save_path}")
