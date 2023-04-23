import torch
import numpy as np

labels = [[1, 2], [1], [1, 5]]

labels_tensor = torch.zeros(len(labels), 6)
for i, label in enumerate(labels):
    labels_tensor[i][label] = 1
print(labels_tensor)

out = torch.sigmoid(labels_tensor)  # 将预测值转化为概率
print(out)

out = torch.where(out > 0.5, torch.ones_like(out), torch.zeros_like(out))  # 找到概率大于0.5的位置，并将其设为1，否则设为0
predicted_labels = []
for j in range(len(out)):
    predicted_label = torch.where(out[j] == 1)[0].tolist()  # 将位置索引转换为标签
    predicted_labels.append(predicted_label)
print(predicted_labels)
