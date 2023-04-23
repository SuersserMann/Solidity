import torch
from Dataset_class import Dataset
from somefunction import collate_fn

train_dataset = Dataset('train')
# 只截取训练集
len(train_dataset), train_dataset[0]
val_dataset = Dataset('validation')
test_dataset = Dataset('test')
# 数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                     batch_size=16,  # 是每个批次的大小，也就是每次处理的样本数量。
                                     collate_fn=collate_fn,  # 是一个函数，用于对每个批次中的样本进行编码和处理。
                                     shuffle=True,  # 是一个布尔值，表示是否对数据进行随机重排。
                                     drop_last=True)  # 是一个布尔值，表示是否在最后一个批次中舍弃不足一个批次大小的数据

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                     batch_size=16,  # 是每个批次的大小，也就是每次处理的样本数量。
                                     collate_fn=collate_fn,  # 是一个函数，用于对每个批次中的样本进行编码和处理。
                                     shuffle=False,  # 是一个布尔值，表示是否对数据进行随机重排。
                                     drop_last=True)  # 是一个布尔值，表示是否在最后一个批次中舍弃不足一个批次大小的数据

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                     batch_size=16,  # 是每个批次的大小，也就是每次处理的样本数量。
                                     collate_fn=collate_fn,  # 是一个函数，用于对每个批次中的样本进行编码和处理。
                                     shuffle=False,  # 是一个布尔值，表示是否对数据进行随机重排。
                                     drop_last=True)  # 是一个布尔值，表示是否在最后一个批次中舍弃不足一个批次大小的数据