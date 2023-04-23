import torch
# 从transformers调用现有的model
from transformers import BertModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 加载预训练模型，使用预训练模型可以加快训练的速度
pretrained = BertModel.from_pretrained('bert-base-uncased')
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