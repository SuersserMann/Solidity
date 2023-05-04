import torch
from transformers import BertModel, BertTokenizer
from transformers import BertTokenizer, AutoModel, AutoTokenizer

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # 定义一个全连接层，输入维度为768，输出维度为6
        # self.fc = torch.nn.Linear(768, 3)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(768, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 3)) # 添加多层神经网络

    def forward(self, input_ids, attention_mask):
        # 将输入传入预训练模型，并记录计算图以计算梯度
        out = pretrained(input_ids=input_ids,
                         attention_mask=attention_mask,
                         )
        # 只保留预训练模型输出的最后一个隐藏状态，并通过全连接层进行分类
        out = self.fc(out.last_hidden_state[:, 0])

        return out


# 加载预训练模型和分词器
pretrained = AutoModel.from_pretrained("microsoft/codebert-base")
token = AutoTokenizer.from_pretrained("microsoft/codebert-base")

# 加载训练好的模型
model = Model()
model.load_state_dict(torch.load("best_model_3.pth",map_location=torch.device('cpu')))
model.eval()

slither = {0: 'reentrancy',1: 'access control',2: 'arithmetic'}


def predict_slither_label(text):
    inputs = token.encode_plus(text, return_tensors='pt', padding='max_length', max_length=512, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)

    out = torch.sigmoid(out)  # 将预测值转化为概率
    #out = torch.where(out > 0.3, torch.ones_like(out), torch.zeros_like(out))  # 找到概率大于0.5的位置，并将其设为1，否则设为0
    predicted_labels = []
    for j in range(len(out)):
        label_probs = {}
        for i in range(out.shape[1]):
            label_probs[slither[i]] = out[j][i].item()
        predicted_labels.append(label_probs)

    return predicted_labels

while True:

    input_text = input("请输入数据：").replace("\n","").replace("\r","").replace("\t","")
    print(input_text)
    predicted_label = predict_slither_label(input_text)
    print(f"Predicted slither label: {predicted_label}")