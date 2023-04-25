import torch
from transformers import BertModel, BertTokenizer


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 6)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)

        out = self.fc(out.last_hidden_state[:, 0])
        out = out.softmax(dim=1)
        return out


# 加载预训练模型和分词器
pretrained = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 加载训练好的模型
model = Model()
model.load_state_dict(torch.load("../Nlp/model.pth"))
model.eval()

slither = {0: 'access-control', 1: 'arithmetic', 2: 'other', 3: 'reentrancy', 4: 'safe', 5: 'unchecked-calls'}


def predict_slither_label(text):
    inputs = tokenizer(text, return_tensors='pt', padding='max_length', max_length=500, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

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

    input_text = input("请输入数据：")
    predicted_label = predict_slither_label(input_text)
    print(f"Predicted slither label: {predicted_label}\n")