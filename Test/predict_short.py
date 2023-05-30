import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

print(torch.__version__)
# 使用cuda
device_ids = [0, 1]
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 如果是cuda则调用cuda反之则cpu
print('device=', device)


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.pretrained = AutoModel.from_pretrained("microsoft/codebert-base")
        self.pretrained.to(device)
        for param in self.pretrained.parameters():
            param.requires_grad_(False)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(768, 100),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(100, 3))

    def forward(self, input_ids, attention_mask):
        out = self.pretrained(input_ids=input_ids,
                              attention_mask=attention_mask,
                              )

        out = self.fc(out.last_hidden_state[:, 0])

        return out


token = AutoTokenizer.from_pretrained("microsoft/codebert-base")

# 加载训练好的模型
model = Model()
model = nn.DataParallel(model, device_ids=device_ids)
model.to(device)
model.load_state_dict(torch.load("best_model_8.pth", map_location=torch.device('cpu')))
model.eval()

slither = {0: 'reentrancy', 1: 'access control', 2: 'arithmetic'}


def predict_slither_label(text):
    inputs = token.encode_plus(text, return_tensors='pt', padding='max_length', max_length=512, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)

    out = torch.sigmoid(out)  # 将预测值转化为概率
    # out = torch.where(out > 0.3, torch.ones_like(out), torch.zeros_like(out))  # 找到概率大于0.5的位置，并将其设为1，否则设为0
    predicted_labels = []
    for j in range(len(out)):
        label_probs = {}
        for i in range(out.shape[1]):
            label_probs[slither[i]] = out[j][i].item()
        predicted_labels.append(label_probs)

    return predicted_labels


while True:
    input_text = input("请输入数据：").replace("\n", "").replace("\r", "").replace("\t", "")
    print(input_text)
    predicted_label = predict_slither_label(input_text)
    print(f"Predicted slither label: {predicted_label}")
