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

    pred_label_idx = torch.argmax(out, dim=1).item()
    pred_label = slither[pred_label_idx]
    return pred_label


while True:
    print("请选择输入类型：")
    print("1. Source Code")
    print("2. Bytecode")
    print("3. 退出")
    choice = input("输入选项（1/2/3）：")

    if choice == '1':
        input_text = input("请输入Source Code：")
    elif choice == '2':
        input_text = input("请输入Bytecode：")
    elif choice == '3':
        print("退出程序")
        break
    else:
        print("无效的输入，请重新输入。")
        continue

    predicted_label = predict_slither_label(input_text)
    print(f"Predicted slither label: {predicted_label}\n")