device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.device_count() < 2:
    print('device=', device)
else:
    print("使用", torch.cuda.device_count(), "张GPU进行训练")


pretrained = AutoModel.from_pretrained("microsoft/codebert-base")

pretrained.to(device)

for param in pretrained.parameters():
    param.requires_grad_(False)

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.fc = torch.nn.Linear(768, 6)
    def forward(self, input_ids, attention_mask):

        out = pretrained(input_ids=input_ids,
                         attention_mask=attention_mask,
                         )

        out = self.fc(out.last_hidden_state[:, 0])

        return out

model = Model()
model.to(device)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.dataset = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split=split,
                                    verification_mode='no_checks')

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, i):
        source_code = self.dataset[i]['source_code']
        bytecode = self.dataset[i]['bytecode']
        label = self.dataset[i]['slither']
        return source_code, bytecode, label


train_dataset = Dataset('train')
len(train_dataset), train_dataset[0]
val_dataset = Dataset('validation')
test_dataset = Dataset('test')
token = AutoTokenizer.from_pretrained("microsoft/codebert-base")
if torch.cuda.device_count() > 1:
    dist.init_process_group(backend='nccl', init_method='tcp://localhost:127.0.0.1:8888', world_size=torch.cuda.device_count(), rank=0)

def collate_fn(data):
    source_codes = [delete_comment(i[0]) for i in data]
    bytecodes = [bytecode_to_opcodes(i[1]) for i in data]
    labels = [i[2] for i in data]
    amount = 0
    cutted_list = []
    cut_labels = []
    for i, cut_bytecode in enumerate(bytecodes):
        new_labels = []

        new_labels.append(cut_bytecode)
        cutted = truncate_list(new_labels, 2048)
        for gg in cutted:
            cutted_list.append(gg)

        for dd in range(len(cutted)):
            cut_labels.insert(i + amount, labels[i])
        amount += len(cutted)
    labels = cut_labels
    bytecodes = cutted_list
    # 编码
    data = token.batch_encode_plus(
        # source_codes,
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
    # token_type_ids = data['token_type_ids'].to(device)

    # 将 labels 转换为多标签格式
    labels_tensor = torch.zeros(len(labels), 6).to(device)

    for i, label in enumerate(labels):
        labels_tensor[i][label] = 1

    return input_ids, attention_mask, labels_tensor

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=4,
                                           collate_fn=collate_fn,
                                           shuffle=True,
                                           drop_last=True,
                                           sampler=train_sampler)

def train_model(learning_rate, num_epochs):
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:8888',
                                world_size=torch.cuda.device_count(),
                                rank=torch.cuda.current_device())

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_f1 = 0
    best_model_state = None

    for epoch in range(num_epochs):
        if torch.cuda.device_count() > 1:
            dist.init_process_group(backend='nccl', init_method='tcp://localhost:8888',
                                    world_size=torch.cuda.device_count(),
                                    rank=torch.cuda.current_device())
        model.train()
        for i, (input_ids, attention_mask, labels) in enumerate(train_loader):

            out = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = criterion(out, labels.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            out = torch.sigmoid(out)
            out = torch.where(out > 0.5, torch.ones_like(out), torch.zeros_like(out))  # 找到概率大于0.5的位置，并将其设为1，否则设为0
            predicted_labels = []
            True_labels = []
            f2 = 0
            for j in range(len(out)):
                predicted_label = torch.where(out[j] == 1)[0].tolist()
                predicted_labels.append(predicted_label)
                True_label = torch.where(labels[j] == 1)[0].tolist()
                True_labels.append(True_label)


                predicted_set = set(predicted_label)
                true_set = set(True_label)

                TP = len(predicted_set.intersection(true_set))
                FP = len(predicted_set - true_set)
                FN = len(true_set - predicted_set)
                precision = TP / (TP + FP) if TP + FP else 0
                recall = TP / (TP + FN) if TP + FN else 0
                f1 = calculate_f1(precision, recall)

                f2 = f1 + f2
            f2 = f2 / len(out)

            if i % 10 == 0:
                print(f"predicted_labels：{predicted_labels}", '\n', f"True_labels：{True_labels}")
                print(f"第{i}轮训练, loss：{loss.item()}, 第{i}次训练集F1准确率为:{f2}")
        model.eval()

        with torch.no_grad():
            for i, (input_ids, attention_mask, labels) in enumerate(val_loader):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(out, labels.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                out = torch.sigmoid(out)
                out = torch.where(out > 0.5, torch.ones_like(out), torch.zeros_like(out))  # 找到概率大于0.5的位置，并将其设为1，否则设为0
                predicted_labels = []
                True_labels = []

                for j in range(len(out)):
                    predicted_label = torch.where(out[j] == 1)[0].tolist()
                    predicted_labels.append(predicted_label)
                    True_label = train_dataset[i * 5 + j][2]
                    True_label = sorted(True_label)
                    True_labels.append(True_label)
                    predicted_set = set(predicted_label)
                    true_set = set(True_label)

                    TP = len(predicted_set.intersection(true_set))
                    FP = len(predicted_set - true_set)
                    FN = len(true_set - predicted_set)
                    precision = TP / (TP + FP) if TP + FP else 0
                    recall = TP / (TP + FN) if TP + FN else 0
                    f1 = calculate_f1(precision, recall)

                    f2 = f1 + f2
                average_val_f1 = f2 / len(out)
                if i % 10 == 0:
                    print(f"predicted_labels：{predicted_labels}", '\n', f"True_labels：{True_labels}")
                    print(f"第{i}轮训练, loss：{loss.item()}, 第{i}次验证集F1准确率为:{average_val_f1}")


        if average_val_f1 > best_val_f1:
            best_val_f1 = average_val_f1
            best_model_state = model.state_dict()


    model.load_state_dict(best_model_state)


    model.eval()

    with torch.no_grad():
        for i, (input_ids, attention_mask, labels) in enumerate(test_loader):

            out = torch.sigmoid(out)
            out = torch.where(out > 0.5, torch.ones_like(out), torch.zeros_like(out))  # 找到概率大于0.5的位置，并将其设为1，否则设为0
            predicted_labels = []
            True_labels = []

            for j in range(len(out)):
                predicted_label = torch.where(out[j] == 1)[0].tolist()
                predicted_labels.append(predicted_label)
                True_label = train_dataset[i * 5 + j][2]
                True_label = sorted(True_label)
                True_labels.append(True_label)

                predicted_set = set(predicted_label)
                true_set = set(True_label)

                TP = len(predicted_set.intersection(true_set))
                FP = len(predicted_set - true_set)
                FN = len(true_set - predicted_set)
                precision = TP / (TP + FP) if TP + FP else 0
                recall = TP / (TP + FN) if TP + FN else 0
                f1 = calculate_f1(precision, recall)

                f2 = f1 + f2
            average_test_f1 = f2 / len(out)

            if i % 10 == 0:
                print(f"predicted_labels：{predicted_labels}", '\n', f"True_labels：{True_labels}")
                print(f"第{i}轮训练, loss：{loss.item()}, 第{i}次测试集F1准确率为:{average_test_f1}")

    print(f"测试集 F1 分数：{average_test_f1}")

    return average_test_f1


learning_rates = [1e-5, 3e-5, 1e-4]
num_epochs_list = [1, 2, 3]

best_hyperparams = None
best_test_f1 = 0

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

    model = train_model(best_lr, best_num_epochs)

    torch.save(model.state_dict(), save_path)
    print(f"使用最佳超参数训练的模型已保存到：{save_path}")


model_save_path = "../Nlp/best_model.pth"
train_and_save_best_model(best_hyperparams, model_save_path)


