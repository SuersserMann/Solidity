import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base")

# 定义文本
text = "hello world"

# 使用tokenizer对文本进行编码
inputs = tokenizer(text, return_tensors="pt")

# 对编码后的文本进行预测
outputs = model(**inputs)
predictions = outputs

# 输出预测结果
print(predictions)
