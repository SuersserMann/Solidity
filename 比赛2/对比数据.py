import json
import jsonlines

text_ids_list = []

# 读取txt文件，获取text_id列表
with open('result_v.txt', 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line.strip())
        text_id = data["text_id"]
        text_ids_list.append(text_id)

# 读取jsonl文件，根据text_id进行过滤，并保存到新的txt文件
filtered_data = []
with jsonlines.open('dev.jsonl', 'r') as reader:
    with open('filtered_data.txt', 'w', encoding='utf-8') as writer:
        for item in reader:
            text_id = item['text_id']
            if text_id in text_ids_list:
                filtered_data.append(item)
                writer.write(json.dumps(item, ensure_ascii=False) + '\n')

# 打印或保存不完全相等的数据到filtered_data.txt
with open('result_v.txt', 'r', encoding='utf-8') as file1, open('filtered_data.txt', 'r', encoding='utf-8') as file2:
    original_data = [json.loads(line) for line in file1]
    filtered_data = [json.loads(line) for line in file2]

different_events_data = []
for original_item, filtered_item in zip(original_data, filtered_data):
    original_events = sorted(original_item['events'], key=lambda x: json.dumps(x, sort_keys=True))
    filtered_events = sorted(filtered_item['events'], key=lambda x: json.dumps(x, sort_keys=True))
    if original_events != filtered_events:
        different_events_data.append(filtered_item)
        different_events_data.append(original_item)
        different_events_data.append('')

# 将不完全相等的数据保存到filtered_data.txt文件
with open('filtered_data.txt', 'w', encoding='utf-8') as file:
    for item in different_events_data:
        file.write(json.dumps(item, ensure_ascii=False) + '\n')

# 打印不完全相等的数据
for item in different_events_data:
    print(item)
