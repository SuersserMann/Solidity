import jsonlines

file1_path = 'result_单.txt'
file2_path = 'result_多.txt'

# 读取文件1的数据
file1_data = []
with jsonlines.open(file1_path, 'r') as f:
    for line in f:
        file1_data.append(line)

# 读取文件2的数据
file2_data = []
with jsonlines.open(file2_path, 'r') as f:
    for line in f:
        file2_data.append(line)

# 合并events字段并去重
merged_events = {}  # 使用字典来记录合并后的events，以text_id作为键
for data in file1_data + file2_data:
    text_id = data['text_id']
    events = data['events']
    if text_id not in merged_events:
        merged_events[text_id] = []

    # 将当前数据的events添加到合并后的列表中
    for event in events:
        event_type = event['type']
        entity = event['entity']
        if {'type': event_type, 'entity': entity} not in merged_events[text_id]:
            merged_events[text_id].append({'type': event_type, 'entity': entity})

# 构建合并后的数据列表
merged_data = [{'text_id': text_id, 'events': events} for text_id, events in merged_events.items()]

# 将合并后的数据写入新文件
output_file = 'merged_file.txt'
with jsonlines.open(output_file, 'w') as f:
    for data in merged_data:
        f.write(data)

print(f"合并后的数据已写入文件: {output_file}")
