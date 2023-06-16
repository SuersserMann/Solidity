import json

# 读取数据文件
with open('filtered_data.txt', 'r',encoding='utf-8') as file:
    data = file.readlines()

result = []
prev_event = None

for i in range(0, len(data), 2):
    # 解析JSON数据
    json_data1 = json.loads(data[i])
    json_data2 = json.loads(data[i+1])

    # 获取事件列表
    events1 = json_data1['events']
    events2 = json_data2['events']

    # 比较事件列表并去除相同事件
    unique_events = []
    for event1 in events1:
        if event1 not in events2:
            unique_events.append(event1)
    for event2 in events2:
        if event2 not in events1:
            unique_events.append(event2)

    # 添加新的数据到结果列表
    if unique_events:
        json_data1['events'] = unique_events
        result.append(json_data1)

    # 更新前一个事件列表
    prev_event = unique_events

# 将结果写入新文件
with open('去掉顺序不同.txt', 'w',encoding='utf-8') as file:
    for item in result:
        file.write(json.dumps(item) + '\n')
