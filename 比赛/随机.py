import json

# 读取JSON文件
with open('B_task2_test.json', 'r') as file:
    data = json.load(file)

# 在每个列表后添加 '实体集'
for item in data:
    item.append('实体集')

# 保存为新的JSON文件
with open('B_task3_test.json', 'w',encoding='utf-8') as file:
    json.dump(data, file,ensure_ascii=False)
