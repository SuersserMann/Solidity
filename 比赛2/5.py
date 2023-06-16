import random

train_dataset = [
    ("1", "这是一个示例文本。", [
        {"entity": "示例", "type": "A"},
        {"entity": "文本", "type": "B"}
    ]),
    ("2", "另一个用于测试的示例。", [
        {"entity": "另一个", "type": "C"},
        {"entity": "测试", "type": "D"}
    ])
]

entity_list = ["实体1", "实体2", "实体3"]

# 用于保存修改后的数据
modified_data = []

for item in train_dataset:
    text_id = item[0]
    text = item[1]
    events = item[2]

    entity_mapping = {}  # 用于映射原实体和新实体的对应关系

    for event in events:
        entity = event.get("entity")
        if entity and text:
            new_entity = random.choice(entity_list)
            text = text.replace(entity, new_entity)
            entity_mapping[entity] = new_entity
        else:
            entity_mapping[entity] = entity

    # 更新events中的实体
    for event in events:
        entity = event.get("entity")
        if entity:
            new_entity = entity_mapping.get(entity)
            if new_entity:
                event["entity"] = new_entity

    modified_data.append({
        "text_id": text_id,
        "text": text,
        "events": events
    })

# 输出修改后的数据
for item in modified_data:
    print(item)
