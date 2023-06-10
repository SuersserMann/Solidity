import json


def is_subset(entity1, entity2):
    return set(entity1).issubset(entity2)


def remove_subset_entities(input_file, output_file):
    # 将输入文件中的数据加载为JSON对象
    with open(input_file, 'r', encoding='utf-8') as file:
        json_data = [json.loads(line) for line in file]
    a = 0
    # 遍历每个JSON对象
    for obj in json_data:

        events = obj['events']
        non_subset_events = []
        for i in range(len(events)):
            current_event = events[i]
            is_subset_entity = False
            for j in range(len(events)):
                if i != j:
                    other_event = events[j]
                    if current_event['type'] == other_event['type'] and is_subset(current_event['entity'],
                                                                                  other_event['entity']):
                        is_subset_entity = True
                        break
            if not is_subset_entity:
                non_subset_events.append(current_event)
        obj['events'] = non_subset_events
        if len(non_subset_events) < len(events):
            print("子集被包含的情况，text_id:", obj['text_id'], '', a)
            a += 1

    # 保存结果到输出文件
    with open(output_file, 'w', encoding='utf-8') as file:
        for obj in json_data:
            file.write(json.dumps(obj, ensure_ascii=False) + '\n')


# 调用函数执行任务
remove_subset_entities('result.txt', 'new_result.txt')
