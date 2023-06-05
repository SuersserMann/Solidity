import json

with open('cfn-test-B.json', 'r', encoding='UTF-8') as file:
    json_data3 = json.load(file)
with open('B_task2_test.json', 'r', encoding='UTF-8') as file:
    json_data2 = json.load(file)
with open('B_task1_test.json', 'r', encoding='UTF-8') as file:
    json_data1 = json.load(file)

for item in json_data1:
    sentence_id, frame = item
    for obj in json_data3:
        if obj['sentence_id'] == sentence_id:
            obj['frame'] = frame
            break

for item in json_data2:
    sentence_id, start, end = item
    for obj in json_data3:
        if obj['sentence_id'] == sentence_id:
            if 'cfn_spans' not in obj:
                obj['cfn_spans'] = []
            obj['cfn_spans'].append({"start": start, "end": end, "fe_abbr": "", "fe_name": ""})
            break

# 将更新后的json_data3写回到文件
with open('new_cfn-test-B.json', 'w', encoding='utf-8') as file:
    json.dump(json_data3, file, ensure_ascii=False)
