json_data = [{
    "sentence_id": 2611,
    "cfn_spans": [
        {"start": 0, "end": 2, "fe_abbr": "ent_1", "fe_name": "实体1"},
        {"start": 4, "end": 12, "fe_abbr": "ent_2", "fe_name": "实体2"}
    ],
    "frame": "等同",
    "target": {"start": 3, "end": 4, "pos": "v"},
    "text": "餐饮业是天津市在海外投资的重点之一。",
    "word": [
        {"start": 0, "end": 2, "pos": "n"},
        {"start": 3, "end": 4, "pos": "v"},
        {"start": 5, "end": 6, "pos": "nz"},
        {"start": 7, "end": 7, "pos": "p"},
        {"start": 8, "end": 9, "pos": "n"},
        {"start": 10, "end": 11, "pos": "v"},
        {"start": 12, "end": 12, "pos": "u"},
        {"start": 13, "end": 14, "pos": "n"},
        {"start": 15, "end": 16, "pos": "n"},
        {"start": 17, "end": 17, "pos": "wp"}
    ]
}]

for item in json_data:
    cfn_spans = item["cfn_spans"]
    word_list = item["word"]
    cfn_spans_start = [span["start"] for span in cfn_spans]
    cfn_spans_end = [span["end"] for span in cfn_spans]
    target_start = item["target"]["start"]
    target_end = item["target"]["end"]
    text = item["text"]

    pos_list = []
    for word in word_list:
        if any(start <= word["start"] <= end for start, end in zip(cfn_spans_start, cfn_spans_end)):
            pos_list.append(word["pos"])
        else:
            pos_list.append('0')
    target_start_index = None
    target_end_index = None
    for i, word in enumerate(word_list):
        if word["start"] == target_start:
            target_start_index = i
        if word["end"] == target_end:
            target_end_index = i

    target_text = [str(num) for num in text[target_start:target_end + 1]]
    pos_list[target_start_index:target_end_index+1] = list(target_text)

    pos_list = [str(pos) for pos in pos_list]

    print(pos_list)
    print(text[target_start:target_end + 1])