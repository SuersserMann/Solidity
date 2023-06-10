label=[1,0,1,2,1,0,1,1,2,2,0,1,2,1]
def get_value(label):
    intervals = []

    start_index = None
    for i, tag in enumerate(label):
        if tag == 1:  # 开始区间
            if i + 1 < len(label) and label[i + 1] == 2:  # 开头是1且后面是2
                start_index = i
        elif tag != 2:  # 非中间或结束区间
            if start_index is not None:
                intervals.append([start_index, i - 1])
                start_index = None

    # 处理最后一个区间
    if start_index is not None and label[start_index] == 1:
        intervals.append([start_index, len(label) - 1])
    return intervals
print(get_value(label))