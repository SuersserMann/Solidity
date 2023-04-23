amount = 0
cutted_list = []
for i, cut_bytecode in enumerate(bytecodes):
    new_labels = []
    if len(bytecodes[i]) / 2048 == 0:
        z = len(bytecodes[i]) // 2048
    else:
        z = len(bytecodes[i]) // 2048 + 1
    amount = i + z
    new_labels.append(cut_bytecode)
    cutted = truncate_list(new_labels, 2048)
    for gg in (cutted):
        cutted_list.append(gg)
    for dd in range(z):
        labels.insert(i + 1, labels[i])