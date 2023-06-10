label = [7, 1, 2, 3, 7, 7, 7, 7]

# Find the indices of the first and second occurrence of 7
first_index = label.index(7)
second_index = label.index(7, first_index + 1)

# Extract the desired data between the first and second occurrence of 7 (including the second occurrence)
modified_label = label[first_index:second_index + 1]

print("Modified label:", modified_label)