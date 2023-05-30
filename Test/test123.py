print("请输入多行文本：")
input_lines = []
while True:
    input_line = input()
    if input_line.strip() == "":
        break
    input_lines.append(input_line)
input_text = "\n".join(input_lines)
predicted_label = input_text
print(f"Predicted slither label: {predicted_label}")