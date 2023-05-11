import os
import pandas as pd

# 指定要读取的文件夹路径和Excel文件路径
folder_path = "C:/Users/13663/Desktop/test/arithmetic"
excel_path = "C:/Users/13663/Desktop/345.xlsx"

# 读取已有的Excel文件（如果存在）并将其内容添加到DataFrame中
if os.path.exists(excel_path):
    # 读取已有数据，并获取最后一行的行号
    existing_df = pd.read_excel(excel_path)
    last_row = existing_df.shape[0] + 1
else:
    # 如果文件不存在，从第一行开始写入数据
    last_row = 1
    existing_df = pd.DataFrame(columns=['source_code', 'slither'])

# 遍历文件夹中的所有文件
for file_name in os.listdir(folder_path):
    # 获取文件路径
    file_path = os.path.join(folder_path, file_name)
    # 如果是文件而不是文件夹，打开文件，将每行内容插入到DataFrame中
    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            content = ''.join(lines)
            content = content.replace("\n", "").replace("\r", "").replace("\t", "").replace(" ","")
            # 在每个数据对应的第一列和第二列添加数据
            existing_df.loc[last_row, 'source_code'] = content
            existing_df.loc[last_row, 'slither'] = 2
            # 更新最后一行的行号
            last_row += 1

# 将数据写入Excel文件
existing_df.to_excel(excel_path, index=False, header=True)
