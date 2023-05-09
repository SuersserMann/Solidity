import os
import pandas as pd
import xlsxwriter

# 指定要读取的文件夹路径和Excel文件路径
folder_path = "C:/Users/13663/Desktop/tx-origin"
excel_path = "C:/Users/13663/Desktop/234.xlsx"

# 创建一个空的DataFrame，用于存储所有文件的内容
df = pd.DataFrame(columns=['content'])

# 读取已有的Excel文件（如果存在）并将其内容添加到DataFrame中
if os.path.exists(excel_path):
    existing_df = pd.read_excel(excel_path)
    df = pd.concat([df, existing_df], ignore_index=True)

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
            # 在每个数据对应的第二列添加数字"1"
            df = pd.concat([df, pd.DataFrame({'content': [content]})], ignore_index=True)
            df.loc[df.index[-1], 'number'] = 1

# 创建一个Excel文件写入器
writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')

# 将DataFrame中的内容写入Excel表格的第一列和第二列
df.to_excel(writer, sheet_name='Sheet1', startrow=0, startcol=0, index=False, header=False)

# 保存并关闭Excel文件
writer.close()
