import pandas as pd
import matplotlib.pyplot as plt

data1 = pd.read_excel(r'C:\Users\13663\Desktop\work.xlsx', sheet_name='Sheet1')
data2 = pd.read_excel(r'C:\Users\13663\Desktop\work.xlsx', sheet_name='×系数')
data3 = pd.read_excel(r'C:\Users\13663\Desktop\work.xlsx', sheet_name='阈值')

AL3 = 108 #固定数值
AL5 = 14074 #固定数值
AL4 = AL5 - AL3 #固定数值
AN2 = 0.00767372459855052 #固定数值
AN3 = 40 #固定数值

c_list = data2['y'].tolist() #读取c行数据保存在c_list
af_list= data2['Xscore'].tolist() #读取Xscore行数据保存在af_list

AN4_list = [] #初始化p1的list
AN5_list = []#初始化p2的list
AL_list = [] #初始化阈值的list

for AL in [x / 100 for x in range(-1000, 1001)]: #让阈值从-10到10之间循环
    ag_list = []
    for value in af_list:
        if value > AL:
            ag_list.append(1)
        else:
            ag_list.append(0)

    ah_list = [] #如果c列表和ag列表的数值相等，则ah列表的值为1，否则我0
    for i in range(len(c_list)):
        if c_list[i] == ag_list[i]:
            ah_list.append(1)
        else:
            ah_list.append(0)

    AL6 = 0 #如果c为0，但是ah为0，则AL6数量+1
    for i in range(len(c_list)):
        if c_list[i] == 0 and ah_list[i] == 0:
            AL6 += 1

    AL7 = 0 #如果c为1，但是ah为0，则AL7数量+1
    for i in range(len(c_list)):
        if c_list[i] == 1 and ah_list[i] == 0:
            AL7 += 1

    AL8 = 0 #如果c为0，但是ah为1，则AL8数量+1
    for i in range(len(c_list)):
        if c_list[i] == 0 and ah_list[i] == 1:
            AL8 += 1

    AL9 = 0 #如果c为1，但是ah为1，则AL8数量+1
    for i in range(len(c_list)):
        if c_list[i] == 1 and ah_list[i] == 1:
            AL9 += 1

    AN4 = AL7 / AL3
    AN5 = AL6 / AL4
    AN6 = AN2 * AN3 * AN4 + (1 - AN2) * (AL6 / AL4)

    AN4_list.append(AN4)
    AN5_list.append(AN5)
    AL_list.append(AL)


plt.plot(AL_list, AN4_list, label='AN4')
plt.plot(AL_list, AN5_list, label='AN5')


plt.title('AN4 and AN5 vs AL')
plt.xlabel('AL')
plt.ylabel('AN4 and AN5')

plt.legend()

plt.show()

