import os
import numpy as np
import csv
import pickle

"""
    Normalization:
        功能介绍：对数据进行归一化操作
"""
def Normalization(data):
    max_value = max(data) # 获取最大值
    min_value = min(data) # 获取最小值
    data = (data - min_value) / (max_value - min_value) # 归一化
    return data, max_value, min_value # 返回归一化后的数据, 最大值, 最小值

# ====================================================================
# 1. 构建数据集 ---> 训练集和测试集
# 1.1 加载CPF数据 数据间隔1min取一个数据点, 与SGP4数据时间间隔一致
CPF_data, SGP4_data = pickle.load(open('./CFP_SGP4_data.p', mode='rb'))
CPF_x = CPF_data['x'][::1] # 每隔1个数据 取一个数据, 得到CPF的位置x数据

# 1.2 加载SGP4
SGP4_x = np.array(SGP4_data['x']) # 得到SGP4的位置x数据

# 1.3 对数据(位置误差、速度、加速度)进行归一化操作
val_x = np.array(CPF_x) - SGP4_x # 位置误差
val_x, max_value, min_value = Normalization(val_x)
vx, _, _ = Normalization(np.array(SGP4_data['vx']))
ax, _, _ = Normalization(np.array(SGP4_data['ax']))

# 1.4 构建数据集
data = []  # 输入数据
label = [] # 真实标签
for i in range(len(val_x)):
    data.append([val_x[i], vx[i], ax[i]]) # 将位置误差, 速度, 加速度整合到一起
    label.append(val_x[i]) # 获取对应的标签数据

data = np.array(data)    # 将列表转为array
label = np.array(label)  # 将列表转为array

# 构建时间步长
time_steps = 150 # 设置时间步长为150
t_data = []      # 构建时间序列数据
t_label = []     # 构建时间序列数据对应的标签
temp = []        # 临时变量

cfp_x_original = []  # 原始CPF 位置x的数据
sgp4_x_original = [] # 原始SGP4 位置x的数据

for i in range(data.shape[0]):
    if i+time_steps < data.shape[0]:
        for j in range(time_steps):
            temp.append([data[i+j][0], data[i+j][1], data[i+j][2]])
        temp = np.array(temp)
        t_data.append(temp)
        t_label.append(data[i+time_steps][0])
        cfp_x_original.append(CPF_x[i+time_steps])
        sgp4_x_original.append(SGP4_x[i+time_steps])
        temp = []
        
t_data = np.array(t_data)   # 将列表转为array
t_label = np.array(t_label) # 将列表转为array

# 对数据集进行拆分
# (train_data, train_label)为训练数据集, (test_data, test_label)为测试数据集
train_data = t_data[:7*1440]        # 09-11 00:00:00---> 09.17 24:00:00 共7天 训练数据
train_label = t_label[:7*1440]
test_data = t_data[7*1440: 8*1440]  # 09-18 00:00:00---> 09.18:24:00:00 共1天 测试数据
test_label = t_label[7*1440: 8*1440]

print('训练集:', train_data.shape, train_label.shape)
print('测试集:', test_data.shape, test_label.shape)

# ====================================================================
# 2. 加载模型
from tensorflow.keras.models import load_model
model = load_model('./model_50_16.keras')  # 加载之前保存的训练好的模型
    
# ====================================================================
# 3. 模型预测
last_one = train_data[-1] # 取出训练数据集的最后一个数据, 预测得到测试集数据集第一个数据的位置误差
last_one = last_one.reshape(1, time_steps, 3) # 对last_one的shape进行变形, 满足模型的输入shape
predict_results = [] # 用来保存模型的预测误差结果

for i in range(test_data.shape[0]):        # 遍历测试数据集
    result = model.predict(last_one)       # 获取模型的预测误差
    
    # 利用预测处理的结果来构建下一次的输入
    temp = test_data[i]                    # 获取测试数据集的一个输入数据        
    temp[-1][0] = result[0][0]             # 将获取到的数据的位置误差的值替换为模型预测误差的值
    
    temp = temp.reshape(-1, time_steps, 3) # 对temp的shape进行变形, 满足模型的输入shape
    last_one = temp
    
    predict_results.append(result[0][0])   # 存储模型预测出来的位置误差


predict_results = np.array(predict_results) # 将列表转为array
predict_results = predict_results * (max_value - min_value) + min_value #将模型预测的位置误差结果进行反归一化
test_label = test_label * (max_value - min_value) + min_value           #将真实位置误差进行反归一化



# ====================================================================
# 4. 绘制
# 4.1 获取CPF 位置x数据
cpf_x = np.array(cfp_x_original)       # 获取CPF 位置x数据
cpf_x = cpf_x[7*1440:8*1440]           # 获取测试数据集对应的CPF 位置x的数据

# 4.2 获取SGP4 位置x数据
sgp4_x = np.array(sgp4_x_original)     # 获取SGP4 位置x数据
sgp4_x = sgp4_x[7*1440:8*1440]         # 获取测试数据集对应的SGP4 位置x的数据

# 4.3 原始位置x的误差
Val_x = cpf_x - sgp4_x                 # 没有加入LSTM之前的位置x误差

# 4.4 加上LSTM模型后的位置x的误差
sgp4_x_lstm = sgp4_x + predict_results # 原始SGP4 位置x的数据加上LSTM预测的位置误差
Val_x_lstm = cpf_x - sgp4_x_lstm       # 加上LSTM模型后的位置x的误差

import matplotlib.pyplot as plt        # 导入绘图包

# 绘制400min 原始位置x的误差与修正之后的位置x的误差
plt.figure(figsize=(8, 6))             # 设置图的大小
plt.plot(Val_x[:400])                  # 绘制原始位置x的误差
plt.plot(Val_x_lstm[:400])             # 绘制加上LSTM模型后的位置x的误差
plt.show()

# 绘制800min 原始位置x的误差与修正之后的位置x的误差
plt.figure(figsize=(8, 6))             # 设置图的大小
plt.plot(Val_x[:800])                  # 绘制原始位置x的误差
plt.plot(Val_x_lstm[:800])             # 绘制加上LSTM模型后的位置x的误差
plt.show()

# 绘制1440min 原始位置x的误差与修正之后的位置x的误差
plt.figure(figsize=(8, 6))             # 设置图的大小
plt.plot(Val_x)                        # 绘制原始位置x的误差
plt.plot(Val_x_lstm)                   # 绘制加上LSTM模型后的位置x的误差
plt.show()



# 绘制400min 修正之后的位置x的误差
plt.figure(figsize=(8, 6))             # 设置图的大小
plt.plot(Val_x_lstm[:400])             # 绘制加上LSTM模型后的位置x的误差
plt.show()

# 绘制800min 修正之后的位置x的误差
plt.figure(figsize=(8, 6))             # 设置图的大小
plt.plot(Val_x_lstm[:800])             # 绘制加上LSTM模型后的位置x的误差
plt.show()

# 绘制1440min 修正之后的位置x的误差
plt.figure(figsize=(8, 6))             # 设置图的大小
plt.plot(Val_x_lstm)                   # 绘制加上LSTM模型后的位置x的误差
plt.show()