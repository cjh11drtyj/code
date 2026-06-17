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
# 1. 构建数据集（训练集和测试集）
# 1.1 加载CPF数据 数据间隔1min取一个数据点, 与SGP4数据时间间隔一致
CPF_data, SGP4_data = pickle.load(open('./CFP_SGP4_data.p', mode='rb'))
CPF_z = CPF_data['z'][::1] # 每隔1个数据 取一个数据, 得到CPF的位置z数据

# 1.2 加载SGP4
SGP4_z = np.array(SGP4_data['z'])            # 得到SGP4的位置z数据

# 1.3 对数据(位置误差、速度、加速度)进行归一化操作
val_z = np.array(CPF_z) - SGP4_z             # 位置误差
val_z, max_value, min_value = Normalization(val_z)
vz, _, _ = Normalization(np.array(SGP4_data['vz']))
az, _, _ = Normalization(np.array(SGP4_data['az']))

# 1.4 构建数据集
data = []                                 # 输入数据
label = []                                # 真实标签
for i in range(len(val_z)):
    data.append([val_z[i], vz[i], az[i]]) # 将位置误差, 速度, 加速度整合到一起
    label.append(val_z[i])                # 获取对应的标签数据

data = np.array(data)                     # 将列表转为array
label = np.array(label)                   # 将列表转为array

# 构建时间步长
time_steps = 150                          # 设置时间步长为150
t_data = []                               # 构建时间序列数据
t_label = []                              # 构建时间序列数据对应的标签
temp = []                                 # 临时变量

for i in range(data.shape[0]):
    if i+time_steps < data.shape[0]:
        for j in range(time_steps):
            temp.append([data[i+j][0], data[i+j][1], data[i+j][2]])
        temp = np.array(temp)
        t_data.append(temp)
        t_label.append(data[i+time_steps][0])
        temp = []

t_data = np.array(t_data)                 # 将列表转为array
t_label = np.array(t_label)               # 将列表转为array

# 对数据集进行拆分
# (train_data, train_label)为训练数据集, (test_data, test_label)为测试数据集
train_data = t_data[:7*1440]              # 09-11 00:00:00---> 09.17 24:00:00 共7天 训练数据
train_label = t_label[:7*1440]
test_data = t_data[7*1440: 8*1440]        # 09-18 00:00:00---> 09.18:24:00:00 共1天 测试数据
test_label = t_label[7*1440: 8*1440]

print('训练集:', train_data.shape, train_label.shape)
print('测试集:', test_data.shape, test_label.shape)


# ====================================================================
#2. 构建模型
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import l2
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, precision_score, recall_score
from multi_head_attention import Multi_Head_Attention

time_steps=150                                                    # 模型的时间步长与时间序列数据步长要保持一致
alpha = 1e-2                                                      # L2正则化因子
input = Input(shape=(time_steps, 3))                              # 模型的输入层
rnn_layer = LSTM(units=64 , return_sequences=True )                                        # LSTM层
h = rnn_layer(input)
mha_output = Multi_Head_Attention()(h)                            # 多头注意力层
output = Dense(units=1, kernel_regularizer=l2(alpha))(mha_output) # 输出层 #activation='linear'
model = Model(input, output)
model.compile(optimizer=Adam(1e-3), loss='mean_squared_error', metrics=['mean_squared_error'])
model.summary()

test_mse = [] # 用于存储测试集MSE误差
epoch = 50
batch_size = 16
min_mse = 1000
if True:
    for l in range(epoch):
        print('iter ' + str(l))
        model.fit(train_data, train_label, epochs=1, batch_size=batch_size, verbose=2) # 模型训练  
        res = model.evaluate(test_data, test_label)                                    # 模型评估
        test_mse.append(res[1])                                                        # 存储mse误差
        
        if min_mse > res[1]:
            #3. 模型保存
            print('.......saving model.......')
            model.save('./lstm_attention_'+str(epoch)+'_'+str(batch_size)+'.keras')
            min_mse = res[1]
            
print('训练mes:', test_mse)






    







