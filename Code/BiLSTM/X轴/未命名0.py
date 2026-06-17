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
    label.append(val_x[i])                # 获取对应的标签数据

data = np.array(data)    # 将列表转为array
label = np.array(label)  # 将列表转为array


# ====================================================================
# 1.5 构建时间序列数据

time_steps = 150 # 设置时间步长为150
t_data = []      # 构建时间序列数据
t_label = []     # 构建时间序列数据对应的标签
temp = []        # 临时变量

cfp_x_original = []  # 原始CPF 位置x的数据
sgp4_x_original = [] # 原始SGP4 位置x的数据

for i in range(data.shape[0]):
    if i + time_steps < data.shape[0]:
        for j in range(time_steps):
            temp.append([
                data[i + j][0],
                data[i + j][1],
                data[i + j][2]
            ])

        temp = np.array(temp)

        t_data.append(temp)
        t_label.append(data[i + time_steps][0])

        cfp_x_original.append(CPF_x[i + time_steps])
        sgp4_x_original.append(SGP4_x[i + time_steps])

        temp = []

t_data = np.array(t_data)   # 将列表转为array
t_label = np.array(t_label) # 将列表转为array


# ====================================================================
# 1.6 对数据集进行拆分

# (train_data, train_label)为训练数据集
# (test_data, test_label)为测试数据集

train_data = t_data[:7 * 1440]        # 7天训练数据
train_label = t_label[:7 * 1440]

test_data = t_data[7 * 1440: 8 * 1440]  # 1天测试数据
test_label = t_label[7 * 1440: 8 * 1440]

print('BiLSTM训练集:', train_data.shape, train_label.shape)
print('BiLSTM测试集:', test_data.shape, test_label.shape)


# ====================================================================
# 2. 加载BiLSTM模型

from tensorflow.keras.models import load_model

# 注意：这里加载的是BiLSTM模型
model = load_model('./bilstm_model_50_16.keras')

model.summary()


# ====================================================================
# 3. BiLSTM模型预测

# 取出训练数据集的最后一个时间窗口
# 用它预测测试集第一个时刻的位置误差
last_one = train_data[-1]

# 变形为模型输入格式: (1, 150, 3)
last_one = last_one.reshape(1, time_steps, 3)

predict_results = [] # 保存BiLSTM预测出来的位置误差

for i in range(test_data.shape[0]):

    # 预测归一化后的位置误差
    result = model.predict(last_one, verbose=0)

    pred_error = result[0][0]

    # 存储模型预测出来的位置误差
    predict_results.append(pred_error)

    # 利用预测结果构建下一次输入
    temp = test_data[i].copy()

    # 将当前位置误差替换为BiLSTM模型预测出来的位置误差
    temp[-1][0] = pred_error

    # 变形为模型输入格式: (1, 150, 3)
    temp = temp.reshape(1, time_steps, 3)

    # 下一轮预测使用更新后的输入
    last_one = temp


predict_results = np.array(predict_results)

# 反归一化
predict_results = predict_results * (max_value - min_value) + min_value
test_label = test_label * (max_value - min_value) + min_value


# ====================================================================
# 4. 绘制结果

# 4.1 获取CPF位置x数据
cpf_x = np.array(cfp_x_original)
cpf_x = cpf_x[7 * 1440: 8 * 1440]

# 4.2 获取SGP4位置x数据
sgp4_x = np.array(sgp4_x_original)
sgp4_x = sgp4_x[7 * 1440: 8 * 1440]

# 4.3 原始位置x误差
Val_x = cpf_x - sgp4_x

# 4.4 加上BiLSTM模型修正后的SGP4位置x
sgp4_x_bilstm = sgp4_x + predict_results

# 4.5 加上BiLSTM模型后的位置x误差
Val_x_bilstm = cpf_x - sgp4_x_bilstm


# ====================================================================
# 5. 绘图

import matplotlib.pyplot as plt

# 绘制400min 原始位置x误差与BiLSTM修正后误差
plt.figure(figsize=(8, 6))
plt.plot(Val_x[:400], label='Original Error')
plt.plot(Val_x_bilstm[:400], label='BiLSTM Corrected Error')
plt.legend()
plt.title('400 min X Position Error')
plt.xlabel('Time / min')
plt.ylabel('X Error')
plt.show()

# 绘制800min 原始位置x误差与BiLSTM修正后误差
plt.figure(figsize=(8, 6))
plt.plot(Val_x[:800], label='Original Error')
plt.plot(Val_x_bilstm[:800], label='BiLSTM Corrected Error')
plt.legend()
plt.title('800 min X Position Error')
plt.xlabel('Time / min')
plt.ylabel('X Error')
plt.show()

# 绘制1440min 原始位置x误差与BiLSTM修正后误差
plt.figure(figsize=(8, 6))
plt.plot(Val_x, label='Original Error')
plt.plot(Val_x_bilstm, label='BiLSTM Corrected Error')
plt.legend()
plt.title('1440 min X Position Error')
plt.xlabel('Time / min')
plt.ylabel('X Error')
plt.show()


# ====================================================================
# 6. 单独绘制BiLSTM修正后的误差

plt.figure(figsize=(8, 6))
plt.plot(Val_x_bilstm[:400], label='BiLSTM Corrected Error')
plt.legend()
plt.title('400 min BiLSTM Corrected X Error')
plt.xlabel('Time / min')
plt.ylabel('X Error')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(Val_x_bilstm[:800], label='BiLSTM Corrected Error')
plt.legend()
plt.title('800 min BiLSTM Corrected X Error')
plt.xlabel('Time / min')
plt.ylabel('X Error')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(Val_x_bilstm, label='BiLSTM Corrected Error')
plt.legend()
plt.title('1440 min BiLSTM Corrected X Error')
plt.xlabel('Time / min')
plt.ylabel('X Error')
plt.show()
