import os
import numpy as np
import csv
import pickle

"""
    Normalization:
        功能介绍：对数据进行归一化操作
"""
def Normalization(data):
    max_value = max(data)  # 获取最大值
    min_value = min(data)  # 获取最小值
    data = (data - min_value) / (max_value - min_value)  # 归一化
    return data, max_value, min_value


# ====================================================================
# 1. 构建数据集 ---> 训练集和测试集

# 1.1 加载CPF数据
CPF_data, SGP4_data = pickle.load(open('./CFP_SGP4_data.p', mode='rb'))
CPF_x = CPF_data['x'][::1]

# 1.2 加载SGP4
SGP4_x = np.array(SGP4_data['x'])

# 1.3 对数据进行归一化
val_x = np.array(CPF_x) - SGP4_x  # 位置误差

val_x, max_value, min_value = Normalization(val_x)
vx, _, _ = Normalization(np.array(SGP4_data['vx']))
ax, _, _ = Normalization(np.array(SGP4_data['ax']))

# 1.4 构建基础数据
data = []
label = []

for i in range(len(val_x)):
    data.append([val_x[i], vx[i], ax[i]])
    label.append(val_x[i])

data = np.array(data)
label = np.array(label)


# ====================================================================
# 1.5 构建时间序列数据

time_steps = 150

t_data = []
t_label = []
temp = []

cfp_x_original = []
sgp4_x_original = []

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

t_data = np.array(t_data)      # shape: (样本数, 150, 3)
t_label = np.array(t_label)    # shape: (样本数,)


# ====================================================================
# 1.6 BP模型需要将输入展平

# LSTM输入: (样本数, 150, 3)
# BP输入:   (样本数, 150 * 3)
bp_data = t_data.reshape(t_data.shape[0], time_steps * 3)


# ====================================================================
# 1.7 划分训练集和测试集

train_data = bp_data[:7 * 1440]
train_label = t_label[:7 * 1440]

test_data = bp_data[7 * 1440: 8 * 1440]
test_label = t_label[7 * 1440: 8 * 1440]

# 为了后面递推预测时方便替换位置误差，保留三维形式的数据
train_data_seq = t_data[:7 * 1440]
test_data_seq = t_data[7 * 1440: 8 * 1440]

print('BP训练集:', train_data.shape, train_label.shape)
print('BP测试集:', test_data.shape, test_label.shape)


# ====================================================================
# 2. 加载BP模型

from tensorflow.keras.models import load_model

# 注意：这里加载的是BP模型，不是LSTM模型
model = load_model('./bp_model_50_16.keras')

model.summary()


# ====================================================================
# 3. BP模型预测

# 取训练集最后一个时间窗口
# shape: (150, 3)
last_one = train_data_seq[-1].copy()

predict_results = []

for i in range(test_data_seq.shape[0]):

    # BP模型输入必须是二维: (1, 150 * 3)
    bp_input = last_one.reshape(1, time_steps * 3)

    # 预测归一化后的位置误差
    result = model.predict(bp_input, verbose=0)

    pred_error = result[0][0]

    predict_results.append(pred_error)

    # 利用预测结果构造下一次输入
    # 这里保持你原来LSTM代码的递推方式
    temp = test_data_seq[i].copy()

    # 将当前位置误差替换为BP模型预测出来的位置误差
    temp[-1][0] = pred_error

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

# 4.3 原始SGP4位置x误差
Val_x = cpf_x - sgp4_x

# 4.4 加上BP模型修正后的位置x误差
sgp4_x_bp = sgp4_x + predict_results
Val_x_bp = cpf_x - sgp4_x_bp


# ====================================================================
# 5. 绘图

import matplotlib.pyplot as plt

# 绘制400min 原始位置x误差与BP修正后误差
plt.figure(figsize=(8, 6))
plt.plot(Val_x[:400], label='Original Error')
plt.plot(Val_x_bp[:400], label='BP Corrected Error')
plt.legend()
plt.title('400 min X Position Error')
plt.xlabel('Time / min')
plt.ylabel('X Error')
plt.show()

# 绘制800min 原始位置x误差与BP修正后误差
plt.figure(figsize=(8, 6))
plt.plot(Val_x[:800], label='Original Error')
plt.plot(Val_x_bp[:800], label='BP Corrected Error')
plt.legend()
plt.title('800 min X Position Error')
plt.xlabel('Time / min')
plt.ylabel('X Error')
plt.show()

# 绘制1440min 原始位置x误差与BP修正后误差
plt.figure(figsize=(8, 6))
plt.plot(Val_x, label='Original Error')
plt.plot(Val_x_bp, label='BP Corrected Error')
plt.legend()
plt.title('1440 min X Position Error')
plt.xlabel('Time / min')
plt.ylabel('X Error')
plt.show()


# ====================================================================
# 6. 单独绘制BP修正后的误差

plt.figure(figsize=(8, 6))
plt.plot(Val_x_bp[:400], label='BP Corrected Error')
plt.legend()
plt.title('400 min BP Corrected X Error')
plt.xlabel('Time / min')
plt.ylabel('X Error')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(Val_x_bp[:800], label='BP Corrected Error')
plt.legend()
plt.title('800 min BP Corrected X Error')
plt.xlabel('Time / min')
plt.ylabel('X Error')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(Val_x_bp, label='BP Corrected Error')
plt.legend()
plt.title('1440 min BP Corrected X Error')
plt.xlabel('Time / min')
plt.ylabel('X Error')
plt.show()
