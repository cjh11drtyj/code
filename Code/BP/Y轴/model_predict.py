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
CPF_y = CPF_data['y'][::1]

# 1.2 加载SGP4
SGP4_y = np.array(SGP4_data['y'])

# 1.3 对数据进行归一化
val_y = np.array(CPF_y) - SGP4_y  # Y轴位置误差

val_y, max_value, min_value = Normalization(val_y)
vy, _, _ = Normalization(np.array(SGP4_data['vy']))
ay, _, _ = Normalization(np.array(SGP4_data['ay']))

# 1.4 构建基础数据
data = []
label = []

for i in range(len(val_y)):
    data.append([val_y[i], vy[i], ay[i]])
    label.append(val_y[i])

data = np.array(data)
label = np.array(label)


# ====================================================================
# 1.5 构建时间序列数据

time_steps = 150

t_data = []
t_label = []
temp = []

cfp_y_original = []
sgp4_y_original = []

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

        cfp_y_original.append(CPF_y[i + time_steps])
        sgp4_y_original.append(SGP4_y[i + time_steps])

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

# 注意：这里加载的是Y轴BP模型
model = load_model('./bp_y_model_50_16.keras')

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

    # 预测归一化后的Y轴位置误差
    result = model.predict(bp_input, verbose=0)

    pred_error = result[0][0]

    predict_results.append(pred_error)

    # 利用预测结果构造下一次输入
    temp = test_data_seq[i].copy()

    # 将当前位置误差替换为BP模型预测出来的Y轴位置误差
    temp[-1][0] = pred_error

    # 下一轮预测使用更新后的输入
    last_one = temp


predict_results = np.array(predict_results)

# 反归一化
predict_results = predict_results * (max_value - min_value) + min_value
test_label = test_label * (max_value - min_value) + min_value


# ====================================================================
# 4. 绘制结果

# 4.1 获取CPF位置y数据
cpf_y = np.array(cfp_y_original)
cpf_y = cpf_y[7 * 1440: 8 * 1440]

# 4.2 获取SGP4位置y数据
sgp4_y = np.array(sgp4_y_original)
sgp4_y = sgp4_y[7 * 1440: 8 * 1440]

# 4.3 原始SGP4位置y误差
Val_y = cpf_y - sgp4_y

# 4.4 加上BP模型修正后的SGP4位置y
sgp4_y_bp = sgp4_y + predict_results

# 4.5 BP修正后的Y轴位置误差
Val_y_bp = cpf_y - sgp4_y_bp


# ====================================================================
# 5. 绘图

import matplotlib.pyplot as plt

# 绘制400min 原始位置y误差与BP修正后误差
plt.figure(figsize=(8, 6))
plt.plot(Val_y[:400], label='Original Error')
plt.plot(Val_y_bp[:400], label='BP Corrected Error')
plt.legend()
plt.title('400 min Y Position Error')
plt.xlabel('Time / min')
plt.ylabel('Y Error')
plt.show()

# 绘制800min 原始位置y误差与BP修正后误差
plt.figure(figsize=(8, 6))
plt.plot(Val_y[:800], label='Original Error')
plt.plot(Val_y_bp[:800], label='BP Corrected Error')
plt.legend()
plt.title('800 min Y Position Error')
plt.xlabel('Time / min')
plt.ylabel('Y Error')
plt.show()

# 绘制1440min 原始位置y误差与BP修正后误差
plt.figure(figsize=(8, 6))
plt.plot(Val_y, label='Original Error')
plt.plot(Val_y_bp, label='BP Corrected Error')
plt.legend()
plt.title('1440 min Y Position Error')
plt.xlabel('Time / min')
plt.ylabel('Y Error')
plt.show()


# ====================================================================
# 6. 单独绘制BP修正后的误差

plt.figure(figsize=(8, 6))
plt.plot(Val_y_bp[:400], label='BP Corrected Error')
plt.legend()
plt.title('400 min BP Corrected Y Error')
plt.xlabel('Time / min')
plt.ylabel('Y Error')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(Val_y_bp[:800], label='BP Corrected Error')
plt.legend()
plt.title('800 min BP Corrected Y Error')
plt.xlabel('Time / min')
plt.ylabel('Y Error')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(Val_y_bp, label='BP Corrected Error')
plt.legend()
plt.title('1440 min BP Corrected Y Error')
plt.xlabel('Time / min')
plt.ylabel('Y Error')
plt.show()
