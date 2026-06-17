import os
import numpy as np
import csv
import pickle

"""
    Normalization:
        功能介绍：对数据进行归一化操作
"""
def Normalization(data):
    data = np.array(data, dtype=np.float32)
    max_value = max(data)
    min_value = min(data)

    if max_value == min_value:
        data = np.zeros_like(data)
    else:
        data = (data - min_value) / (max_value - min_value)

    return data, max_value, min_value


# ====================================================================
# 1. 构建数据集 ---> 训练集和测试集

# 1.1 加载CPF数据
CPF_data, SGP4_data = pickle.load(open('./CFP_SGP4_data.p', mode='rb'))
CPF_y = CPF_data['y'][::1]  # 获取CPF的位置y数据

# 1.2 加载SGP4数据
SGP4_y = np.array(SGP4_data['y'])  # 获取SGP4的位置y数据

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

t_data = np.array(t_data)
t_label = np.array(t_label)


# ====================================================================
# 1.6 划分训练集和测试集

train_data = t_data[:7 * 1440]
train_label = t_label[:7 * 1440]

test_data = t_data[7 * 1440: 8 * 1440]
test_label = t_label[7 * 1440: 8 * 1440]

print('Y轴BiLSTM训练集:', train_data.shape, train_label.shape)
print('Y轴BiLSTM测试集:', test_data.shape, test_label.shape)


# ====================================================================
# 2. 加载BiLSTM模型

from tensorflow.keras.models import load_model

# 加载Y轴BiLSTM模型
model = load_model('./bilstm_y_model_50_16.keras')

model.summary()


# ====================================================================
# 3. BiLSTM模型预测

# 取训练集最后一个时间窗口
last_one = train_data[-1]

# 调整为模型输入格式: (1, 150, 3)
last_one = last_one.reshape(1, time_steps, 3)

predict_results = []

for i in range(test_data.shape[0]):

    # 预测归一化后的Y轴位置误差
    result = model.predict(last_one, verbose=0)

    pred_error = result[0][0]

    predict_results.append(pred_error)

    # 利用预测结果构建下一次输入
    temp = test_data[i].copy()

    # 将当前位置误差替换为BiLSTM模型预测出来的Y轴位置误差
    temp[-1][0] = pred_error

    # 调整为模型输入格式
    temp = temp.reshape(1, time_steps, 3)

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

# 4.3 原始Y轴位置误差
Val_y = cpf_y - sgp4_y

# 4.4 加上BiLSTM模型修正后的SGP4位置y
sgp4_y_bilstm = sgp4_y + predict_results

# 4.5 加上BiLSTM模型后的Y轴位置误差
Val_y_bilstm = cpf_y - sgp4_y_bilstm


# ====================================================================
# 5. 绘图

import matplotlib.pyplot as plt

# 绘制400min 原始Y轴误差与BiLSTM修正后误差
plt.figure(figsize=(8, 6))
plt.plot(Val_y[:400], label='Original Error')
plt.plot(Val_y_bilstm[:400], label='BiLSTM Corrected Error')
plt.legend()
plt.title('400 min Y Position Error')
plt.xlabel('Time / min')
plt.ylabel('Y Error')
plt.show()

# 绘制800min 原始Y轴误差与BiLSTM修正后误差
plt.figure(figsize=(8, 6))
plt.plot(Val_y[:800], label='Original Error')
plt.plot(Val_y_bilstm[:800], label='BiLSTM Corrected Error')
plt.legend()
plt.title('800 min Y Position Error')
plt.xlabel('Time / min')
plt.ylabel('Y Error')
plt.show()

# 绘制1440min 原始Y轴误差与BiLSTM修正后误差
plt.figure(figsize=(8, 6))
plt.plot(Val_y, label='Original Error')
plt.plot(Val_y_bilstm, label='BiLSTM Corrected Error')
plt.legend()
plt.title('1440 min Y Position Error')
plt.xlabel('Time / min')
plt.ylabel('Y Error')
plt.show()


# ====================================================================
# 6. 单独绘制BiLSTM修正后的Y轴误差

plt.figure(figsize=(8, 6))
plt.plot(Val_y_bilstm[:400], label='BiLSTM Corrected Error')
plt.legend()
plt.title('400 min BiLSTM Corrected Y Error')
plt.xlabel('Time / min')
plt.ylabel('Y Error')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(Val_y_bilstm[:800], label='BiLSTM Corrected Error')
plt.legend()
plt.title('800 min BiLSTM Corrected Y Error')
plt.xlabel('Time / min')
plt.ylabel('Y Error')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(Val_y_bilstm, label='BiLSTM Corrected Error')
plt.legend()
plt.title('1440 min BiLSTM Corrected Y Error')
plt.xlabel('Time / min')
plt.ylabel('Y Error')
plt.show()
