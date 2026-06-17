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
CPF_z = CPF_data['z'][::1]  # 获取CPF的位置z数据

# 1.2 加载SGP4数据
SGP4_z = np.array(SGP4_data['z'])  # 获取SGP4的位置z数据

# 1.3 对数据进行归一化
val_z = np.array(CPF_z) - SGP4_z  # Z轴位置误差

val_z, max_value, min_value = Normalization(val_z)
vz, _, _ = Normalization(np.array(SGP4_data['vz']))
az, _, _ = Normalization(np.array(SGP4_data['az']))

# 1.4 构建基础数据
data = []
label = []

for i in range(len(val_z)):
    data.append([val_z[i], vz[i], az[i]])
    label.append(val_z[i])

data = np.array(data)
label = np.array(label)


# ====================================================================
# 1.5 构建时间序列数据

time_steps = 150

t_data = []
t_label = []
temp = []

cfp_z_original = []
sgp4_z_original = []

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

        cfp_z_original.append(CPF_z[i + time_steps])
        sgp4_z_original.append(SGP4_z[i + time_steps])

        temp = []

t_data = np.array(t_data)
t_label = np.array(t_label)


# ====================================================================
# 1.6 划分训练集和测试集

train_data = t_data[:7 * 1440]
train_label = t_label[:7 * 1440]

test_data = t_data[7 * 1440: 8 * 1440]
test_label = t_label[7 * 1440: 8 * 1440]

print('Z轴BiLSTM训练集:', train_data.shape, train_label.shape)
print('Z轴BiLSTM测试集:', test_data.shape, test_label.shape)


# ====================================================================
# 2. 加载BiLSTM模型

from tensorflow.keras.models import load_model

# 加载Z轴BiLSTM模型
model = load_model('./bilstm_z_model_50_16.keras')

model.summary()


# ====================================================================
# 3. BiLSTM模型预测

# 取训练集最后一个时间窗口
last_one = train_data[-1]

# 调整为模型输入格式: (1, 150, 3)
last_one = last_one.reshape(1, time_steps, 3)

predict_results = []

for i in range(test_data.shape[0]):

    # 预测归一化后的Z轴位置误差
    result = model.predict(last_one, verbose=0)

    pred_error = result[0][0]

    predict_results.append(pred_error)

    # 利用预测结果构建下一次输入
    temp = test_data[i].copy()

    # 将当前位置误差替换为BiLSTM模型预测出来的Z轴位置误差
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

# 4.1 获取CPF位置z数据
cpf_z = np.array(cfp_z_original)
cpf_z = cpf_z[7 * 1440: 8 * 1440]

# 4.2 获取SGP4位置z数据
sgp4_z = np.array(sgp4_z_original)
sgp4_z = sgp4_z[7 * 1440: 8 * 1440]

# 4.3 原始Z轴位置误差
Val_z = cpf_z - sgp4_z

# 4.4 加上BiLSTM模型修正后的SGP4位置z
sgp4_z_bilstm = sgp4_z + predict_results

# 4.5 加上BiLSTM模型后的Z轴位置误差
Val_z_bilstm = cpf_z - sgp4_z_bilstm


# ====================================================================
# 5. 绘图

import matplotlib.pyplot as plt

# 绘制400min 原始Z轴误差与BiLSTM修正后误差
plt.figure(figsize=(8, 6))
plt.plot(Val_z[:400], label='Original Error')
plt.plot(Val_z_bilstm[:400], label='BiLSTM Corrected Error')
plt.legend()
plt.title('400 min Z Position Error')
plt.xlabel('Time / min')
plt.ylabel('Z Error')
plt.show()

# 绘制800min 原始Z轴误差与BiLSTM修正后误差
plt.figure(figsize=(8, 6))
plt.plot(Val_z[:800], label='Original Error')
plt.plot(Val_z_bilstm[:800], label='BiLSTM Corrected Error')
plt.legend()
plt.title('800 min Z Position Error')
plt.xlabel('Time / min')
plt.ylabel('Z Error')
plt.show()

# 绘制1440min 原始Z轴误差与BiLSTM修正后误差
plt.figure(figsize=(8, 6))
plt.plot(Val_z, label='Original Error')
plt.plot(Val_z_bilstm, label='BiLSTM Corrected Error')
plt.legend()
plt.title('1440 min Z Position Error')
plt.xlabel('Time / min')
plt.ylabel('Z Error')
plt.show()


# ====================================================================
# 6. 单独绘制BiLSTM修正后的Z轴误差

plt.figure(figsize=(8, 6))
plt.plot(Val_z_bilstm[:400], label='BiLSTM Corrected Error')
plt.legend()
plt.title('400 min BiLSTM Corrected Z Error')
plt.xlabel('Time / min')
plt.ylabel('Z Error')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(Val_z_bilstm[:800], label='BiLSTM Corrected Error')
plt.legend()
plt.title('800 min BiLSTM Corrected Z Error')
plt.xlabel('Time / min')
plt.ylabel('Z Error')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(Val_z_bilstm, label='BiLSTM Corrected Error')
plt.legend()
plt.title('1440 min BiLSTM Corrected Z Error')
plt.xlabel('Time / min')
plt.ylabel('Z Error')
plt.show()
