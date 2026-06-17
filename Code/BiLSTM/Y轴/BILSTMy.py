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
# 1. 构建数据集（训练集和测试集）

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
# 2. 构建BiLSTM模型

from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

time_steps = 150
alpha = 1e-2

input_layer = Input(shape=(time_steps, 3))

# BiLSTM层
h = Bidirectional(
    LSTM(
        units=64,
        return_sequences=False
    )
)(input_layer)

# 输出层，回归任务
output_layer = Dense(
    units=1,
    kernel_regularizer=l2(alpha)
)(h)

model = Model(input_layer, output_layer)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='mean_squared_error',
    metrics=['mean_squared_error']
)

model.summary()


# ====================================================================
# 3. 模型训练

test_mse = []

epoch = 50
batch_size = 16
min_mse = 1000

for l in range(epoch):
    print('iter ' + str(l))

    model.fit(
        train_data,
        train_label,
        epochs=1,
        batch_size=batch_size,
        verbose=2
    )

    res = model.evaluate(
        test_data,
        test_label,
        verbose=0
    )

    mse = res[1]
    test_mse.append(mse)

    print('当前Y轴测试集MSE:', mse)

    if min_mse > mse:
        print('.......saving Y-axis BiLSTM model.......')
        model.save('./bilstm_y_model_' + str(epoch) + '_' + str(batch_size) + '.keras')
        min_mse = mse

print('Y轴训练mse:', test_mse)
print('Y轴最小测试集MSE:', min_mse)
