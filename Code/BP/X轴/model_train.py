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
    max_value = np.max(data)
    min_value = np.min(data)

    if max_value == min_value:
        data = np.zeros_like(data)
    else:
        data = (data - min_value) / (max_value - min_value)

    return data, max_value, min_value


# ====================================================================
# 1. 构建数据集（训练集和测试集）

# 1.1 加载CPF数据
CPF_data, SGP4_data = pickle.load(open('./CFP_SGP4_data.p', mode='rb'))

CPF_x = CPF_data['x'][::1]          # CPF位置x数据
SGP4_x = np.array(SGP4_data['x'])   # SGP4位置x数据

# 1.2 计算位置误差，并归一化
val_x = np.array(CPF_x) - SGP4_x    # 位置误差

val_x, max_value, min_value = Normalization(val_x)
vx, _, _ = Normalization(np.array(SGP4_data['vx']))
ax, _, _ = Normalization(np.array(SGP4_data['ax']))

# 1.3 构建基础数据
data = []
label = []

for i in range(len(val_x)):
    data.append([val_x[i], vx[i], ax[i]])
    label.append(val_x[i])

data = np.array(data)
label = np.array(label)

# ====================================================================
# 1.4 构建时间序列数据

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
# 1.5 将时间序列数据展平，适配BP神经网络

# 原始 t_data.shape: (样本数, 150, 3)
# BP模型输入需要二维数据: (样本数, 150 * 3)
bp_data = t_data.reshape(t_data.shape[0], time_steps * 3)

# ====================================================================
# 1.6 划分训练集和测试集

train_data = bp_data[:7 * 1440]
train_label = t_label[:7 * 1440]

test_data = bp_data[7 * 1440: 8 * 1440]
test_label = t_label[7 * 1440: 8 * 1440]

print('训练集:', train_data.shape, train_label.shape)
print('测试集:', test_data.shape, test_label.shape)


# ====================================================================
# 2. 构建BP神经网络模型

from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

input_layer = Input(shape=(time_steps * 3,))

# BP隐藏层
h = Dense(
    units=128,
    activation='relu',
    kernel_regularizer=l2(1e-4)
)(input_layer)

h = Dense(
    units=64,
    activation='relu',
    kernel_regularizer=l2(1e-4)
)(h)

h = Dense(
    units=32,
    activation='relu',
    kernel_regularizer=l2(1e-4)
)(h)

# 可选Dropout，防止过拟合
h = Dropout(0.2)(h)

# 输出层，回归任务，所以不用激活函数
output_layer = Dense(units=1, activation='linear')(h)

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

    print('当前测试集MSE:', mse)

    if min_mse > mse:
        print('.......saving model.......')
        model.save('./bp_model_' + str(epoch) + '_' + str(batch_size) + '.keras')
        min_mse = mse

print('训练mse:', test_mse)
print('最小测试集MSE:', min_mse)
