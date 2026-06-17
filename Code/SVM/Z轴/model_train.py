import os
import numpy as np
import csv
import pickle
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ====================================================================
# 1. 构建数据集（训练集和测试集）
# 1.1 加载CPF数据 数据间隔1min取一个数据点, 与SGP4数据时间间隔一致
CPF_data, SGP4_data = pickle.load(open('./CFP_SGP4_data.p', mode='rb'))
CPF_z = CPF_data['z'][::1]  # 每隔1个数据 取一个数据, 得到CPF的位置z数据

# 1.2 加载SGP4
SGP4_z = np.array(SGP4_data['z'])  # 得到SGP4的位置z数据

# 1.3 计算位置误差
val_z = np.array(CPF_z) - SGP4_z  # 位置误差
vz = np.array(SGP4_data['vz'])
az = np.array(SGP4_data['az'])

# 1.4 构建数据集
data = []  # 输入数据
label = []  # 真实标签
for i in range(len(val_z)):
    data.append([val_z[i], vz[i], az[i]])  # 将位置误差, 速度, 加速度整合到一起
    label.append(val_z[i])  # 获取对应的标签数据

data = np.array(data)  # 将列表转为array
label = np.array(label)  # 将列表转为array

# 构建时间步长
time_steps = 150  # 设置时间步长为150
t_data = []  # 构建时间序列数据
t_label = []  # 构建时间序列数据对应的标签
temp = []  # 临时变量
for i in range(data.shape[0]):
    if i + time_steps < data.shape[0]:
        for j in range(time_steps):
            temp.append([data[i + j][0], data[i + j][1], data[i + j][2]])
        temp = np.array(temp)
        t_data.append(temp)
        t_label.append(data[i + time_steps][0])
        temp = []

t_data = np.array(t_data)  # 将列表转为array
t_label = np.array(t_label)  # 将列表转为array
# 1.5 调整数据结构以适应SVM
# 将三维时间序列数据转换为二维特征矩阵
t_data_2d = t_data.reshape(t_data.shape[0], -1)  # 新形状：(n_samples, 150*3=450)
# 对数据集进行拆分

train_data = t_data_2d[:7 * 1440]  # 训练集
train_label = t_label[:7 * 1440]
test_data = t_data_2d[7 * 1440: 8 * 1440]  # 测试集
test_label = t_label[7 * 1440: 8 * 1440]

print('训练集:', train_data.shape, train_label.shape)
print('测试集:', test_data.shape, test_label.shape)

# ====================================================================
# 2. 数据标准化（SVM对特征尺度敏感）
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# 对标签进行标准化
label_scaler = StandardScaler()
train_label_scaled = label_scaler.fit_transform(train_label.reshape(-1, 1)).ravel()
test_label_scaled = label_scaler.transform(test_label.reshape(-1, 1)).ravel()

# ====================================================================
# 3. 构建SVM回归模型
"""
参数说明：
- kernel : 核函数类型（'linear', 'rbf', 'poly'等）
- C      : 正则化参数（值越小正则化越强）
- epsilon: 控制回归容忍度的epsilon值
- gamma  : RBF核的系数（'scale'自动计算或'auto'）
"""
svm_model = SVR(
    kernel='rbf',
    C=1.0,
    epsilon=0.1,
    gamma='scale',
    verbose=True  # 显示训练进度
)

# ====================================================================
# 4. 模型训练
print("开始训练SVM...")
svm_model.fit(train_data_scaled, train_label_scaled)

# ====================================================================
# 5. 模型评估
# 5.1 训练集评估
train_pred_scaled = svm_model.predict(train_data_scaled)
train_pred = label_scaler.inverse_transform(train_pred_scaled.reshape(-1, 1)).ravel()
train_mse = mean_squared_error(train_label, train_pred)

# 5.2 测试集评估
test_pred_scaled = svm_model.predict(test_data_scaled)
test_pred = label_scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).ravel()
test_mse = mean_squared_error(test_label, test_pred)

print(f"训练集MSE: {train_mse:.6f}")
print(f"测试集MSE: {test_mse:.6f}")

# ====================================================================
# 6. 模型保存
with open('./svm_model_z.pkl', 'wb') as f:
    pickle.dump({
        'model': svm_model,
        'scaler': scaler,
        'label_scaler': label_scaler
    }, f)
    