import os
import numpy as np
import csv
import pickle

"""
    功能说明: 读取对应目录下的CPF数据和SGP4数据, 并将读取后的数据进行存储, 以便后续神经网络的训练和预测用
"""

# ====================================================================
# 1. 读取CPF星历文件 09-11 00:00:00---> 09.17 24:00:00 共7天 训练数据
#                  09-18 00:00:00---> 09.18:24:00:00 共1天 测试数据
#                  09-18 00:00:00---> 09.24:24:00:00 共7天 预测时长与输入步长对LSTM模型性能的影响的测试数据
def Load_CPF_data(dir_path):
    files_path = os.listdir(dir_path) # 获取该目录下的所有文件
    date = ['60381 ','60382 ','60383 ','60384 ','60385 ','60386 ','60387 ',
            '60388 ','60389 '] # 根据date选取对应CPF文件中一天的数据
    CPF_data = {'time':[], 'x':[], 'y':[], 'z':[]} # 用来保存时间(time), 位置x(x), 位置y(y), 位置z(z)
    
    i = 0
    for file_path in files_path:  # 遍历每一个文件
        with open(dir_path+file_path, 'r') as f: # 读取文件中的数据
            for line in f: # 遍历文件中每行数据
                if line.startswith('10') and date[i] in line: # 找到对应的数据行
                    line = line.strip()  # 去掉字符串前后的空格
                    line_list = line.split(' ') # 对字符串按照空格进行拆分, 变成列表
                    line_list = [x for x in line_list if x != ""] # 将列表中的”“ 去掉
                    
                    CPF_data['time'].append(line_list[3]) # 存储时间数据
                    CPF_data['x'].append(float(line_list[5])) # 存储位置x数据
                    CPF_data['y'].append(float(line_list[6])) # 存储位置y数据
                    CPF_data['z'].append(float(line_list[7])) # 存储位置z数据
            i = i + 1
    return CPF_data # 返回数据

dir_path = './CPF_Files/'           # CPF数据目录的地址
CPF_data = Load_CPF_data(dir_path)  # 获取CPF数据


# ====================================================================
# 2. 读取SGP4预测数据 09-11 00:00:00---> 09.17 24:00:00 共7天
def Load_SGP4_data(root_path):
    SGP4_data = {'x':[], 'y':[], 'z':[],
                 'vx':[], 'vy':[], 'vz':[],
                 'ax':[], 'ay':[], 'az':[]} # 用来保存位置x(x), 位置y(y), 位置z(z), 速度vx(vx), 速度vy(vy), 速度vz(vz), 加速度ax(ax), 加速度ay(ay), 加速度az(az)
    
    
    dir_paths = os.listdir(root_path) # 获取对应目录下的文件路径
    
    for dir_path in dir_paths:
        if os.path.isdir(root_path+'/'+dir_path): # 获取对应目录下的文件路径
            
            # 获取位置和速度
            if '位置和速度' in dir_path: # 判断目录字符串是否包含"位置和速度"
                file_paths = os.listdir(root_path+'/'+dir_path) # 获取对应目录下的文件路径
                
                # 读取.csv文件
                for file_path in file_paths: # 遍历该目录下的所有.csv文件
                    with open(root_path+'/'+dir_path+'/'+file_path, newline='') as csvfile: # 打开csv文件
                        # print('reading...'+root_path+'/'+dir_path+'/'+file_path)
                        csv_reader = csv.reader(csvfile) # 转换成csv对象
                        
                        for row in csv_reader: # 逐行读取数据
                         if row[1] and row[2] and row[3] and row[4] and row[5] and row[6]:  # 检查每个字段是否为空
                            SGP4_data['x'].append(float(row[1])*1000)  # 存储位置x数据
                            SGP4_data['y'].append(float(row[2])*1000)  # 存储位置y数据
                            SGP4_data['z'].append(float(row[3])*1000)  # 存储位置z数据
                            SGP4_data['vx'].append(float(row[4])*1000) # 存储位置vx数据
                            SGP4_data['vy'].append(float(row[5])*1000) # 存储位置vy数据
                            SGP4_data['vz'].append(float(row[6])*1000) # 存储位置vz数据
                              
                # print(len(SGP4_data['x']), len(SGP4_data['y']), len(SGP4_data['z']), len(SGP4_data['vx']), len(SGP4_data['vy']), len(SGP4_data['vz']))
            # 获取加速度
            if '加速度' in dir_path: # 判断目录字符串是否包含"加速度"
                file_paths = os.listdir(root_path+'/'+dir_path) # 获取对应目录下的文件路径
                
                # 读取.csv文件
                for file_path in file_paths: # 遍历该目录下的所有.csv文件
                    with open(root_path+'/'+dir_path+'/'+file_path, newline='') as csvfile: # 打开csv文件
                        # print('reading...'+root_path+'/'+dir_path+'/'+file_path)
                        csv_reader = csv.reader(csvfile) # 转换成csv对象
                        
                        for row in csv_reader: # 逐行读取数据
                         if row[1] and row[2] and row[3]:  # 检查每个字段是否为空
                            SGP4_data['ax'].append(float(row[1])*1000) # 存储位置ax数据
                            SGP4_data['ay'].append(float(row[2])*1000) # 存储位置ay数据
                            SGP4_data['az'].append(float(row[3])*1000) # 存储位置az数据
                # print(len(SGP4_data['ax']), len(SGP4_data['ay']), len(SGP4_data['az']))
    return SGP4_data # 返回数据
    
SGP4_data = Load_SGP4_data('./SGP4_Files') # 获取SGP4数据


# ====================================================================
# 3. 保存数据
file_path = './CFP_SGP4_data.p'   # 存储数据的路径

with open(file_path, 'wb') as f:
    pickle.dump((CPF_data, SGP4_data), f) # 存储








    







