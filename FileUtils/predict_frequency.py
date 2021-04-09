import pickle

import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from FileUtils.functionUtils import loadFile, readVIB_FPData, FFTvibData, timeWindows, ifftOrigin, ifftPredict
import matplotlib.pyplot as plt
import numpy as np

# 设置plt能够正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def train(vib_Name, flightTime, modelPath, num,vib_value,vib_time):
    # 飞参与vib的相关性及其变量名
    vib_191207_FP_corr = loadFile("Savedfile/" + flightTime + "_" + vib_Name + "_FP_corr")
    vib_191207_FP_col = loadFile("Savedfile/" + flightTime + "_" + vib_Name + "_FP_col")

    # list转换为np.array
    vib_191207_FP_corr_ary = np.array(vib_191207_FP_corr)
    vib_191207_FP_col_ary = np.array(vib_191207_FP_col)

    # 选择相关性最高的20个飞参
    num = -1 * int(num)
    vib_FP_highCol_FP = vib_191207_FP_col_ary[np.argsort(np.abs(vib_191207_FP_corr_ary))][num:]
    # 表示进行频域变换的每个单元监控点的数量
    NUM = 500

    # # 读取经过转换后的测点数据和飞参数据
    # vib_value = np.array(vib_value[:(len(vib_value)-len(vib_value) % NUM)])
    # vib_value = np.reshape(vib_value, (-1, NUM))
    # vib_time = np.array(vib_time[:(len(vib_time) - len(vib_time) % NUM)])
    # vib_time = np.reshape(vib_time, (-1, NUM))
    # vib_time = vib_time[:, 0]
    # FP_time =

    vib_value_191207, FP_value_191207 = readVIB_FPData(vib_Name, vib_FP_highCol_FP, '191207', NUM)

    vib_abs_191207, vib_ang_191207 = FFTvibData(vib_value_191207, NUM)

    # 需要预测的测点的频率
    select_fre = range(0, 200)
    vib_191207_ary = vib_abs_191207[:, select_fre]
    # 对转换后的频域信号进行滑动时间窗移动平均
    vib_191207_ary = timeWindows(vib_191207_ary, length=10)

    if modelPath == None:
        '''
          线性回归
          预测及结果展示
          '''
        LRreg = LinearRegression().fit(FP_value_191207, vib_191207_ary)
    else:
        with open(modelPath, 'rb') as f:
            LRreg = pickle.load(f)  # 从f文件中提取出模型赋给clf2
        LRreg.fit(FP_value_191207, vib_191207_ary)
    return LRreg


def predict(vib_Name, num):
    # 飞参与vib的相关性及其变量名
    vib_191207_FP_corr = loadFile('Savedfile/' + vib_Name + '_FP_corr')
    vib_191207_FP_col = loadFile('Savedfile/' + vib_Name + '_FP_col')

    # list转换为np.array
    vib_191207_FP_corr_ary = np.array(vib_191207_FP_corr)
    vib_191207_FP_col_ary = np.array(vib_191207_FP_col)

    # 选择相关性最高的20个飞参
    num = -1 * num
    vib_FP_highCol_FP = vib_191207_FP_col_ary[np.argsort(np.abs(vib_191207_FP_corr_ary))][num:]

    # 表示进行频域变换的每个单元监控点的数量
    NUM = 500
    # 读取经过转换后的测点数据和飞参数据
    vib_value_191207, FP_value_191207 = readVIB_FPData(vib_Name, vib_FP_highCol_FP, '191207', NUM)
    vib_value_200102, FP_value_200102 = readVIB_FPData(vib_Name, vib_FP_highCol_FP, '200102', NUM)
    vib_value_200106, FP_value_200106 = readVIB_FPData(vib_Name, vib_FP_highCol_FP, '200106', NUM)

    vib_abs_191207, vib_ang_191207 = FFTvibData(vib_value_191207, NUM)
    vib_abs_200102, vib_ang_200102 = FFTvibData(vib_value_200102, NUM)
    vib_abs_200106, vib_ang_200106 = FFTvibData(vib_value_200106, NUM)

    # 需要预测的测点的频率
    select_fre = range(0, 200)

    vib_191207_ary = vib_abs_191207[:, select_fre]
    vib_200102_ary = vib_abs_200102[:, select_fre]
    vib_200106_ary = vib_abs_200106[:, select_fre]

    # 对转换后的频域信号进行滑动时间窗移动平均
    vib_191207_ary = timeWindows(vib_191207_ary, length=10)
    vib_200102_ary = timeWindows(vib_200102_ary, length=10)
    vib_200106_ary = timeWindows(vib_200106_ary, length=10)

    '''
    线性回归
    预测及结果展示
    '''
    LRreg = LinearRegression().fit(FP_value_191207, vib_191207_ary)
    vib_pred_LR_200102 = LRreg.predict(FP_value_200102)

    '''
    支持向量机回归
    预测及结果展示
    '''
    SVRreg = []
    base_SVRreg = make_pipeline(SVR(kernel='rbf', C=100, tol=0.1, gamma=0.1,
                                    epsilon=0.5, max_iter=1000))
    for i in range(len(select_fre)):
        SVRreg.append(sklearn.base.clone(base_SVRreg).fit(FP_value_191207, vib_191207_ary[:, i]))
    vib_pred_SVR_200102 = np.array([SVRreg[i].predict(FP_value_200102) for i in range(len(select_fre))]).T

    '''
    人工神经网络
    预测及结果展示
    '''
    ANNreg = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100,),
                                                          alpha=0.001, learning_rate_init=0.001,
                                                          random_state=1, max_iter=100))
    ANNreg.fit(FP_value_191207, vib_191207_ary)
    vib_pred_ANN_200102 = ANNreg.predict(FP_value_200102)

    '''
    算法集合体
    预测及结果展示
    '''
    LR_meta = []
    base_LR_meta = LinearRegression()
    for i in range(len(select_fre)):
        vib_pred_ALL_200102 = np.hstack((vib_pred_LR_200102[:, i].reshape(-1, 1),
                                         vib_pred_SVR_200102[:, i].reshape(-1, 1),
                                         vib_pred_ANN_200102[:, i].reshape(-1, 1)))
        LR_meta.append(sklearn.base.clone(base_LR_meta).fit(vib_pred_ALL_200102, vib_200102_ary[:, i]))

    # vib_pred_ALL_200102, vib_200102_ary = TrainDataUnSamping(vib_pred_ALL_200102, vib_200102_ary, 10000)

    vib_pred_LR_200106 = LRreg.predict(FP_value_200106)
    vib_pred_SVR_200106 = np.array([SVRreg[i].predict(FP_value_200106) for i in range(len(select_fre))]).T
    vib_pred_ANN_200106 = ANNreg.predict(FP_value_200106)

    vib_pred_final_200106 = np.array([LR_meta[i].predict(np.hstack((vib_pred_LR_200106[:, i].reshape(-1, 1),
                                                                    vib_pred_SVR_200106[:, i].reshape(-1, 1),
                                                                    vib_pred_ANN_200106[:, i].reshape(-1, 1)))) \
                                      for i in range(len(select_fre))]).T

    # 将预测的频域信号通过快速傅里叶逆变换转换成时域信号，并进行评价和可视化
    # vib_true_200106 = vib_value_200106.reshape(-1)
    vib_true_200106 = ifftOrigin(timeWindows(vib_abs_200106, length=10), vib_ang_200106)
    vib_pred_200106 = ifftPredict(vib_pred_final_200106, select_fre, NUM)
    print(abs(np.sqrt(np.mean(vib_true_200106 ** 2)) - np.sqrt(np.mean(vib_pred_200106 ** 2))) / np.sqrt(
        np.mean(vib_true_200106 ** 2)))
    plt.figure()
    plt.plot(vib_true_200106)
    plt.plot(vib_pred_200106)
    plt.title('算法集合体预测结果')
