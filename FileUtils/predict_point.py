import matplotlib.pyplot as plt
import numpy as np
#设置plt能够正确显示中文
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from FileUtils.functionUtils import loadFile, vibMatFetch, findFolder, readData

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def predict():
    # vib_name表示需要预测的测点名称
    vib_Name = 'ANT6'
    # 飞参与vib的相关性及其变量名
    vib_191207_vib_corr = loadFile('Savedfile/' + vib_Name + '_vib_corr')
    vib_191207_vib_col = loadFile('Savedfile/' + vib_Name + '_vib_col')

    # list转换为np.array
    vib_191207_vib_corr_ary = np.array(vib_191207_vib_corr)
    vib_191207_vib_col_ary = np.array(vib_191207_vib_col)

    # 选择相关性最高的10个其他测点
    vib_highCol_vib = vib_191207_vib_col_ary[np.argsort(np.abs(vib_191207_vib_corr_ary))][-11:-1]

    # 读取相关的测点数据
    vib_191207 = vibMatFetch(vib_highCol_vib, "191207")
    vib_200102 = vibMatFetch(vib_highCol_vib, "200102")
    vib_200106 = vibMatFetch(vib_highCol_vib, "200106")

    # 读取需要预测的测点数据
    ANT6_191207 = np.array(
        readData([findFolder(vib_Name)[j] for j in range(3) if "191207" in findFolder(vib_Name)[j]][0], vib_Name))
    ANT6_200102 = np.array(
        readData([findFolder(vib_Name)[j] for j in range(3) if "200102" in findFolder(vib_Name)[j]][0], vib_Name))
    ANT6_200106 = np.array(
        readData([findFolder(vib_Name)[j] for j in range(3) if "200106" in findFolder(vib_Name)[j]][0], vib_Name))
    # vib_191207_1, ANT6_191207_1 = TrainDataUnSamping(vib_191207, ANT6_191207, 10000)

    '''
    线性回归
    预测及结果展示
    '''
    LRreg = LinearRegression().fit(vib_191207, ANT6_191207)
    vib_pred_LR_200102 = LRreg.predict(vib_200102)
    print(abs(np.sqrt(np.mean(ANT6_200102 ** 2)) - np.sqrt(np.mean(vib_pred_LR_200102 ** 2))) / np.sqrt(
        np.mean(ANT6_200102 ** 2)))
    plt.figure()
    plt.plot(ANT6_200102)
    plt.plot(vib_pred_LR_200102)
    plt.title('线性回归预测结果')

    '''
    支持向量机回归
    预测及结果展示
    '''
    SVRreg = make_pipeline(SVR(kernel='rbf', C=100, tol=0.1, gamma=0.1, epsilon=0.5, max_iter=1000))
    SVRreg.fit(vib_191207, ANT6_191207)
    vib_pred_SVR_200102 = SVRreg.predict(vib_200102)
    print(abs(np.sqrt(np.mean(ANT6_200102 ** 2)) - np.sqrt(np.mean(vib_pred_SVR_200102 ** 2))) / np.sqrt(
        np.mean(ANT6_200102 ** 2)))
    plt.figure()
    plt.plot(ANT6_200102)
    plt.plot(vib_pred_SVR_200102)
    plt.title('支持向量回归预测结果')

    '''
    人工神经网络
    预测及结果展示
    '''
    ANNreg = make_pipeline(StandardScaler(),
                           MLPRegressor(hidden_layer_sizes=(100,), alpha=0.001, learning_rate_init=0.01, random_state=1,
                                        max_iter=10))
    ANNreg.fit(vib_191207, ANT6_191207)
    vib_pred_ANN_200102 = ANNreg.predict(vib_200102)
    print(abs(np.sqrt(np.mean(ANT6_200102 ** 2)) - np.sqrt(np.mean(vib_pred_ANN_200102 ** 2))) / np.sqrt(
        np.mean(ANT6_200102 ** 2)))
    plt.figure()
    plt.plot(ANT6_200102)
    plt.plot(vib_pred_ANN_200102)
    plt.title('人工神经网络预测结果')

    '''
    算法集合体
    预测及结果展示
    '''
    vib_pred_ALL_200102 = np.hstack(
        (vib_pred_LR_200102.reshape(-1, 1), vib_pred_SVR_200102.reshape(-1, 1), vib_pred_ANN_200102.reshape(-1, 1)))
    # vib_pred_ALL_200102, ANT6_200102 = TrainDataUnSamping(vib_pred_ALL_200102, ANT6_200102, 10000)
    LR_meta = LinearRegression().fit(vib_pred_ALL_200102, ANT6_200102)

    vib_pred_LR_200106 = LRreg.predict(vib_200106)
    vib_pred_SVR_200106 = SVRreg.predict(vib_200106)
    vib_pred_ANN_200106 = ANNreg.predict(vib_200106)

    vib_pred_ALL_200106 = np.hstack(
        (vib_pred_LR_200106.reshape(-1, 1), vib_pred_SVR_200106.reshape(-1, 1), vib_pred_ANN_200106.reshape(-1, 1)))
    vib_pred_final_200106 = LR_meta.predict(vib_pred_ALL_200106)
    print(abs(np.sqrt(np.mean(ANT6_200106 ** 2)) - np.sqrt(np.mean(vib_pred_final_200106 ** 2))) / np.sqrt(
        np.mean(ANT6_200106 ** 2)))
    plt.figure()
    plt.plot(ANT6_200106)
    plt.plot(vib_pred_final_200106)
    plt.title('算法集合体预测结果')