import numpy as np
import matplotlib.pyplot as plt
#设置plt能够正确显示中文
from sklearn.linear_model import LinearRegression

from FileUtils.functionUtils import readData, findFolder, ensembleTrain

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def predict():
    # vib_name表示需要预测的测点名称
    vib_Name = 'ANH4'
    # 水平尾翼测点名称
    LT_vib_names = ["ANH1", "ANH2", "ANH3", "ANH4", "ANH5", "ANH11", "ANH13"]
    # 水平尾翼数据读取
    LT_vib_data = []
    for i in range(len(LT_vib_names)):
        LT_vib_data.append(readData(findFolder(LT_vib_names[i])[0], LT_vib_names[i]))
        print(LT_vib_names[i] + " Finished!")

    LT_vib_ary = np.array(LT_vib_data).T

    # 提取需要预测测点的监测序列
    ANH4_191207 = np.array(LT_vib_data[3])

    # 水平尾翼模态叠加法训练及预测
    # 水平尾翼固有模态
    LTZMode_phi1 = 17.960 * np.array([0.198240, 0.209660, 0.060092, 0.079628, 0.007122, 0.172180, 0.096400]).reshape(-1,
                                                                                                                     1)
    LTZMode_phi2 = 51.337 * np.array([0.186680, 0.196690, 0.069147, 0.079404, 0.014269, 0.164790, 0.101410]).reshape(-1,
                                                                                                                     1)
    LTZMode_phi3 = 64.446 * np.array([0.136150, 0.219270, 0.150160, 0.072657, 0.013771, 0.065595, 0.033473]).reshape(-1,
                                                                                                                     1)
    LTZMode_phi4 = 80.292 * np.array([0.170690, 0.083437, 0.087226, 0.093101, 0.042739, 0.156430, 0.179670]).reshape(-1,
                                                                                                                     1)
    LTZMode_mat = np.concatenate((LTZMode_phi1, LTZMode_phi2, LTZMode_phi3, LTZMode_phi4), axis=1)

    ANH4_191207_pred_MT = []
    X_i = np.vstack((LTZMode_mat[:3, :], LTZMode_mat[4:, :]))
    for i in range(len(ANH4_191207)):
        y_i = np.concatenate((LT_vib_ary[i, :3], LT_vib_ary[i, 4:]))
        reg = LinearRegression(fit_intercept=True).fit(X_i, y_i)
        # ANH4_191207_pred_MT.append(np.dot(reg.coef_, LTZMode_mat[3, :]))
        ANH4_191207_pred_MT.append(reg.predict(LTZMode_mat[3, :].reshape(1, -1))[0])
        if (i % 400000 == 0):
            print(str(round(i / len(ANH4_191207) * 100, 2)) + "%")

    ANH4_191207_pred_MT = np.array(ANH4_191207_pred_MT)
    print(abs(np.sqrt(np.mean(ANH4_191207 ** 2)) - np.sqrt(np.mean(0.04 * ANH4_191207_pred_MT ** 2))) / np.sqrt(
        np.mean(ANH4_191207 ** 2)))
    plt.figure()
    plt.plot(ANH4_191207)
    plt.plot(0.2 * ANH4_191207_pred_MT)
    plt.title('0.2*水平尾翼模态叠加法预测结果')

    # 最高次项为2的平面插值方法
    # 水平尾翼测点坐标（2维）
    LT_vib_pos = np.array([[-0.15, 3.5], [-0.45, 3.5], [0.75, 2.3], [0.05, 2.3], [0, 0.9],
                           [0, 3.3], [-0.3, 2.4]])

    ANH4_191207_pred_CZ = []
    X_i = np.vstack((LT_vib_pos[:3, :], LT_vib_pos[4:, :]))
    X_i_dg2 = np.hstack((X_i, X_i ** 2))
    for i in range(len(ANH4_191207)):
        y_i = np.concatenate((LT_vib_ary[i, :3], LT_vib_ary[i, 4:]))
        lr = LinearRegression().fit(X_i_dg2, y_i)
        ANH4_191207_pred_CZ.append(lr.predict(np.hstack((LT_vib_pos[3, :], LT_vib_pos[3, :] ** 2)).reshape(1, -1))[0])
        if (i % 400000 == 0):
            print(str(round(i / len(ANH4_191207) * 100, 2)) + "%")

    ANH4_191207_pred_CZ = np.array(ANH4_191207_pred_CZ)
    print(abs(np.sqrt(np.mean(ANH4_191207 ** 2)) - np.sqrt(np.mean(ANH4_191207_pred_CZ ** 2))) / np.sqrt(
        np.mean(ANH4_191207 ** 2)))
    plt.figure()
    plt.plot(ANH4_191207_pred_CZ)
    plt.plot(ANH4_191207)
    plt.title('水平尾翼平面插值法预测结果')

    '''
    算法集合体
    预测及结果展示
    '''
    ANH4_pred_ALL_191207 = np.hstack((ANH4_191207_pred_MT.reshape(-1, 1), ANH4_191207_pred_CZ.reshape(-1, 1)))
    # LR_meta = LinearRegression().fit(ANH4_pred_ALL_191207, ANH4_191207)
    LR_meta = ensembleTrain(ANH4_pred_ALL_191207, ANH4_191207, step=len(ANH4_191207))
    ANH4_pred_final_191207 = LR_meta.predict(ANH4_pred_ALL_191207)
    print(abs(np.sqrt(np.mean(ANH4_191207 ** 2)) - np.sqrt(np.mean(ANH4_pred_final_191207 ** 2))) / np.sqrt(
        np.mean(ANH4_191207 ** 2)))
    plt.figure()
    plt.plot(ANH4_191207)
    plt.plot(ANH4_pred_final_191207)
    plt.title('水平尾翼算法集合体预测结果')