from sklearn.linear_model import LinearRegression

from FileUtils.functionUtils import findFolder, readData, ensembleTrain
import numpy as np
import matplotlib.pyplot as plt

#设置plt能够正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def predict():
    # vib_name表示需要预测的测点名称
    vib_Name = 'ANT4'
    # 垂直尾翼测点名称
    VT_vib_names = ["ANT1", "ANT2", "ANT3", "ANT4", "ANT5", "ANT6", "ANT7"]
    # 垂直尾翼数据读取
    VT_vib_data = []
    for i in range(len(VT_vib_names)):
        VT_vib_data.append(readData(findFolder(VT_vib_names[i])[0], VT_vib_names[i]))
        print(VT_vib_names[i] + " Finished!")

    VT_vib_ary = np.array(VT_vib_data).T

    # 提取需要预测测点的监测序列
    ANT4_191207 = np.array(VT_vib_data[3])

    # 垂直尾翼模态叠加法训练及预测
    # 垂直尾翼固有模态
    VTZMode_phi1 = 14.368 * np.array([0.112540, 0.028934, 0.131050, 0.037767, 0.043148, 0.094470, 0.134830]).reshape(-1,
                                                                                                                     1)
    VTZMode_phi2 = 45.686 * np.array([0.000378, 0.092578, 0.136430, 0.038831, 0.015865, 0.035344, 0.163180]).reshape(-1,
                                                                                                                     1)
    VTZMode_phi3 = 56.507 * np.array([0.102730, 0.038422, 0.116900, 0.039560, 0.047495, 0.090431, 0.120570]).reshape(-1,
                                                                                                                     1)
    VTZMode_phi4 = 61.685 * np.array([0.126060, 0.032371, 0.089103, 0.044842, 0.085259, 0.116170, 0.072661]).reshape(-1,
                                                                                                                     1)
    VTZMode_mat = np.concatenate((VTZMode_phi1, VTZMode_phi2, VTZMode_phi3, VTZMode_phi4), axis=1)

    ANT4_191207_pred_MT = []
    X_i = np.vstack((VTZMode_mat[:3, :], VTZMode_mat[4:, :]))
    for i in range(len(ANT4_191207)):
        y_i = np.concatenate((VT_vib_ary[i, :3], VT_vib_ary[i, 4:]))
        reg = LinearRegression(fit_intercept=True).fit(X_i, y_i)
        # ANT4_191207_pred_MT.append(np.dot(reg.coef_, VTZMode_mat[3, :]))
        ANT4_191207_pred_MT.append(reg.predict(VTZMode_mat[3, :].reshape(1, -1))[0])
        if (i % 400000 == 0):
            print(str(round(i / len(ANT4_191207) * 100, 2)) + "%")

    ANT4_191207_pred_MT = np.array(ANT4_191207_pred_MT)
    print(abs(np.sqrt(np.mean(ANT4_191207 ** 2)) - np.sqrt(np.mean(ANT4_191207_pred_MT ** 2))) / np.sqrt(
        np.mean(ANT4_191207 ** 2)))
    plt.figure()
    plt.plot(ANT4_191207)
    plt.plot(ANT4_191207_pred_MT)
    plt.title('垂直尾翼模态叠加法预测结果')

    # 最高次项为2的平面插值方法
    # 垂直尾翼测点坐标（2维）
    VT_vib_pos = np.array([[3.2, 3.7], [2, 2.2], [3.85, 3.7], [3.1, 2], [3.55, 2],
                           [3.1, 3.4], [4.2, 3.7]])

    ANT4_191207_pred_CZ = []
    X_i = np.vstack((VT_vib_pos[:3, :], VT_vib_pos[4:, :]))
    X_i_dg2 = np.hstack((X_i, X_i ** 2))
    for i in range(len(ANT4_191207)):
        y_i = np.concatenate((VT_vib_ary[i, :3], VT_vib_ary[i, 4:]))
        lr = LinearRegression().fit(X_i_dg2, y_i)
        ANT4_191207_pred_CZ.append(lr.predict(np.hstack((VT_vib_pos[3, :], VT_vib_pos[3, :] ** 2)).reshape(1, -1))[0])
        if (i % 400000 == 0):
            print(str(round(i / len(ANT4_191207) * 100, 2)) + "%")

    ANT4_191207_pred_CZ = np.array(ANT4_191207_pred_CZ)
    print(abs(np.sqrt(np.mean(ANT4_191207 ** 2)) - np.sqrt(np.mean(ANT4_191207_pred_CZ ** 2))) / np.sqrt(
        np.mean(ANT4_191207 ** 2)))
    plt.figure()
    plt.plot(ANT4_191207_pred_CZ)
    plt.plot(ANT4_191207)
    plt.title('垂直尾翼平面插值法预测结果')

    '''
    算法集合体
    预测及结果展示
    '''
    ANT4_pred_ALL_191207 = np.hstack((ANT4_191207_pred_MT.reshape(-1, 1), ANT4_191207_pred_CZ.reshape(-1, 1)))
    # LR_meta = LinearRegression().fit(ANT4_pred_ALL_191207, ANT4_191207)
    LR_meta = ensembleTrain(ANT4_pred_ALL_191207, ANT4_191207, step=5000)
    ANT4_pred_final_191207 = LR_meta.predict(ANT4_pred_ALL_191207)
    print(abs(np.sqrt(np.mean(ANT4_191207 ** 2)) - np.sqrt(np.mean(ANT4_pred_final_191207 ** 2))) / np.sqrt(
        np.mean(ANT4_191207 ** 2)))
    plt.figure()
    plt.plot(ANT4_191207)
    plt.plot(ANT4_pred_final_191207)
    plt.title('垂直尾翼算法集合体预测结果')