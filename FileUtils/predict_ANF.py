import numpy as np
from sklearn.linear_model import LinearRegression

from FileUtils.functionUtils import readData, findFolder, ensembleTrain
import matplotlib.pyplot as plt
#设置plt能够正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def predict():
    # vib_name表示需要预测的测点名称
    vib_Name = 'ANF6'
    # 机翼测点名称,ANF1和ANF2完全对称，振动高度相关，所以只取一个就可以，降低运算时间
    body_vib_names = ["ANF1", "ANF5", "ANF6", "ANF9"]
    # 机翼数据读取
    body_vib_data = []
    for i in range(len(body_vib_names)):
        body_vib_data.append(readData(findFolder(body_vib_names[i])[0], body_vib_names[i]))
        print(body_vib_names[i] + " Finished!")

    body_vib_ary = np.array(body_vib_data).T

    # 提取需要预测测点的监测序列
    ANF6_191207 = np.array(body_vib_data[2])

    # 机身模态叠加法训练及预测
    # 机身固有模态
    bodyZMode_phi1 = 19800 * np.array(
        [5.517 * 10 ** (-4), 6.068 * 10 ** (-4), 6.620 * 10 ** (-4), 1.655 * 10 ** (-4)]).reshape(-1, 1)
    bodyZMode_phi2 = 97184 * np.array([0, 2.191 * 10 ** (-4), 0, 8.764 * 10 ** (-4)]).reshape(-1, 1)
    bodyZMode_phi3 = 192000 * np.array(
        [5.671 * 10 ** (-5), 6.805 * 10 ** (-4), 5.671 * 10 ** (-4), 1.134 * 10 ** (-4)]).reshape(-1, 1)
    bodyZMode_phi4 = 830000 * np.array([0, 1.104 * 10 ** (-3), 0, 0]).reshape(-1, 1)
    bodyZMode_mat = np.concatenate((bodyZMode_phi1, bodyZMode_phi2, bodyZMode_phi3, bodyZMode_phi4), axis=1)

    ANF6_191207_pred_MT = []
    X_i = bodyZMode_mat[[0, 1, 3], :]
    for i in range(len(ANF6_191207)):
        y_i = np.concatenate((body_vib_ary[i, :2], body_vib_ary[i, 3].reshape(1, )))
        reg = LinearRegression(fit_intercept=True).fit(X_i, y_i)
        # ANF6_191207_pred_MT.append(np.dot(reg.coef_, bodyZMode_mat[2, :])[0])
        ANF6_191207_pred_MT.append(reg.predict(bodyZMode_mat[2, :].reshape(1, -1))[0])
        if (i % 400000 == 0):
            print(str(round(i / len(ANF6_191207) * 100, 2)) + "%")

    ANF6_191207_pred_MT = np.array(ANF6_191207_pred_MT)
    print(abs(np.sqrt(np.mean(ANF6_191207 ** 2)) - np.sqrt(np.mean(ANF6_191207_pred_MT ** 2))) / np.sqrt(
        np.mean(ANF6_191207 ** 2)))
    plt.figure()
    plt.plot(ANF6_191207)
    plt.plot(ANF6_191207_pred_MT)
    plt.title('机身模态叠加法预测结果')

    # 机身线性插值法训练及预测，机身部分仅考虑1维情况
    # 机身测点坐标（1维）
    body_vib_pos = np.array([[1.2, ], [7.0, ], [7.4, ], [18.7, ]])
    ANF6_191207_pred_CZ = []
    X_i = body_vib_pos[[0, 1, 3], :]
    for i in range(len(ANF6_191207)):
        y_i = np.concatenate((body_vib_ary[i, :2], body_vib_ary[i, 3].reshape(1, )))
        lr = LinearRegression().fit(X_i, y_i)
        ANF6_191207_pred_CZ.append(lr.predict(body_vib_pos[3, :].reshape(1, -1))[0])
        if (i % 400000 == 0):
            print(str(round(i / len(ANF6_191207) * 100, 2)) + "%")

    ANF6_191207_pred_CZ = np.array(ANF6_191207_pred_CZ)
    print(abs(np.sqrt(np.mean(ANF6_191207 ** 2)) - np.sqrt(np.mean(ANF6_191207_pred_CZ ** 2))) / np.sqrt(
        np.mean(ANF6_191207 ** 2)))
    plt.figure()
    plt.plot(ANF6_191207)
    plt.plot(ANF6_191207_pred_CZ)
    plt.title('机身线性插值法预测结果')

    '''
    算法集合体
    预测及结果展示
    '''
    ANF6_pred_ALL_191207 = np.hstack((ANF6_191207_pred_MT.reshape(-1, 1), ANF6_191207_pred_CZ.reshape(-1, 1)))
    # LR_meta = LinearRegression().fit(ANF6_pred_ALL_191207, ANF6_191207)
    LR_meta = ensembleTrain(ANF6_pred_ALL_191207, ANF6_191207, step=5000)
    ANF6_pred_final_191207 = LR_meta.predict(ANF6_pred_ALL_191207)
    print(abs(np.sqrt(np.mean(ANF6_191207 ** 2)) - np.sqrt(np.mean(ANF6_pred_final_191207 ** 2))) / np.sqrt(
        np.mean(ANF6_191207 ** 2)))
    plt.figure()
    plt.plot(ANF6_191207)
    plt.plot(ANF6_pred_final_191207)
    plt.title('机身算法集合体预测结果')