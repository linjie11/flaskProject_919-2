from sklearn.linear_model import LinearRegression

from FileUtils.functionUtils import readData, findFolder, ensembleTrain
import numpy as np
import matplotlib.pyplot as plt

#设置plt能够正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def predict():
    # vib_name表示需要预测的测点名称
    vib_Name = 'ANW5'
    # 机翼测点名称
    wing_vib_names = ["ANW1", "ANW2", "ANW3", "ANW4", "ANW5", "ANW6", "ANW7", "ANW8", "ANW17", "ANW18"]
    # 机翼数据读取
    wing_vib_data = []
    for i in range(len(wing_vib_names)):
        wing_vib_data.append(readData(findFolder(wing_vib_names[i])[0], wing_vib_names[i]))
        print(wing_vib_names[i] + " Finished!")

    wing_vib_ary = np.array(wing_vib_data).T

    # 提取需要预测测点的监测序列
    ANW5_191207 = np.array(wing_vib_data[4])

    # 机翼模态叠加法训练及预测
    # 机翼固有模态
    wingZMode_phi1 = 4.6242 * np.array(
        [0.053323, 0.054902, 0.052748, 0.039911, 0.040730, 0.019372, 0.020486, 0.048519, 0.002433, 0.018903]).reshape(
        -1, 1)
    wingZMode_phi2 = 10.135 * np.array(
        [0.057075, 0.059566, 0.056369, 0.039238, 0.040464, 0.015998, 0.016806, 0.050925, 0.002292, 0.016526]).reshape(
        -1, 1)
    wingZMode_phi3 = 20.535 * np.array(
        [0.044158, 0.053486, 0.040799, 0.013048, 0.014958, 0.032946, 0.029549, 0.034825, 0.008309, 0.026757]).reshape(
        -1, 1)
    wingZMode_phi4 = 33.196 * np.array(
        [0.041207, 0.052499, 0.039350, 0.007710, 0.010925, 0.031769, 0.032244, 0.028927, 0.008433, 0.031172]).reshape(
        -1, 1)
    wingZMode_mat = np.concatenate((wingZMode_phi1, wingZMode_phi2, wingZMode_phi3, wingZMode_phi4), axis=1)

    ANW5_191207_pred_MT = []
    X_i = np.vstack((wingZMode_mat[:4, :], wingZMode_mat[5:-2, :]))
    for i in range(len(ANW5_191207)):
        y_i = np.concatenate((wing_vib_ary[i, :4], wing_vib_ary[i, 5:-2]))
        reg = LinearRegression(fit_intercept=True).fit(X_i, y_i)
        # ANW5_191207_pred_MT.append(np.dot(reg.coef_, wingZMode_mat[4, :]))
        ANW5_191207_pred_MT.append(reg.predict(wingZMode_mat[4, :].reshape(1, -1))[0])
        if (i % 400000 == 0):
            print(str(round(i / len(ANW5_191207) * 100, 2)) + "%")

    ANW5_191207_pred_MT = np.array(ANW5_191207_pred_MT)
    print(abs(np.sqrt(np.mean(ANW5_191207 ** 2)) - np.sqrt(np.mean(ANW5_191207_pred_MT ** 2))) / np.sqrt(
        np.mean(ANW5_191207 ** 2)))
    plt.figure()
    plt.plot(ANW5_191207)
    plt.plot(ANW5_191207_pred_MT)
    plt.title('机翼模态叠加法预测结果')

    # 最高次项为2的平面插值方法
    # 机翼测点坐标（2维）
    wing_vib_pos = np.array([[-0.9, 8.3], [-1.45, 8.3], [-0.7, 8.3], [-0.5, 6.9], [-0.8, 6.9],
                             [0.7, 4.7], [0.2, 4.7], [-1.4, 7.35], [0.15, 1.3], [-0.45, 4.3]])

    ANW5_191207_pred_CZ = []
    X_i = np.vstack((wing_vib_pos[:4, :], wing_vib_pos[5:-2, :]))
    X_i_dg2 = np.hstack((X_i, X_i ** 2))
    for i in range(len(ANW5_191207)):
        y_i = np.concatenate((wing_vib_ary[i, :4], wing_vib_ary[i, 5:-2]))
        lr = LinearRegression().fit(X_i_dg2, y_i)
        ANW5_191207_pred_CZ.append(
            lr.predict(np.hstack((wing_vib_pos[4, :], wing_vib_pos[4, :] ** 2)).reshape(1, -1))[0])
        if (i % 400000 == 0):
            print(str(round(i / len(ANW5_191207) * 100, 2)) + "%")

    ANW5_191207_pred_CZ = np.array(ANW5_191207_pred_CZ)
    print(abs(np.sqrt(np.mean(ANW5_191207 ** 2)) - np.sqrt(np.mean(ANW5_191207_pred_CZ ** 2))) / np.sqrt(
        np.mean(ANW5_191207 ** 2)))
    plt.figure()
    plt.plot(ANW5_191207_pred_CZ)
    plt.plot(ANW5_191207)
    plt.title('机翼平面插值法预测结果')

    '''
    算法集合体
    预测及结果展示
    '''
    ANW5_pred_ALL_191207 = np.hstack((ANW5_191207_pred_MT.reshape(-1, 1), ANW5_191207_pred_CZ.reshape(-1, 1)))
    # LR_meta = LinearRegression().fit(ANW5_pred_ALL_191207, ANW5_191207)
    LR_meta = ensembleTrain(ANW5_pred_ALL_191207, ANW5_191207, step=5000)
    ANW5_pred_final_191207 = LR_meta.predict(ANW5_pred_ALL_191207)
    print(abs(np.sqrt(np.mean(ANW5_191207 ** 2)) - np.sqrt(np.mean(ANW5_pred_final_191207 ** 2))) / np.sqrt(
        np.mean(ANW5_191207 ** 2)))
    plt.figure()
    plt.plot(ANW5_191207)
    plt.plot(ANW5_pred_final_191207)
    plt.title('机翼算法集合体预测结果')