import pickle
import datetime
import matplotlib.pyplot as plt
import sklearn
from flask import Flask, request, Response, json
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import os

from FileUtils.functionUtils import readData, readTime, matchData, loadFile, writeFile, \
    FFTvibData, findKey, matchFPData, timeWindows, ifftOrigin, ifftPredict, printColumns, read_process, \
    write_process
import scipy.stats as scs
from flask_cors import *
import numpy as np

app = Flask(__name__)
# 解决跨域问题
CORS(app, supports_credentials=True)
# 存储进度数据
progress_data = {'getCorr': 0, 'train': 0, 'predict': 0}

# 存储录入的数据
VIBParamsTrain = {}
VIBParamsVerify = {}
FPParamsTrain = {}
FPParamsVerify = {}
SubSampling = False
# predictTime = '200106'
# trainTime = []
vibNames = []
# 设置文件路径
baseURL = "data/"


# vibData = []
# fpData = {}


############################################数据录入#######################################
@app.route('/paramList', methods=['GET'])
def paramList():
    fileName = request.args.get("fileName")
    filePath = baseURL + fileName
    params = printColumns(filePath)
    params.remove("TIME")
    return Response(json.dumps(params), mimetype='application/json')

@app.route('/dataLoad', methods=['GET', 'POST'])
def dataLoad():
    global SubSampling
    global VIBParamsTrain
    global FPParamsTrain
    global FPParamsVerify
    global VIBParamsVerify
    if request.method == 'POST':
        input_data = json.loads(str(request.get_data().decode()))
        FPParamsTrain = input_data.get("FPParams_train")
        VIBParamsTrain = input_data.get("VIBParams_train")
        FPParamsVerify = input_data.get("FPParams_verify")
        VIBParamsVerify = input_data.get("VIBParams_verify")
        SubSampling = input_data.get("SubSampling")
        if SubSampling is None:
            SubSampling = False
        else:
            SubSampling = eval(SubSampling)
        # predictTime = input_data.get("predictTime")
        # trainTime = input_data.get("trainTime")
    return Response(json.dumps('录入数据成功'), mimetype='application/json')


############################################数据预处理#######################################
# 展示用户可选择的振动参数
@app.route('/vibParamList', methods=['GET', 'POST'])
def vibParamList():
    vibs = []
    for vibParam in VIBParamsTrain.keys():
        if type(VIBParamsTrain[vibParam]) is list:
            vibs = vibs + VIBParamsTrain[vibParam]
        else:
            vibs.append(VIBParamsTrain[vibParam])
    return Response(json.dumps(list(set(vibs))), mimetype='application/json')


# 获取指定振动参数的相关性
@app.route('/getCorr', methods=['GET', 'POST'])
def getCorr():
    # global vibData
    # global vibName
    if request.method == 'GET':
        progress_data['getCorr'] = 0
        vibName = request.args.get("vibName")
        dataProcess = request.args.get("dataProcess", type=bool, default=False)
        if vibName + "_FP_col" in os.listdir("Savedfile") and vibName + "_FP_corr" in os.listdir("Savedfile"):
            corr_list = loadFile('Savedfile/' + vibName + '_FP_corr')
            col_list = loadFile('Savedfile/' + vibName + '_FP_col')
            corr = np.array(corr_list)
            col = np.array(col_list)
            vib_FP_highCorr_FP = corr[np.argsort(np.abs(corr))[::-1]].tolist()
            vib_FP_highCol_FP = col[np.argsort(np.abs(corr))[::-1]].tolist()
            progress_data['getCorr'] = float(100)
        else:
            corr_list = []  # 初始化相关性list
            col_list = []  # 初始化变量名list
            zeroCor_191207 = loadFile("Savedfile/zeroCor_191207")
            vibFolder = baseURL + findKey(VIBParamsTrain, vibName)
            # vibFolder = "FTPD-C919-10101-YL-" + trainTime[0] + "-F-01-" + vibFile + ".txt"  # 查找振动数据所在数据包名称
            vibData = readData(vibFolder, vibName, SubSampling)  # 读取振动数据
            vibTime = readTime(vibFolder, SubSampling)  # 读取数据包时间序列
            # 统计飞参个数
            FpNumber = 0
            for file in FPParamsTrain.keys():
                FpNumber += len(FPParamsTrain[file])
            i = 1
            for file in FPParamsTrain.keys():
                time_i = readTime(baseURL + file, SubSampling)  # 读取飞参时间
                for FPParam in FPParamsTrain[file]:
                    if (FPParam in zeroCor_191207):  # 如果某个变量在zeroCor_191207中
                        print(FPParam + " Break!")  # 告知并跳过该变量的计算
                    elif ("Accel" in FPParam):  # 如果某个变量包括"Accel"在变量名称中，该变量为加速度传感器数据
                        print(FPParam + " Break!")  # 告知并跳过该变量的计算
                    else:
                        jCol = readData(baseURL + file, FPParam, SubSampling)  # 读取飞参数据
                        jCol_matched = matchData(vibTime, time_i, jCol)  # 将飞参数据与振动数据时间轴匹配
                        corrJ = scs.pearsonr(jCol_matched, vibData)[0]  # 计算该飞参数据与振动响应数据的相关性
                        corr_list.append(corrJ)  # 将计算的相关性加入corr_list中
                        col_list.append(FPParam)  # 将变量名称加入col_list
                        print("Correlation between " + FPParam + " and " + str(vibName) + " is " + str(
                            corrJ))  # 告知变量名与相关性
                    num_progress = round(i * 100 / FpNumber, 2)
                    print(num_progress)
                    i += 1
                    print(i)
                    progress_data['getCorr'] = num_progress
            print(" Fetching complete!")  # 告知飞参变量的读取完毕
            writeFile("Savedfile/" + vibName + "_FP_corr", corr_list)
            writeFile("Savedfile/" + vibName + "_FP_col", col_list)
            corr = np.array(corr_list)
            col = np.array(col_list)
            vib_FP_highCorr_FP = corr[np.argsort(np.abs(corr))[::-1]].tolist()
            vib_FP_highCol_FP = col[np.argsort(np.abs(corr))[::-1]].tolist()
        result = {"corr_list": vib_FP_highCorr_FP, "col_list": vib_FP_highCol_FP}
        # vibNames.append(vibName)
        return Response(json.dumps(result), mimetype='application/json')


# @app.route('/dataProcess', methods=['GET', 'POST'])
# def dataProcess():
#     global vibData
#     global fpData
#     vibFolder = [findFolder(vibName)[i] for i in range(len(findFolder(vibName))) if flightTime in findFolder(vibName)[i]][0] #查找振动数据所在数据包名称
#     vibTime = readTime(vibFolder)  # 读取数据包时间序列
#     vibData = readData(vibFolder, vibName)  # 读取振动数据
#     zeroCor_191207 = loadFile("Savedfile/zeroCor_191207")
#     for file in FPParams.keys():
#         time_i = readTime(file)
#         for FPParam in  FPParams[file]:
#             if (FPParam in zeroCor_191207):  # 如果某个变量在zeroCor_191207中
#                 print(FPParam + " Break!")  # 告知并跳过该变量的计算
#             elif ("Accel" in FPParam):  # 如果某个变量包括"Accel"在变量名称中，该变量为加速度传感器数据
#                 print(FPParam + " Break!")  # 告知并跳过该变量的计算
#             else:
#                 jCol = readData(file, FPParam)  # 读取飞参数据
#                 jCol_matched = matchData(vibTime, time_i, jCol)  # 将飞参数据与振动数据时间轴匹配
#                 fpData[FPParam] = jCol_matched
#     return '数据处理成功'
#
# @app.route(('/getCorr'), methods=['GET'])
# def getCorr():
#     corr_list = []  # 初始化相关性list
#     col_list = []  # 初始化变量名list
#     zeroCor_191207 = loadFile("Savedfile/zeroCor_191207")
#     for fpPara in fpData.keys():
#         if (fpPara in zeroCor_191207):  # 如果某个变量在zeroCor_191207中
#             print(fpPara + " Break!")  # 告知并跳过该变量的计算
#         elif ("Accel" in fpPara):  # 如果某个变量包括"Accel"在变量名称中，该变量为加速度传感器数据
#             print(fpPara + " Break!")  # 告知并跳过该变量的计算
#         else:
#             corrJ = scs.pearsonr(fpData[fpPara], vibData)[0]
#             corr_list.append(corrJ)  # 将计算的相关性加入corr_list中
#             col_list.append(fpPara)  # 将变量名称加入col_list
#             print("Correlation between " + fpPara + " and " + str(vibName) + " is " + str(
#                 corrJ))  # 告知变量名与相关性
#     print(" Fetching complete!")  # 告知飞参变量的读取完毕
#     # 保存文件
#     writeFile("Savedfile/" + vibName + "_FP_corr", corr_list)
#     writeFile("Savedfile/" + "_" + vibName + "_FP_col", col_list)
#     result = {"corr_list": corr_list, "col_list": col_list}
#     return Response(json.dumps(result), mimetype='application/json')

############################################模型训练#######################################
# 振动参数选择，只能选择做过相关性分析的
@app.route('/vibNameList', methods=['GET'])
def vibNameList():
    for i in os.listdir("Savedfile"):
        if "FP_col" in i:
            vibNames.append(i.split("_")[0])
    return Response(json.dumps(vibNames), mimetype='application/json')


@app.route('/getNum', methods=['GET'])
def getNum():
    global vibNameforTrain
    vibNameforTrain = request.args.get('vibName')
    # vibNameforTrain = "ANF6"
    num = request.args.get('num')
    # 飞参与vib的相关性及其变量名
    vib_191207_FP_corr = loadFile('Savedfile/' + vibNameforTrain + '_FP_corr')
    vib_191207_FP_col = loadFile('Savedfile/' + vibNameforTrain + '_FP_col')
    # list转换为np.array
    vib_191207_FP_corr_ary = np.array(vib_191207_FP_corr)
    vib_191207_FP_col_ary = np.array(vib_191207_FP_col)
    vib_FP_highCol_FP = vib_191207_FP_col_ary[np.argsort(np.abs(vib_191207_FP_corr_ary))[::-1]].tolist()
    # 选择相关性最高的num个飞参
    num = -1 * int(num)
    FPPre = vib_191207_FP_col_ary[np.argsort(np.abs(vib_191207_FP_corr_ary))][num:]
    result = {"col": vib_FP_highCol_FP, "checked": FPPre.tolist()}
    return result


@app.route('/train', methods=['POST'])
def train():
    global LRreg
    global SVRreg
    global ANNreg
    global LR_meta
    global vib_FP_highCol_FP
    global start
    global spendTime
    progress_data['train'] = 0
    input_data = json.loads(str(request.get_data().decode()))
    modelPath = input_data.get("modelPath")
    vib_FP_highCol_FP = input_data.get("FPSelected")
    start = datetime.datetime.now()
    length = 500
    # 需要预测的测点的频率
    select_fre = range(0, 200)
    vib_value_train, fp_value_train = readVIB_FPData(vibNameforTrain, VIBParamsTrain, vib_FP_highCol_FP, FPParamsTrain,
                                                     baseURL, length, 'train', SubSampling)
    vib_value_verify, fp_value_verify = readVIB_FPData(vibNameforTrain, VIBParamsVerify, vib_FP_highCol_FP,
                                                       FPParamsVerify, baseURL, length, 'train', SubSampling)
    vib_abs_train, vib_ang_train = FFTvibData(vib_value_train, length)
    vib_abs_verify, vib_ang_verify = FFTvibData(vib_value_verify, length)
    vib_ary_train = vib_abs_train[:, select_fre]
    vib_ary_verify = vib_abs_verify[:, select_fre]
    # 对转换后的频域信号进行滑动时间窗移动平均
    vib_ary_train = timeWindows(vib_ary_train, length=10)
    progress_data['train'] = progress_data['train'] + 5.00              # 65%
    vib_ary_verify = timeWindows(vib_ary_verify, length=10)
    progress_data['train'] = progress_data['train'] + 5.00              # 70%
    if modelPath is None or modelPath == "":
        '''
        线性回归
        '''
        LRreg = LinearRegression().fit(fp_value_train, vib_ary_train)
        vib_pred_LR_verify = LRreg.predict(fp_value_verify)
        '''
        支持向量机回归
        '''
        SVRreg = []
        base_SVRreg = make_pipeline(SVR(kernel='rbf', C=100, tol=0.1, gamma=0.1,
                                        epsilon=0.5, max_iter=1000))
        base_process = progress_data['train']
        for i in range(len(select_fre)):
            SVRreg.append(sklearn.base.clone(base_SVRreg).fit(fp_value_train, vib_ary_train[:, i]))
            num_progress = round((i + 1) * 100 * 0.25 / len(select_fre), 2) + base_process  # 95%
            progress_data['train'] = num_progress
        vib_pred_SVR_verify = np.array([SVRreg[i].predict(fp_value_verify) for i in range(len(select_fre))]).T

        '''
        人工神经网络
        '''
        ANNreg = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100,),
                                                              alpha=0.001, learning_rate_init=0.001,
                                                              random_state=1, max_iter=100))
        ANNreg.fit(fp_value_train, vib_ary_train)
        vib_pred_ANN_verify = ANNreg.predict(fp_value_verify)
        '''
        算法集合体
        预测及结果展示
        '''
        LR_meta = []
        base_LR_meta = LinearRegression()
        for i in range(len(select_fre)):
            vib_pred_ALL_verify = np.hstack((vib_pred_LR_verify[:, i].reshape(-1, 1),
                                             vib_pred_SVR_verify[:, i].reshape(-1, 1),
                                             vib_pred_ANN_verify[:, i].reshape(-1, 1)))
            LR_meta.append(sklearn.base.clone(base_LR_meta).fit(vib_pred_ALL_verify, vib_ary_verify[:, i]))
    else:
        with open('model/' + modelPath, 'rb') as f:
            LRreg = pickle.load(f)  # 从f文件中提取出模型赋给clf2
            SVRreg = pickle.load(f)
            ANNreg = pickle.load(f)
            LR_meta = pickle.load(f)
        '''
       线性回归
       '''
        LRreg = LRreg.fit(fp_value_train, vib_ary_train)
        vib_pred_LR_verify = LRreg.predict(fp_value_verify)
        '''
        支持向量机回归
        '''
        SVRreg = []
        base_SVRreg = make_pipeline(SVR(kernel='rbf', C=100, tol=0.1, gamma=0.1,
                                        epsilon=0.5, max_iter=1000))
        base_process = progress_data['train']
        for i in range(len(select_fre)):
            SVRreg.append(sklearn.base.clone(base_SVRreg).fit(fp_value_train, vib_ary_train[:, i]))
            num_progress = round((i + 1) * 100 * 0.25 / len(select_fre), 2) + base_process  # 95%
            progress_data['train'] = num_progress
        vib_pred_SVR_verify = np.array([SVRreg[i].predict(fp_value_verify) for i in range(len(select_fre))]).T
        progress_data['train'] = num_progress
        '''
        人工神经网络
        '''
        ANNreg.fit(fp_value_train, vib_ary_train)
        vib_pred_ANN_verify = ANNreg.predict(fp_value_verify)
        '''
        算法集合体
        预测及结果展示
        '''
        LR_meta = []
        base_LR_meta = LinearRegression()
        for i in range(len(select_fre)):
            vib_pred_ALL_verify = np.hstack((vib_pred_LR_verify[:, i].reshape(-1, 1),
                                             vib_pred_SVR_verify[:, i].reshape(-1, 1),
                                             vib_pred_ANN_verify[:, i].reshape(-1, 1)))
            LR_meta.append(sklearn.base.clone(base_LR_meta).fit(vib_pred_ALL_verify, vib_ary_verify[:, i]))
    end = datetime.datetime.now()
    spendTime = str((end - start).seconds)
    progress_data['train'] = 100.00
    result = {"VIBParamsTrain": VIBParamsTrain, "FPParamsTrain": FPParamsTrain, "vibNameforTrain": vibNameforTrain,
              "fpNameforTrain": vib_FP_highCol_FP, "beginTime": str(start.strftime('%Y-%m-%d %H:%M:%S')),
              "spendTime": spendTime}
    return Response(json.dumps(result), mimetype='application/json')


# 模型保存
@app.route('/SaveModel', methods=['GET'])
def SaveModel():
    savePath = request.args.get("saveName")
    with open('model/' + savePath, 'wb') as f:  # python路径要用反斜杠
        pickle.dump(LRreg, f)  # 将模型dump进f里面
        pickle.dump(SVRreg, f)
        pickle.dump(ANNreg, f)
        pickle.dump(LR_meta, f)
    modelName = savePath.split('.')[0]
    writeFile("model/" + modelName + "_input", vib_FP_highCol_FP)
    with open('model/' + modelName + '.txt', 'w', encoding='utf-8') as f1:
        f1.write("该组预测模型参数应用了" + str(VIBParamsTrain) + "数据文件与标识符、该组预测模型参数应用了" + str(FPParamsTrain) + "飞参数据文件与标识符、"
                                                                                                     "针对" + str(
            vibNameforTrain) + "振动数据进行的训练、敏感飞参有" + str(vib_FP_highCol_FP)
                 + "、" + str(start.strftime('%Y-%m-%d %H:%M:%S')) + "开始训练、训练花了" + spendTime + "秒")
    print(vib_FP_highCol_FP)
    return Response(json.dumps('保存模型成功'), mimetype='application/json')


############################################模型预测#######################################

@app.route('/predict', methods=['POST'])
def predict():
    input_data = json.loads(str(request.get_data().decode()))
    modelPath = input_data.get("modelPath")
    VIBForPredict = input_data.get('VIBForPredict')
    vib_FP_highCol_FP = input_data.get('FPForPredict')
    zeroCor_191207 = loadFile("Savedfile/zeroCor_191207")
    fpPredict = []
    for key in vib_FP_highCol_FP.keys():
        for v in vib_FP_highCol_FP[key]:
            if v in zeroCor_191207.tolist():
                continue
            fpPredict.append(v)
    with open('model/' + modelPath, 'rb') as f:  # python路径要用反斜杠
        LR = pickle.load(f)
        SVR = pickle.load(f)
        ANN = pickle.load(f)
        LRmeta = pickle.load(f)
    features = loadFile("model/" + modelPath.split('.')[0] + "_input")
    # fpPredict.sort()
    # features.sort()
    # if fpPredict != features:
    #     return Response(json.dumps('模型需要的特征与选择的不一致'), mimetype='application/json')
    # if features !=
    # 表示进行频域变换的每个单元监控点的数量
    if len(fpPredict) < len(features):
        return Response(json.dumps({"code": 500, "msg": "选择的特征数少于该模型需要的特征数，请重选"}), mimetype='application/json')
    else:
        fpPredict = fpPredict[0:len(features)]
    NUM = 500
    # 需要预测的测点的频率
    select_fre = range(0, 200)
    vibNameforPredict = list(VIBForPredict.values())[0]
    vib_value_pre, FP_value_pre = readVIB_FPData(vibNameforPredict, VIBForPredict, features, vib_FP_highCol_FP,
                                                 baseURL, NUM, 'predict')
    vib_abs_pre, vib_ang_pre = FFTvibData(vib_value_pre, NUM)
    # vib_200102_ary = vib_abs_200102[:, select_fre]
    # 对转换后的频域信号进行滑动时间窗移动平均
    # vib_200102_ary = timeWindows(vib_200102_ary, length=10)
    # model = LRreg
    vib_pred_LR = LR.predict(FP_value_pre)
    vib_pred_SVR = np.array([SVR[i].predict(FP_value_pre) for i in range(len(select_fre))]).T
    vib_pred_ANN = ANN.predict(FP_value_pre)
    vib_pred_final = np.array([LRmeta[i].predict(np.hstack((vib_pred_LR[:, i].reshape(-1, 1),
                                                            vib_pred_SVR[:, i].reshape(-1, 1),
                                                            vib_pred_ANN[:, i].reshape(-1, 1)))) \
                               for i in range(len(select_fre))]).T
    progress_data['predict'] = 75.00  # 75%
    vib_true_pre = ifftOrigin(timeWindows(vib_abs_pre, length=10), vib_ang_pre)
    vib_pred_pre = ifftPredict(vib_pred_final, select_fre, 500)
    progress_data['predict'] = 90.00  # 90%
    plt.figure()
    plt.plot(vib_true_pre, label='true')
    plt.plot(vib_pred_pre, label='predict')
    plt.legend()
    plt.savefig(vibNameforPredict + ".png")
    RMSE = abs(np.sqrt(np.mean(vib_true_pre ** 2)) - np.sqrt(np.mean(vib_pred_pre ** 2))) * 100 / np.sqrt(
        np.mean(vib_true_pre ** 2))
    rms = abs(np.sqrt(np.mean(vib_true_pre ** 2)) - np.sqrt(np.mean(vib_pred_pre ** 2)))
    progress_data['predict'] = 100.00
    result = {"code": 200, "msg": "模型预测成功",
              "data": {"trueValue": vib_true_pre.tolist()[::1000], "predictValue": vib_pred_pre.tolist()[::1000],
                       "RMS": rms, "RMSE": RMSE}}
    return Response(json.dumps(result), mimetype='application/json')


######################################进度条服务####################################
@app.route('/show_progress', methods=['GET'])
def show_progress():
    name = request.args.get("serviceName")
    process = progress_data[name]
    #如果进度条到了100，则置零
    if process == 100.00:
        progress_data[name] = 0
    return Response(json.dumps(process), mimetype='application/json')



def readVIB_FPData(vib_Name, VIBParams, FPnames, FPParams, baseURL, length, methodName, SubSampling=False):
    '''
    抓取vibNames和FPnames中变量在flightTime架次的数据
    其中vib变量按照length进行重构，方便后期进行频域变换
    FP变量按照重构后的vib变量时间轴进行匹配，取每段时间内的均值
    返回时间匹配后的振动变量和飞参变量
    返回值类型为np.array
    '''

    vibFile = findKey(VIBParams, vib_Name)
    vibFolder = baseURL + vibFile
    vib_value = readData(vibFolder, vib_Name, SubSampling)  # 读取相应的振动响应变量数据
    vib_value = np.array(vib_value[:(len(vib_value) - len(vib_value) % length)])
    vib_value = np.reshape(vib_value, (-1, length))
    progress_data[methodName] = progress_data[methodName] + 2.00

    vib_time = readTime(vibFolder, SubSampling)  # 读取振动响应数据的时间轴
    vib_time = np.array(vib_time[:(len(vib_time) - len(vib_time) % length)])
    vib_time = np.reshape(vib_time, (-1, length))
    vib_time = vib_time[:, 0]
    progress_data[methodName] = progress_data[methodName] + 2.00

    FPnames = list(reversed(FPnames))
    fpFile = findKey(FPParams, FPnames[0])
    FPFolder = baseURL + fpFile
    FP_time = readTime(FPFolder, SubSampling)  # 读取飞参时间轴
    progress_data[methodName] = progress_data[methodName] + 2.00

    FP_value = readFPData(FPnames, FPParams, baseURL, methodName)
    FP_value = np.array(matchFPData(FP_time, vib_time, FP_value))
    progress_data[methodName] = progress_data[methodName] + 2.00
    return (vib_value, FP_value)  # 输出振动变量矩阵


def readFPData(FPnames, FPParams, baseURL, methodName):
    '''
    抓取FPnames中变量在flightTime架次的数据
    返回时间匹配后的飞参变量矩阵
    返回值类型为ndarray
    '''
    base_process = progress_data[methodName]
    for i in range(len(FPnames)):  # 在目标飞参（FPnames）中开始循环
        fpFile = findKey(FPParams, FPnames[i])  # 提取飞参所在的数据包名称
        fpFolder = baseURL + fpFile
        # fpFolder = "FTPD-C919-10101-YL-" + flightTime + "-F-01-" + fpFile + ".txt"
        if (i == 0):  # 第一次读取飞参时
            FP_i = readData(fpFolder, FPnames[i])  # 读取飞参数据
            FP_i_matched = np.array(FP_i).reshape(-1, 1)  # 将飞参数据从list转换成numpy
            FP_mat = FP_i_matched.copy()  # 初始化飞参矩阵

        else:  # 不是第一次读取飞参时
            FP_i = readData(fpFolder, FPnames[i])  # 读取飞参数据
            FP_i_matched = np.array(FP_i).reshape(-1, 1)  # 将飞参数据从list转换成numpy
            FP_mat = np.hstack((FP_mat, FP_i_matched))  # 将新的飞参变量加入现有的飞参矩阵
        print(str((i + 1) / len(FPnames) * 100) + "%")  # 打印百分比进度
        ## 保存进度变量
        if methodName == 'predict':
            num_progress = round((i + 1) * 100 * 0.22 * 2 / len(FPnames), 2) + base_process
        else:
            num_progress = round((i + 1) * 100 * 0.22 / len(FPnames), 2) + base_process
        progress_data[methodName] = num_progress
    return (FP_mat)  # 输出飞参矩阵


if __name__ == '__main__':
    app.run(host='0.0.0.0')
