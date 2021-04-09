import os
import pickle
import pandas as pd

import numpy as np
from sklearn.linear_model import LinearRegression


# basePath = "E:\民机结构振动响应预测平台\data_919"


def printFiles(path):
    '''
    返回所有的txt文件
    返回值类型为list
    '''
    filenameList_ = []  # 初始化输出list
    # path = "E:\民机结构振动响应预测平台\data_919"
    for filename_ in os.listdir(path):  # 搜索目标文件夹中的全部文件
        if (filename_.endswith(".txt")):  # 匹配以".txt"结尾的文件名
            # filenameList_.append(filename_.rsplit('.')[0])  # 在输出list中添加匹配到的文件名
            filenameList_.append(filename_)
            # print (filename)
    return (filenameList_)  # 输出list


def printColumns(filename_):
    '''
    返回输入文件名（filename_)中的所有变量名
    返回值类型为list
    '''

    f_ = open(filename_, "r")  # 打开目标文件
    colname = f_.readline().split("\t")  # 读取变量名称
    colname[-1] = colname[-1].strip()  # 去掉空格
    # print (colname)
    f_.close()
    return (colname)  # 输出变量名称


def readTime(filename_, subsampling=False):
    '''
    返回输入文件(filename_)的时间序列并统一为毫秒（1000毫秒=1秒）单位。
    返回值类型为list
    '''
    j = 0
    targetFile = open(filename_, "r")  # 打开目标文件
    timeAry = []  # 初始化输出list

    while True:

        line = targetFile.readline()  # 以行为单位读取目标文件

        if (line == ""):  # 如果读取到的行为空
            break  # 跳过

        if (line[:4] != "TIME"):  # 如果读取到的行不是"TIME"
            j += 1
            if subsampling:
                if (j-0) % 5 == 0:
                    timeList = line.split("\t")[0].split(":")  # 将目标行以":"分割，时：分：秒：毫秒
                    timei = 0  # 初始化以毫秒为单位的时间计数器
                    for i in range(4):  # 读取由时、分、秒、毫秒构成的timeList
                        if (i < 3):
                            timei += int(timeList[i]) * 60 ** (2 - i)  # 转换时分秒
                        else:
                            timei *= 1000
                            timei += int(timeList[i])  # 转换为毫秒
                    timeAry.append(timei)  # 将timei加入输出list
            else:
                timeList = line.split("\t")[0].split(":")  # 将目标行以":"分割，时：分：秒：毫秒
                timei = 0  # 初始化以毫秒为单位的时间计数器
                for i in range(4):  # 读取由时、分、秒、毫秒构成的timeList
                    if (i < 3):
                        timei += int(timeList[i]) * 60 ** (2 - i)  # 转换时分秒
                    else:
                        timei *= 1000
                        timei += int(timeList[i])  # 转换为毫秒
                timeAry.append(timei)  # 将timei加入输出list
    return (timeAry)  # 输出timeAry


def readData(filename_, colname_, subsampling=False):
    df = pd.read_table(filename_, sep='\t')
    if subsampling:
        return df[colname_].fillna(0, inplace=True).tolist()[::5]
    else:
        return df[colname_].fillna(0, inplace=True).tolist()
    # i = 0
    # '''
    # 返回输入文件(filename_)中某个变量（column_）
    # 返回值类型为list
    # '''
    # targetFile = open(filename_, "r")  # 打开目标文件
    #
    # colAry = []  # 初始化输出list
    #
    # while True:
    #
    #     line = targetFile.readline()  # 以行为单位读取文件
    #
    #     if (line == ""):  # 若行为空
    #         break  # 跳过
    #
    #     if (line[:4] == "TIME"):  # 如果读取行前4个字符为“TIME”，则该行是第一行，则记录变量名称和序号
    #         firstLine = line.split("\t")  # 分割第一行（包含全部变量名称的行）
    #         # firstLine[-1] = firstLine[-1][:-2]
    #         # print (firstLine[-1])
    #         for i in range(len(firstLine)):  # 搜索第一行
    #             # print (firstLine[i])
    #             if (firstLine[i].strip() == colname_):  # 如果搜索变量名与目标变量匹配（colname_）
    #
    #                 colIndex = i  # 记录该变量所在位置的序号
    #                 # print (i)
    #
    #     else:  # 如果读取行前4个字符不为“TIME”，则该行不是第一行，则记录变量具体数据
    #         i += 1
    #         if subsampling:
    #             if i % 5 == 0:
    #                 colAry.append(float(line.split("\t")[colIndex]))  # 分割行后，读取对应序号的数据，转换为float类型，加入输出list
    #         else:
    #             colAry.append(float(line.split("\t")[colIndex]))
    #
    # return (colAry)  # 返回输出list


def matchData(timeA, timeB, aviB):
    '''
    timeB为输入数据aviB的时间序列
    将根据timeA的时间序列匹配
    并返回时间匹配后的aviB
    返回值类型为list
    '''

    counter = 0  # 计数器初始化为0，计数器代表aviB所在位置
    returnList = []  # 初始化输出list为空

    for i in range(len(timeA)):  # 以timeA的序列为基准进行匹配

        if (counter + 1 >= len(timeB)):  # 如果计数大小已经超过B的长度
            counter -= 1  # 计数器停止增长

        elif (timeA[i] == timeB[counter + 1]):  # 或者如果timeB的下一位置可匹配timeA的当前位置
            counter += 1  # 计数器加1

        elif (timeA[i] == timeB[counter + 1] + 1):  # 或者如果timeB的下一位置加1可匹配timeA的当前位置（time的取值频率不固定，如果对照实际数据理解会更加清晰）
            counter += 1  # 计数器加1

        returnList.append(aviB[counter])  # 输出list加入当前位置B的变量

    return (returnList)  # 返回输出list


def writeFile(writeName, writeFile):
    '''
    将文件（writeFile）存入文件地址（writeName）
    无返回
    '''
    f_ = open(writeName, "wb")  # 在writeName地址创建新文件，如果新文件已经存在将打开并进行修改
    pickle.dump(writeFile, f_)  # 将变量writeFile写入打开的文件
    f_.close()  # 关闭文件


def loadFile(loadName):
    '''
    读取地址（loadName）的文件
    返回值类型为string
    '''

    f_ = open(loadName, "rb")  # 以已读方式打开loadName地址的文件
    returnFile = pickle.load(f_)  # 读取打开的文件，并将其存入returnFile变量
    f_.close()  # 关闭打开的文件

    return (returnFile)  # 输出returnFile


def findFolder(colName):
    '''
    返回变量（colname）所在的文件夹
    返回值类型为list
    '''
    fList = printFiles(os.getcwd())  # 将全部文件名称存入fList变量
    returnList = []  # 初始化returnList变量为空
    for i in range(len(fList)):  # 检索文件列表
        if (colName in printColumns(fList[i])):  # 如果目标变量名colName包含在检索到的文件中
            returnList.append(fList[i])  # 将该文件名加入returnList变量

    return (returnList)  # 输出returnList变量


def readFPData(FPnames, FPParams, baseURL, is_process=False):
    '''
    抓取FPnames中变量在flightTime架次的数据
    返回时间匹配后的飞参变量矩阵
    返回值类型为ndarray
    '''
    if is_process:
        base_process = read_process('train_process')
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
        if is_process:
            ## 保存进度变量
            num_progress = round((i + 1) * 100 * 0.26 / len(FPnames), 2) + base_process
            write_process('train_process', num_progress)
    return (FP_mat)  # 输出飞参矩阵


def matchFPData(timeA, timeB, aviA):
    '''
    timeA为输入数据aviA的时间序列
    将根据timeB的时间序列匹配
    并返回时间匹配后的aviA
    返回值类型为list
    '''
    counter = 0  # 计数器初始化为0，计数器代表aviB所在位置
    returnList = []  # 初始化输出list为空

    for i in range(len(timeB) - 1):  # 以timeB的序列为基准进行匹配
        temp = []
        for j in range(1000):
            if (timeA[counter] >= timeB[i]):
                temp.append(aviA[counter, :])
                counter += 1
                if (timeA[counter] >= timeB[i + 1]):
                    break
        returnList.append(np.mean(temp, axis=0))  # 输出list加入当前位置B的变量

    temp = aviA[counter:-1, :]
    returnList.append(np.mean(temp, axis=0))  # 输出list加入当前位置B的变量

    return (returnList)  # 输出returnList变量


def readVIB_FPData(vib_Name, VIBParams, FPnames, FPParams, baseURL, length, SubSampling=False, is_process=False):
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
    if is_process:
        write_process('train_process', 2.00 + read_process('train_process'))
    vib_time = readTime(vibFolder, SubSampling)  # 读取振动响应数据的时间轴
    vib_time = np.array(vib_time[:(len(vib_time) - len(vib_time) % length)])
    vib_time = np.reshape(vib_time, (-1, length))
    vib_time = vib_time[:, 0]
    if is_process:
        write_process('train_process', 2.00 + read_process('train_process'))

    FPnames = list(reversed(FPnames))
    fpFile = findKey(FPParams, FPnames[0])
    # FPFolder = "FTPD-C919-10101-YL-" + flightTime + "-F-01-" + fpFile + ".txt"
    FPFolder = baseURL + fpFile
    # FPFolder = [findFolder(FPnames[0])[j] for j in range(3) if flightTime in findFolder(FPnames[0])[j]][
    #     0]  # 提取飞参所在的数据包名称
    FP_time = readTime(FPFolder, SubSampling)  # 读取飞参时间轴
    if is_process:
        write_process('train_process', 2.00 + read_process('train_process'))
    FP_value = readFPData(FPnames, FPParams, baseURL, is_process)
    FP_value = np.array(matchFPData(FP_time, vib_time, FP_value))
    if is_process:
        write_process('train_process', 2.00 + read_process('train_process'))
    return (vib_value, FP_value)  # 输出振动变量矩阵


def FFTvibData(vib_value, length):
    '''
    对处理后的振动数据进行快速傅里叶变换（FFT）
    将时域信号转变为频域信号
    返回值类型为np.array，分别为不同频率下振动的幅值和角度
    '''
    vib_fft = [np.fft.fft(vib_value[i, :]) for i in range(vib_value.shape[0])]
    vib_fft_abs = np.abs(vib_fft)
    vib_fft_ang = np.angle(vib_fft)

    return (vib_fft_abs, vib_fft_ang)


def timeWindows(vib_fft, length=10):
    '''
    对转换后的频域信号进行滑动时间窗平均，窗口的长度为length
    返回值类型为np.array，是经过移动平均之后的频域信号
    '''
    vib_fft_TW = []
    for i in range(vib_fft.shape[1]):
        temp = vib_fft[:length, i].tolist()
        for j in range(vib_fft.shape[0] - length):
            a = np.mean(vib_fft[j:(j + length), i])
            temp.append(a)
        vib_fft_TW.append(temp)

    vib_fft_TW = np.array(vib_fft_TW).T

    return (vib_fft_TW)


def ifftOrigin(vib_fft_abs, vib_fft_ang):
    '''
    根据经过滑动时间窗平均得到的频域信号进行快速傅里叶逆变换得到原始的时域信号
    返回值类型为np.array
    '''
    vib_fft = abs(vib_fft_abs) * np.exp(1j * vib_fft_ang)
    vib_origin = np.array([np.fft.ifft(vib_fft[i, :]).real for i in range(vib_fft.shape[0])]).reshape(-1)

    return (vib_origin)


def ifftPredict(vib_fft_pre, select_fre, NUM=500):
    '''
    根据预测得到的频域信号进行快速傅里叶逆变换得到原始的时域信号
    返回值类型为np.array
    '''
    vib_fft_pre = abs(vib_fft_pre) * np.exp(1j * np.random.uniform(0, 2 * np.pi, (vib_fft_pre.shape)))
    vib_fft = np.zeros([vib_fft_pre.shape[0], NUM], dtype=complex)
    vib_fft[:, select_fre] = vib_fft_pre
    select_fre = list(NUM - np.array(select_fre))
    select_fre.remove(NUM)
    vib_fft_pre = vib_fft_pre.real - 1j * vib_fft_pre.imag
    vib_fft[:, select_fre] = vib_fft_pre[:, 1:]
    vib_pre = np.array([np.fft.ifft(vib_fft[i, :]).real for i in range(vib_fft_pre.shape[0])]).reshape(-1)

    return (vib_pre)


def ensembleTrain(X, y, step=5000):
    '''
    训练算法集合体的权重，使得均方根值误差尽可能小，并且通过分段避免过拟合
    step表示每段含有的监控点数量
    返回值类型为sklearn定义的linermodel
    '''
    X, y = X[:(len(X) - len(X) % step), :], y[:(len(y) - len(y) % step), ]
    X_new = []
    for i in range(X.shape[1]):
        temp = X[:, i].reshape([-1, step])
        temp = [np.sqrt(np.mean(temp[j, :] ** 2)) for j in range(temp.shape[0])]
        X_new.append(temp)
    X_new = np.array(X_new).T
    y_new = y.reshape([-1, step])
    y_new = [np.sqrt(np.mean(y_new[j, :] ** 2)) for j in range(y_new.shape[0])]

    LR_meta = LinearRegression(fit_intercept=False).fit(X_new, y_new)

    return (LR_meta)


def vibMatFetch(vibNames, flightTime):
    '''
    抓取vibNames中变量在flightTime架次的数据
    返回值类型为np.array
    '''
    vib_data_time = []
    for i in range(len(vibNames)):
        vibFolder = [findFolder(vibNames[i])[j] for j in range(len(findFolder(vibNames[i]))) if
                     flightTime in findFolder(vibNames[i])[j]][0]
        vib_data_time.append(readData(vibFolder, vibNames[i]))
        print(flightTime + vibNames[i] + " Finished!")

    vib_data_time = np.array(vib_data_time).T

    return (vib_data_time)  # 输出振动变量矩阵


def TrainDataUnSamping(TrainX, TrainY, SaveNum):
    '''
    对训练集进行随机欠采样
    最终保留SaveNum个样本作为模型的输入
    '''

    SaveIndex = np.random.randint(0, len(TrainX), SaveNum)
    TrainSaveX, TrainSaveY = TrainX[SaveIndex,], TrainY[SaveIndex,]

    return (TrainSaveX, TrainSaveY)


def findKey(map, value):
    '''
     获取参数在那个文件当中
     '''
    for fileName in map.keys():
        if value in map[fileName]:
            return fileName


# 读取进度信息
def read_process(process_name):
    if os.path.exists(process_name):
        with open(process_name, 'rb') as f:
            process = pickle.load(f)
    else:
        process = 0
    return process


# 存储进度信息
def write_process(process_name, process_value):
    f = open(process_name, 'wb')
    # 将变量存储到目标文件中区
    pickle.dump(process_value, f)
    # 关闭文件
    f.close()
