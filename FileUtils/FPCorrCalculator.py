# 飞参和振动相关性分析
from FileUtils.functionUtils import findFolder, readTime, readData, loadFile, printFiles, printColumns, matchData
import scipy.stats as scs
import os


def FPCorrCalculator(vibName, flightTime):
    '''
    计算输入变量（vibName）与某个架次（flightTime）飞行参数的Pearson's Correlation
    返回2个变量corr_list, col_list
    corr_list: 相关性系数
    col_list: 与相关性系数对应的变量名
    (corr_list[i]为col_list[i]与vibName的Person's Correlation)
    返回值类型均为list
    '''

    vibFolder = \
        [findFolder(vibName)[i] for i in range(len(findFolder(vibName))) if flightTime in findFolder(vibName)[i]][
            0]  # 查找振动数据所在数据包名称
    vibTime = readTime(vibFolder)  # 读取数据包时间序列
    vibData = readData(vibFolder, vibName)  # 读取振动数据
    zeroCor_191207 = loadFile("Savedfile/zeroCor_191207")  # zeroCor_191207为191207架次中的衡量变量，与振动响应数据无任何相关性

    print("Vib Fetching complete!")  # 告知振动数据已经提取完毕

    corr_list = []  # 初始化相关性list
    col_list = []  # 初始化变量名list

    # fileList_ = printFiles++()
    # fileFT = [fileList_[i] for i in range(len(fileList_)) if flightTime in fileList_[i]]

    for i in range(7):  # 在全部数据包中搜索
        filenameList = printFiles(os.getcwd())  # 将全部数据包名称存入filenameList变量
        filenameStr = filenameList[i]  # 将搜索至数据包名称存至filenameStr变量

        if (filenameStr.endswith("CAOWEN-664002-32.txt") or filenameStr.endswith("CAOWEN-664003-32.txt") or
                filenameStr.endswith("CAOWEN-ANA003-32.txt") or filenameStr.endswith("FCS-664002-16.txt")):  # 筛选飞参数据包
            print(filenameStr + " Fetching Started!")  # 告知开始提取某飞参数据包的数据
            # print (filenameStr)
            time_i = readTime(filenameStr)  # 将该飞参数据的时间数据存至time_i变量
            columns_i = printColumns(filenameStr)  # 将该飞参数据的变量名称存至columns_i变量
            print("In total of " + str(len(columns_i)) + " columns")  # 告知一共有多少变量

            for j in range(1, len(columns_i)):  # 搜索提取的变量名称

                print("Start " + str(j) + "/" + str(len(columns_i)))  # 告知开始计算某个变量

                if (columns_i[j] in zeroCor_191207):  # 如果某个变量在zeroCor_191207中
                    print(columns_i[j] + " Break!")  # 告知并跳过该变量的计算

                elif ("Accel" in columns_i[j]):  # 如果某个变量包括"Accel"在变量名称中，该变量为加速度传感器数据
                    print(columns_i[j] + " Break!")  # 告知并跳过该变量的计算

                else:  # 如果不符合以上两种情况
                    jCol = readData(filenameStr, columns_i[j])  # 读取飞参数据
                    jCol_matched = matchData(vibTime, time_i, jCol)  # 将飞参数据与振动数据时间轴匹配
                    corrJ = scs.pearsonr(jCol_matched, vibData)[0]  # 计算该飞参数据与振动响应数据的相关性
                    corr_list.append(corrJ)  # 将计算的相关性加入corr_list中
                    col_list.append(columns_i[j])  # 将变量名称加入col_list
                    print("Correlation between " + columns_i[j] + " and " + str(vibName) + " is " + str(
                        corrJ))  # 告知变量名与相关性

            print(filenameStr + " Fetching complete!")  # 告知飞参变量的读取完毕

    return (corr_list, col_list)  # 输出corr_list与col_list
