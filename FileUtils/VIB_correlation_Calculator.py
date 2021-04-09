# 指定测点与其他测点振动相关性分析
from FileUtils.functionUtils import printFiles, findFolder, readData, printColumns
import scipy.stats as scs


def VIB_correlation_Calculator(VIBname, VIBtime):
    '''
    计算名为VIBname的振动数据与其他振动数据在VIBtime架次的Pearson's corrletion
    返回2个变量corr_list, col_list
    corr_list: 相关性系数
    col_list: 与相关性系数对应的变量名
    (corr_list[i]为col_list[i]与VIBname的Person's Correlation)
    返回值类型均为list
    '''
    fileList = printFiles()  # 读取全部文件名

    corList = []  # 初始化corList为空
    colList = []  # 初始化colList为空

    VIBList = findFolder(VIBname)  # 将VIBname所在的数据包名称存入VIBList变量
    for i in range(len(VIBList)):  # 搜索VIBList
        if (VIBtime in VIBList[i]):  # 如果检索到VIBtime架次
            VIBtarget = readData(VIBList[i], VIBname)  # 读取目标振动响应数据

    for i in range(len(fileList)):  # 搜索fileList

        if (VIBtime in fileList[i]):  # 如果搜索至的fileList变量包含VIBtime（架次）
            filenameStr = fileList[i]  # 提取该文件名称
            if (filenameStr.endswith("00VIB-ANA003-512_new.txt")):  # 如果文件名以"00VIB-ANA003-512_new.txt"结尾
                print("00VIB-ANA003-512 Fetching Started!")  # 告知开始提取"00VIB-ANA003-512_new.txt"的变量

                columns_ANA003 = printColumns(filenameStr)  # 提取该数据包中的变量名称
                print("In total of " + str(len(columns_ANA003)) + " columns")  # 告知一共有多少变量

                for j in range(1, len(columns_ANA003)):  # 搜索变量名称

                    print("Start " + str(j) + "/" + str(len(columns_ANA003)))  # 告知已开始某变量的计算
                    jCol = readData(filenameStr, columns_ANA003[j])  # 读取该变量代表振动数据

                    corrJ = scs.pearsonr(jCol, VIBtarget)[0]  # 计算该振动数据与目标振动数据间的皮尔逊相关系数
                    corList.append(corrJ)  # 将计算好的相关系数加入corList
                    colList.append(columns_ANA003[j])  # 将变量名称加入colList
                    print("Correlation between " + columns_ANA003[j] + " and " + VIBname + " is " + str(
                        corrJ))  # 告知变量名称和相关性系数

                print("00VIB-ANA003-512 Fetching complete!")  # 告知该数据包的振动数据提取和计算已完成

            if (filenameStr.endswith("FLUTTER-ANA001-512_new.txt")):  # 如果文件以"FLUTTER-ANA001-512_new.txt"结尾
                print("FLUTTER-ANA001-512 Fetching Started!")  # 告知"FLUTTER-ANA001-512_new.txt"的数据读取开始

                columns_ANA001 = printColumns(filenameStr)  # 提取该数据包中的全部变量名称
                print("In total of " + str(len(columns_ANA001)) + " columns")  # 告知一共有多少变量

                for j in range(1, len(columns_ANA001)):  # 搜索变量名称

                    print("Start " + str(j) + "/" + str(len(columns_ANA001)))  # 告知已开始某变量的计算
                    jCol = readData(filenameStr, columns_ANA001[j])  # 读取该变量代表振动数据

                    corrJ = scs.pearsonr(jCol, VIBtarget)[0]  # 计算该振动数据与目标振动数据间的皮尔逊相关系数
                    corList.append(corrJ)  # 将计算好的相关系数加入corList
                    colList.append(columns_ANA001[j])  # 将变量名称加入colList
                    print("Correlation between " + columns_ANA001[j] + " and " + VIBname + " is " + str(
                        corrJ))  # 告知变量名称和相关性系数

                print("FLUTTER-ANA001-512 Fetching complete!")  # 告知该数据包的振动数据提取和计算已完成

            if (filenameStr.endswith("FLUTTER-ANA002-512_new.txt")):  # 如果文件以"FLUTTER-ANA002-512_new.txt"结尾
                print("FLUTTER-ANA002-512 Fetching Started!")  # 告知"FLUTTER-ANA002-512_new.txt"的数据读取开始

                columns_ANA002 = printColumns(filenameStr)  # 提取该数据包中的全部变量名称
                print("In total of " + str(len(columns_ANA002)) + " columns")  # 告知一共有多少变量

                for j in range(1, len(columns_ANA002)):  # 搜索变量名称

                    print("Start " + str(j) + "/" + str(len(columns_ANA002)))  # 告知已开始某变量的计算
                    jCol = readData(filenameStr, columns_ANA002[j])  # 读取该变量代表振动数据

                    corrJ = scs.pearsonr(jCol, VIBtarget)[0]  # 计算该振动数据与目标振动数据间的皮尔逊相关系数
                    corList.append(corrJ)  # 将计算好的相关系数加入corList
                    colList.append(columns_ANA002[j])  # 将变量名称加入colList
                    print("Correlation between " + columns_ANA002[j] + " and " + VIBname + " is " + str(
                        corrJ))  # 告知变量名称和相关性系数

                print("FLUTTER-ANA002-512 Fetching complete!")  # 告知该数据包的振动数据提取和计算已完成

    return (corList, colList)  # 输出corList与colList