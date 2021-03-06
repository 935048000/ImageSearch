import os
import h5py
import numpy as np
import argparse
# from core.feature_extraction import extract_feat
from core.feature_extraction import feature
# from memory_profiler import profile
from pyprind import ProgBar
from core.base import base
from time import time

b = base ()

# 命令行参数功能
def comdArgs():
    """
    command argument
    :return:train path and feat file
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", required = True, help = "训练集的路径")
    ap.add_argument("-f", required = True, help = "特征索引文件的路径名称")
    args = vars(ap.parse_args())
    return b.getFileList(args["d"],"JPEG"),args["f"]


# 按指定格式读取h5文件
def rH5FileData(Key,filename):
    """
    Read the h5 file in the specified format.
    :param Key:index
    :param filename:file name
    :return:feat,image name
    """
    try:
        with h5py.File (filename, 'r') as h5f:
            feats = h5f["data" + str (Key)][:]
            imgNames = h5f["name" + str (Key)][:]
            return feats,imgNames[0].decode("utf-8")
    except KeyError:
        print("Read HDF5 File Key Error")
        return 1

def rH5FileData2(Key1, key2, filename):
    """
    Read the h5 file in the specified format.
    :param Key1: feats index
    :param key2: image name index
    :param filename: file name
    :return: feat list,image name list
    """
    NameList = []
    featsArrayList = []
    try:
        with h5py.File (filename, 'r') as h5f:
            feats = h5f[Key1][:]
            imgNames = h5f[key2][:]
            for i in imgNames:
                NameList.append(i.decode("utf-8"))
            featsArrayList = np.array (feats)
            return featsArrayList, NameList
    except KeyError:
        print ("Read HDF5 File Key Error")
        return 1


# 按指定格式写入h5文件（按key存入两个list）
def wH5FileData(Key,feats,names,filename):
    """
    Write the h5 file in the specified format (save two lists by key)
    :param Key:index
    :param feats:feature
    :param names:image name
    :param filename:file name
    :return:
    """
    namess = []
    # 数据编码转换
    if type(names) is list:
        for j in names:
            namess.append (j.encode ())
    else:
        names.encode ()
    try:
        with h5py.File (filename, 'a') as h5f:
            h5f.create_dataset ("data"+str(Key), data=feats)
            h5f.create_dataset ("name"+str(Key), data=namess)
    except RuntimeError:
        raise NameError('Unable to create link (name already exists)')
    return 0


# 按指定格式写入h5文件（一次性存入两个list）
def wH5FileData2(Key1,Key2,feats,names,filename):
    """
    Write the h5 file in the specified format (save two lists by key)
    :param Key1: feature index
    :param Key2: image name index
    :param feats: feature
    :param names: image name
    :param filename: file name
    :return:
    """
    namess = []
    # 数据编码转换
    if type(names) is list:
        for j in names:
            namess.append (j.encode ())
    else:
        names.encode ()
    try:
        with h5py.File (filename, 'a') as h5f:
            h5f.create_dataset (Key1, data=feats)
            h5f.create_dataset (Key2, data=namess)
    except RuntimeError:
        raise NameError('Unable to create link (name already exists)')
    return 0

# 提取特征并写入文件
# @profile (precision=6)
def etlFeature(post,img_list,h5filename):
    """
    Extract features and write to files.
    :param post:index
    :param img_list:image list
    :param h5filename:hdf5 file
    :return:
    """
    # 迭代方式，提取特征值写入h5文件
    feat = feature()
    # bar = ProgBar (len(img_list), monitor=True, title="提取图片特征,Image Total:%d" % len (img_list))
    for i, img_path in enumerate (img_list):
        norm_feat = feat.extract_feat (img_path)
        img_name = os.path.split (img_path)[1]
        names = []
        names.append (img_name)
        feats2 = np.array (norm_feat)
        try:
            wH5FileData (i+post, feats2, names,h5filename)
        except:
            print("Feats Write Error")
            return 1
        # bar.update ()
        # print ("提取图片特征！进度: %d/%d" % ((i + 1), len (img_list)))
    # print (bar)
    return 0


# 获取HDF5文件内数据条数，便于追加和读取。
def showHDF5Len(filename):
    """
    Get the number of data in the HDF5 file, easy to append and read.
    :param filename:HDF5 file
    :return:file length
    """
    # 文件不存在则重写，不追加
    if not os.path.exists(filename):
        return 0
    # 存在则追加
    with h5py.File (filename, 'r') as h5f:
        return int(len(h5f)/2)


def main():
    feats = []
    h5filename = "./imageCNN.h5"
    dataset = " "
    img_list = b.getFileList(dataset,"JPEG")
    etlFeature (showHDF5Len (h5filename), img_list, h5filename)
    return 0

# 特征值数据库功能测试用例
def testDatabase():
    """
    Feature value database functional test cases.
    :return:
    """
    featsList = []
    nameList = []
    h5filename = "./models/image_Feature_Test.h5"
    dataset = "./imagetest"
    # 取图像数据集列表
    img_list = b.getFileList (dataset, "JPEG")
    # 提取图像数据集列表特征后存入HDF5文件
    etlFeature (showHDF5Len (h5filename), img_list, h5filename)
    # 读取图像特征值和图像名称
    t = time ()
    for i in range (showHDF5Len (h5filename)):
        feats, imgNames = rH5FileData (i, h5filename)
        featsList.append (feats)
        nameList.append (imgNames)
    print ("特征数据库读取耗时(秒)：%.2f s" % (time () - t))
    print("特征值列表长度：", len(featsList))
    print("特征值对应的图像名称列表：", nameList)
    print("图像名称列表长度:", len(nameList))
    return 0

if __name__ == "__main__":
    pass
    feats = []
    # 数据文件
    h5filename = "./models/imageCNN6442.h5"

    t = time()
    testDatabase()
    print ("数据库读写总耗时(秒)：%.2f s" % (time () - t))

    # 文件条数
    # lens = showHDF5Len (h5filename)
    # print(lens)


    
    
    # h5filename = "./imageCNN4_1.h5"
    # img_list = b.getFileList("./temp_image1","JPEG")
    # etlFeature (showHDF5Len (h5filename), img_list, h5filename)
    #
    # pass
    # h5filename = "./imageCNN4_2.h5"
    # img_list = b.getFileList ("./temp_image2", "JPEG")
    # etlFeature (showHDF5Len (h5filename), img_list, h5filename)


    # featsList = []
    # nameList = []
    # for i in range (showHDF5Len (h5filename)):
    #     feats, imgNames = rH5FileData (i, h5filename)
    #     featsList.append (feats)
    #     nameList.append (imgNames)
    #
    #
    # print(showHDF5Len (h5filename))
    #
    # wH5FileData2 ("feature", "imagename", featsList, nameList, "./models/imageCNNAll.h5")

    # print(showHDF5Len ("./models/imageCNNAll.h5"))

    # feats, imgNames = rH5FileData2 ("feature", "imagename", "./models/imageCNNAll.h5")
    # print(len(imgNames))

    # 读取数据
    # featsList = []
    # nameList = []
    # for i in range (lens):
    #     feats, imgNames = rH5FileData (i, h5filename)
    #     featsList.append (feats)
    #     nameList.append (imgNames)

    # queryVec = extract_feat("D:/datasets/testingset1-1/19700102125648863.JPEG")
    # featsList = np.array(featsList)
    # scores = np.dot(queryVec, feats.T)
    # rank_ID = np.argsort(scores)[::-1] # 排序,倒序，大到小
    # rank_score = scores[rank_ID] # 计算评分
    # print(rank_score)


    
