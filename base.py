import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import os
from pyprind import ProgBar
import time

class base(object):
    """
    这是一个基础方法类，封装常用方法。
    This is a basic method class that encapsulates common methods.
    """

    # 返回目录中指定后缀的文件列表。
    def getFileList(self,path,type):
        """
        Returns a list of files with the specified suffix in the directory
        :param path:
        :param type:
        :return: list
        """
        return [os.path.join (path, f) for f in os.listdir (path) if f.endswith ('.'+type)]

    # 获取文件名字,有文件类别后缀的，无路径。
    def getFileName(self,FilePath):
        """
        Get file name, with file type suffix, no path
        :param FilePath:
        :return: image name
        """
        _temp = FilePath.split ("/")[-1]
        return _temp.split ("\\")[-1]

    # 获取文件名字,无文件类别后缀。
    def getFileNameNoSuffix(self,File):
        """
        Get name, no file category suffix.
        :param File:
        :return: image name
        """
        return File.split (".")[0]

    # 自定义文件后缀。
    def definedFileSuffix(self,file,suffix):
        """
        Custom file suffix
        :param file:
        :param suffix:
        :return:
        """
        return file + "." + suffix

    # 提取同类的图片
    def getImageList(self,filename,class_name):
        """
        Extract the same kind of picture
        :param filename:
        :param class_name:
        :return:list
        """
        imgLists = []
    
        with open (filename, "r", encoding="utf-8") as f:
            _temp = f.readlines ()
    
        for i in range (len (_temp)):
            _t = _temp[i]
            _tt = _t.strip ("\n").split (",")
            if _tt[0] == class_name:
                imgLists.append (_tt[1])
    
        return imgLists
    

if __name__ == '__main__':
    
    b = base()
    
    # function test
    lists = b.getFileList ("H:/datasets/testingset", "txt")
    print (lists[0])
    
    file = b.getFileName (lists[0])
    print (file)
    
    file2 = b.getFileNameNoSuffix (file)
    print (file2)
    
    file3 = b.definedFileSuffix (file2, "avi")
    print (file3)
    
    pass