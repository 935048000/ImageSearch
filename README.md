# ImageSearch
This is an image search application implemented in python. 

It can extract the features of the image into a file, and then compare the feature extraction of the image to be searched with the image in the file to match the image with the highest similarity.

本系统能够提供图像检索服务，实现基于图像检索的位置识别。

The system uses the VGG16 architecture of the deep learning convolutional neural network to achieve image feature extraction. The feature storage is implemented by Python's HDF5 library, and the image similarity is calculated by vector dot product.

本系统使用深度学习卷积神经网络的VGG16架构实现图像特征提取，通过Python的HDF5库实现特征存储，通过向量点积，计算图像相似度。


## base.py
This is a basic class program, and some basic file operations are defined inside.

这个程序为字符串处理，基础库程序。

## imageAdjust.py
This is an image transformation program that USES the specified algorithm to customize the image.

这个程序为图像变换程序，对图像进行大小变换。

## feature_extraction.py
This is a program that can extract image features.

The program uses keras as the front end and the TensorFlow as the back end to extract the image features.

This program's keras uses the pre-training model VGG16 for feature extraction of the CNN model.

这个程序为特征提取程序，提供图像特征提取服务。

## feat2file.py

This is an application that extracts images in bulk and writes to the hdf5 file.

这个程序为图像特征数据库程序，提供特征数据存储服务。

## imageQuery.py

This is an image retrieval program that only needs to specify the retrieved image and feature files to get the closest image.

这个程序为图像检索程序，提供图像检索服务。

## models

The file under this directory is a feature index file.

此目录为特征数据库文件目录。

## log

The files in this directory are test logs.

此目录为日志文件目录。


