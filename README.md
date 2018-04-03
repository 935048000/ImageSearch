# ImageSearch
This is an image search application implemented in python. 

It can extract the features of the image into a file, and then compare the feature extraction of the image to be searched with the image in the file to match the image with the highest similarity.

## base.py
This is a basic class program, and some basic file operations are defined inside.



## imageAdjust.py
This is an image transformation program that USES the specified algorithm to customize the image.



## feature_extraction.py
This is a program that can extract image features.

The program uses keras as the front end and the TensorFlow as the back end to extract the image features.

This program's keras uses the pre-training model VGG16 for feature extraction of the CNN model.


## feat2file.py

This is an application that extracts images in bulk and writes to the hdf5 file.


## imageQuery.py

This is an image retrieval program that only needs to specify the retrieved image and feature files to get the closest image.

## model

The file under this directory is a feature index file.

## log

The files in this directory are test logs.

