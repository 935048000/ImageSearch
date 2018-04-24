import numpy as np
from numpy import linalg as LA

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from PIL import Image
from cv2 import imread,resize,cvtColor,COLOR_BGR2RGB,INTER_AREA,imshow

'''
VGG16模型,权重由ImageNet训练而来
使用vgg16模型提取特征
输出归一化特征向量
'''
class feature():
    """
    Feature extraction class
    """
    def __init__(self):
        pass
    
    
    def extract_feat(self,img_path):
        input_shape = (272, 480, 3)
        model = VGG16(input_shape = (input_shape[0],input_shape[1],input_shape[2]), pooling = 'max', include_top = False)
        _img = imread (img_path)
        res = resize (_img, (input_shape[1], input_shape[0]), interpolation=INTER_AREA)
        img = Image.fromarray (cvtColor (res, COLOR_BGR2RGB))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = model.predict(img)
        norm_feat = feat[0]/LA.norm(feat[0])
        return norm_feat


if __name__ == '__main__':
    print("local run .....")


    # models = VGG16 (weights='imagenet', pooling = 'max', include_top=False)
    # img_path = './database/001_accordion_image_0001.jpg'
    # img = image.load_img (img_path, target_size=(224, 224))
    # x = image.img_to_array (img)
    # x = np.expand_dims (x, axis=0)
    # x = preprocess_input (x)
    # features = models.predict (x)
    # norm_feat = features[0]/LA.norm(features[0])
    # feats = np.array(norm_feat)
    # print(norm_feat.shape)
    # print(feats.shape)

    img_path = "H:/datasets/testingset/19700102125648863.JPEG"
    f = feature()
    norm_feat = f.extract_feat(img_path)
    print(norm_feat)
    print(norm_feat.shape)

    