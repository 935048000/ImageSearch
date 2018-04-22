from core.feature_extraction import feature
# import feature_extraction

def testCase(img_path):
    f = feature()
    norm_feat = f.extract_feat (img_path)
    print (norm_feat)
    print (norm_feat.shape)
    print(type(norm_feat))
    return 0


if __name__ == '__main__':
    print(feature.__doc__)
    # testCase("H:/datasets/testingset/19700102125648863.JPEG")