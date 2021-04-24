import numpy as np
import os
import cv2
import csv
import warnings
from sklearn.preprocessing import MinMaxScaler
from FeatureGenerator import FeatureGenerator

warnings.filterwarnings('ignore')

feature_generator = FeatureGenerator()
train_path = "../train"
test_path = "../test"
train_images_path = []
test_images_path = []
labels = []
breed_dic = {}

fixed_size = tuple((80, 80))

def generator_train_npy():
    with open(os.path.join("./", "train.csv")) as f:
        reader = csv.reader(f)
        for row in reader:
            img_file_name, breed, label = row
            train_images_path.append(os.path.join(train_path, img_file_name + ".jpg"))
            label = int(label)
            labels.append(label)
            breed_dic[label] = breed
    assert len(train_images_path) == len(labels)

    print(train_images_path)
    print(labels)
    print(breed_dic)


    global_features = []
    for i in range(len(train_images_path)):
        image_i = cv2.imread(train_images_path[i])
        image_i = cv2.resize(image_i, fixed_size)
        fv_hu_moments_i = feature_generator.fd_hu_moments(image_i)
        fv_histogram_i = feature_generator.fd_histogram(image_i)
        global_feature = np.hstack([fv_hu_moments_i, fv_histogram_i])
        global_features.append(global_feature)
    assert len(global_features) == len(labels)


    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_features = scaler.fit_transform(global_features)

    X = np.array(rescaled_features)
    y = np.array(labels)
    with open('train_X.npy', 'wb') as f:
        np.save(f, X)
    with open('train_y.npy', 'wb') as f:
        np.save(f, y)

if not os.path.exists("train_X.npy") or not os.path.exists("train_y.npy"):
    generator_train_npy()


