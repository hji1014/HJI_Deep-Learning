""" lazy loading을 통한 효율적 학습 """

import numpy as np
import cv2
from keras.utils import to_categorical, Sequence

class DataGenerator(Sequence):
    def __init__(self, path, list_IDs, labels, batch_size, img_size, img_channel, num_classes):
        """ initialization method """
        # 데이터셋 경로
        self.path = path
        # 데이터 이미지 개별 주소 [ DataFrame 형식 (image 주소, image 클래스) ]
        self.list_IDs = list_IDs
        # 데이터 라벨 리스트 [ DataFrame 형식 (image 주소, image 클래스) ]
        self.labels = labels
        # 학습 Batch 사이즈
        self.batch_size = batch_size
        # 이미지 리사이징 사이즈
        self.img_size = img_size
        # 이미지 채널 [RGB or Gray]
        self.img_channel = img_channel
        # 데이터 라벨의 클래스 수
        self.num_classes = num_classes
        # 전체 데이터 수
        self.indexes = np.arange(len(self.list_IDs))

    def __len__(self):
        """ number of batches in the sequence """
        len_ = int(len(self.list_IDs) / self.batch_size)
        if len_ * self.batch_size < len(self.list_IDs):
            len_ += 1
        return len_

    def __getitem__(self, index):
        """ indexing """
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def __data_generation(self, list_IDs_temp):
        """ 지정된 인덱스의 데이터 배치를 전처리 이후 반환 """
        X = np.zeros((self.batch_size, self.img_size, self.img_size, self.img_channel))
        y = np.zeros((self.batch_size, self.num_classes), dtype=int)
        for i, ID in enumerate(list_IDs_temp):
            img = cv2.imread(self.path + ID)
            img = cv2.resize(img, (self.img_size, self.img_size))
            X[i,] = img / 255
            y[i,] = to_categorical(self.labels[i], num_classes=self.num_classes)
        return X, y

import keras
import pandas as pd

if keras.backend.image_data_format == "channels_first":
    input_shape = (1, 12, 120, 120)
else:
    input_shape = (12, 120, 120, 1)

data_address = './01_DL_practice/04_Data/train.csv'
train_labels = pd.read_csv(data_address)

clss_num = len(train_labels['labels'].unique())
labels_dict = dict(zip(train_labels['labels'].unique(), range(clss_num)))
train_labels = train_labels.replace({"labels": labels_dict})

tartget_size = 150
img_ch = 3
num_class = 12
batch_size = 32

train_generator = DataGenerator(path=data_address, list_IDs=train_labels['image'],
                                labels=train_labels['labels'], batch_size=128,
                                img_size=tartget_size, img_channel=img_ch,
                                num_classes=num_class)

# 학습
# history = model.fit_generator(train_generator, epochs=1)

""" MNIST data folder 만든 후 직접 lazy loading 시켜보기 """