################################
# 공통 패키지 불러오기
################################
from keras.datasets import mnist
import numpy as np
from PIL import Image
import math
import os

import keras.backend as K
import tensorflow as tf

K.set_image_data_format('channels_first')
print(K.image_data_format)

################################
# GAN 모델링
################################
from keras import models, layers, optimizers


def mse_4d(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=(1, 2, 3))


def mse_4d_tf(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true), axis=(1, 2, 3))


class GAN(models.Sequential):
    def __init__(self, input_dim=64):
        """
        self, self.generator, self.discriminator are all models
        """
        super().__init__()
        self.input_dim = input_dim

        self.generator = self.GENERATOR()
        self.discriminator = self.DISCRIMINATOR()
        self.add(self.generator)
        self.discriminator.trainable = False
        self.add(self.discriminator)

        self.compile_all()

    def compile_all(self):
        # Compiling stage
        d_optim = optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)
        g_optim = optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)
        self.generator.compile(loss=mse_4d_tf, optimizer="SGD")
        self.compile(loss='binary_crossentropy', optimizer=g_optim)
        self.discriminator.trainable = True
        self.discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

    def GENERATOR(self):
        input_dim = self.input_dim

        model = models.Sequential()
        model.add(layers.Dense(1024, activation='tanh', input_dim=input_dim))
        model.add(layers.Dense(128 * 7 * 7, activation='tanh'))
        model.add(layers.BatchNormalization())
        model.add(layers.Reshape((128, 7, 7), input_shape=(128 * 7 * 7,)))
        model.add(layers.UpSampling2D(size=(2, 2)))
        model.add(layers.Conv2D(64, (5, 5), padding='same', activation='tanh'))
        model.add(layers.UpSampling2D(size=(2, 2)))
        model.add(layers.Conv2D(1, (5, 5), padding='same', activation='tanh'))
        return model

    def DISCRIMINATOR(self):
        model = models.Sequential()
        model.add(layers.Conv2D(64, (5, 5), padding='same', activation='tanh',
                                input_shape=(1, 28, 28)))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(128, (5, 5), activation='tanh'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation='tanh'))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    def get_z(self, ln):
        input_dim = self.input_dim
        return np.random.uniform(-1, 1, (ln, input_dim))

    def train_both(self, x):
        ln = x.shape[0]
        # First trial for training discriminator
        z = self.get_z(ln)
        w = self.generator.predict(z, verbose=0)
        xw = np.concatenate((x, w))
        y2 = [1] * ln + [0] * ln
        d_loss = self.discriminator.train_on_batch(xw, y2)

        # Second trial for training generator
        z = self.get_z(ln)
        self.discriminator.trainable = False
        g_loss = self.train_on_batch(z, [1] * ln)
        self.discriminator.trainable = True

        return d_loss, g_loss


################################
# GAN 학습하기
################################
def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[2:]
    image = np.zeros((height * shape[0], width * shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0],
        j * shape[1]:(j + 1) * shape[1]] = img[0, :, :]
