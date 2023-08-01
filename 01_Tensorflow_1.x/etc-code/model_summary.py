"""
주의사항 : tf 1.x에서만 사용 가능. tf 2.x에서 변환 시 사용 불가
"""

import numpy as np

from tensorflow.python.layers import base
import tensorflow as tf
import tensorflow.contrib.slim as slim

x = np.zeros((1,4,4,3))
x_tf = tf.convert_to_tensor(x, np.float32)
z_tf = tf.layers.conv2d(x_tf, filters=32, kernel_size=(3,3))

def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

model_summary()