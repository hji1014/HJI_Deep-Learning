import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils

# 데이터 로드
from tensorflow.keras import datasets
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images, test_images = (train_images / 255.0), (test_images / 255.0)       # normalization 0 to 1
train_images = np.reshape(train_images, (np.shape(train_images)[0], np.shape(train_images)[1] * np.shape(train_images)[2]))
test_images = np.reshape(test_images, (np.shape(test_images)[0], np.shape(test_images)[1] * np.shape(test_images)[2]))
train_labels = np_utils.to_categorical(train_labels)
test_labels = np_utils.to_categorical(test_labels)

#########
# 신경망 모델 구성
######

X = tf.placeholder(tf.float32, [None, 784])     # features : 28 * 28 = 784
Y = tf.placeholder(tf.float32, [None, 10])      # label : 0 ~ 9
keep_prob = tf.placeholder(tf.float32)

# 텐서플로우에 내장된 함수를 이용하여 dropout 을 적용합니다.
# 함수에 적용할 레이어와 확률만 넣어주면 됩니다. 겁나 매직!!
W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))
L1 = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))
L2 = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L2, W3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

#########
# 신경망 모델 학습
######
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(np.shape(train_images)[0] / batch_size)

for epoch in range(15):
    total_cost = 0

    for i in range(total_batch):
        # 텐서플로우의 mnist 모델의 next_batch 함수를 이용해
        # 지정한 크기만큼 학습할 데이터를 가져옵니다.
        # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = train_images[(0 + i * batch_size):(batch_size + i * batch_size), :]
        batch_ys = train_labels[(0 + i * batch_size):(batch_size + i * batch_size), :]

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.8})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print('최적화 완료!')

#########
# 결과 확인
######
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(accuracy,
                        feed_dict={X: test_images,
                                   Y: test_labels,
                                   keep_prob: 1}))

#########
# 결과 확인 (matplot)
######
# predicted value
labels = sess.run(model,
                  feed_dict={X: test_images,
                             Y: test_labels,
                             keep_prob: 1})

fig = plt.figure()
for i in range(10):
    subplot = fig.add_subplot(2, 5, i + 1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title('%d' % np.argmax(labels[i]))
    subplot.imshow(test_images[i].reshape((28, 28)),
                   cmap=plt.cm.gray_r)

plt.show()
