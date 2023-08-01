import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
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

W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L2, W3)

# 텐서플로우에서 기본적으로 제공되는 크로스 엔트로피 함수를 이용해
# 복잡한 수식을 사용하지 않고도 최적화를 위한 비용 함수를 다음처럼 간단하게 적용할 수 있습니다.
# tf.nn.softmax_cross_entropy_with_logits_v2 : softmax 함수와 크로스 엔트로피*Cross-Entropy 손실함수를 구현한 API.
# softmax 계산까지 함께 해주기 때문에 소프트맥스 함수(tf.nn.softmax)를 씌운 normalize된 출력값이 아닌 소프트맥스 함수를 씌우기 전의 출력값인 logits를 인자로 넣어주어야 함.
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

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print('최적화 완료!')

#########
# 결과 확인
######
# model 로 예측한 값과 실제 레이블인 Y의 값을 비교합니다.
# tf.argmax 함수를 이용해 예측한 값에서 가장 큰 값을 예측한 레이블이라고 평가합니다.
# 예) [0.1 0 0 0.7 0 0.2 0 0 0 0] -> 3
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(accuracy,
                        feed_dict={X: test_images,
                                   Y: test_labels}))
