# X와 Y의 상관관계를 분석하는 기초적인 선형 회귀 모델 실습
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))         # tf.random_uniform([텐서 형태], 최솟값, 최댓값)
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32, name='X')                   # name : 추후 텐서보드 사용할 때 이름을 통해 쉽게 확인할 수 있음
Y = tf.placeholder(tf.float32, name='Y')
#X = tf.placeholder(tf.float32)
#Y = tf.placeholder(tf.float32)

hypothesis = W * X + b                                     # W, X가 행렬이 아니므로 행렬곱이 아닌 곱셈기호 사용

cost = tf.reduce_mean(tf.square(hypothesis - Y))                    # 손실함수 : mse / tf.reduce_mean(a, 0(1)) : 열(행) 평균
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)    # 옵티마이저 : 가중치 업데이트 방법(경사하강법 최적화)
train_op = optimizer.minimize(cost)                                 # 비용을 최소화하는 것이 최종 목표

# 세션을 생성하고 초기화
with tf.Session() as sess:                                          # with ~ as 구문 사용 시 sess.close() 함수 자동 호출
    sess.run(tf.global_variables_initializer())

    # 최적화 100번 수행
    for step in range(100):
        # sess.run()을 통해 train_op와 cost 그래프 계산
        # 이 때, 가설 수식에 넣어야 할 실제 값을 feed_dict를 통해 전달함
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})

        print(step, cost_val, sess.run(W), sess.run(b))

    # 테스트 데이터 입력
    print(sess.run(hypothesis, feed_dict={X: 5}))
    print(sess.run(hypothesis, feed_dict={X: 12243.32}))
