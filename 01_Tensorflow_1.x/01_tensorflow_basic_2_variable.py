# 플레이스홀더와 변수의 개념
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# tf.placeholder : 계산을 실행할 때 입력값을 받는 변수로 사용
# None : 크기가 정해지지 않았음을 의미
X = tf.placeholder(tf.float32, [None, 3])
print(X)

# X 플레이스 홀더에 넣을 값 만들기
# 플레이스홀더에서 설정한 것처럼 두 번째 차원의 요소의 개수 : 3
x_data = [[1, 2, 3], [4, 5, 6]]

# tf.Variable : 그래프를 계산하면서 최적화 할 변수들.
# tf.random_normal : 각 변수들의 초기값을 정규분포 랜덤 값으로 초기화
W = tf.Variable(tf.random_normal([3, 2]))
b = tf.Variable(tf.random_normal([2, 1]))

# 입력값과 변수들을 계산할 수식
# tf.matmul() : 행렬곱 수행
expr = tf.matmul(X, W) + b

sess = tf.Session()
sess.run(tf.global_variables_initializer())     # 변수 다시 한 번 초기화

print(x_data)
print(sess.run(W))
print(sess.run(b))
print(sess.run(expr, feed_dict={X: x_data}))    # expr 수식에 x_data 입력하는 방법

#a = sess.run(expr, feed_dict={X: x_data})       # 출력값은 numpy 배열로 나옴
#print(type(a))

sess.close()