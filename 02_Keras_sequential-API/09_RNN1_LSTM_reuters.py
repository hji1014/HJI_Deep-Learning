import numpy
import tensorflow as tf
import matplotlib.pyplot as plt

# 로이터 뉴스 데이터셋 불러오기
from keras.datasets import reuters
from keras.models import Sequential
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense,LSTM,Embedding
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.random.set_seed(3)

# 불러온 데이터를 학습셋, 테스트셋으로 나누기
(X_train, Y_train), (X_test, Y_test) = reuters.load_data(num_words=1000, test_split=0.2)

# 데이터 확인하기
category = numpy.max(Y_train) + 1
print(category, '카테고리')
print(len(X_train), '학습용 뉴스 기사')
print(len(X_test), '테스트용 뉴스 기사')
print(X_train[0])

# 데이터 전처리
x_train = sequence.pad_sequences(X_train, maxlen=100)       # 입력 데이터 크기를 100으로 맞춰 줌
x_test = sequence.pad_sequences(X_test, maxlen=100)         # 모자라면 0으로 채우고, 100보다 크면 100개째 단어까지 선택
y_train = np_utils.to_categorical(Y_train)                  # one-hot encoding
y_test = np_utils.to_categorical(Y_test)

# 모델의 설정
model = Sequential()
model.add(Embedding(1000, 100))
model.add(LSTM(100, activation='tanh'))
model.add(Dense(46, activation='softmax'))

# 모델의 컴파일
model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

# 모델의 실행
history = model.fit(x_train, y_train, batch_size=100, epochs=20, validation_data=(x_test, y_test), verbose=1)

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(x_test, y_test)[1]))


# 테스트 셋의 오차
y_vloss = history.history['val_loss']

# 학습셋의 오차
y_loss = history.history['loss']

# 그래프로 표현
x_len = numpy.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

# 그래프에 그리드를 주고 레이블을 표시
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


from knockknock import discord_sender
webhook_url = "https://discord.com/api/webhooks/1014081244734173274/kCGlk4rXRPb4LSOdf4ECz9P0lvbiHr5tq4EOR2Zf6RPd8OgU8eytgtG2-IjER1abFt4y"
@discord_sender(webhook_url=webhook_url)
def DL_notification():
    #import time
    #time.sleep(3)
    #return {'averaged test acc of premodel' : avg_pt}, {'averaged test acc of transfer model' : avg_tl}, {'소요시간' :(terminate_time - start_time)} # Optional return value
    return {'averaged test acc of premodel': model.evaluate(x_test, y_test)[1]}
DL_notification()
