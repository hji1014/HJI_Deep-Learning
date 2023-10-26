"""
ref : https://wikidocs.net/24996, https://github.com/ukairia777/tensorflow-nlp-tutorial/blob/main/14.%20Seq2Seq%20(NMT)/14-1.%20char_level_seq2seq.ipynb
"""

import pandas as pd
import urllib3
import zipfile
import shutil
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

print(tf.__version__)


""" Data Downloading (parallel corpus data) """
# http = urllib3.PoolManager()
# url = 'http://www.manythings.org/anki/fra-eng.zip'
# filename = 'fra-eng.zip'
# path = os.getcwd()
# zipfilename = os.path.join(path, filename)
# with http.request('GET', url, preload_content=False) as r, open(zipfilename, 'wb') as out_file:
#     # Zip File Downloading
#     shutil.copyfileobj(r, out_file)
#
# with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
#     zip_ref.extractall(path)


""" 데이터 확인 및 불필요 feature 제거 """
# 파일 실제/절대 경로 확인 : os.path.realpath(__file__), os.path.abspath(__file__)
lines = pd.read_csv('./01_DL_practice/01_Tensorflow_2.x/01_Basic_Seq2Seq/fra-eng/fra.txt', names=['src', 'tar', 'lic'], sep='\t')
del lines['lic']
#print('전체 샘플의 개수 :', len(lines))

lines = lines.loc[:, 'src':'tar']
lines = lines[0: 60000]     # 6만개만 저장
#print(lines.sample(10))

lines.tar = lines.tar.apply(lambda x : '\t '+ x + ' \n')        # \t : <sos>, \n : <eos>
#print(lines.sample(10))


""" 글자 집합 구축 """
# 글자 집합 구축
src_vocab = set()
for line in lines.src:      # 1줄씩 읽음
    for char in line:       # 1개의 글자씩 읽음
        src_vocab.add(char)

tar_vocab = set()
for line in lines.tar:
    for char in line:
        tar_vocab.add(char)

src_vocab_size = len(src_vocab)+1
tar_vocab_size = len(tar_vocab)+1
#print('source 문장의 char 집합 :', src_vocab_size)
#print('target 문장의 char 집합 :', tar_vocab_size)

src_vocab = sorted(list(src_vocab))
tar_vocab = sorted(list(tar_vocab))
#print(src_vocab[45:75])
#print(tar_vocab[45:75])

src_to_index = dict([(word, i+1) for i, word in enumerate(src_vocab)])
tar_to_index = dict([(word, i+1) for i, word in enumerate(tar_vocab)])
#print(src_to_index)
#print(tar_to_index)


encoder_input = []
# 1개의 문장
for line in lines.src:
      encoded_line = []
      # 각 줄에서 1개의 char
      for char in line:
          # 각 char을 정수로 변환
          encoded_line.append(src_to_index[char])
      encoder_input.append(encoded_line)
#print('source 문장의 정수 인코딩 :', encoder_input[:5])

decoder_input = []
for line in lines.tar:
      encoded_line = []
      for char in line:
          encoded_line.append(tar_to_index[char])
      decoder_input.append(encoded_line)
#print('target 문장의 정수 인코딩 :', decoder_input[:5])

decoder_target = []         # 디코더의 예측값과 비교하기 위한 실제값
for line in lines.tar:
    timestep = 0
    encoded_line = []
    for char in line:
        if timestep > 0:
            encoded_line.append(tar_to_index[char])
        timestep = timestep + 1
    decoder_target.append(encoded_line)
#print('target 문장 레이블의 정수 인코딩 :',decoder_target[:5])

max_src_len = max([len(line) for line in lines.src])
max_tar_len = max([len(line) for line in lines.tar])
#print('source 문장의 최대 길이 :',max_src_len)
#print('target 문장의 최대 길이 :',max_tar_len)

# 영어(max:22)/프랑스어(max:76) 문장 최대 길이에 맞춰 padding(뒤쪽으로 zero padding)
encoder_input = pad_sequences(encoder_input, maxlen=max_src_len, padding='post')
decoder_input = pad_sequences(decoder_input, maxlen=max_tar_len, padding='post')
decoder_target = pad_sequences(decoder_target, maxlen=max_tar_len, padding='post')

# one-hot encoding
encoder_input = to_categorical(encoder_input)
decoder_input = to_categorical(decoder_input)
decoder_target = to_categorical(decoder_target)


""" Seq2Seq Model 구현 및 학습 """
import numpy as np
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model

encoder_inputs = Input(shape=(None, src_vocab_size))
encoder_lstm = LSTM(units=256, return_state=True)           # hidden state 크기(=step_size) : 256, 인코더의 내부 상태를 디코더로 넘겨주어야 하기 때문에 True

# encoder_outputs은 여기서는 불필요
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

# LSTM은 바닐라 RNN과는 달리 상태가 두 개. 은닉 상태와 셀 상태.
encoder_states = [state_h, state_c]     # context vector

decoder_inputs = Input(shape=(None, tar_vocab_size))
decoder_lstm = LSTM(units=256, return_sequences=True, return_state=True)

# 디코더에게 인코더의 은닉 상태, 셀 상태를 전달.
decoder_outputs, _, _= decoder_lstm(decoder_inputs, initial_state=encoder_states)       # 디코더는 인코더의 마지막 hidden state를 초기 hidden state로 사용

decoder_softmax_layer = Dense(tar_vocab_size, activation='softmax')
decoder_outputs = decoder_softmax_layer(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy")

model.fit(x=[encoder_input, decoder_input], y=decoder_target, batch_size=64, epochs=40, validation_split=0.2)


""" Seq2Seq 기계 번역기 동작시키기 """
encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)        #  encoder_inputs와 encoder_states는 훈련 과정에서 이미 정의한 것들을 재사용하는 것
encoder_model.summary()

# 이전 시점의 상태들을 저장하는 텐서
decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# 문장의 다음 단어를 예측하기 위해서 초기 상태(initial_state)를 이전 시점의 상태로 사용.
# 뒤의 함수 decode_sequence()에 동작을 구현 예정
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)

# 훈련 과정에서와 달리 LSTM의 리턴하는 은닉 상태와 셀 상태를 버리지 않음.
decoder_states = [state_h, state_c]
decoder_outputs = decoder_softmax_layer(decoder_outputs)
decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs, outputs=[decoder_outputs] + decoder_states)

index_to_src = dict((i, char) for char, i in src_to_index.items())
index_to_tar = dict((i, char) for char, i in tar_to_index.items())

def decode_sequence(input_seq):
    # 입력으로부터 인코더의 상태를 얻음
    states_value = encoder_model.predict(input_seq)

    # 에 해당하는 원-핫 벡터 생성
    target_seq = np.zeros((1, 1, tar_vocab_size))
    target_seq[0, 0, tar_to_index['\t']] = 1.

    stop_condition = False
    decoded_sentence = ""

    # stop_condition이 True가 될 때까지 루프 반복
    while not stop_condition:
        # 이점 시점의 상태 states_value를 현 시점의 초기 상태로 사용
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # 예측 결과를 문자로 변환
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = index_to_tar[sampled_token_index]

        # 현재 시점의 예측 문자를 예측 문장에 추가
        decoded_sentence += sampled_char

        # <eos>에 도달하거나 최대 길이를 넘으면 중단.
        if (sampled_char == '\n' or
            len(decoded_sentence) > max_tar_len):
            stop_condition = True

        # 현재 시점의 예측 결과를 다음 시점의 입력으로 사용하기 위해 저장
        target_seq = np.zeros((1, 1, tar_vocab_size))
        target_seq[0, 0, sampled_token_index] = 1.

        # 현재 시점의 상태를 다음 시점의 상태로 사용하기 위해 저장
        states_value = [h, c]

    return decoded_sentence

for seq_index in [34, 505, 130, 340, 1001]: # 입력 문장의 인덱스
    input_seq = encoder_input[seq_index:seq_index+1]
    decoded_sentence = decode_sequence(input_seq)
    print(35 * "-")
    print('입력 문장:', lines.src[seq_index])
    print('정답 문장:', lines.tar[seq_index][2:len(lines.tar[seq_index])-1]) # '\t'와 '\n'을 빼고 출력
    print('번역 문장:', decoded_sentence[1:len(decoded_sentence)-1]) # '\n'을 빼고 출력
