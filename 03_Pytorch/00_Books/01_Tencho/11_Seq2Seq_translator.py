""" 데이터 확인 """
import string

l = []

# 한글 텍스트 파일을 읽기 위해 utf-8 인코딩으로 읽어옴('\t' : tap)
with open("C:/Users/user02/OpenDatasets/eng_kor/kor_modified.txt", 'r', encoding="utf-8") as f:
    lines = f.read().split("\n")
    for line in lines:
       # 특수 문자를 지우고 모든 글자를 소문자로 변경
       txt = "".join(v for v in line if v not in string.punctuation).lower()
       l.append(txt)
#print(l[:5])


""" BOW를 만드는 함수 정의 """
import numpy as np
import torch

from torch.utils.data.dataset import Dataset

def get_BOW(corpus):  # 문장들로부터 BOW를 만드는 함수
    BOW = {"<SOS>": 0, "<EOS>": 1}  # ❶ <SOS> 토큰과 <EOS> 토큰을 추가

    # ❷ 문장 내 단어들을 이용해 BOW를 생성
    for line in corpus:
        for word in line.split():
            if word not in BOW.keys():
                BOW[word] = len(BOW.keys())

    return BOW


""" 학습에 사용할 데이터셋 정의 """
class Eng2Kor(Dataset):  # 학습에 이용할 데이터셋
    def __init__(self, pth2txt="C:/Users/user02/OpenDatasets/eng_kor/kor_modified.txt"):
        self.eng_corpus = []  # 영어 문장이 들어가는 변수
        self.kor_corpus = []  # 한글 문장이 들어가는 변수
        # ➊ 텍스트 파일을 읽어서 영어 문장과 한글 문장을 저장
        with open(pth2txt, 'r', encoding="utf-8") as f:
            lines = f.read().split("\n")
            for line in lines:
                # 특수 문자와 대문자 제거
                txt = "".join(v for v in line if v not in string.punctuation).lower()
                engtxt = txt.split("\t")[0]
                kortxt = txt.split("\t")[1]

                # 길이가 10 이하인 문장만을 사용
                if len(engtxt.split()) <= 10 and len(kortxt.split()) <= 10:
                    self.eng_corpus.append(engtxt)
                    self.kor_corpus.append(kortxt)

        self.engBOW = get_BOW(self.eng_corpus)  # 영어 BOW
        self.korBOW = get_BOW(self.kor_corpus)  # 한글 BOW
    # 문장을 단어별로 분리하고 마지막에 <EOS>를 추가
    def gen_seq(self, line):
        seq = line.split()
        seq.append("<EOS>")

        return seq
    def __len__(self): # ❶
        return len(self.eng_corpus)

    def __getitem__(self, i): # ❷
        # 문자열로 되어 있는 문장을 숫자 표현으로 변경
        data = np.array([self.engBOW[txt] for txt in self.gen_seq(self.eng_corpus[i])])

        label = np.array([self.korBOW[txt] for txt in self.gen_seq(self.kor_corpus[i])])

        return data, label


""" 학습에 사용할 데이터 로더 정의 """
def loader(dataset):  # 데이터셋의 문장을 한문장씩 불러오기 위한 함수(여기서는 padding을 안 했음)
    for i in range(len(dataset)):
        data, label = dataset[i]

        # ❶ 데이터와 정답을 반환(yield는 return이랑 비슷하지만, 값을 반복적으로 반환할 때 쓰임
        yield torch.tensor(data), torch.tensor(label)


""" 인코더 정의 """
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, x, h):
        # ❶ 배치차원과 시계열 차원 추가
        x = self.embedding(x).view(1, 1, -1)  # veiw : reshape와 비슷, embedding 거친 shape : [64], reshape 이후 shape : [1(배치 차원), 1(시계열 차원), 64]
        output, hidden = self.gru(x, h)       # GRU output shape : [1, 1, 64], GRU hidden shape : [1, 1, 64]
        return output, hidden


""" 디코더 정의 """
# 아마도 Luong Attention을 구현한 듯 함
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=11):
        super(Decoder, self).__init__()

        # 임베딩층 정의
        self.embedding = nn.Embedding(output_size, hidden_size)

        # 어텐션 가중치를 계산하기 위한 MLP층
        self.attention = nn.Linear(hidden_size * 2, max_length)

        #특징 추출을 위한 MLP층
        self.context = nn.Linear(hidden_size * 2, hidden_size)

        # 과적합을 피하기 위한 드롭아웃 층
        self.dropout = nn.Dropout(dropout_p)

        # GRU층
        self.gru = nn.GRU(hidden_size, hidden_size)

        # 단어 분류를 위한 MLP층
        self.out = nn.Linear(hidden_size, output_size)

        # 활성화 함수
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, h, encoder_outputs):
        # ➊입력을 밀집 표현으로
        x = self.embedding(x).view(1, 1, -1)
        x = self.dropout(x)

        # ➋어텐션 가중치 계산
        attn_weights = self.softmax(self.attention(torch.cat((x[0], h[0]), -1)))

        # ➌어텐션 가중치와 인코더의 출력을 내적
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        # ➍인코더 각 시점의 중요도와 밀집표현을 합쳐
        # MLP층으로 특징 추출
        output = torch.cat((x[0], attn_applied[0]), 1)
        output = self.context(output).unsqueeze(0)
        output = self.relu(output)

        # ➎GRU층으로 입력
        output, hidden = self.gru(output, h)

        # ➏예측된 단어 출력
        output = self.out(output[0])

        return output


""" 학습에 필요한 요소 정의 """
import random
import tqdm

from torch.optim.adam import Adam

# 학습에 사용할 프로세서 정의
device = "cuda" if torch.cuda.is_available() else "cpu"
# 학습에 사용할 데이터셋 정의
dataset = Eng2Kor()

# 인코더 디코더 정의
encoder = Encoder(input_size=len(dataset.engBOW), hidden_size=64).to(device)
decoder = Decoder(64, len(dataset.korBOW), dropout_p=0.1).to(device)
# 인코더 디코더 학습을 위한 최적화 정의
encoder_optimizer = Adam(encoder.parameters(), lr=0.0001)
decoder_optimizer = Adam(decoder.parameters(), lr=0.0001)


""" 학습 루프 정의 """
for epoch in range(200):
   iterator = tqdm.tqdm(loader(dataset), total=len(dataset))
   total_loss = 0

   for data, label in iterator:
       data = torch.tensor(data, dtype=torch.long).to(device)
       label = torch.tensor(label, dtype=torch.long).to(device)

       # 인코더의 초기 은닉 상태
       encoder_hidden = torch.zeros(1, 1, 64).to(device)
       # 인코더의 모든 시점의 출력을 저장하는 변수
       encoder_outputs = torch.zeros(11, 64).to(device)

       encoder_optimizer.zero_grad()
       decoder_optimizer.zero_grad()

       loss = 0
       for ei in range(len(data)):
           # ➊한 단어씩 인코더에 넣어줌
           encoder_output, encoder_hidden = encoder(
               data[ei], encoder_hidden)
           # ❷인코더의 은닉 상태를 저장
           encoder_outputs[ei] = encoder_output[0, 0]

       decoder_input = torch.tensor([[0]]).to(device)

       # ❸인코더의 마지막 은닉 상태를 디코더의 초기 은닉 상태로 저장
       decoder_hidden = encoder_hidden
       use_teacher_forcing = True if random.random() < 0.5 else False  # ❶

       if use_teacher_forcing:
           for di in range(len(label)):
               decoder_output = decoder(
                   decoder_input, decoder_hidden, encoder_outputs)

               # 직접적으로 정답을 다음 시점의 입력으로 넣어줌
               target = torch.tensor(label[di], dtype=torch.long).to(device)
               target = torch.unsqueeze(target, dim=0).to(device)
               loss += nn.CrossEntropyLoss()(decoder_output, target)
               decoder_input = target
       else:
           for di in range(len(label)):
               decoder_output = decoder(
                   decoder_input, decoder_hidden, encoder_outputs)

               # ➊ 가장 높은 확률을 갖는 단어의 인덱스가 topi
               topv, topi = decoder_output.topk(1)
               decoder_input = topi.squeeze().detach()

               # 디코더의 예측값을 다음 시점의 입력으로 넣어줌
               target = torch.tensor(label[di], dtype=torch.long).to(device)
               target = torch.unsqueeze(target, dim=0).to(device)
               loss += nn.CrossEntropyLoss()(decoder_output, target)

               if decoder_input.item() == 1:  #  토큰을 만나면 중지
                   break
       # 전체 손실 계산
       total_loss += loss.item()/len(dataset)
       iterator.set_description(f"epoch:{epoch+1} loss:{total_loss}")
       loss.backward()

       encoder_optimizer.step()
       decoder_optimizer.step()

torch.save(encoder.state_dict(), "./01_DL_practice/01_Pytorch/modelattn_enc.pth")
torch.save(decoder.state_dict(), "./01_DL_practice/01_Pytorch/modelattn_dec.pth")


""" 모델 성능 평가에 필요한 요소 정의 """
# 인코더 가중치 불러오기
encoder.load_state_dict(torch.load("./01_DL_practice/01_Pytorch/attn_enc.pth", map_location=device))
# 디코더 가중치 불러오기
decoder.load_state_dict(torch.load("./01_DL_practice/01_Pytorch/attn_dec.pth", map_location=device))

# ❶불러올 영어 문장을 랜덤하게 지정
idx = random.randint(0, len(dataset))
# 테스트에 사용할 문장
input_sentence = dataset.eng_corpus[idx]
# 신경망이 번역한 문장
pred_sentence = ""

data, label = dataset[idx]
data = torch.tensor(data, dtype=torch.long).to(device)
label = torch.tensor(label, dtype=torch.long).to(device)

# ➋인코더의 초기 은닉 상태 정의
encoder_hidden = torch.zeros(1, 1, 64).to(device)
# 인코더 출력을 담기위한 변수
encoder_outputs = torch.zeros(11, 64).to(device)


""" 인코더 동작 """
for ei in range(len(data)):
    # ➊한 단어씩 인코더에 넣어줌
    encoder_output, encoder_hidden = encoder(
        data[ei], encoder_hidden)

    # ➋인코더의 출력을 저장
    encoder_outputs[ei] = encoder_output[0, 0]

# ➌디코더의 초기 입력
# 0은 토큰
decoder_input = torch.tensor([[0]]).to(device)

# ➍인코더의 마지막 은닉 상태를 디코더의 초기 은닉 상태로
decoder_hidden = encoder_hidden


""" 디코더 동작 """
for di in range(11):
    # ➊가장 높은 확률을 갖는 단어의 요소를 구함
   decoder_output = decoder(
                       decoder_input, decoder_hidden, encoder_outputs)
   topv, topi = decoder_output.topk(1)
   decoder_input = topi.squeeze().detach()

   # ➋ 토큰을 만나면 중지
   if decoder_input.item() == 1:
       break

   # ➌가장 높은 단어를 문자열에 추가
   pred_sentence += list(dataset.korBOW.keys())[decoder_input] + " "

print(input_sentence)  # 영어 문장
print(pred_sentence)  # 한글 문장
