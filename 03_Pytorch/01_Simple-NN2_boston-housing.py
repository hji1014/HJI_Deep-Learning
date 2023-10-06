""" 보스턴 데이터셋의 특징 출력 """
from sklearn.datasets import load_boston

# 경고 무시
import warnings
warnings.filterwarnings('ignore')

dataset = load_boston()     # 데이터셋을 불러옴
print(dataset.keys())       # 데이터셋의 키(요소들의 이름)를 출력


""" 데이터의 구성요소 확인 """
import pandas as pd

from sklearn.datasets import load_boston

dataset = load_boston()
dataFrame = pd.DataFrame(dataset["data"])       # ❶ 데이터셋의 데이터 불러오기
dataFrame.columns = dataset["feature_names"]    # ❷ 특징의 이름 불러오기
dataFrame["target"] = dataset["target"]         # ❸ 데이터 프레임에 정답을 추가

print(dataFrame.head())                         # ➍ 데이터프레임을 요약해서 출력


""" 선형회귀를 위한 MLP 모델의 설계 """
import torch
import torch.nn as nn

from torch.optim.adam import Adam

# ❶ 모델 정의
model = nn.Sequential(
   nn.Linear(13, 100),
   nn.ReLU(),
   nn.Linear(100, 1)
)

X = dataFrame.iloc[:, :13].values               # ❷ 정답을 제외한 특징을 X에 입력
Y = dataFrame["target"].values                  # 데이터프레임의 target의 값을 추출

batch_size = 100
learning_rate = 0.001

# ❸ 가중치를 수정하기 위한 최적화 정의
optim = Adam(model.parameters(), lr=learning_rate)

# 에포크 반복
for epoch in range(200):

   # 배치 반복
   for i in range(len(X)//batch_size):          # len(X) // batch_size : iteration -> 1 epoch를 완성시키는 데 필요한 배치의 반복 횟수
       start = i*batch_size                     # ➍ 배치 크기에 맞게 인덱스를 지정
       end = start + batch_size

       # 파이토치 실수형 텐서로 변환
       x = torch.FloatTensor(X[start:end])
       y = torch.FloatTensor(Y[start:end])

       optim.zero_grad()                        # ❺ 가중치의 기울기를 0으로 초기화
       preds = model(x)                         # ❻ 모델의 예측값 계산
       loss = nn.MSELoss()(preds, y)            # ❼ MSE 손실 계산
       loss.backward()                          # 오차 역전파
       optim.step()                             # 최적화 진행

   if epoch % 20 == 0:
       print(f"epoch{epoch} loss:{loss.item()}")


""" 모델 성능 평가 """
prediction = model(torch.FloatTensor(X[0, :13]))
real = Y[0]
print('\n')
print(f"prediction:{prediction.item()} real:{real}")