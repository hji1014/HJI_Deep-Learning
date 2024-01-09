""" 손글씨 데이터 살펴보기 """
import matplotlib.pyplot as plt

from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor

# ❶ 학습용 데이터와 평가용 데이터 분리
training_data = MNIST(root="./01_DL_practice/01_Pytorch/data", train=True, download=True, transform=ToTensor())
test_data = MNIST(root="./01_DL_practice/01_Pytorch/data", train=False, download=True, transform=ToTensor())

print(len(training_data))           # 학습에 사용할 데이터 개수
print(len(test_data))               # 평가에 사용할 데이터 개수

for i in range(9):                  # 샘플 이미지를 9개 출력
   plt.subplot(3, 3, i+1)
   plt.imshow(training_data.data[i])
plt.show()


""" 학습 데이터와 평가 데이터의 데이터로더 정의 """
from torch.utils.data.dataloader import DataLoader

train_loader = DataLoader(training_data, batch_size=32, shuffle=True)

# ❶평가용은 데이터를 섞을 필요가 없음
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


""" 손글씨 분류 모델 학습하기 """
import torch
import torch.nn as nn

from torch.optim.adam import Adam

device = "cuda" if torch.cuda.is_available() else "cpu"        # ❶ 학습에 사용할 프로세서를 지정

model = nn.Sequential(
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
model.to(device) # 모델의 파라미터를 GPU로 보냄

lr = 1e-3
optim = Adam(model.parameters(), lr=lr)

for epoch in range(20):
    for data, label in train_loader:
        optim.zero_grad()
        # ❷ 입력 데이터를 모델의 입력에 맞게 모양을 변환
        data = torch.reshape(data, (-1, 784)).to(device)
        preds = model(data)

        loss = nn.CrossEntropyLoss()(preds, label.to(device))      # ❸ 손실 계산
        loss.backward()
        optim.step()

    print(f"epoch{epoch+1} loss:{loss.item()}")

torch.save(model.state_dict(), "./01_DL_practice/01_Pytorch/model/MNIST.pth") # ➍ 모델을 MNIST.pth라는 이름으로 저장


""" 모델의 성능 평가 """
# ❶ 모델 가중치 불러오기
model.load_state_dict(torch.load("./01_DL_practice/01_Pytorch/model/MNIST.pth", map_location=device))        # map_location : 불러올 위치 -> 여기서는 GPU로 불러옴

num_corr = 0 # 분류에 성공한 전체 개수

with torch.no_grad():                                                       # ❷ 기울기를 계산하지 않음
    for data, label in test_loader:
        data = torch.reshape(data, (-1, 784)).to(device)

        output = model(data.to(device))
        preds = output.data.max(1)[1] # ❸ 모델의 예측값 계산
        # ❹ 올바르게 분류한 개수
        corr = preds.eq(label.to(device).data).sum().item()
        num_corr += corr

    print(f"Accuracy:{num_corr/len(test_data)}")                             # 분류 정확도를 출력합니다.
