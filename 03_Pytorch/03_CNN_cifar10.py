""" 데이터 살펴보기 """
# import matplotlib.pyplot as plt
#
# from torchvision.datasets.cifar import CIFAR10
# from torchvision.transforms import ToTensor
#
# # ❶ CIFAR10 데이터셋을 불러옴
# training_data = CIFAR10(
#     root="./01_DL_practice/01_Pytorch/data",
#     train=True,
#     download=True,
#     transform=ToTensor())
#
# test_data = CIFAR10(
#     root="./01_DL_practice/01_Pytorch/data",
#     train=False,
#     download=True,
#     transform=ToTensor())
#
# for i in range(9):
#    plt.subplot(3, 3, i+1)
#    plt.imshow(training_data.data[i])
# plt.show()


""" 데이터 전처리에 크롭핑과 뒤집기를 추가 """
# import matplotlib.pyplot as plt
# import torchvision.transforms as T
#
# from torchvision.datasets.cifar import CIFAR10
# from torchvision.transforms import Compose
# from torchvision.transforms import RandomHorizontalFlip, RandomCrop
#
# transforms = Compose([                               # ❶ 데이터 전처리 함수들
#     T.ToPILImage(),
#     RandomCrop((32, 32), padding=4),            # ➋ 랜덤으로 이미지 일부 제거 후 패딩
#     RandomHorizontalFlip(p=0.5),                     # ➌ y축으로 기준으로 대칭
# ])
#
# training_data = CIFAR10(
#     root="./01_DL_practice/01_Pytorch/data",
#     train=True,
#     download=True,
#     transform=transforms)                           # transform에는 데이터를 변환하는 함수가 들어감
#
# test_data = CIFAR10(
#     root="./01_DL_practice/01_Pytorch/data",
#     train=False,
#     download=True,
#     transform=transforms)
#
# for i in range(9):
#     plt.subplot(3, 3, i+1)
#     plt.imshow(transforms(training_data.data[i]))
# plt.show()

""" 데이터 전처리에 크롭핑, 뒤집기, 정규화 추가 """
# import matplotlib.pyplot as plt
# import torchvision.transforms as T
#
# from torchvision.datasets.cifar import CIFAR10
# from torchvision.transforms import Compose, ToTensor
# from torchvision.transforms import RandomHorizontalFlip, RandomCrop, Normalize
#
# transforms = Compose([
#     T.ToPILImage(),
#     RandomCrop((32, 32), padding=4),
#     RandomHorizontalFlip(p=0.5),
#     T.ToTensor(),
#     # ➊ 데이터 정규화
#     Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
#     T.ToPILImage()
# ])
#
# training_data = CIFAR10(
#     root="./01_DL_practice/01_Pytorch/data",
#     train=True,
#     download=True,
#     transform=transforms)
#
# test_data = CIFAR10(
#     root="./01_DL_practice/01_Pytorch/data",
#     train=False,
#     download=True,
#     transform=transforms)
#
# for i in range(9):
#     plt.subplot(3, 3, i+1)
#     plt.imshow(transforms(training_data.data[i]))
# plt.show()


""" 데이터셋의 평균과 표준편차 """
# import torch
#
# training_data = CIFAR10(
#     root="./01_DL_practice/01_Pytorch/data",
#     train=True,
#     download=True,
#     transform=ToTensor())
#
# # item[0]은 이미지, item[1]은 정답 레이블
# imgs = [item[0] for item in training_data]
#
# # ❶imgs를 하나로 합침
# imgs = torch.stack(imgs, dim=0).numpy()
#
# # rgb 각각의 평균
# mean_r = imgs[:, 0, :, :].mean()
# mean_g = imgs[:, 1, :, :].mean()
# mean_b = imgs[:, 2, :, :].mean()
# print(mean_r, mean_g, mean_b)
#
# # rgb 각각의 표준편차
# std_r = imgs[:, 0, :, :].std()
# std_g = imgs[:, 1, :, :].std()
# std_b = imgs[:, 2, :, :].std()
# print(std_r, std_g, std_b)


""" VGG 기본 블록 정의 """
import torch
import torch.nn as nn

class BasicBlock(nn.Module):        # ❶ 기본 블록을 정의합니다.
    # 기본블록을 구성하는 계층의 정의
    def __init__(self, in_channels, out_channels, hidden_dim):
        # ❷ nn.Module 클래스의 요소 상속
        super(BasicBlock, self).__init__()

        # ❸ 합성곱층 정의
        self.conv1 = nn.Conv2d(in_channels, hidden_dim,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, out_channels,
                               kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # stride는 커널의 이동 거리를 의미합니다.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):           # ➍  기본블록의 순전파 정의
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        return x


""" VGG 모델 정의 """
class CNN(nn.Module):
    def __init__(self, num_classes): # num_classes는 클래스의 개수를 의미합니다
        super(CNN, self).__init__()

        # ❶ 합성곱 기본 블록의 정의
        self.block1 = BasicBlock(in_channels=3, out_channels=32, hidden_dim=16)
        self.block2 = BasicBlock(in_channels=32, out_channels=128, hidden_dim=64)
        self.block3 = BasicBlock(in_channels=128, out_channels=256,
                                hidden_dim=128)

        # ❷ 분류기 정의
        self.fc1 = nn.Linear(in_features=4096, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=num_classes)


        # ❸ 분류기의 활성화 함수
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)  # 출력 모양: (-1, 256, 4, 4)
        x = torch.flatten(x, start_dim=1) # ➍ 2차원 특징맵을 1차원으로

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x


""" 데이터 증강 정의 """
import matplotlib.pyplot as plt
import torchvision.transforms as T

from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, ToTensor
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, Normalize
from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam

transforms = Compose([
    RandomCrop((32, 32), padding=4),  # ❶ 랜덤 크롭핑
    RandomHorizontalFlip(p=0.5),  # ❷ y축으로 뒤집기
    ToTensor(),  # ❸ 텐서로 변환
    # ❹ 이미지 정규화
    Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
])


""" 데이터 로드 및 모델 정의 """
# ❶ 학습 데이터와 평가 데이터 불러오기
training_data = CIFAR10(root="./01_DL_practice/01_Pytorch/data", train=True, download=True, transform=transforms)
test_data = CIFAR10(root="./01_DL_practice/01_Pytorch/data", train=False, download=True, transform=transforms)

# ❷ 데이터로더 정의
train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# ❸ 학습을 진행할 프로세서 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# ➍ CNN 모델 정의
model = CNN(num_classes=10)

# ➎ 모델을 device로 보냄
model.to(device)


""" 모델 학습하기 """
# ❶ 학습률 정의
lr = 1e-3

# ❷ 최적화 기법 정의
optim = Adam(model.parameters(), lr=lr)

# 학습 루프 정의
for epoch in range(100):
    for data, label in train_loader:  # ➌ 데이터 호출
        optim.zero_grad()  # ➍ 기울기 초기화

        preds = model(data.to(device))  # ➎ 모델의 예측

        # ➏ 오차역전파와 최적화
        loss = nn.CrossEntropyLoss()(preds, label.to(device))
        loss.backward()
        optim.step()

    if epoch==0 or epoch%10==9:  # 10번마다 손실 출력
        print(f"epoch{epoch+1} loss:{loss.item()}")


# 모델 저장
torch.save(model.state_dict(), "CIFAR.pth")

# 모델 평가
model.load_state_dict(torch.load("CIFAR.pth", map_location=device))
num_corr = 0
with torch.no_grad():
    for data, label in test_loader:
        output = model(data.to(device))
        preds = output.data.max(1)[1]
        corr = preds.eq(label.to(device).data).sum().item()
        num_corr += corr
    
    print(f"Accuracy:{num_corr/len(test_data)}")
