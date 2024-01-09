""" Resnet의 기본 블럭"""
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(BasicBlock, self).__init__()

        # ❶ 합성곱층 정의
        self.c1 = nn.Conv2d(in_channels, out_channels,
                            kernel_size=kernel_size, padding=1)
        self.c2 = nn.Conv2d(out_channels, out_channels,
                            kernel_size=kernel_size, padding=1)

        self.downsample = nn.Conv2d(in_channels, out_channels,
                                    kernel_size=1)

        # ❷ 배치 정규화층 정의
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        # ❸스킵 커넥션을 위해 초기 입력을 저장
        x_ = x

        x = self.c1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.c2(x)
        x = self.bn2(x)

        # ➍합성곱의 결과와 입력의 채널 수를 맞춤
        x_ = self.downsample(x_)

        # ➎합성곱층의 결과와 저장해놨던 입력값을 더해줌
        x += x_
        x = self.relu(x)

        return x


""" Resnet 모델 정의하기 """
class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()


        # ❶ 기본 블록
        self.b1 = BasicBlock(in_channels=3, out_channels=64)
        self.b2 = BasicBlock(in_channels=64, out_channels=128)
        self.b3 = BasicBlock(in_channels=128, out_channels=256)


        # ❷ 풀링을 최댓값이 아닌 평균값으로
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # ❸ 분류기
        self.fc1 = nn.Linear(in_features=4096, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=num_classes)

        self.relu = nn.ReLU()
    def forward(self, x):
        # ❶ 기본 블록과 풀링층을 통과
        x = self.b1(x)
        x = self.pool(x)
        x = self.b2(x)
        x = self.pool(x)
        x = self.b3(x)
        x = self.pool(x)


        # ❷ 분류기의 입력으로 사용하기 위해 flatten
        x = torch.flatten(x, start_dim=1)

        # ❸ 분류기로 예측값 출력
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x


""" 데이터 전처리 정의 """
import tqdm

from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, ToTensor
from torchvision.transforms import RandomHorizontalFlip, RandomCrop
from torchvision.transforms import Normalize
from torch.utils.data.dataloader import DataLoader

from torch.optim.adam import Adam

from torchinfo import summary

transforms = Compose([
    RandomCrop((32, 32), padding=4), #❶ 랜덤 크롭핑
    RandomHorizontalFlip(p=0.5), #❷ 랜덤 y축 대칭
    ToTensor(),
    Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
])


""" 데이터 불러오기 """
training_data = CIFAR10(root="./01_DL_practice/01_Pytorch/data",
                        train=True,
                        download=True,
                        transform=transforms)
test_data = CIFAR10(root="./01_DL_practice/01_Pytorch/data",
                    train=False,
                    download=True,
                    transform=transforms)

train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


""" 모델 정의 """
device = "cuda" if torch.cuda.is_available() else "cpu"

model = ResNet(num_classes=10)
model.to(device)
# print(model)
# summary(model)


""" 학습루프 정의 """
lr = 1e-4
optim = Adam(model.parameters(), lr=lr)

for epoch in range(30):
    iterator = tqdm.tqdm(train_loader)
    for data, label in iterator:
        # 최적화를 위해 기울기를 초기화
        optim.zero_grad()

        # 모델의 예측값
        preds = model(data.to(device))

        # 손실 계산 및 역전파
        loss = nn.CrossEntropyLoss()(preds, label.to(device))
        loss.backward()
        optim.step()

        iterator.set_description(f"epoch:{epoch+1} loss:{loss.item()}")

torch.save(model.state_dict(), "ResNet.pth")


""" Resnet 성능 확인해보기 """
model.load_state_dict(torch.load("./01_DL_practice/01_Pytorch/model/ResNet.pth", map_location=device))

num_corr = 0

with torch.no_grad():
    for data, label in test_loader:

        output = model(data.to(device))
        preds = output.data.max(1)[1]
        corr = preds.eq(label.to(device).data).sum().item()
        num_corr += corr

    print(f"Accuracy:{num_corr/len(test_data)}")
