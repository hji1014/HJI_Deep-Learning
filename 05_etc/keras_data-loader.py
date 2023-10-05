import numpy as np
import math
from tensorflow.keras.utils import Sequence


class Dataloader(Sequence):

    def __init__(self, x_set, y_set, batch_size, shuffle=False):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    # batch 단위로 직접 묶어줘야 함
    def __getitem__(self, idx):
        # sampler의 역할(index를 batch_size만큼 sampling해줌)
        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = [self.x[i] for i in indices]
        batch_y = [self.y[i] for i in indices]

        return np.array(batch_x), np.array(batch_y)

    # epoch이 끝날때마다 실행
    def on_epoch_end(self):
        self.indices = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indices)


train_loader = Dataloader(x, y, 128, shuffle=True)
valid_loader = Dataloader(x, y, 128)
test_loader = Dataloader(x, y, 128)

# 방법 1
model.fit(train_loader, validation_data=valid_loader, epochs=10, workers=4)  # multi로 처리할 개수
model.evaluate(test_loader)

# 방법 2
for e in range(epochs):
    for x, y in train_loader:
        train_step(x, y)
    for x, y in valid_loader:
        valid_step(x, y)
