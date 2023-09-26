"""

- sequential model은 간단히 구현할 수 있지만, 복잡도가 높은 모델에 적용하기에는 한계가 있음
- 인스턴스 : layers.Dense(Nh)는 layers.Dense 객체의 인스턴스임
  -> 객체를 함수처럼 사용할 수 있기 때문에 ()를 사용해 호출이 가능한 것
  -> def __call__(self, ...)라는 멤버 함수를 사용하면 위의 줄 구현 가능
  -> ex) Dense라는 객체를 만들어 함수처럼 사용하려면...
  ->     class Dense:
            def __call__(self, x):
                print(x)

- model.Model(x, y) : 딥러닝 구조가 여러 가지 딥러닝에 필요한 함수와 연계되도록 만드는 역할
- 예를 들어, 신경망에서 사용하는 학습, 예측, 평가와 같은 다양한 함수를 제공

- 케라스는 컴파일을 수행하여 타깃 플랫폼에 맞게 딥러닝 코드를 구성

- 객체지향 model은 재사용성을 높일 수 있음
- 즉, 일반 사용자의 경우 전문가가 만든 인공지능 모델을 객체로 불로 쉽게 활용할 수 있다는 장점

"""

##############################################
# Modeling
##############################################
from keras import layers, models


# 분산 방식 모델링을 포함하는 함수형 구현
def ANN_models_func(Nin, Nh, Nout):
    x = layers.Input(shape=(Nin,))
    h = layers.Activation('relu')(layers.Dense(Nh)(x))
    y = layers.Activation('softmax')(layers.Dense(Nout)(h))
    model = models.Model(x, y)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# 연쇄 방식 모델링을 포함하는 함수형 구현
def ANN_seq_func(Nin, Nh, Nout):
    model = models.Sequential()
    model.add(layers.Dense(Nh, activation='relu', input_shape=(Nin,)))
    model.add(layers.Dense(Nout, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# 분산 방식 모델링을 포함하는 객체지향형 구현
class ANN_models_class(models.Model):       # 클래스를 만들고, models.Model로부터 특성을 상속
    def __init__(self, Nin, Nh, Nout):      # 클래스의 초기화 함수 정의
        # Prepare network layers and activate functions
        hidden = layers.Dense(Nh)
        output = layers.Dense(Nout)
        relu = layers.Activation('relu')
        softmax = layers.Activation('softmax')

        # Connect network elements
        x = layers.Input(shape=(Nin,))
        h = relu(hidden(x))
        y = softmax(output(h))

        super().__init__(x, y)              # 상속받은 부모 클래스의 초기화를 진행
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# 연쇄 방식 모델링을 포함하는 객체지향형 구현
class ANN_seq_class(models.Sequential):
    def __init__(self, Nin, Nh, Nout):
        super().__init__()                  # 연쇄 방식에서는 부모 클래스의 초기화 함수를 자기 자신의 초기화 함수 가장 앞 단에서 부름
        self.add(layers.Dense(Nh, activation='relu', input_shape=(Nin,)))
        self.add(layers.Dense(Nout, activation='softmax'))
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


##############################################
# Data
##############################################
import numpy as np
from keras import datasets  # mnist
from keras.utils import np_utils  # to_categorical


def Data_func():
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

    Y_train = np_utils.to_categorical(y_train)
    Y_test = np_utils.to_categorical(y_test)

    L, W, H = X_train.shape
    X_train = X_train.reshape(-1, W * H)
    X_test = X_test.reshape(-1, W * H)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return (X_train, Y_train), (X_test, Y_test)


##############################################
# Plotting
##############################################
import matplotlib.pyplot as plt


def plot_acc(history, title=None):
    # summarize history for accuracy
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    if title is not None:
        plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Verification'], loc=0)
    # plt.show()


def plot_loss(history, title=None):
    # summarize history for loss
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    if title is not None:
        plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Verification'], loc=0)
    # plt.show()


##############################################
# Main
##############################################
def main():
    Nin = 784
    Nh = 100
    number_of_class = 10
    Nout = number_of_class

    # model = ANN_models_func(Nin, Nh, Nout)
    # model = ANN_models_class(Nin, Nh, Nout)
    model = ANN_seq_class(Nin, Nh, Nout)
    (X_train, Y_train), (X_test, Y_test) = Data_func()

    ##############################################
    # Training
    ##############################################
    history = model.fit(X_train, Y_train, epochs=15, batch_size=100, validation_split=0.2)      # training set 6만개에서 1.2만개를 validation에 사용한다는 의미
    performace_test = model.evaluate(X_test, Y_test, batch_size=100)
    print('Test Loss and Accuracy ->', performace_test)

    plot_loss(history)
    plt.show()
    plot_acc(history)
    plt.show()


# Run code
if __name__ == '__main__':
    main()
