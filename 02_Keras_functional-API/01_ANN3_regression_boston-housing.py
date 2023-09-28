##############################################
# Modeling
##############################################
from keras import layers, models
from sklearn import preprocessing


class ANN(models.Model):
    def __init__(self, Nin, Nh, Nout):
        # Prepare network layers and activate functions
        hidden = layers.Dense(Nh)
        output = layers.Dense(Nout)
        relu = layers.Activation('relu')

        # Connect network elements
        x = layers.Input(shape=(Nin,))
        h = relu(hidden(x))
        y = output(h)

        super().__init__(x, y)

        self.compile(loss='mse', optimizer='sgd')


##############################################
# Data
##############################################
from keras import datasets


def Data_func():
    (X_train, y_train), (X_test, y_test) = datasets.boston_housing.load_data()
    scaler = preprocessing.MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return (X_train, y_train), (X_test, y_test)


##############################################
# Plotting
##############################################
import matplotlib.pyplot as plt
from keraspp.skeras import plot_loss


##############################################
# Main
##############################################
def main():
    Nin = 13
    Nh = 5
    Nout = 1

    model = ANN(Nin, Nh, Nout)
    (X_train, y_train), (X_test, y_test) = Data_func()

    history = model.fit(X_train, y_train, epochs=100, batch_size=100, validation_split=0.2, verbose=2)

    performace_test = model.evaluate(X_test, y_test, batch_size=100)
    print('\nTest Loss -> {:.2f}'.format(performace_test))

    plot_loss(history, title='happy day')
    # plot_loss(history, title='loss graph')
    plt.show()

    # 예측값 뽑기 -> y_test랑 비교
    # predict_value = model.predict(X_test)


if __name__ == '__main__':
    main()
