from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split


class FashionMNISTModel:

    def __init__(self):

        self.model = None

    def preprocess_training_data(self, X, y):

        X = X.astype('float32')
        X /= 255
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

        return X, y

    def fit(self, X, y):

        batch_size = 128
        epochs = 50

        model2 = Sequential()
        model2.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same',
                          activation='relu', input_shape=(28, 28, 1)))
        model2.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                          activation='relu'))
        model2.add(MaxPooling2D(pool_size=(3, 3)))
        model2.add(Dropout(0.35))
        model2.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                          activation='relu'))
        model2.add(MaxPooling2D(pool_size=(2, 2)))
        model2.add(Dropout(0.3))
        model2.add(Flatten())
        model2.add(Dense(units=256, activation='relu'))
        model2.add(Dropout(0.4))
        model2.add(Dense(units=10, activation='softmax'))

        rmsprop = RMSprop()

        model2.compile(optimizer=rmsprop, loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

        x_train, x_validate, y_train, y_validate = train_test_split(X, y, test_size=0.1, random_state=42)

        model2.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                   validation_data=(x_validate, y_validate), verbose=2)

        self.model = model2

    def preprocess_unseen_data(self, X):

        X = X.astype('float32')
        X /= 255
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

        return X

    def predict(self, X):

        return self.model.predict_classes(X)
