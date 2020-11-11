# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.regularizers import l2

class AlexNet():
    @staticmethod
    def build(width, height, depth, classes, reg=0.0002):
        model = Sequential()

        # Block #1: first CONV => RELU => BN => POOL => DO layer set
        model.add(Conv2D(96, (11, 11), strides=(4, 4), padding="same", activation="relu", input_shape=(height, width, depth), kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # Block #2: second CONV => RELU => BN => POOL => DO layer set
        model.add(Conv2D(256, (5, 5), padding="same", activation="relu", kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # Block #3: [[CONV => RELU => BN]*3 => POOL => DO] layer set
        model.add(Conv2D(384, (3, 3), padding="same", activation="relu", kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=-1))
        model.add(Conv2D(384, (3, 3), padding="same", activation="relu", kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=-1))
        model.add(Conv2D(256, (3, 3), padding="same", activation="relu", kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # Block #4: first FC => RELU => BN => DO layer set
        model.add(Flatten())
        model.add(Dense(4096, activation="relu", kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=-1))
        model.add(Dropout(0.5))

        # Block #5: second FC => RELU => BN => DO layer set
        model.add(Dense(4096, activation="relu", kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=-1))
        model.add(Dropout(0.5))

        # Block #6: FC => SOFTMAX
        model.add(Dense(classes, activation="softmax"))

        # return the constructed network architecture
        return model