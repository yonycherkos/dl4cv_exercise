from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout


class MiniVGGNet():
    @staticmethod
    def build(width, height, depth, classes):
        inputShape = (height, width, depth)
        # INPUT => [[CONV => RELU => BN]*2 => POOL => DO]*2 => [FC => RELU => DO] => [FC => SOFTMAX]
        model = Sequential()

        # first [CONV => RELU => BN]*2 => POOL => DO layers set
        model.add(Conv2D(32, (3, 3), padding="same",
                         activation="relu", input_shape=inputShape))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), padding="same",
                         activation="relu", input_shape=inputShape))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Dropout(0.25))

        # second [CONV => RELU => BN]*2 => POOL => DO layers set
        model.add(Conv2D(64, (3, 3), padding="same",
                         activation="relu", input_shape=inputShape))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), padding="same",
                         activation="relu", input_shape=inputShape))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Dropout(0.25))

        # first (and only) FC => RELU => DO layers
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes, activation="softmax"))

        # return
        return model
