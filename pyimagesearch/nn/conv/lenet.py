from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras import backend as K


class LeNet():
    @staticmethod
    def build(width, height, depth, classes):
        inputShape = (height, width, depth)
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # INPUT => [[CONV => RELU] => POOL]*2 => [FC => RELU] => FC
        model = Sequential()

        model.add(Conv2D(20, (5, 5), padding="same", activation="relu", input_shape=inputShape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        model.add(Flatten())
        model.add(Dense(500, activation="relu"))
        model.add(Dense(classes, activation="softmax"))

        return model
