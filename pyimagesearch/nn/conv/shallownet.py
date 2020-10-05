from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, Activation, Dense, Flatten
from tensorflow.keras import backend as K


class ShallowNet():
    @staticmethod
    def build(width, height, depth, classes):
        inputShape = (height, width, depth)
        if K.image_data_format() == "channels_first":
            inputShape = (width, height, depth)

        # INPUT => CONV => RELU => FC
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
