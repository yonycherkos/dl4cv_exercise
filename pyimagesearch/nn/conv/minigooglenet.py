# import the necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, MaxPooling2D, AveragePooling2D, BatchNormalization, concatenate, Dropout, Dense, Flatten

class MiniGoogleNet():
    @staticmethod
    def conv_module(x, K, kX, kY, stride, padding="same"):
        # define a CONV => RELU => BN layer pattern
        x = Conv2D(K, (kX, kY), strides=stride, padding=padding, activation="relu")(x)
        x = BatchNormalization(axis=-1)(x)

        # return the block
        return x

    @staticmethod
    def inception_module(x, numK1x1, numK3x3):
        # define two conv modules, then concatenate them along channel dimension.
        conv_1x1 = MiniGoogleNet.conv_module(x, numK1x1, 1, 1, (1, 1))
        conv_3x3 = MiniGoogleNet.conv_module(x, numK3x3, 3, 3, (1, 1))
        x = concatenate([conv_1x1, conv_3x3], axis=-1)

        # return the block
        return x

    @staticmethod
    def downsample_module(x, K):
        # define a CONV module and POOL layer, then concatenate them along channel dimension.
        conv_3x3 = MiniGoogleNet.conv_module(x, K, 3, 3, (2, 2), padding="valid")
        pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = concatenate([conv_3x3, pool], axis=-1)

        # return the block
        return x

    @staticmethod
    def build(height, width, depth, classes):
        # define the model input and first CONV module
        inputs = Input(shape=(height, width, depth))
        x = MiniGoogleNet.conv_module(inputs, 96, 3, 3, (1, 1))

        # two Inception modules followed by a downsample module
        x = MiniGoogleNet.inception_module(x, 32, 32)
        x = MiniGoogleNet.inception_module(x, 32, 48)
        x = MiniGoogleNet.downsample_module(x, 80)

        # four Inception modules followed by a downsample module
        x = MiniGoogleNet.inception_module(x, 112, 48)
        x = MiniGoogleNet.inception_module(x, 96, 64)
        x = MiniGoogleNet.inception_module(x, 80, 80)
        x = MiniGoogleNet.inception_module(x, 48, 96)
        x = MiniGoogleNet.downsample_module(x, 96)

        # two Inception modules followed by global POOL and dropout
        x = MiniGoogleNet.inception_module(x, 176, 160)
        x = MiniGoogleNet.inception_module(x, 176, 160)
        x = AveragePooling2D((7, 7))(x)
        x = Dropout(0.5)(x)

        # softmax classifier
        x = Flatten()(x)
        x = Dense(classes, activation="softmax")(x)

        # create the model
        model = Model(inputs, x, name="miniGoogleNet")

        # return the constructed network architecture
        return model