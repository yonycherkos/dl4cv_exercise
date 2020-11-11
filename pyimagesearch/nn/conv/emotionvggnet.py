# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ReLU, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout

class EmotionVggNet():
    @staticmethod
    def build(width, height, depth, classes):
        inputShape = (height, width, depth)
        model = Sequential()

        # Block #1: first [CONV => RELU => BN]*2 => POOL layer set
        model.add(Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation="elu", kernel_initializer="he_normal", input_shape=inputShape))
        model.add(BatchNormalization(axis=-1))
        model.add(Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation="elu", kernel_initializer="he_normal"))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Block #2: second [CONV => RELU => BN]*2 => POOL layer set
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation="elu", kernel_initializer="he_normal"))
        model.add(BatchNormalization(axis=-1))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation="elu", kernel_initializer="he_normal"))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Block #3: third [CONV => RELU => BN]*2 => POOL layer set
        model.add(Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation="elu", kernel_initializer="he_normal"))
        model.add(BatchNormalization(axis=-1))
        model.add(Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation="elu", kernel_initializer="he_normal"))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Block #4: first set of [FC => RELU => BN] layers
        model.add(Flatten())
        model.add(Dense(64, activation="elu", kernel_initializer="he_normal"))
        model.add(BatchNormalization(axis=-1))

        # Block #5: second set of [FC => RELU => BN] layers
        model.add(Dense(64, activation="elu", kernel_initializer="he_normal"))
        model.add(BatchNormalization(axis=-1))

        # Block #6: softmax classifier
        model.add(Dense(classes, activation="softmax", kernel_initializer="he_normal"))

        # return the constructed network architecture
        return model