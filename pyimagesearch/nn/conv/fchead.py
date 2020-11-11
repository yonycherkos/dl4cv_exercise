# import the necessary packages
from tensorflow.keras.layers import Flatten, Dense, Activation, Dropout

class FCHeadNet():
    @staticmethod
    def build(baseModel, classes, D):
        # INPUT => FC => RELU => DO => FC => SOFTMAX
        headModel = baseModel.output
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(D, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(classes, activation="softmax")

        return headModel