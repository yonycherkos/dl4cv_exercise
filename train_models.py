# import the necessary packages
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv.minivggnet import MiniVGGNet
import argparse
import glob
import os

# construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to output")
ap.add_argument("-m", "--model", required=True, help="path to output model")
ap.add_argument("-n", "num-models", required=True,
                help="number of models to train")
ap.add_argument("-l", "--logs", required=True,
                help="path to output tensorboard log")
args = vars(ap.parse_args())

print("[INFO] loading data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainx = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
                         shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

for i in args["num_models"]:
    print("[INFO] training model {}/{}".format(i + 1, args["num_models"]))
    # build a model
    model = MiniVGGNet.build(
        width=32, height=32, depth=3, classes=len(lb.classes_))

    # compile the model
    opt = SGD(lr=0.001)
    model.compile(optimizers=opt, loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # train the model
    p = os.path.join.sep([args["logs"], "log_{}".format(i + 1)])
    tensorboard = TensorBoard()
    model.fit(aug.flow(trainX, trainY, batch_size=32), validation_data=(
        testX, testY), batch_size=32, epochs=100, callbacks=[tensorboard], verbose=1)

    # evaluate the model
    report = classification_report(testY.argmax(axis=1), model.predict(
        testX).argmax(axis=1), target_names=lb.classes_)
    p = os.path.sep.join(args["output"], "report_{}.txt".format(i + 1))
    f = open(p, "w")
    f.write(report)
    f.close()

    # saving model
    p = os.path.sep.join(args["model"], "model_{}.hdf5".format(i + 1))
    model.save(p)
