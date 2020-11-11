# import the necessary packages
import sys
sys.path.append("../")

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from pyimagesearch.nn.conv.emotionvggnet import EmotionVggNet
from config import emotion_config as config
import argparse
import os

# define train, val and test imageDataGenerator
trainAug = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
                              shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode="nearest", rescale=1./255)
valAug = ImageDataGenerator(rescale=1./255)

print("[INFO] genrating data...")
trainGen = trainAug.flow_from_directory(
    config.TRAIN_PATH,
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical')
valGen = valAug.flow_from_directory(
    config.VAL_PATH,
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical')

testGen = valAug.flow_from_directory(
    config.TEST_PATH,
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical')

if not os.path.exists(config.MODEL):
    print("[INFO] compiling model...")
    model = EmotionVggNet.build(48, 48, 1, len(config.CLASS_NAMES))
    opt = Adam(lr=1e-3)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
else:
    print("[INFO] loading {}...".format(config.MODEL))
    model = load_model(config.MODEL)

    # updating learning rate
    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-3)
    print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))

# construct set of callbacks
tensorboard = TensorBoard(log_dir=config.LOG_DIR)
checkpoint = ModelCheckpoint(config.MODEL, monitor="val_loss", verbose=1, save_best_only=True, period=5)
reduceLR = ReduceLROnPlateau(monitor="val_loss", patience=5)
callbacks = [tensorboard, checkpoint, reduceLR]

# compute classWeight to handle data imbalance
classTotal = trainGen.labels.sum(axis=0)
classWeight = classTotal.max()/classTotal

print("[INFO] training model...")
model.fit(trainGen, validation_data=valGen, batch_size=32, epochs=100, verbose=1, callbacks=callbacks, class_weight=classWeight)