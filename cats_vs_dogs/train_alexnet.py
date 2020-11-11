# import the necessary packages
import sys
sys.path.append("../")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv.alexnet import AlexNet
from config import cats_vs_dogs_config as config

# construct training and validation image data generator for data augmentation
trainAug = ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,
                              shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
valAug = ImageDataGenerator(rescale=1./255)

# initialize train and validatin data generator
trainGen = trainAug.flow_from_directory(
    "dataset/train", target_size=(227, 227), batch_size=32, class_mode="binary")
valGen = valAug.flow_from_directory(
    "dataset/val", target_size=(227, 227), batch_size=32, class_mode="binary")
testGen = valAug.flow_from_directory(
    "dataset/test", target_size=(227, 227), batch_size=32, class_mode="binary")

# build model architecture
model = AlexNet.build(width=227, height=227, depth=3, reg=0.0002)
print("[INFO] Compiling model...")
opt = Adam(lr=1e-3)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

print("[INFO] Training model...")
checkpoint = ModelCheckpoint(config.MODEL, monitor="val_loss", save_best_only=True, verbose=1) 
tensorboard = TensorBoard(log_dir=config.LOG)
model.fit(trainGen, validation_data=valGen, epochs=50, verbose=1, callbacks=[checkpoint, tensorboard])

print("[INFO] Evaluating...")
report = classification_report(testGen.labels, model.predict(testGen), target_names=["cats", "dogs"])
print(report)
f = open(config.REPORT, "w")
f.write(report)
f.close()