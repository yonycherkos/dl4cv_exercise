# import the necessary packages
import sys
sys.path.append("../")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv.fchead import FCHeadNet
from config import cats_vs_dogs_config as config

print("[INFO] image data generating...")
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

print("[INFO] building model...")
baseModel = ResNet50(weights="imagenet", include_top=False)
headModel = FCHeadNet.build(baseModel, 2, 256)
model = Model(inputs=baseModel.input, outputs=headModel.output)

# experiment #1
# freeze the base conv layers
for layer in baseModel.layers:
    layer.trainable = False

print("[INFO] compiling model...")
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

print("[INFO] training model...")
checkpoint = ModelCheckpoint(config.MODEL, monitor="val_loss", save_best_only=True, verbose=1) 
tensorboard = TensorBoard(log_dir=config.LOG)
model.fit(trainGen, validation_data=valGen, batch_size=32, epochs=15, verbose=1)

print("[INFO] evaluating...")
print("[INFO] Evaluating...")
report = classification_report(testGen.labels, model.predict(testGen), target_names=["cats", "dogs"])
print(report)
f = open(config.REPORT, "w")
f.write(report)
f.close()

# # experiment #2
# # unfreeze the last conv layers
# for layer in baseModel[:-5]:
#     layer.trainable = True

# print("[INFO] re-compiling model...")
# opt = Adam(lr=0.001)
# model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

# print("[INFO] re-training model...")
# model.fit(trainGen, validation_data=valGen, batch_size=32, epochs=15, verbose=1)

# print("[INFO] re-evaluating...")
# print("[INFO] Evaluating...")
# report = classification_report(testGen.labels, model.predict(testGen), target_names=["cats", "dogs"])
# print(report)
# f = open(config.REPORT, "w")
# f.write(report)
# f.close()