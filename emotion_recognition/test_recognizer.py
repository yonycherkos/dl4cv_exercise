# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from config import emotion_config as config

print("[INFO] generating test data...")
testAug = ImageDataGenerator(rescale=1./255)

testGen = testAug.flow_from_directory(
    config.TEST_PATH,
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical')

print("[INFO] loading {}...".format(config.MODEL))
model = load_model(config.MODEL)

print("[INFO] evaluating model...")
report = classification_report(testGen.labels.argmax(axis=1), model.predict(testGen).argmax(axis=1), target_names=config.CLASS_NAMES)
print(report)
f = open(config.REPORT, "w")
f.write(report)
f.close()