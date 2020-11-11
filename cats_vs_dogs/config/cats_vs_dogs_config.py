# define path to input dataset
IMAGES_PATH = "../datasets/cats_vs_dogs/"

# define val and test split ratio
NUM_CLASSES = 2
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1

# define path to train, val, and test hdf5
TRAIN_PATH = "dataset/train/"
VAL_PATH = "dataset/val/"
TEST_PATH = "dataset/test/"

# define path output directory
OUTPUT = "output"

# define path to store and load model
MODEL = "output/models/resnet1_cats_vs_dogs.hdf5"

# define path to store tensorboard log file
LOG = "output/logs/resnet/"

# define path to store model evaluation report
REPORT = "output/resnet1_report.txt"

# # define path to datset mean
# DATASET_MEAN = "output/cats_vs_dogs_mean.json"