# import the necessary packages
from pyimagesearch.nn.conv.lenet import LeNet
from tensorflow.keras.utils import plot_model

model = LeNet.build(28, 28, 3, 3)
plot_model(model, show_shapes=True, to_file="lenet.png")