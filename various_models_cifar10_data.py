import keras
from pathlib import Path
import numpy as np
from keras.datasets import cifar10
from keras.applications import resnet50, vgg16, mobilenet
from keras.models import Sequential
from keras.layers import Dense, Flatten

# These are the class labels that correspond to the images in the cifar10 dataset
cifar10_class_labels = {
    0: "Plane",
    1: "Car",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Boat",
    9: "Truck"
}

# Loading in our data
# x are the actual images , while y are the class number corresponding to those images
# 50000 total training images and labels
# 10000 total test images and labels (used for validation, just different naming)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Numpy is usually used to deal with multi dimensional arrays
x_train = np.array(x_train)
x_test = np.array(x_test)

# Setting up our class labels for keras
y_train = np.array(y_train)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = np.array(y_test)
y_test = keras.utils.to_categorical(y_test, 10)

# Normalize data set to values between 0 and 1
x_train = resnet50.preprocess_input(x_train)
x_test = resnet50.preprocess_input(x_test)

# Load an existing neural  network to use as a feature extractor
# image_top = False means we cut off the last layer of this neural network
model = resnet50.ResNet50(weights=None,
                          input_shape=(32, 32, 3),
                          classes=10)

# Compile model
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

# Train model
model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=30,
    shuffle=True
)

# Save neural network structure
model_structure = model.to_json()
file_path = Path("model_cifar10_data_structure.json")
file_path.write_text(model_structure)

# Save neural network weights
model.save_weights("model_cifar10_data_weights.h5")

