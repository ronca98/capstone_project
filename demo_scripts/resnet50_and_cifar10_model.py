import keras
from pathlib import Path
import numpy as np
from keras.datasets import cifar10
from keras.applications import resnet50
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
pre_trained_nn = resnet50.ResNet50(weights="imagenet",
                                   input_shape=(32, 32, 3),
                                   include_top=False)
x_train = pre_trained_nn.predict(x_train)
x_test = pre_trained_nn.predict(x_test)

# Create a model and add layers
model = Sequential()

# Since we have features extracted, we don't require any layers besides the
# last classification layers of a CNN
model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(256, activation="relu"))
model.add(Dense(10, activation="softmax"))

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
file_path = Path("resnet50_cifar10_model_structure.json")
file_path.write_text(model_structure)

# Save neural network weights
model.save_weights("resnet50_cifar10_model_weights.h5")

