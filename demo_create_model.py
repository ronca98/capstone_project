import keras
import numpy as np
# import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from pathlib import Path

# Specify our 10 classes for the cifar_10 data set
# Since we know the class numbers for the data set, these numbers and corresponding
# labels are already known.
cifar10_class_names = {
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
# 10000 total test images and labels
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
sample_first_img = x_train[0]
class_number_first_img = y_train[0][0]

# Showing the first image happens to be a frog which is class 6: Frog
# plt.imshow(sample_first_img)
# plt.title(class_number_first_img)
# plt.show()

# --------------------------------------------------------------------
# Now are the steps for setting up CNN

# Normalize data set to values between 0 and 1
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train = x_train / 255
x_test = x_test / 255

# Setting up our class labels for keras
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Add Convolutional layers
model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation="relu"))
# MaxPooling to reduce size of images but keeping the most important information
model.add(MaxPooling2D(pool_size=(2, 2)))
# randomly cut 25% of neural network
model.add(Dropout(0.25))

# Not yet familiar why the guide adds more of these layers
model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# We need to flatten the 2D x,y pixel data
# When going from Convolution to Dense Layers
model.add(Flatten())

model.add(Dense(512, activation="relu"))
model.add(Dropout(0.50))
model.add(Dense(10, activation="softmax"))

# Compile the Model
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# Print a summary of model
# Note, Param # means total number of weights in that layer
model.summary()

# Train the model
model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=30,
    validation_data=(x_test, y_test),
    shuffle=True
)

# Save neural network structure
model_structure = model.to_json()
file_path = Path("model_structure.json")
file_path.write_text(model_structure)

# Save neural network's trained weights
model.save_weights("model_weights.h5")


