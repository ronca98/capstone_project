import keras
import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from pathlib import Path

# Training data
normal_images = Path("normal_images")
under_extruded_images = Path("under_extruded_images")
images = []
labels = []

# Load all normal training images
for img in normal_images.glob("*.png"):
    img = image.load_img(img)

    image_array = image.img_to_array(img)
    images.append(image_array)

    # we associate 0 as the label number for not-dog images
    labels.append(0)

# Load all under extruded training images
for img in under_extruded_images.glob("*.png"):
    img = image.load_img(img)

    image_array = image.img_to_array(img)
    images.append(image_array)

    # associate 1 as the label number for dog images
    labels.append(1)

# Validation data
vd_normal_images = Path("validation_images/normal")
vd_under_extruded_images = Path("validation_images/under_extruded")
vd_images = []
vd_labels = []

# Load all normal validation images
for img in vd_normal_images.glob("*.png"):
    img = image.load_img(img)

    image_array = image.img_to_array(img)
    vd_images.append(image_array)

    # we associate 0 as the label number for not-dog images
    vd_labels.append(0)

# Load all under extruded validation images
for img in vd_under_extruded_images.glob("*.png"):
    img = image.load_img(img)

    image_array = image.img_to_array(img)
    vd_images.append(image_array)

    # associate 1 as the label number for dog images
    vd_labels.append(1)

# Numpy is usually used to deal with multi dimensional arrays
x_train = np.array(images)
x_val = np.array(vd_images)

# Setting up our class labels for keras
y_train = np.array(labels)
y_val = np.array(vd_labels)
y_train = keras.utils.to_categorical(y_train, 2)
y_val = keras.utils.to_categorical(y_val, 2)

# Normalize data set to values between 0 and 1
x_train = x_train / 255
x_val = x_val / 255

# Add Convolutional layers
model = Sequential()
model.add(Conv2D(64, (4, 4),
                 padding="same",
                 activation="relu",
                 input_shape=(224, 224, 3)))
model.add(Conv2D(64, (4, 4), activation="relu"))
# MaxPooling to reduce size of images but keeping the most important information
model.add(MaxPooling2D(pool_size=(4, 4)))
# randomly cut 25% of neural network
model.add(Dropout(0.25))

# Not yet familiar why the guide adds more of these layers
#model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
#model.add(Conv2D(64, (3, 3), activation="relu"))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

# We need to flatten the 2D x,y pixel data
# When going from Convolution to Dense Layers
model.add(Flatten())

model.add(Dense(512, activation="relu"))
#model.add(Dropout(0.50))
model.add(Dense(2, activation="softmax"))

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
    epochs=30,
    shuffle=True,
    validation_data=(x_val, y_val)
)

# Save neural network structure
model_structure = model.to_json()
file_path = Path("model_structure.json")
file_path.write_text(model_structure)

# Save neural network's trained weights
model.save_weights("model_weights.h5")


