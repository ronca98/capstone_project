import keras
from pathlib import Path
import numpy as np
from keras.preprocessing import image
from keras.applications import resnet50
from keras.models import Sequential
from keras.layers import Dense, Flatten

normal_images = Path("normal_images")
under_extruded_images = Path("under_extruded_images")

images = []
labels = []

# Load all not-dog images
for img in normal_images.glob("*.png"):
    img = image.load_img(img)

    image_array = image.img_to_array(img)
    images.append(image_array)

    # we associate 0 as the label number for not-dog images
    labels.append(0)


# Load all dog images
for img in under_extruded_images.glob("*.png"):
    img = image.load_img(img)

    image_array = image.img_to_array(img)
    images.append(image_array)

    # associate 1 as the label number for dog images
    labels.append(1)

# Numpy is usually used to deal with multi dimensional arrays
x_train = np.array(images)

# Setting up our class labels for keras
y_train = np.array(labels)
y_train = keras.utils.to_categorical(y_train, 2)

# Normalize data set to values between 0 and 1
x_train = resnet50.preprocess_input(x_train)

# Load an existing neural  network to use as a feature extractor
# image_top = False means we cut off the last layer of this neural network
pre_trained_nn = resnet50.ResNet50(weights="imagenet",
                                   input_shape=(224, 224, 3),
                                   include_top=False)
x_train = pre_trained_nn.predict(x_train)

# Create a model and add layers
model = Sequential()

# Since we have features extracted, we don't require any layers besides the
# last classification layers of a CNN
model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(512, activation="relu"))
model.add(Dense(2, activation="softmax"))

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
    epochs=50,
    shuffle=True
)

# Save neural network structure
model_structure = model.to_json()
file_path = Path("model_structure_TL.json")
file_path.write_text(model_structure)

# Save neural network weights
model.save_weights("model_weights_TL.h5")

