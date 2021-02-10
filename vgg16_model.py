import keras
from pathlib import Path
import numpy as np
from keras.preprocessing import image
from keras.applications import vgg16
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

# Path to folders with training data
# 64*64 image resolution .png images
not_dog_path = Path("demo_training_data") / "not_dogs"
dog_path = Path("demo_training_data") / "dogs"

images = []
labels = []

# Load all not-dog images
for img in not_dog_path.glob("*.png"):
    img = image.load_img(img)

    image_array = image.img_to_array(img)
    images.append(image_array)

    # we associate 0 as the label number for not-dog images
    labels.append(0)


# Load all dog images
for img in dog_path.glob("*.png"):
    img = image.load_img(img)

    image_array = image.img_to_array(img)
    images.append(image_array)

    # associate 1 as the label number for dog images
    labels.append(1)

# create a single numpy array with all images
x_train = np.array(images)

# convert labels to a numpy array and convert to categorical
y_train = np.array(labels)
y_train = keras.utils.to_categorical(y_train)

# Normalize image data to 0-1 range
x_train = vgg16.preprocess_input(x_train)

# Load a pre-trained neural network to use as a feature extractor
# image_top = False means we cut off the last layer of this neural network
# This is a common step for transfer learning
pre_trained_nn = vgg16.VGG16(weights="imagenet",
                             include_top=False,
                             input_shape=(64, 64, 3))

x_train = pre_trained_nn.predict(x_train)

# Create a model and add layers
model = Sequential()

# Since we have features extracted, we don't require any convolutional layers
model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation="sigmoid"))

# Compile model
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# Train model
model.fit(
    x_train,
    y_train,
    epochs=10,
    shuffle=True
)

# Save neural network structure
model_structure = model.to_json()
file_path = Path("vgg16_model_structure.json")
file_path.write_text(model_structure)

# Save neural network weights
model.save_weights("vgg16_model_weights.h5")



