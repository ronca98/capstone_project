import keras
import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.applications import mobilenet, resnet50, xception
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from pathlib import Path


# Function for creating our array of images and labels to feed into model
def img_array_and_labels(images,
                         labels,
                         folder_path,
                         class_num):

    for img in folder_path.glob("*.png"):
        img = image.load_img(img,
                             color_mode="rgb")

        image_array = image.img_to_array(img)
        images.append(image_array)

        labels.append(class_num)

    return images, labels


# We can use this function to create our own ConvNet
def generate_model():
    # Create a model and add layers
    model = Sequential()

    # Add Convolutional layers to convert image data into feature data
    model.add(Conv2D(32, (3, 3),
                     padding="same",
                     activation="relu",
                     input_shape=(224, 224, 3)))
    model.add(Conv2D(32, (5, 5), activation="relu"))
    model.add(Conv2D(32, (5, 5), activation="relu"))
    # MaxPooling to reduce size and complexity of feature data before classification
    model.add(MaxPooling2D(pool_size=(4, 4)))
    # randomly cut 25% of neural network
    # model.add(Dropout(0.25))

    # Feature data needs to be converted to 1D vector
    # before going into classification layers.
    model.add(Flatten())

    # Classification layers
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(3, activation="softmax"))

    return model


# This function is used for transfer learning with an existing neural net
def generate_model_TL(x_train, x_val):

    # Grabs the convolutional layers from an existing neural net
    conv_layers = mobilenet.MobileNet(weights="imagenet",
                                      input_shape=(224, 224, 3),
                                      include_top=False)
    # print(len(pre_trained_nn.layers))

    # .predict is NOT used for prediction right now but to send
    # image data into the convolutional layers of the existing
    # neural net so that it will convert it into feature data
    x_train = conv_layers.predict(x_train)
    x_val = conv_layers.predict(x_val)

    # Create a model and add layers
    model = Sequential()

    # Feature data needs to be converted to 1D vector
    # before going into classification layers.
    model.add(Flatten(input_shape=x_train.shape[1:]))

    # Since we have feature data extracted
    # we will input them to our own classification layers
    model.add(Dense(2048, activation="relu"))
    model.add(Dense(3, activation="softmax"))

    return model, x_train, x_val


def main():

    data_set_folder = Path("data_set")

    # Training data folders
    normal_images = data_set_folder / "normal_images"
    under_extruded_images = data_set_folder / "under_extruded_images"
    over_extruded_images = data_set_folder / "over_extruded_images"
    td_images = []
    td_labels = []

    # Load all normal training images
    td_images, td_labels = img_array_and_labels(td_images,
                                                td_labels,
                                                normal_images,
                                                class_num=0)

    # Load all under extruded training images
    td_images, td_labels = img_array_and_labels(td_images,
                                                td_labels,
                                                under_extruded_images,
                                                class_num=1)

    # Load all over extruded training images
    td_images, td_labels = img_array_and_labels(td_images,
                                                td_labels,
                                                over_extruded_images,
                                                class_num=2)

    # Validation data
    vd_normal_images = data_set_folder / "validation_images/normal"
    vd_under_extruded_images = data_set_folder / "validation_images/under_extruded"
    vd_over_extruded_images = data_set_folder / "validation_images/over_extruded"
    vd_images = []
    vd_labels = []

    # Load all validation training images
    vd_images, vd_labels = img_array_and_labels(vd_images,
                                                vd_labels,
                                                vd_normal_images,
                                                class_num=0)

    # Load all under extruded validation images
    vd_images, vd_labels = img_array_and_labels(vd_images,
                                                vd_labels,
                                                vd_under_extruded_images,
                                                class_num=1)

    # Load all over extruded validation images
    vd_images, vd_labels = img_array_and_labels(vd_images,
                                                vd_labels,
                                                vd_over_extruded_images,
                                                class_num=2)

    # Numpy is usually used to deal with multi dimensional arrays properly
    x_train = np.array(td_images)
    x_val = np.array(vd_images)

    # Setting up our class labels for keras
    y_train = np.array(td_labels)
    y_val = np.array(vd_labels)
    y_train = keras.utils.to_categorical(y_train, 3)
    y_val = keras.utils.to_categorical(y_val, 3)

    # Normalize data set to values between 0 and 1
    x_train = x_train / 255
    x_val = x_val / 255

    # We call either one of the CNN methodologies

    # with TL, x_train and x_val will be feature data and not image data
    # Since image data was converted to feature data, the model
    # will ONLY consist of the classification layers for this
    model, x_train, x_val = generate_model_TL(x_train, x_val)

    # model = generate_model()

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
        epochs=20,
        shuffle=True,
        validation_data=(x_val, y_val)
    )

    # Save neural network structure
    model_structure = model.to_json()
    file_path = Path("model_structure.json")
    file_path.write_text(model_structure)

    # Save neural network's trained weights
    model.save_weights("model_weights.h5")


if __name__ == '__main__':
    main()




