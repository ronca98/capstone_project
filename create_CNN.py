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
    model = Sequential()
    # Add Convolutional layer
    model.add(Conv2D(32, (3, 3),
                     padding="same",
                     activation="relu",
                     input_shape=(224, 224, 1)))
    # MaxPooling to reduce size of images but keeping the most important information
    model.add(MaxPooling2D(pool_size=(4, 4)))
    # randomly cut 25% of neural network
    model.add(Dropout(0.25))

    # We need to flatten the 2D x,y pixel data
    # When going from Convolution to Dense Layers
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(3, activation="softmax"))

    return model


def generate_model_TL(x_train, x_val):

    pre_trained_nn = mobilenet.MobileNet(weights="imagenet",
                                         input_shape=(224, 224, 3),
                                         include_top=False)
    print(len(pre_trained_nn.layers))
    x_train = pre_trained_nn.predict(x_train)
    x_val = pre_trained_nn.predict(x_val)

    # Create a model and add layers
    model = Sequential()

    # Since we have features extracted, we don't require any layers besides the
    # last classification layers of a CNN
    model.add(Flatten(input_shape=x_train.shape[1:]))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(3, activation="softmax"))

    return model, x_train, x_val


def main():

    # Training data
    normal_images = Path("normal_images")
    under_extruded_images = Path("under_extruded_images")
    over_extruded_images = Path("over_extruded_images")
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
    vd_normal_images = Path("validation_images/normal")
    vd_under_extruded_images = Path("validation_images/under_extruded")
    vd_over_extruded_images = Path("validation_images/over_extruded")
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

    # Numpy is usually used to deal with multi dimensional arrays
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
        epochs=100,
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




