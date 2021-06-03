from keras.models import Sequential
from keras.applications import mobilenet
from keras.layers import Dense, Flatten
from pathlib import Path
from tensorflow import keras
import tensorflow as tf


def process(image_file, label):
    image_file = tf.cast(image_file/255, tf.float32)
    return image_file, label


# This function is used for transfer learning with an existing neural net
def generate_model_TL(x_train):

    # Grabs the feature learning layers from an existing neural net
    feature_learning_layers = mobilenet.MobileNet(weights="imagenet",
                                                  input_shape=(224, 224, 3),
                                                  include_top=False)
    # print(len(pre_trained_nn.layers))

    # .predict is NOT used for prediction right now but to send
    # image data into the feature learning layers of the existing
    # neural net so that it will convert it into feature data
    x_train = feature_learning_layers.predict(x_train)

    # Create a model and add layers
    model = Sequential()

    # Feature data needs to be converted to 1D vector
    # before going into classification layers.
    model.add(Flatten(input_shape=x_train.shape[1:]))

    # Since we have feature data extracted
    # we will input them to our own classification layers
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(3, activation="softmax"))

    return model, x_train


# Transfer learning but feature learning layers are also being trained to have their weights adjusted
def generate_model_TF_fine_tune():
    base_model = keras.applications.MobileNet(
        weights="imagenet",
        input_shape=(224, 224, 3),
        include_top=False)

    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))

    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)

    outputs = keras.layers.Dense(4, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    return model, base_model


def main():

    training_data = tf.keras.preprocessing.image_dataset_from_directory("data_set/",
                                                                        image_size=(224, 224),
                                                                        label_mode="categorical",
                                                                        shuffle=True,
                                                                        class_names=["normal_images",
                                                                                     "under_extruded_images",
                                                                                     "over_extruded_images",
                                                                                     "no_pattern"])

    training_data = training_data.map(process)

    # with TL, x_train and x_val will be feature data and not image data
    # Since image data was converted to feature data, the model
    # will ONLY consist of the classification layers for this
    # model, x_train, x_val = generate_model_TL(x_train)

    model, base_model = generate_model_TF_fine_tune()
    base_model.trainable = True
    model_name = base_model.name

    # Compile the Model
    model.compile(loss="categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(1e-5),
                  metrics=["accuracy"])

    # Print a summary of model
    # Note, Param # means total number of weights in that layer
    model.summary()

    # Train the model
    model.fit(
        training_data,
        epochs=10
    )

    # Save neural network structure
    model_structure = model.to_json()
    file_path = Path(f"model_{model_name}_structure.json")
    file_path.write_text(model_structure)

    # Save neural network's trained weights
    model.save_weights(f"model_{model_name}_weights.h5")


if __name__ == '__main__':
    main()




