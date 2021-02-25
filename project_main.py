from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np
from keras.applications import resnet50
import tensorflow as tf


def predict_with_CNN(img):

    # Load the model's structure in the .json file
    file_path = Path("model_structure.json")
    model_structure = file_path.read_text()
    model = model_from_json(model_structure)

    # Load the weight's file from the .h5 file
    model.load_weights("model_weights.h5")

    # Convert img to numpy array and normalize data between 0 and 1
    img = image.img_to_array(img) / 255

    # Keras let's you send a batch of images but we need this step just to send 1
    # image to model to see if it'll classify the picture of the cat as cat.
    list_of_images = np.expand_dims(img, axis=0)

    # Make a prediction using our created model
    results = model.predict(list_of_images)
    class_number = np.argmax(results[0])
    likelihood = np.max(results[0])

    # Print the result
    print(f"Belongs to class number: {class_number}")
    print(f"Likelihood: {likelihood * 100}%")
    cifar10_class_names = {
        0: "normal",
        1: "under_extruded",

    }
    print(f"Class number {class_number} corresponds to class: {cifar10_class_names[class_number]}.")

    tf.keras.backend.clear_session()


def main():
    resolution = (224, 224)
    # Load an image file to test, resizing it to 32*32 pixels as required by our model
    for img_num in range(189, 236):
        img = image.load_img(fr"validation_images/under_extruded/UF_block_65_30_{img_num}.png",
                             target_size=resolution)
        predict_with_CNN(img)


if __name__ == "__main__":
    main()
