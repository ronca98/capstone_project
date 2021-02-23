from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np
from keras.applications import resnet50


def predict_with_CNN(img):

    # Specify the classes
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

    # Load the json file that contains the model's structure
    file_path = Path("model_structure_TL.json")
    model_structure = file_path.read_text()

    # Recreate the Keras model object from the json data
    model = model_from_json(model_structure)

    # Load the model's trained weights
    model.load_weights("model_weights_TL.h5")

    # Convert single image to numpy array
    image_array = image.img_to_array(img)

    # Keras needs a list of images as input, so 1 image inside a list
    images = np.expand_dims(image_array,
                            axis=0)

    # Normalize data between 0 and 1
    images = resnet50.preprocess_input(images)

    # Use pre-trained model to extract features from our test image
    feature_extraction_model = resnet50.ResNet50(weights="imagenet",
                                                 include_top=False,
                                                 input_shape=(32, 32, 3))

    features = feature_extraction_model.predict(images)

    # Given the extracted features, make a final prediction using our own model
    results = model.predict(features)

    class_number = np.argmax(results[0])
    likelihood = np.max(results[0])

    # Print the result
    print(f"Belongs to class number: {class_number}")
    print(f"Likelihood: {likelihood * 100}%")
    print(f"Class number {class_number} corresponds to class: {cifar10_class_names[class_number]}.")


def main():
    resolution = (32, 32)
    # Load an image file to test, resizing it to 32*32 pixels as required by our model
    img = image.load_img("plane.png",
                         target_size=resolution)
    predict_with_CNN(img)


if __name__ == "__main__":
    main()
