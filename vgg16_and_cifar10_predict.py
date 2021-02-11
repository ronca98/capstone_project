from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np
from keras.applications import vgg16

# Load the json file that contains the model's structure
file_path = Path("vgg16_cifar10_model_structure.json")
model_structure = file_path.read_text()

# Recreate the Keras model object from the json data
model = model_from_json(model_structure)

# Re-load the model's trained weights
model.load_weights("vgg16_cifar10_model_weights.h5")

# Load an image file to test, resizing it to 64*64 pixels as required by our model
img = image.load_img("cat.png",
                     target_size=(32, 32))

# Convert single image to numpy array
image_array = image.img_to_array(img)

# Keras needs a list of images as input, so 1 image inside a list
images = np.expand_dims(image_array, axis=0)

# Normalize data
images = vgg16.preprocess_input(images)

# Use pre-trained model to extract features from our test image
feature_extraction_model = vgg16.VGG16(weights="imagenet",
                                       include_top=False,
                                       input_shape=(32, 32, 3))

features = feature_extraction_model.predict(images)

# Given the extracted features, make a final prediction using our own model
results = model.predict(features)

class_number = np.argmax(results[0])

# Print the result
print(f"Belongs to class number: {class_number}")

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

print(f"Class number {class_number} corresponds to class: {cifar10_class_names[class_number]}.")


