from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np
from keras.applications import resnet50

# Load the json file that contains the model's structure
file_path = Path("model_cifar10_data_structure.json")
model_structure = file_path.read_text()

# Recreate the Keras model object from the json data
model = model_from_json(model_structure)

# Re-load the model's trained weights
model.load_weights("model_cifar10_data_weights.h5")

# Load an image file to test, resizing it to 64*64 pixels as required by our model
img = image.load_img("deer.png",
                     target_size=(32, 32))

# Convert single image to numpy array
image_array = image.img_to_array(img)

# Keras needs a list of images as input, so 1 image inside a list
images = np.expand_dims(image_array, axis=0)

# Convert image data to 0-1
images = resnet50.preprocess_input(images)


# Given the extracted features, make a final prediction using our own model
results = model.predict(images)

class_number = np.argmax(results[0])
likelihood = np.max(results[0])

# Print the result
print(f"Belongs to class number: {class_number}")
print(f"Likelihood: {likelihood*100}%")

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


