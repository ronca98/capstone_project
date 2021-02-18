from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np

# Load the model's structure in the .json file
file_path = Path("model_structure_less_layers.json")
model_structure = file_path.read_text()

model = model_from_json(model_structure)

# Load the weight's file from the .h5 file
model.load_weights("model_weights_less_layers.h5")

# Load the cat .png image for testing
img = image.load_img("deer.png", target_size=(32, 32))

# Convert img to numpy array
image_to_try = image.img_to_array(img) / 255

# Keras let's you send a batch of images but we need this step just to send 1
# image to model to see if it'll classify the picture of the cat as cat.
list_of_images = np.expand_dims(image_to_try, axis=0)

# Make a prediction using our created model
results = model.predict(list_of_images)
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