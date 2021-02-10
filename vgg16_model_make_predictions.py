from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np
from keras.applications import vgg16

# Load the json file that contains the model's structure
file_path = Path("vgg16_model_structure.json")
model_structure = file_path.read_text()

# Recreate the Keras model object from the json data
model = model_from_json(model_structure)

# Re-load the model's trained weights
model.load_weights("vgg16_model_weights.h5")

# Load an image file to test, resizing it to 64*64 pixels as required by our model
img = image.load_img("not_dog.png", target_size=(64, 64))

# Convert single image to numpy array
image_array = image.img_to_array(img)

# Keras needs a list of images as input, so 1 image inside a list
images = np.expand_dims(image_array, axis=0)

# Normalize data
images = vgg16.preprocess_input(images)

# Use pre-trained model to extract features from our test image
feature_extraction_model = vgg16.VGG16(weights="imagenet",
                                       include_top=False,
                                       input_shape=(64, 64, 3))

features = feature_extraction_model.predict(images)

# Given the extracted features, make a final prediction using our own model
results = model.predict(features)

single_result = np.argmax(results[0])

# Print the result
print(f"Belongs to class number: {single_result}")


