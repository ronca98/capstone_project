from tensorflow.keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications import mobilenet, resnet50, xception
import cv2


# Function used to feed individual images such at the CNN can classify for each
def predict_with_CNN(img, model):

    # Convert img to numpy array and normalize data between 0 and 1
    img = image.img_to_array(img) / 255

    # Keras let's you send a batch of images but we need this step just to send 1
    # image to model to see if it'll classify the picture of the cat as cat.
    list_of_images = np.expand_dims(img, axis=0)

    # for TL we convert the images we want to predict into feature data
    # feature_learning_layers = mobilenet.MobileNet(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    # features = feature_learning_layers.predict(list_of_images)

    # The use of .predict here is the expected use as we now have
    # feature data to feed into a model with strictly classification layers
    # results = model.predict(features)

    # Make a prediction using our created model
    results = model.predict(list_of_images)

    class_number = np.argmax(results[0])
    likelihood = np.max(results[0])

    # Print the result
    print(f"Belongs to class number: {class_number}")
    print(f"Likelihood: {likelihood * 100}%")
    class_names = {
        0: "normal",
        1: "under_extruded",
        2: "over_extruded",
        3: "no_pattern"

    }
    class_name = class_names[class_number]
    print(f"Class number {class_number} corresponds to class: {class_name}.")

    return (class_number,
            class_name,
            likelihood*100)


def main():

    file_names = []
    class_numbers = []
    class_names = []
    likelihoods = []
    img_numbers = []

    # Load the model's structure in the .json file
    file_path = Path("model_mobilenet_1.00_224_structure.json")
    model_structure = file_path.read_text()
    model = model_from_json(model_structure)

    # Load the weight's file from the .h5 file
    model.load_weights("model_mobilenet_1.00_224_weights.h5")
    img_array = []
    # This for loop will eventually be replaced with a live feed of images coming in
    for img_num in range(71, 527):
        file_path = Path(f"images_to_try/Princess_Leia_normal_{img_num}.png")
        file_name = file_path.name
        file_names.append(file_name)
        img = image.load_img(file_path,
                             target_size=(224, 224),
                             color_mode="rgb")
        print(img_num)
        img_numbers.append(img_num)
        img_opencv = cv2.imread(file_path.as_posix())
        img_array.append(img_opencv)

        class_num, class_name, likelihood = predict_with_CNN(img, model)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_opencv, class_name, (10, 224), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        class_numbers.append(class_num)
        class_names.append(class_name)
        likelihoods.append(likelihood)

    # Code for outputting results in video .avi form
    out = cv2.VideoWriter("prediction_results.avi",
                          cv2.VideoWriter_fourcc(*'DIVX'),
                          15,
                          (224, 224))

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    # Code for outputting results in .csv form
    prediction_results_csv = pd.DataFrame({"File Name": file_names,
                                           "Image Number": img_numbers,
                                           "Class Name": class_names,
                                           "Likelihood": likelihoods})

    prediction_results_csv.to_csv("prediction_results.csv", index=False)

    # Code for plotting results
    fig, ax1 = plt.subplots()
    ax1.plot(img_numbers, class_numbers, "o", color="tab:red")
    ax1.set(xlabel="Image Number",
            ylabel="Class Number",
            title="Prediction Results 0 - N, 1 - U, 2 - O, 3 - NP")
    ax1.grid()
    ax1.set_ylim([0, 3])
    ax1.set_ylabel("Class Number", color="tab:red")
    ax1.set_yticks([0, 1, 2, 3])

    ax2 = ax1.twinx()
    ax2.plot(img_numbers, likelihoods, "o", color="tab:blue")
    ax2.set_ylabel("Likelihood %", color="tab:blue")
    ax2.set_ylim([50, 105])

    fig.tight_layout()
    fig.savefig("prediction_results.png")

    plt.show()


if __name__ == "__main__":
    main()
