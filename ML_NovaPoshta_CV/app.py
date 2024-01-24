import gradio as gr
import easyocr
import subprocess
import numpy as np
from ultralytics import YOLO
from PIL import Image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from scipy.spatial.distance import cosine
import json
from matplotlib import pyplot as plt
import cv2
import os


def calculate_cnn_feature_vector(image):
    image = cv2.resize(image, (224, 224))

    # Preprocess the image for the VGG16 model
    image = preprocess_input(image)

    # Expand dimensions to create a batch of size 1
    image = np.expand_dims(image, axis=0)

    # Get the feature vector using the pre-trained VGG16 model
    feature_vector = model.predict(image)

    # Flatten the feature vector
    flattened_feature_vector = feature_vector.flatten()

    return flattened_feature_vector


def define_class(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    results = ys.predict(
        source=image, max_det=1, conf=0.03, classes=[5, 11, 37, 69, 72]
    )

    for res in results:
        mask_found = False

        if res.masks is None:
            continue

        for j in range(len(res.masks)):
            mask = (res.masks[j].data.numpy() * 255).astype("uint8")

            resized_mask = cv2.resize(mask[0], (image.shape[1], image.shape[0]))

            inverse_mask_colored = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)

            result_image = cv2.bitwise_and(image, inverse_mask_colored)

            gray_img = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_img = clahe.apply(gray_img)

            # CHECK OUT IF MASK HAS SOME TEXT OTHERWISE CHOOSE NEXT MASK
            text_found = False
            number_found = False
            reader = easyocr.Reader(["en"], gpu=False)
            result_ocr = reader.readtext(result_image)

            for r_ocr in result_ocr:
                if r_ocr[2] < 0.7 or r_ocr[1].isnumeric() is False:
                    pass
                top_left = tuple(r_ocr[0][0])
                bottom_right = tuple(r_ocr[0][2])
                top_left = (int(top_left[0]), int(top_left[1]))
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                text = r_ocr[1]
                text_found = True
                numbers = 0
                for c in text:
                    if c.isnumeric():
                        numbers += 1
                if numbers >= 3:
                    number_found = True
                print(r_ocr[1])

                if number_found == True:
                    break

            if number_found == False:
                continue

            plt.imsave("./ML_NovaPoshta_CV/res.png", clahe_img)
            print("running script")
            script_to_run = "C:/Users/Tymof/OneDrive/Desktop/education/Practice/Meduzzen_Intership/ML_NovaPoshta_CV/models/lcnn/demo.py"

            script_parameters = [
                "-d 0",
                "C:/Users/Tymof/OneDrive/Desktop/education/Practice/Meduzzen_Intership/ML_NovaPoshta_CV/models/lcnn/config/wireframe.yaml",
                "C:/Users/Tymof/OneDrive/Desktop/education/Practice/Meduzzen_Intership/ML_NovaPoshta_CV/models/lcnn/190418-201834-f8934c6-lr4d10-312k.pth",
                "C:/Users/Tymof/OneDrive/Desktop/education/Practice/Meduzzen_Intership/ML_NovaPoshta_CV/res.png",
            ]

            result = subprocess.run(
                ["python", script_to_run] + script_parameters,
                stdout=subprocess.PIPE,
                text=True,
            )
            print("script done")
            new_image = cv2.imread(
                "C:/Users/Tymof/OneDrive/Desktop/education/Practice/Meduzzen_Intership/ML_NovaPoshta_CV/res-0.94.png"
            )
            mask_found = True
            break
        if mask_found == True:
            break

    # pil_image = Image.fromarray(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

    vec = calculate_cnn_feature_vector(new_image)

    best_match = 0
    best_match_name = ""

    for key in loaded_feature_vectors:
        cosine_similarity_cnn = 1 - cosine(vec, loaded_feature_vectors[key])
        if cosine_similarity_cnn > best_match:
            best_match = cosine_similarity_cnn
            best_match_name = key

    print(f"Euclidean Distance between CNN feature vectors: {best_match}")
    print(best_match_name)
    path_result = "C:/Users/Tymof/OneDrive/Desktop/education/Practice/Meduzzen_Intership/ML_NovaPoshta_CV" + best_match_name[1:]
    pil_image = Image.open(path_result)
    return pil_image


demo = gr.Interface(
    define_class,
    gr.Image(type="pil"),
    "image",
    live=True,
)


if __name__ == "__main__":
    ys = YOLO("./ML_NovaPoshta_CV/models/yolov8s-seg.pt")

    base_model = VGG16(weights="imagenet")
    model = Model(inputs=base_model.input, outputs=base_model.get_layer("fc2").output)
    loaded_feature_vectors = {}

    with open(
        "C:/Users/Tymof/OneDrive/Desktop/education/Practice/Meduzzen_Intership/ML_NovaPoshta_CV/vectors.json",
        "r",
    ) as infile:
        loaded_feature_vectors = json.load(infile)

    for key, value in loaded_feature_vectors.items():
        loaded_feature_vectors[key] = np.array(value)

    demo.launch()
