import gradio as gr
import os
import cv2
import os
import easyocr
import subprocess
import numpy as np
from matplotlib import pyplot as plt
from ultralytics import YOLO
from PIL import Image

def define_class(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    results = ys.predict(
        source=image, max_det=1, conf=0.05, classes=[5, 11, 37, 69, 72]
    )

    res = results[0]

    if res.masks is not None:
        # TODO: CHECK OUT IF MASK HAS SOME TEXT OTHERWISE CHOOSE NEXT MASK

        mask = (res.masks[0].data.numpy() * 255).astype("uint8")

        # Resize the mask to match the size of the original image

        resized_mask = cv2.resize(mask[0], (image.shape[1], image.shape[0]))

        # Create a 3-channel inverse mask
        inverse_mask_colored = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)

        # Apply the inverse mask to the original image
        result_image = cv2.bitwise_and(image, inverse_mask_colored)

        # Convert the image to grayscale
        gray_img = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray_img)

        plt.imsave("./ML_NovaPoshta_CV/res.png", clahe_img)

        script_to_run = "./ML_NovaPoshta_CV/models/lcnn/demo.py"

        # Parameters to pass to the script
        script_parameters = [
            "-d 0",
            "./ML_NovaPoshta_CV/models/lcnn/config/wireframe.yaml",
            "C:/Users/Tymof/OneDrive/Desktop/education/Practice/Meduzzen_Intership/ML_NovaPoshta_CV/models/lcnn/190418-201834-f8934c6-lr4d10-312k.pth",
            "C:/Users/Tymof/OneDrive/Desktop/education/Practice/Meduzzen_Intership/ML_NovaPoshta_CV/res.png",
        ]
        # Run the script with parameters
        result = subprocess.run(
            ["python", script_to_run] + script_parameters,
            stdout=subprocess.PIPE,
            text=True,
        )
        new_image = cv2.imread("./ML_NovaPoshta_CV/res-0.94.png")
        pil_image = Image.fromarray(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))



    return pil_image


demo = gr.Interface(
    define_class,
    gr.Image(type="pil"),
    "image",
    live=True,
)


if __name__ == "__main__":
    ys = YOLO("./ML_NovaPoshta_CV/models/yolov8s-seg.pt")
    demo.launch()
