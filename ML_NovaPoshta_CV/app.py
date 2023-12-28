import gradio as gr
import os


def define_class(image):
    print(
        os.system(
            f'python ./models/lcnn/demo.py -d 0 ./models/lcnn/config/wireframe.yaml "C:/Users/Tymof/OneDrive/Desktop/education/Practice/Meduzzen_Intership/ML_NovaPoshta_CV/models/lcnn/190418-201834-f8934c6-lr4d10-312k.pth" "C:/Users/Tymof/OneDrive/Desktop/education/Practice/Meduzzen_Intership/ML_NovaPoshta_CV/data/images_data/1001(2023-06-07T08_06_44).jpg"  '
        )
    )
    return image


demo = gr.Interface(
    define_class,
    gr.Image(type="pil"),
    "text",
    live=True,
)


if __name__ == "__main__":
    demo.launch()
