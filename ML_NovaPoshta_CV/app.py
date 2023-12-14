import gradio as gr


def define_class(image):
    return image


demo = gr.Interface(
    define_class,
    gr.Image(type="pil"),
    "text",
    live=True,
)

if __name__ == "__main__":
    demo.launch()
