import gradio as gr


def your_function(input_text):
    output_text = f"You entered: {input_text}"
    return output_text


interface = gr.Interface(
    fn=your_function,
    inputs=gr.Textbox(),
    outputs=gr.Textbox(),
    live=True,
)

interface.launch()
