from fastapi import FastAPI
import gradio as gr
from gradio.routes import mount_gradio_app
from PIL import Image
import numpy as np

def process_image(input_image, resolution):
    # Convert to PIL Image if needed
    if isinstance(input_image, np.ndarray):
        input_image = Image.fromarray(input_image)
    
    # Resize the image
    resized_image = input_image.resize((resolution, resolution))
    
    # Create a new image with 4x3 repetitions
    new_width = resolution * 4
    new_height = resolution * 3
    repeated_image = Image.new('RGB', (new_width, new_height))
    
    # Paste the resized image in a 4x3 grid
    for y in range(3):  # 3 rows
        for x in range(4):  # 4 columns
            repeated_image.paste(resized_image, (x * resolution, y * resolution))
    
    return resized_image, repeated_image

with gr.Blocks() as demo:
    with gr.Row():
        input_image = gr.Image(type="pil", label="Input Image")
        resolution = gr.Slider(minimum=128, maximum=1024, step=32, value=256, label="Resolution")
    
    with gr.Row():
        resized_image = gr.Image(type="pil", label="Resized Image")
        repeated_image = gr.Image(type="pil", label="Repeated Image")
    
    generate_btn = gr.Button("Generate")
    generate_btn.click(
        fn=process_image,
        inputs=[input_image, resolution],
        outputs=[resized_image, repeated_image]
    )

web_app = FastAPI()
app = mount_gradio_app(
    app=web_app,
    blocks=demo,
    path="/",
)

if __name__ == "__main__":
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str)
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)