import argparse
import glob

import gradio as gr
import numpy as np

from inference import get_mask
from train import AnimeSegmentation, net_names


def rmbg_fn(img, img_size):
    mask = get_mask(model, img, False, int(img_size))
    img = (mask * img + 255 * (1 - mask)).astype(np.uint8)
    mask = (mask * 255).astype(np.uint8)
    img = np.concatenate([img, mask], axis=2, dtype=np.uint8)
    mask = mask.repeat(3, axis=2)
    return mask, img


def load_model(path, net_name, img_size):
    global model
    model = AnimeSegmentation.try_load(
        net_name=net_name, img_size=int(img_size), ckpt_path=path, map_location="cpu"
    )
    model.eval()
    return "success"


def get_model_path():
    model_paths = sorted(glob.glob("**/*.ckpt", recursive=True))
    return model_path_input.update(choices=model_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=6006, help="gradio server port,")
    opt = parser.parse_args()
    model = None

    app = gr.Blocks()
    with app:
        with gr.Accordion(label="Model option", open=False):
            load_model_path_btn = gr.Button("Get Models")
            model_path_input = gr.Dropdown(label="model")
            model_type = gr.Dropdown(
                label="model type",
                value="isnet_is",
                choices=net_names,
            )
            model_image_size = gr.Slider(
                label="image size", value=1024, minimum=0, maximum=1280, step=32
            )
            load_model_path_btn.click(get_model_path, [], model_path_input)
            load_model_btn = gr.Button("Load")
            model_msg = gr.Textbox()
            load_model_btn.click(
                load_model, [model_path_input, model_type, model_image_size], model_msg
            )
        input_img = gr.Image(label="input image")
        run_btn = gr.Button(variant="primary")
        with gr.Row():
            output_mask = gr.Image(label="mask")
            output_img = gr.Image(label="result", image_mode="RGBA")
        run_btn.click(rmbg_fn, [input_img, model_image_size], [output_mask, output_img])
    app.launch(server_port=opt.port)
