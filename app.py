import argparse
import glob
import torch
import gradio as gr
import numpy as np
import cv2
import os

from inference import get_mask
from train import AnimeSegmentation, net_names

# global model state
class ModelState:
    def __init__(self):
        self.model = None
        self.is_loaded = False
        self.current_device = None
        self.current_path = None

state = ModelState()

def rmbg_fn(img, img_size, white_bg_checkbox, only_matted_checkbox):
    if not state.is_loaded or state.model is None:
        raise gr.Error("Please load the model first!")
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = get_mask(state.model, img, False, int(img_size))
        if white_bg_checkbox and only_matted_checkbox:
            img = np.concatenate((mask * img + 255 * (1 - mask), mask * 255), axis=2).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif only_matted_checkbox:
            img = np.concatenate((mask * img + 1 - mask, mask * 255), axis=2).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        else:
            img = np.concatenate((img, mask * img, mask.repeat(3, 2) * 255), axis=1).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        mask = mask.repeat(3, axis=2)
        return mask, img
    except Exception as e:
        raise gr.Error(f"Error processing image: {str(e)}")


def get_available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])
    return devices


def auto_load_model():
    global state
    if state.is_loaded:
        return gr.Info("Model already loaded successfully")
        
    try:
        model_paths = sorted(glob.glob("**/*.ckpt", recursive=True))
        if not model_paths:
            raise gr.Error("No model files found")
            
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        result = load_model(model_paths[0], "isnet_is", 1024, device)
        return result
    except Exception as e:
        return gr.Error(f"Failed to auto-load model: {str(e)}")


def load_model(path, net_name, img_size, device="cuda:0"):
    global state
    
    # check if model is already loaded 
    if state.is_loaded and state.current_path == path and state.current_device == device:
        return gr.Info("Model already loaded successfully")
    
    try:
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
            
        new_model = AnimeSegmentation.try_load(
            net_name=net_name, 
            img_size=int(img_size), 
            ckpt_path=path, 
            map_location=device
        )
        new_model.eval()
        new_model.to(device)
        
        # update state
        state.model = new_model
        state.is_loaded = True
        state.current_device = device
        state.current_path = path
        
        return gr.Info(f"Model loaded successfully on {device}")
    except Exception as e:
        state.is_loaded = False
        state.model = None
        return gr.Error(f"Failed to load model: {str(e)}")


def get_model_path():
    model_paths = sorted(glob.glob("**/*.ckpt", recursive=True))
    if model_paths:
        return gr.update(choices=model_paths, value=model_paths[0])
    
    raise gr.Error("No model files found")


def batch_inference(input_dir, output_dir, img_size, white_bg_checkbox, only_matted_checkbox):
    if not os.path.exists(input_dir):
        raise gr.Error("Input directory does not exist")
    
    img_paths = sorted(glob.glob(os.path.join(input_dir, "*.*")))
    if not img_paths:
        raise gr.Error("No image files found")
    
    progress = gr.Progress(track_tqdm=True)
    total_images = len(img_paths)
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        for i, path in enumerate(progress.tqdm(img_paths, desc="Processing images")):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise gr.Error(f"Failed to read image: {path}")
            
            # no need mask for batch processing
            _, processed_img = rmbg_fn(img, img_size, white_bg_checkbox, only_matted_checkbox)
            
            cv2.imwrite(f'{output_dir}/{i:06d}.png', processed_img)
    
    except Exception as e:
        raise gr.Error(f"Processing error: {str(e)}")
    
    return f"Batch processing completed: {total_images} images processed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=6006, help="gradio server port")
    opt = parser.parse_args()

    app = gr.Blocks()
    with app:
        app.load(auto_load_model)

        with gr.Accordion(label="Model Settings", open=False):
            load_model_path_btn = gr.Button("Get Models")
            model_path_input = gr.Dropdown(label="Model Path")
            model_type = gr.Dropdown(
                label="Model Type",
                value="isnet_is",
                choices=net_names,
            )
            model_image_size = gr.Slider(
                label="Image Size", value=1024, minimum=0, maximum=1280, step=32
            )
            device_dropdown = gr.Dropdown(
                label="Device",
                choices=get_available_devices(),
                value="cuda:0" if torch.cuda.is_available() else "cpu"
            )
            load_model_path_btn.click(get_model_path, [], model_path_input)
            load_model_btn = gr.Button("Load")
            load_model_btn.click(
                load_model,
                inputs=[model_path_input, model_type, model_image_size, device_dropdown],
                outputs=[]
            )
            
        with gr.Tabs():
            with gr.TabItem("Image Inference"):
                input_img = gr.Image(label="Input Image")

                white_bg_checkbox = gr.Checkbox(label="White Background", value=False)
                only_matted_checkbox = gr.Checkbox(label="Only Matted", value=True)

                run_btn = gr.Button("Process", variant="primary")

                with gr.Row():
                    output_mask = gr.Image(label="Mask")
                    output_img = gr.Image(label="Result", image_mode="RGBA")

                run_btn.click(
                    fn=rmbg_fn,
                    inputs=[
                        input_img,
                        model_image_size,
                        white_bg_checkbox,
                        only_matted_checkbox
                    ],
                    outputs=[output_mask, output_img]
                )
                
            with gr.TabItem("Batch Processing"):
                input_dir = gr.Textbox(label="Input Directory")
                output_dir = gr.Textbox(label="Output Directory")

                batch_white_bg_checkbox = gr.Checkbox(label="White Background", value=False)
                batch_only_matted_checkbox = gr.Checkbox(label="Only Matted", value=True)
                status_text = gr.Textbox(label="Status", interactive=False)
                batch_run_btn = gr.Button("Start Processing", variant="primary")

                batch_run_btn.click(
                    batch_inference,
                    inputs=[input_dir, output_dir, model_image_size,
                           batch_white_bg_checkbox, batch_only_matted_checkbox],
                    outputs=[status_text]
                )

    app.launch(server_port=opt.port)