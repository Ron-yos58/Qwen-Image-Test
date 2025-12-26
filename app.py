import os
import gc
import gradio as gr
import numpy as np
import spaces
import torch
import random
import uuid
import tempfile
from PIL import Image
from typing import Iterable
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

import rerun as rr
from gradio_rerun import Rerun

# --- Theme Configuration ---
colors.orange_red = colors.Color(
    name="orange_red",
    c50="#FFF0E5",
    c100="#FFE0CC",
    c200="#FFC299",
    c300="#FFA366",
    c400="#FF8533",
    c500="#FF4500",
    c600="#E63E00",
    c700="#CC3700",
    c800="#B33000",
    c900="#992900",
    c950="#802200",
)

class OrangeRedTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.orange_red,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

orange_red_theme = OrangeRedTheme()

# --- Hardware Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

from diffusers import FlowMatchEulerDiscreteScheduler
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

dtype = torch.bfloat16

# --- Model Loading ---
pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    transformer=QwenImageTransformer2DModel.from_pretrained(
        "linoyts/Qwen-Image-Edit-Rapid-AIO",
        subfolder='transformer',
        torch_dtype=dtype,
        device_map='cuda'
    ),
    torch_dtype=dtype
).to(device)

try:
    pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
    print("Flash Attention 3 Processor set successfully.")
except Exception as e:
    print(f"Warning: Could not set FA3 processor: {e}")

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp_rerun')
os.makedirs(TMP_DIR, exist_ok=True)

# --- Adapters ---
ADAPTER_SPECS = {
    "Multiple-Angles": {
        "repo": "dx8152/Qwen-Edit-2509-Multiple-angles",
        "weights": "镜头转换.safetensors",
        "adapter_name": "multiple-angles"
    },
    "Photo-to-Anime": {
        "repo": "autoweeb/Qwen-Image-Edit-2509-Photo-to-Anime",
        "weights": "Qwen-Image-Edit-2509-Photo-to-Anime_000001000.safetensors",
        "adapter_name": "photo-to-anime"
    },
}

LOADED_ADAPTERS = set()

def update_dimensions_on_upload(image):
    if image is None:
        return 1024, 1024
    
    original_width, original_height = image.size
    
    if original_width > original_height:
        new_width = 1024
        aspect_ratio = original_height / original_width
        new_height = int(new_width * aspect_ratio)
    else:
        new_height = 1024
        aspect_ratio = original_width / original_height
        new_width = int(new_height * aspect_ratio)
        
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8
    
    return new_width, new_height

@spaces.GPU
def infer(
    input_gallery,
    prompt,
    lora_adapter,
    seed,
    randomize_seed,
    guidance_scale,
    steps,
    progress=gr.Progress(track_tqdm=True)
):
    gc.collect()
    torch.cuda.empty_cache()

    if not input_gallery:
        raise gr.Error("Please upload at least one image to edit.")

    # --- Adapter Loading ---
    spec = ADAPTER_SPECS.get(lora_adapter)
    if not spec:
        raise gr.Error(f"Configuration not found for: {lora_adapter}")

    adapter_name = spec["adapter_name"]

    if adapter_name not in LOADED_ADAPTERS:
        print(f"--- Downloading and Loading Adapter: {lora_adapter} ---")
        try:
            pipe.load_lora_weights(
                spec["repo"], 
                weight_name=spec["weights"], 
                adapter_name=adapter_name
            )
            LOADED_ADAPTERS.add(adapter_name)
        except Exception as e:
            raise gr.Error(f"Failed to load adapter {lora_adapter}: {e}")
    else:
        print(f"--- Adapter {lora_adapter} is already loaded. ---")

    pipe.set_adapters([adapter_name], adapter_weights=[1.0])

    # --- Setup Rerun ---
    run_id = str(uuid.uuid4())
    if hasattr(rr, "new_recording"):
        rec = rr.new_recording(application_id="Qwen-Image-Edit", recording_id=run_id)
    elif hasattr(rr, "RecordingStream"):
        rec = rr.RecordingStream(application_id="Qwen-Image-Edit", recording_id=run_id)
    else:
        rr.init("Qwen-Image-Edit", recording_id=run_id, spawn=False)
        rec = rr

    # --- Processing Loop ---
    # gr.Gallery(type="pil") returns a list of tuples: [(PIL.Image, str_caption), ...]
    # We iterate over them.
    
    total_images = len(input_gallery)
    
    for i, item in enumerate(input_gallery):
        # Handle format: item might be (image, caption) tuple or just image depending on version/updates
        if isinstance(item, (tuple, list)):
            input_pil = item[0]
        else:
            input_pil = item

        if randomize_seed:
            current_seed = random.randint(0, MAX_SEED)
        else:
            current_seed = seed

        generator = torch.Generator(device=device).manual_seed(current_seed)
        negative_prompt = "worst quality, low quality, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry"

        input_pil = input_pil.convert("RGB")
        width, height = update_dimensions_on_upload(input_pil)

        try:
            progress((i + 0.5) / total_images, desc=f"Processing Image {i+1}/{total_images}...")
            
            with torch.inference_mode():
                result_image = pipe(
                    image=input_pil,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    generator=generator,
                    true_cfg_scale=guidance_scale,
                ).images[0]
            
            # --- Log to Rerun ---
            # We use set_time_sequence to create a timeline slider in the Rerun viewer
            # allowing the user to slide through their batch of images.
            rec.set_time_sequence("batch_index", i)
            rec.log("images/original", rr.Image(np.array(input_pil)))
            rec.log("images/edited", rr.Image(np.array(result_image)))
            
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            continue
        finally:
            # Clear VRAM after every image to avoid stacking up memory usage
            gc.collect()
            torch.cuda.empty_cache()

    # Save RRD
    rrd_path = os.path.join(TMP_DIR, f"{run_id}.rrd")
    rec.save(rrd_path)
    
    return rrd_path, seed

@spaces.GPU
def infer_example(input_gallery, prompt, lora_adapter):
    # Wrapper for examples
    if not input_gallery:
        return None, 0
    
    # input_gallery comes as a list of paths from Examples,
    # we need to load them as PIL images to mimic the Gallery output structure for the main function if needed,
    # BUT gr.Gallery in examples usually passes list of paths. 
    # The main logic above expects tuples of (PIL, caption) OR PIL. 
    # Let's ensure we convert paths to PIL here.
    
    processed_gallery = []
    for path in input_gallery:
        if isinstance(path, str):
            processed_gallery.append((Image.open(path), ""))
        else:
            processed_gallery.append((path, "")) # Already PIL or weird format
            
    result_rrd, seed = infer(
        processed_gallery, 
        prompt, 
        lora_adapter, 
        0,      # seed
        True,   # randomize
        1.0,    # guidance
        4       # steps
    )
    return result_rrd, seed

# --- Gradio UI Layout ---
css="""
#col-container {
    margin: 0 auto;
    max-width: 1000px;
}
#main-title h1 {font-size: 2.2em !important;}
"""

with gr.Blocks() as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# **Qwen-Image-Edit-2511-LoRAs-Fast (Multi-Image)**", elem_id="main-title")
        gr.Markdown("Perform diverse image edits using specialized adapters. Upload multiple images to process them in a batch. Use the timeline slider in the output to view results.")

        with gr.Row(equal_height=True):
            with gr.Column():
                # Changed to Gallery for multi-upload
                input_gallery = gr.Gallery(
                    label="Upload Images", 
                    type="pil", 
                    columns=2, 
                    height=300,
                    allow_preview=True
                )
                
                prompt = gr.Text(
                    label="Edit Prompt",
                    show_label=True,
                    placeholder="e.g., transform into anime..",
                )

                run_button = gr.Button("Edit Batch", variant="primary")

            with gr.Column():
                rerun_output = Rerun(
                    label="Rerun Visualization", 
                    height=353
                )
                
                with gr.Row():
                    lora_adapter = gr.Dropdown(
                        label="Choose Editing Style",
                        choices=list(ADAPTER_SPECS.keys()),
                        value="Photo-to-Anime"
                    )
                with gr.Accordion("Advanced Settings", open=False, visible=False):
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                    randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                    guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=10.0, step=0.1, value=1.0)
                    steps = gr.Slider(label="Inference Steps", minimum=1, maximum=50, step=1, value=4)
        
        # Updated Examples to be lists of paths
        gr.Examples(
            examples=[
                [["examples/B.jpg"], "Transform into anime.", "Photo-to-Anime"],
                [["examples/A.jpeg"], "Rotate the camera 45 degrees to the right.", "Multiple-Angles"],
                [["examples/B.jpg", "examples/A.jpeg"], "Transform into sketches.", "Photo-to-Anime"],
            ],
            inputs=[input_gallery, prompt, lora_adapter],
            outputs=[rerun_output, seed],
            fn=infer_example,
            cache_examples=False,
            label="Examples"
        )
        
        gr.Markdown("[*](https://huggingface.co/spaces/prithivMLmods/Qwen-Image-Edit-2511-LoRAs-Fast) Experimental Space.")

    run_button.click(
        fn=infer,
        inputs=[input_gallery, prompt, lora_adapter, seed, randomize_seed, guidance_scale, steps],
        outputs=[rerun_output, seed]
    )

if __name__ == "__main__":
    demo.queue(max_size=30).launch(css=css, theme=orange_red_theme, mcp_server=True, ssr_mode=False, show_error=True)