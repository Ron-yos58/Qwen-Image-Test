import os
import spaces
import gc
import shutil
import gradio as gr
import torch
import safetensors.torch
from huggingface_hub import hf_hub_download, HfApi, login
from accelerate import init_empty_weights

# --- Imports from your local modules ---
# Ensure the folder 'qwenimage' is present in the root directory
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel

# Configuration for the specific Qwen Transformer
TRANSFORMER_CONFIG = {
    "attention_head_dim": 128,
    "axes_dims_rope": [16, 56, 56],
    "guidance_embeds": False,
    "in_channels": 64,
    "joint_attention_dim": 3584,
    "num_attention_heads": 24,
    "num_layers": 60,
    "out_channels": 16,
    "patch_size": 2
}

@spaces.GPU(duration=300)
def convert_and_upload(hf_token, target_repo_id, private_repo):
    """
    Downloads raw weights, converts keys, saves locally, and uploads to HF.
    """
    local_dir = "converted_qwen_transformer"
    source_repo = "Phr00t/Qwen-Image-Edit-Rapid-AIO"
    source_filename = "v19/Qwen-Rapid-AIO-NSFW-v19.safetensors"
    
    yield f"🚀 Starting process...\nAuthenticating with Hugging Face..."
    
    if not hf_token:
        raise gr.Error("Please provide a Write-enabled Hugging Face Token.")
    
    try:
        login(token=hf_token)
        api = HfApi(token=hf_token)
    except Exception as e:
        raise gr.Error(f"Authentication failed: {e}")

    # 1. Download
    yield f"📥 Downloading {source_filename} from {source_repo}..."
    try:
        checkpoint_path = hf_hub_download(repo_id=source_repo, filename=source_filename)
    except Exception as e:
        raise gr.Error(f"Download failed: {e}")

    # 2. Initialize Empty Model
    yield "🏗️ Initializing empty model architecture..."
    with init_empty_weights():
        model = QwenImageTransformer2DModel(**TRANSFORMER_CONFIG)

    # 3. Load and Filter Keys
    yield "🔑 Loading state dict and filtering keys (removing 'model.diffusion_model.')..."
    try:
        state_dict = safetensors.torch.load_file(checkpoint_path, device="cpu")
        
        new_state_dict = {}
        prefix = "model.diffusion_model."
        ignored_keys = ["__index_timestep_zero__", "iteration", "global_step"]

        for key, value in state_dict.items():
            if key in ignored_keys:
                continue
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                new_state_dict[new_key] = value
        
        del state_dict
        gc.collect()
    except Exception as e:
        raise gr.Error(f"Error processing keys: {e}")

    # 4. Load Weights into Model
    yield "⚖️ Loading weights into the model object..."
    try:
        # assign=True is needed for accelerate's init_empty_weights
        model.load_state_dict(new_state_dict, assign=True, strict=False)
        del new_state_dict
        gc.collect()
    except Exception as e:
        raise gr.Error(f"Error loading weights into model: {e}")

    # 5. Save Locally
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    yield f"💾 Saving converted model to local directory: {local_dir}..."
    try:
        # This saves both config.json and diffusion_pytorch_model.safetensors
        model.save_pretrained(local_dir, safe_serialization=True)
    except Exception as e:
        raise gr.Error(f"Error saving local model: {e}")

    # 6. Upload to Hugging Face
    yield f"☁️ Uploading to Hugging Face Repo: {target_repo_id}..."
    try:
        # Create repo if it doesn't exist
        api.create_repo(repo_id=target_repo_id, private=private_repo, exist_ok=True)
        
        api.upload_folder(
            folder_path=local_dir,
            repo_id=target_repo_id,
            commit_message="Upload converted Qwen-Image-Edit Transformer"
        )
    except Exception as e:
        raise gr.Error(f"Upload failed: {e}")
        
    # Cleanup
    shutil.rmtree(local_dir)
    gc.collect()
    
    yield f"✅ Success! Model uploaded to https://huggingface.co/{target_repo_id}"

# --- Gradio UI ---

css = """
#col-container { max_width: 700px; margin: 0 auto; }
"""

with gr.Blocks() as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# 🔄 Qwen Transformer Converter & Uploader")
        gr.Markdown(
            "This tool downloads the raw checkpoints for `Qwen-Image-Edit`, extracts the transformer, "
            "fixes the key names, and uploads the clean `diffusers`-ready model to your Hugging Face account."
        )
        
        with gr.Group():
            hf_token = gr.Textbox(
                label="Hugging Face Token (Write Access)", 
                placeholder="hf_...", 
                type="password"
            )
            target_repo = gr.Textbox(
                label="Target Repository ID", 
                placeholder="username/my-converted-qwen-transformer"
            )
            is_private = gr.Checkbox(label="Make Repo Private", value=True)
            
        convert_btn = gr.Button("Convert & Upload", variant="primary")
        status_output = gr.Textbox(label="Status Log", interactive=False, lines=6)

    convert_btn.click(
        fn=convert_and_upload,
        inputs=[hf_token, target_repo, is_private],
        outputs=[status_output]
    )

if __name__ == "__main__":
    demo.queue().launch(theme=gr.themes.Soft(), css=css)