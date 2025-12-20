import gradio as gr
import spaces
import torch
import torchaudio
import numpy as np
import os
import tempfile
from typing import List

# Import SAM-Audio
# Ensure sam_audio is installed/in path
try:
    from sam_audio import SAMAudio, SAMAudioProcessor
except ImportError:
    print("Error: 'sam_audio' library not found. Please ensure it is installed.")
    exit(1)

# Initialize Device and Model globally to save loading time
print("Loading SAM-Audio Model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    # Using the ID provided in your snippet
    MODEL_ID = "facebook/sam-audio-large" 
    model = SAMAudio.from_pretrained(MODEL_ID).to(device).eval()
    processor = SAMAudioProcessor.from_pretrained(MODEL_ID)
    print("SAM-Audio Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    processor = None

def save_audio(tensor, sr, prefix="out"):
    """Helper to save audio tensor to a temp file and return path."""
    # tensor shape: [channels, time]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix=prefix) as tmp:
        path = tmp.name
    # Ensure CPU for saving
    torchaudio.save(path, tensor.detach().cpu(), sr)
    return path

# ==========================================
# 1. Text Prompting Logic
# ==========================================
@spaces.GPU
def process_text_prompt(audio_path, description, reranking_candidates):
    if not model:
        return None, None
    if not audio_path:
        raise gr.Error("Please upload an audio file.")

    # Process inputs
    inputs = processor(audios=[audio_path], descriptions=[description]).to(device)
    
    # Inference
    with torch.inference_mode():
        if reranking_candidates > 1:
            # Using the re-ranking logic from snippet
            result = model.separate(inputs, predict_spans=True, reranking_candidates=reranking_candidates)
        else:
            result = model.separate(inputs, predict_spans=True)

    # Save outputs
    # result.target and result.residual are usually [1, channels, time] or [channels, time]
    # The snippet implies result.target[0]
    target_path = save_audio(result.target[0], processor.audio_sampling_rate, "text_target")
    residual_path = save_audio(result.residual[0], processor.audio_sampling_rate, "text_residual")
    
    return target_path, residual_path

# ==========================================
# 2. Visual Prompting Logic
# ==========================================
@spaces.GPU
def process_visual_prompt(video_path, text_prompt_for_mask):
    if not model:
        return None, None
    if not video_path:
        raise gr.Error("Please upload a video file.")

    # Lazy import for SAM3 dependencies to avoid crash if not using this tab
    try:
        from torchcodec.decoders import VideoDecoder
        from sam3.model_builder import build_sam3_video_predictor
    except ImportError as e:
        raise gr.Error(f"SAM3 or TorchCodec not installed. Required for visual prompting. Error: {e}")

    print("Initializing SAM3 for mask generation...")
    decoder = VideoDecoder(video_path)
    frames = decoder[:] # Extract frames

    # Create mask using SAM3
    try:
        video_predictor = build_sam3_video_predictor()
        # Start session
        response = video_predictor.handle_request({
            "type": "start_session",
            "resource_path": video_path,
        })
        session_id = response["session_id"]

        print(f"Generating masks for {len(decoder)} frames based on prompt: '{text_prompt_for_mask}'...")
        masks = []
        for frame_index in range(len(decoder)):
            response = video_predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": frame_index,
                "text": text_prompt_for_mask, 
            })
            mask = response["outputs"]["out_binary_masks"]
            # Handle empty masks
            if mask.shape[0] == 0:
                mask = np.zeros_like(frames[0, [0]], dtype=bool)
            masks.append(mask[:1]) # Append first mask

        # Concatenate masks: [Time, 1, H, W] -> [Time, H, W] -> unsqueeze appropriately
        # Snippet logic: mask = torch.from_numpy(np.concatenate(masks)).unsqueeze(1)
        mask_tensor = torch.from_numpy(np.concatenate(masks)).unsqueeze(1)

    except Exception as e:
        raise gr.Error(f"Error during SAM3 masking: {e}")

    # Process with SAM-Audio visual prompting
    # Note: audios input accepts video file path to extract audio
    inputs = processor(
        audios=[video_path],
        descriptions=[""], # Description empty as we use visual mask
        masked_videos=processor.mask_videos([frames], [mask_tensor]),
    ).to(device)

    with torch.inference_mode():
        result = model.separate(inputs)

    target_path = save_audio(result.target[0], processor.audio_sampling_rate, "vis_target")
    residual_path = save_audio(result.residual[0], processor.audio_sampling_rate, "vis_residual")
    
    return target_path, residual_path

# ==========================================
# 3. Span Prompting Logic
# ==========================================
@spaces.GPU
def process_span_prompt(audio_path, description, anchors_text):
    if not model:
        return None, None
    if not audio_path:
        raise gr.Error("Please upload an audio file.")

    # Parse Anchors from text area
    # Expected format per line: + 2.0 3.5
    parsed_anchors = []
    try:
        for line in anchors_text.strip().split('\n'):
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) != 3:
                continue
            # [Type (+/-), Start, End]
            parsed_anchors.append([parts[0], float(parts[1]), float(parts[2])])
    except Exception as e:
        raise gr.Error(f"Error parsing anchors: {e}")

    if not parsed_anchors:
        raise gr.Error("No valid anchors found. Use format: + 2.0 3.5")

    # Wrap in list of lists as per snippet: anchors=[anchors]
    # The snippet implies the input to processor expects a list of anchor_groups (one per audio)
    final_anchors = [parsed_anchors] 

    inputs = processor(
        audios=[audio_path],
        descriptions=[description], # Description is still used alongside anchors
        anchors=final_anchors,
    ).to(device)

    with torch.inference_mode():
        result = model.separate(inputs)

    target_path = save_audio(result.target[0], processor.audio_sampling_rate, "span_target")
    residual_path = save_audio(result.residual[0], processor.audio_sampling_rate, "span_residual")
    
    return target_path, residual_path


# ==========================================
# UI Construction
# ==========================================
css = """
#main-title h1 {font-size: 2.3em !important; text-align: center;}
.subtitle {text-align: center; font-size: 1.1em;}
"""

with gr.Blocks() as demo:
    gr.Markdown("# **SAM-Audio**", elem_id="main-title")
    gr.Markdown("Segment Any Model for Audio: Text, Visual, and Temporal Span Prompting", elem_classes="subtitle")

    with gr.Tabs():
        # ---------------------------------------------
        # TAB 1: Text Prompting
        # ---------------------------------------------
        with gr.Tab("1. Text Prompting"):
            gr.Markdown("### Isolate audio events using a text description.")
            with gr.Row():
                with gr.Column():
                    t1_input_audio = gr.Audio(type="filepath", label="Input Audio")
                    t1_desc = gr.Textbox(label="Description", placeholder="e.g., A man speaking, A dog barking")
                    t1_rerank = gr.Slider(minimum=1, maximum=16, step=1, value=1, label="Reranking Candidates (Higher = Better quality, Slower)")
                    t1_btn = gr.Button("Separate Audio", variant="primary")
                with gr.Column():
                    t1_out_target = gr.Audio(label="Target Audio (Isolated)")
                    t1_out_residual = gr.Audio(label="Residual Audio (Background)")

            t1_btn.click(
                fn=process_text_prompt,
                inputs=[t1_input_audio, t1_desc, t1_rerank],
                outputs=[t1_out_target, t1_out_residual]
            )

        # ---------------------------------------------
        # TAB 2: Visual Prompting
        # ---------------------------------------------
        with gr.Tab("2. Visual Prompting"):
            gr.Markdown("### Isolate audio by describing a visual object in the video.")
            gr.Markdown("*(Requires SAM3 and TorchCodec)*")
            with gr.Row():
                with gr.Column():
                    t2_input_video = gr.Video(label="Input Video", format="mp4")
                    t2_visual_prompt = gr.Textbox(label="Visual Object Prompt (for SAM3)", placeholder="e.g., The person on the left")
                    t2_btn = gr.Button("Segment via Video Mask", variant="primary")
                with gr.Column():
                    t2_out_target = gr.Audio(label="Target Audio")
                    t2_out_residual = gr.Audio(label="Residual Audio")

            t2_btn.click(
                fn=process_visual_prompt,
                inputs=[t2_input_video, t2_visual_prompt],
                outputs=[t2_out_target, t2_out_residual]
            )

        # ---------------------------------------------
        # TAB 3: Span Prompting
        # ---------------------------------------------
        with gr.Tab("3. Span Prompting"):
            gr.Markdown("### Isolate audio using temporal anchors (Time Intervals).")
            with gr.Row():
                with gr.Column():
                    t3_input_audio = gr.Audio(type="filepath", label="Input Audio")
                    t3_desc = gr.Textbox(label="Context Description", placeholder="e.g., A horn honking")
                    t3_anchors = gr.TextArea(
                        label="Anchors (Format: +/- StartTime EndTime)", 
                        value="+ 2.0 3.5\n+ 8.0 9.0\n- 0.0 1.0",
                        info="'+' means sound is present, '-' means sound is absent."
                    )
                    t3_btn = gr.Button("Separate by Spans", variant="primary")
                with gr.Column():
                    t3_out_target = gr.Audio(label="Target Audio")
                    t3_out_residual = gr.Audio(label="Residual Audio")

            t3_btn.click(
                fn=process_span_prompt,
                inputs=[t3_input_audio, t3_desc, t3_anchors],
                outputs=[t3_out_target, t3_out_residual]
            )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=css)