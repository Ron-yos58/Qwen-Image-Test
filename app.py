import gradio as gr
import torch
import torchaudio
import numpy as np
import os
import spaces

# ---------------------------------------------------------
# Import Custom Libraries
# ---------------------------------------------------------
try:
    from sam_audio import SAMAudio, SAMAudioProcessor
    # Visual Prompting dependencies
    from torchcodec.decoders import VideoDecoder
    from sam3.model_builder import build_sam3_video_predictor
except ImportError as e:
    print(f"Warning: Essential libraries (sam_audio, sam3, torchcodec) not found. {e}")

# ---------------------------------------------------------
# Model Initialization
# ---------------------------------------------------------
MODEL_ID = "facebook/sam-audio-large"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading {MODEL_ID} on {device}...")

# Load SAM-Audio Model & Processor Globally
try:
    model = SAMAudio.from_pretrained(MODEL_ID).to(device).eval()
    processor = SAMAudioProcessor.from_pretrained(MODEL_ID)
    print("SAM-Audio loaded successfully.")
except Exception as e:
    print(f"Error loading SAM-Audio: {e}")
    model = None
    processor = None

# Lazy loader for SAM3 to save VRAM if not used immediately
video_predictor = None

def get_sam3_predictor():
    global video_predictor
    if video_predictor is None:
        print("Loading SAM3 Video Predictor...")
        video_predictor = build_sam3_video_predictor()
    return video_predictor

# ---------------------------------------------------------
# Tab 1: Text Prompting
# ---------------------------------------------------------
@spaces.GPU
def process_text_prompting(audio_file, description, rerank_candidates):
    if audio_file is None or not description:
        return None, None

    # Process and separate
    inputs = processor(audios=[audio_file], descriptions=[description]).to(device)
    
    with torch.inference_mode():
        # Use reranking if specified (improves quality, increases latency)
        if rerank_candidates > 1:
            result = model.separate(inputs, predict_spans=True, reranking_candidates=rerank_candidates)
        else:
            result = model.separate(inputs, predict_spans=True)

    # Save results
    sr = processor.audio_sampling_rate
    target_path = "target_text.wav"
    residual_path = "residual_text.wav"
    
    torchaudio.save(target_path, result.target[0].unsqueeze(0).cpu(), sr)
    torchaudio.save(residual_path, result.residual[0].unsqueeze(0).cpu(), sr)
    
    return target_path, residual_path

# ---------------------------------------------------------
# Tab 2: Visual Prompting
# ---------------------------------------------------------
@spaces.GPU
def process_visual_prompting(video_file, visual_prompt_text):
    if video_file is None or not visual_prompt_text:
        return None, None

    # 1. Initialize Video Decoder
    decoder = VideoDecoder(video_file)
    frames = decoder[:] # Get all frames
    
    # 2. Generate Masks using SAM3
    predictor = get_sam3_predictor()
    
    # Start SAM3 Session
    response = predictor.handle_request({
        "type": "start_session",
        "resource_path": video_file,
    })
    session_id = response["session_id"]

    print(f"Generating masks for {len(decoder)} frames using prompt: '{visual_prompt_text}'...")
    masks = []
    
    # Iterate through frames to generate mask for the described object
    # Note: In a production app, you might use tracking instead of per-frame prompting
    # depending on SAM3's optimal usage, but following provided snippet logic:
    for frame_index in range(len(decoder)):
        response = predictor.handle_request({
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": frame_index,
            "text": visual_prompt_text,
        })
        mask = response["outputs"]["out_binary_masks"]
        
        # Handle empty masks
        if mask.shape[0] == 0:
            mask = np.zeros_like(frames[0, [0]], dtype=bool)
        masks.append(mask[:1]) # Take first mask if multiple

    # Concatenate masks: (Time, 1, H, W) -> (Time, 1, H, W)
    # Ensure dimensions match what SAM-Audio expects
    final_mask = torch.from_numpy(np.concatenate(masks)).unsqueeze(1)

    # 3. Process with SAM-Audio using the visual mask
    # Note: descriptions is empty string as we rely on the mask
    inputs = processor(
        audios=[video_file],
        descriptions=[""],
        masked_videos=processor.mask_videos([frames], [final_mask]),
    ).to(device)

    with torch.inference_mode():
        result = model.separate(inputs)

    # Save results
    sr = processor.audio_sampling_rate
    target_path = "target_visual.wav"
    residual_path = "residual_visual.wav"
    
    torchaudio.save(target_path, result.target[0].unsqueeze(0).cpu(), sr)
    torchaudio.save(residual_path, result.residual[0].unsqueeze(0).cpu(), sr)

    return target_path, residual_path

# ---------------------------------------------------------
# Tab 3: Span Prompting (Temporal Anchors)
# ---------------------------------------------------------
@spaces.GPU
def process_span_prompting(audio_file, description, anchors_df):
    if audio_file is None:
        return None, None

    # Convert Dataframe to list format: [["+", 6.3, 7.0], ...]
    # anchors_df columns: ["Type (+/-)", "Start (s)", "End (s)"]
    formatted_anchors = []
    if anchors_df is not None:
        for val in anchors_df:
            # val is likely [type, start, end]
            t, start, end = val[0], float(val[1]), float(val[2])
            formatted_anchors.append([t, start, end])
    
    if not formatted_anchors:
        print("No anchors provided.")
        # Proceed with just text description if anchors are empty
        anchor_input = None
    else:
        anchor_input = [formatted_anchors]

    # Process
    inputs = processor(
        audios=[audio_file],
        descriptions=[description if description else ""],
        anchors=anchor_input,
    ).to(device)

    with torch.inference_mode():
        result = model.separate(inputs)

    # Save results
    sr = processor.audio_sampling_rate
    target_path = "target_span.wav"
    residual_path = "residual_span.wav"
    
    torchaudio.save(target_path, result.target[0].unsqueeze(0).cpu(), sr)
    torchaudio.save(residual_path, result.residual[0].unsqueeze(0).cpu(), sr)

    return target_path, residual_path

# ---------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------
css = """
#main-title h1 {font-size: 2.3em !important; text-align: center;}
.gradio-container {max-width: 1000px !important; margin: auto;}
"""

with gr.Blocks() as demo:
    gr.Markdown("# **SAM-Audio** 🔊", elem_id="main-title")
    gr.Markdown("Segment and isolate sounds using Text, Visual Masks, or Temporal Anchors.")

    with gr.Tabs():
        # ================= TAB 1 =================
        with gr.Tab("Text Prompting"):
            gr.Markdown("### Isolate sound using a text description")
            with gr.Row():
                with gr.Column():
                    t1_input_audio = gr.Audio(label="Input Audio", type="filepath")
                    t1_desc = gr.Textbox(label="Description", placeholder="e.g., 'A man speaking', 'Glass breaking'")
                    t1_rerank = gr.Slider(minimum=1, maximum=16, value=1, step=1, label="Reranking Candidates (Higher = Better quality, slower)")
                    t1_btn = gr.Button("Separate Audio", variant="primary")
                
                with gr.Column():
                    t1_out_target = gr.Audio(label="Target Audio (Isolated)", type="filepath")
                    t1_out_residual = gr.Audio(label="Residual Audio (Background)", type="filepath")
            
            t1_btn.click(
                fn=process_text_prompting,
                inputs=[t1_input_audio, t1_desc, t1_rerank],
                outputs=[t1_out_target, t1_out_residual]
            )

        # ================= TAB 2 =================
        with gr.Tab("Visual Prompting"):
            gr.Markdown("### Isolate sound corresponding to a visual object in a video")
            gr.Markdown("*Note: This uses SAM3 to generate masks for the object described.*")
            with gr.Row():
                with gr.Column():
                    t2_input_video = gr.Video(label="Input Video", format="mp4")
                    t2_visual_prompt = gr.Textbox(label="Visual Object Description", placeholder="e.g., 'The person on the left', 'The red car'")
                    t2_btn = gr.Button("Generate Mask & Separate", variant="primary")
                
                with gr.Column():
                    t2_out_target = gr.Audio(label="Target Audio (From Visual Object)", type="filepath")
                    t2_out_residual = gr.Audio(label="Residual Audio", type="filepath")

            t2_btn.click(
                fn=process_visual_prompting,
                inputs=[t2_input_video, t2_visual_prompt],
                outputs=[t2_out_target, t2_out_residual]
            )

        # ================= TAB 3 =================
        with gr.Tab("Span Prompting"):
            gr.Markdown("### Isolate sound using Temporal Anchors (Time Ranges)")
            gr.Markdown("Define when the sound happens (+) or does not happen (-).")
            with gr.Row():
                with gr.Column():
                    t3_input_audio = gr.Audio(label="Input Audio", type="filepath")
                    t3_desc = gr.Textbox(label="Description (Optional)", placeholder="e.g., 'A horn honking'")
                    
                    t3_anchors = gr.Dataframe(
                        headers=["Type (+/-)", "Start (s)", "End (s)"],
                        datatype=["str", "number", "number"],
                        row_count=3,
                        col_count=(3, "fixed"),
                        label="Temporal Anchors",
                        value=[["+", 0.0, 1.0], ["-", 1.5, 2.0], ["+", 3.0, 4.0]]
                    )
                    t3_btn = gr.Button("Separate with Anchors", variant="primary")
                
                with gr.Column():
                    t3_out_target = gr.Audio(label="Target Audio", type="filepath")
                    t3_out_residual = gr.Audio(label="Residual Audio", type="filepath")

            t3_btn.click(
                fn=process_span_prompting,
                inputs=[t3_input_audio, t3_desc, t3_anchors],
                outputs=[t3_out_target, t3_out_residual]
            )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="teal", secondary_hue="emerald"), css=css, ssr_mode=False)