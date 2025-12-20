import gradio as gr
import torch
import torchaudio
import numpy as np
import os
import tempfile
import spaces
import pandas as pd
from typing import Optional

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
    SAMAudio = None
    SAMAudioProcessor = None
    build_sam3_video_predictor = None

# ---------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------
MODEL_ID = "facebook/sam-audio-large"
DEFAULT_CHUNK_DURATION = 30.0  # seconds
OVERLAP_DURATION = 2.0         # seconds
MAX_DURATION_WITHOUT_CHUNKING = 30.0 # seconds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# Direct Model Loading (No Lazy Loading)
# ---------------------------------------------------------
print(f"----------------------------------------------------------------")
print(f"Initializing Models on {device}...")
print(f"----------------------------------------------------------------")

# 1. Load SAM-Audio
print(f"Loading {MODEL_ID}...")
try:
    model = SAMAudio.from_pretrained(MODEL_ID).to(device).eval()
    processor = SAMAudioProcessor.from_pretrained(MODEL_ID)
    print("✅ SAM-Audio loaded successfully.")
except Exception as e:
    print(f"❌ Error loading SAM-Audio: {e}")
    model = None
    processor = None

# 2. Load SAM3 (Visual Predictor)
print("Loading SAM3 Video Predictor...")
try:
    video_predictor = build_sam3_video_predictor()
    print("✅ SAM3 loaded successfully.")
except Exception as e:
    print(f"❌ Error loading SAM3: {e}")
    video_predictor = None

print(f"----------------------------------------------------------------")
print(f"All models initialized.")
print(f"----------------------------------------------------------------")

# ---------------------------------------------------------
# Audio Processing Helpers (Chunking & Merging)
# ---------------------------------------------------------
def load_audio(file_path):
    """Load audio from file (supports both audio and video files)."""
    waveform, sample_rate = torchaudio.load(file_path)
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sample_rate

def split_audio_into_chunks(waveform, sample_rate, chunk_duration, overlap_duration):
    """Split audio waveform into overlapping chunks."""
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap_duration * sample_rate)
    stride = chunk_samples - overlap_samples
    
    chunks = []
    total_samples = waveform.shape[1]
    
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunk = waveform[:, start:end]
        chunks.append(chunk)
        
        if end >= total_samples:
            break
        start += stride
    
    return chunks

def merge_chunks_with_crossfade(chunks, sample_rate, overlap_duration):
    """Merge audio chunks with crossfade on overlapping regions."""
    if len(chunks) == 1:
        chunk = chunks[0]
        # Ensure 2D tensor
        if chunk.dim() == 1:
            chunk = chunk.unsqueeze(0)
        return chunk
    
    overlap_samples = int(overlap_duration * sample_rate)
    
    # Ensure all chunks are 2D [channels, samples]
    processed_chunks = []
    for chunk in chunks:
        if chunk.dim() == 1:
            chunk = chunk.unsqueeze(0)
        processed_chunks.append(chunk)
    
    result = processed_chunks[0]
    
    for i in range(1, len(processed_chunks)):
        prev_chunk = result
        next_chunk = processed_chunks[i]
        
        # Handle case where chunks are shorter than overlap
        actual_overlap = min(overlap_samples, prev_chunk.shape[1], next_chunk.shape[1])
        
        if actual_overlap <= 0:
            # No overlap possible, just concatenate
            result = torch.cat([prev_chunk, next_chunk], dim=1)
            continue
        
        # Create fade curves
        fade_out = torch.linspace(1.0, 0.0, actual_overlap).to(prev_chunk.device)
        fade_in = torch.linspace(0.0, 1.0, actual_overlap).to(next_chunk.device)
        
        # Get overlapping regions
        prev_overlap = prev_chunk[:, -actual_overlap:]
        next_overlap = next_chunk[:, :actual_overlap]
        
        # Crossfade mix
        crossfaded = prev_overlap * fade_out + next_overlap * fade_in
        
        # Concatenate: non-overlap of prev + crossfaded + non-overlap of next
        result = torch.cat([
            prev_chunk[:, :-actual_overlap],
            crossfaded,
            next_chunk[:, actual_overlap:]
        ], dim=1)
    
    return result

def save_audio_temp(tensor, sample_rate):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        torchaudio.save(tmp.name, tensor, sample_rate)
        return tmp.name

# ---------------------------------------------------------
# Tab 1: Text Prompting (with Chunking)
# ---------------------------------------------------------
@spaces.GPU(duration=120)
def process_text_prompting(file_path, text_prompt, chunk_duration=DEFAULT_CHUNK_DURATION, progress=gr.Progress()):
    global model, processor
    
    progress(0.05, desc="Checking inputs...")
    
    if model is None or processor is None:
        return None, None, "❌ Model failed to load at startup. Check logs."

    if not file_path:
        return None, None, "❌ Please upload an audio or video file."
    if not text_prompt or not text_prompt.strip():
        return None, None, "❌ Please enter a text prompt."
    
    try:
        progress(0.15, desc="Loading audio...")
        waveform, sample_rate = load_audio(file_path)
        duration = waveform.shape[1] / sample_rate
        
        # Decide whether to use chunking
        use_chunking = duration > MAX_DURATION_WITHOUT_CHUNKING
        
        if use_chunking:
            progress(0.2, desc=f"Audio is {duration:.1f}s, splitting into chunks...")
            chunks = split_audio_into_chunks(waveform, sample_rate, chunk_duration, OVERLAP_DURATION)
            num_chunks = len(chunks)
            
            target_chunks = []
            residual_chunks = []
            
            for i, chunk in enumerate(chunks):
                chunk_progress = 0.2 + (i / num_chunks) * 0.6
                progress(chunk_progress, desc=f"Processing chunk {i+1}/{num_chunks}...")
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    torchaudio.save(tmp.name, chunk, sample_rate)
                    chunk_path = tmp.name
                
                try:
                    inputs = processor(audios=[chunk_path], descriptions=[text_prompt.strip()]).to(device)
                    with torch.inference_mode():
                        result = model.separate(inputs, predict_spans=False, reranking_candidates=1)
                    target_chunks.append(result.target[0].cpu())
                    residual_chunks.append(result.residual[0].cpu())
                finally:
                    if os.path.exists(chunk_path):
                        os.unlink(chunk_path)
            
            progress(0.85, desc="Merging chunks...")
            target_merged = merge_chunks_with_crossfade(target_chunks, sample_rate, OVERLAP_DURATION)
            residual_merged = merge_chunks_with_crossfade(residual_chunks, sample_rate, OVERLAP_DURATION)
            
            progress(0.95, desc="Saving results...")
            target_path = save_audio_temp(target_merged, sample_rate)
            residual_path = save_audio_temp(residual_merged, sample_rate)
            
            return target_path, residual_path, f"✅ Isolated '{text_prompt}' ({num_chunks} chunks)"
        else:
            # Process without chunking
            progress(0.3, desc="Processing audio...")
            inputs = processor(audios=[file_path], descriptions=[text_prompt.strip()]).to(device)
            
            progress(0.6, desc="Separating sounds...")
            with torch.inference_mode():
                result = model.separate(inputs, predict_spans=False, reranking_candidates=1)
            
            progress(0.9, desc="Saving results...")
            sr = processor.audio_sampling_rate
            target_path = save_audio_temp(result.target[0].unsqueeze(0).cpu(), sr)
            residual_path = save_audio_temp(result.residual[0].unsqueeze(0).cpu(), sr)
            
            return target_path, residual_path, f"✅ Isolated '{text_prompt}'"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, f"❌ Error: {str(e)}"

# ---------------------------------------------------------
# Tab 2: Visual Prompting
# ---------------------------------------------------------
@spaces.GPU(duration=180)
def process_visual_prompting(video_file, visual_prompt_text, progress=gr.Progress()):
    global model, processor, video_predictor

    if model is None or video_predictor is None:
         return None, None, "❌ Models failed to load at startup. Check logs."

    if video_file is None or not visual_prompt_text:
        return None, None, "❌ Please provide both a video and a description."
    
    try:
        progress(0.1, desc="Initializing Video Decoder...")
        decoder = VideoDecoder(video_file)
        frames = decoder[:]
        
        progress(0.2, desc="Starting SAM3 Session...")
        response = video_predictor.handle_request({
            "type": "start_session",
            "resource_path": video_file,
        })
        session_id = response["session_id"]

        progress(0.3, desc=f"Generating masks for {len(decoder)} frames...")
        masks = []
        step = 1 
        total_frames = len(decoder)
        
        for frame_index in range(0, total_frames, step):
            if frame_index % 10 == 0:
                progress(0.3 + (frame_index/total_frames)*0.4, desc=f"Masking frame {frame_index}/{total_frames}")
            
            response = video_predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": frame_index,
                "text": visual_prompt_text,
            })
            mask = response["outputs"]["out_binary_masks"]
            
            if mask.shape[0] == 0:
                mask = np.zeros_like(frames[0, [0]], dtype=bool)
            masks.append(mask[:1]) 

        final_mask = torch.from_numpy(np.concatenate(masks)).unsqueeze(1)

        progress(0.8, desc="Separating Audio with SAM-Audio...")
        inputs = processor(
            audios=[video_file],
            descriptions=[""], 
            masked_videos=processor.mask_videos([frames], [final_mask]),
        ).to(device)

        with torch.inference_mode():
            result = model.separate(inputs)

        progress(0.95, desc="Saving results...")
        sr = processor.audio_sampling_rate
        target_path = save_audio_temp(result.target[0].unsqueeze(0).cpu(), sr)
        residual_path = save_audio_temp(result.residual[0].unsqueeze(0).cpu(), sr)

        return target_path, residual_path, f"✅ Isolated object: '{visual_prompt_text}'"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, f"❌ Error: {str(e)}"

# ---------------------------------------------------------
# Tab 3: Span Prompting (Temporal Anchors)
# ---------------------------------------------------------
@spaces.GPU(duration=120)
def process_span_prompting(audio_file, description, anchors_df, progress=gr.Progress()):
    global model, processor
    
    if model is None:
        return None, None, "❌ Model not loaded."
    if audio_file is None:
        return None, None, "❌ Please upload an audio file."
    
    try:
        progress(0.1, desc="Parsing anchors...")
        # Parse Dataframe to List: [["+", 6.3, 7.0], ...]
        formatted_anchors = []
        
        # Validating anchors_df (it usually comes as a list of lists from Gradio Dataframe)
        if anchors_df is not None:
            # Handle standard list of lists
            if isinstance(anchors_df, list):
                iterable = anchors_df
            # Handle pandas DataFrame if Gradio returns that (depends on config)
            elif hasattr(anchors_df, "values"):
                iterable = anchors_df.values.tolist()
            else:
                iterable = []

            for row in iterable:
                # row structure: [Type, Start, End]
                # Ensure we have data
                if len(row) >= 3 and row[0] and row[1] is not None and row[2] is not None:
                    # Validate Type
                    anc_type = str(row[0]).strip()
                    if anc_type not in ["+", "-"]:
                        continue # Skip invalid rows
                    
                    try:
                        start_t = float(row[1])
                        end_t = float(row[2])
                        formatted_anchors.append([anc_type, start_t, end_t])
                    except ValueError:
                        continue # Skip if numbers aren't valid
        
        if not formatted_anchors:
            return None, None, "❌ No valid anchors provided. Use '+' or '-' and valid time numbers."
        
        print(f"Anchors: {formatted_anchors}")

        progress(0.3, desc="Processing with Anchors...")
        
        # Note: We do NOT use chunking here because anchors rely on absolute timestamps 
        # of the original file.
        inputs = processor(
            audios=[audio_file],
            descriptions=[description if description else "Sound"], # Desc can be generic if anchors are strong
            anchors=[formatted_anchors],
        ).to(device)

        with torch.inference_mode():
            result = model.separate(inputs)

        progress(0.9, desc="Saving results...")
        sr = processor.audio_sampling_rate
        target_path = save_audio_temp(result.target[0].unsqueeze(0).cpu(), sr)
        residual_path = save_audio_temp(result.residual[0].unsqueeze(0).cpu(), sr)

        return target_path, residual_path, f"✅ Processed with {len(formatted_anchors)} anchors."

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, f"❌ Error: {str(e)}"

# ---------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------
css = """
#main-title h1 {font-size: 2.4em !important; text-align: center;}
.gradio-container {max-width: 1200px !important; margin: auto;}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# **SAM-Audio** 🔊", elem_id="main-title")
    gr.Markdown("Segment and isolate sounds using **Text**, **Visual**, or **Time-based** prompts.")

    with gr.Tabs():
        # ================= TAB 1: TEXT =================
        with gr.Tab("Text Prompting"):
            gr.Markdown("### Isolate sound using a text description")
            
            with gr.Row():
                with gr.Column():
                    t1_input = gr.Audio(label="Input Audio/Video", type="filepath")
                    t1_desc = gr.Textbox(label="Description", placeholder="e.g., 'A man speaking', 'Glass breaking'")
                    t1_chunk = gr.Slider(minimum=10, maximum=60, value=30, step=5, label="Chunk Duration (s)", info="Split long audio into chunks.")
                    t1_btn = gr.Button("Separate Audio", variant="primary")
                
                with gr.Column():
                    t1_status = gr.Textbox(label="Status", interactive=False)
                    t1_target = gr.Audio(label="Target Audio", type="filepath")
                    t1_residual = gr.Audio(label="Residual Audio", type="filepath")
            
            t1_btn.click(
                fn=process_text_prompting,
                inputs=[t1_input, t1_desc, t1_chunk],
                outputs=[t1_target, t1_residual, t1_status]
            )

        # ================= TAB 2: VISUAL =================
        with gr.Tab("Visual Prompting"):
            gr.Markdown("### Isolate sound corresponding to a visual object")
            
            with gr.Row():
                with gr.Column():
                    t2_input = gr.Video(label="Input Video", format="mp4")
                    t2_visual_desc = gr.Textbox(label="Visual Object Description", placeholder="e.g., 'The person on the left'")
                    t2_btn = gr.Button("Generate Mask & Separate", variant="primary")
                
                with gr.Column():
                    t2_status = gr.Textbox(label="Status", interactive=False)
                    t2_target = gr.Audio(label="Target Audio", type="filepath")
                    t2_residual = gr.Audio(label="Residual Audio", type="filepath")

            t2_btn.click(
                fn=process_visual_prompting,
                inputs=[t2_input, t2_visual_desc],
                outputs=[t2_target, t2_residual, t2_status]
            )

        # ================= TAB 3: SPAN =================
        with gr.Tab("Span Prompting"):
            gr.Markdown("### Isolate sound using Temporal Anchors (Time Ranges)")
            gr.Markdown("Specify when a sound **IS (+)** or **IS NOT (-)** present.")
            
            with gr.Row():
                with gr.Column():
                    t3_input = gr.Audio(label="Input Audio", type="filepath")
                    t3_desc = gr.Textbox(label="Description (Optional)", placeholder="e.g. 'A horn honking'")
                    
                    t3_anchors = gr.Dataframe(
                        headers=["Type (+/-)", "Start (s)", "End (s)"],
                        datatype=["str", "number", "number"],
                        row_count=3,
                        col_count=(3, "fixed"),
                        label="Temporal Anchors",
                        value=[["+", 0.0, 5.0], ["-", 5.0, 10.0], ["", None, None]]
                    )
                    t3_btn = gr.Button("Separate with Anchors", variant="primary")
                
                with gr.Column():
                    t3_status = gr.Textbox(label="Status", interactive=False)
                    t3_target = gr.Audio(label="Target Audio", type="filepath")
                    t3_residual = gr.Audio(label="Residual Audio", type="filepath")

            t3_btn.click(
                fn=process_span_prompting,
                inputs=[t3_input, t3_desc, t3_anchors],
                outputs=[t3_target, t3_residual, t3_status]
            )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="indigo",
            neutral_hue="slate",
        ), css=css, mcp_server=True, ssr_mode=False)