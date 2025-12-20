import gradio as gr
import torch
import torchaudio
import numpy as np
import os
import tempfile
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
# Constants & Configuration
# ---------------------------------------------------------
MODEL_ID = "facebook/sam-audio-large"
DEFAULT_CHUNK_DURATION = 30.0  # Process 30 seconds at a time
OVERLAP_DURATION = 2.0         # 2 seconds overlap for crossfading
MAX_DURATION_WITHOUT_CHUNKING = 45.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# Global Model Loading
# ---------------------------------------------------------
print(f"Loading {MODEL_ID} on {device}...")
model = None
processor = None
video_predictor = None

def load_models():
    global model, processor
    if model is None:
        try:
            model = SAMAudio.from_pretrained(MODEL_ID).to(device).eval()
            processor = SAMAudioProcessor.from_pretrained(MODEL_ID)
            print("✅ SAM-Audio loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading SAM-Audio: {e}")

def get_sam3_predictor():
    global video_predictor
    if video_predictor is None:
        print("⏳ Loading SAM3 Video Predictor...")
        video_predictor = build_sam3_video_predictor()
        print("✅ SAM3 loaded.")
    return video_predictor

# Load audio model on startup
load_models()

# ---------------------------------------------------------
# Robust Audio Helpers (Chunking & Merging)
# ---------------------------------------------------------
def load_audio(file_path):
    """Load audio from file (supports both audio and video files)."""
    waveform, sample_rate = torchaudio.load(file_path)
    # Convert to mono if stereo for consistency
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
        
        # Concatenate
        result = torch.cat([
            prev_chunk[:, :-actual_overlap],
            crossfaded,
            next_chunk[:, actual_overlap:]
        ], dim=1)
    
    return result

def save_audio(tensor, sample_rate):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        torchaudio.save(tmp.name, tensor, sample_rate)
        return tmp.name

# ---------------------------------------------------------
# Tab 1: Text Prompting (With Robust Chunking)
# ---------------------------------------------------------
@spaces.GPU(duration=120)
def process_text_prompting(audio_file, description, progress=gr.Progress()):
    if not audio_file:
        return None, None, "❌ Please upload an audio or video file."
    if not description or not description.strip():
        return None, None, "❌ Please enter a text prompt."

    load_models() # Ensure model is loaded
    
    try:
        progress(0.1, desc="Loading audio...")
        waveform, sample_rate = load_audio(audio_file)
        duration = waveform.shape[1] / sample_rate
        
        # Decide whether to use chunking
        use_chunking = duration > MAX_DURATION_WITHOUT_CHUNKING
        
        if use_chunking:
            progress(0.2, desc=f"Audio is {duration:.1f}s, splitting into chunks...")
            chunks = split_audio_into_chunks(waveform, sample_rate, DEFAULT_CHUNK_DURATION, OVERLAP_DURATION)
            num_chunks = len(chunks)
            
            target_chunks = []
            residual_chunks = []
            
            for i, chunk in enumerate(chunks):
                chunk_progress = 0.2 + (i / num_chunks) * 0.6
                progress(chunk_progress, desc=f"Processing chunk {i+1}/{num_chunks}...")
                
                # Save chunk to temp file for processor
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    torchaudio.save(tmp.name, chunk, sample_rate)
                    chunk_path = tmp.name
                
                try:
                    inputs = processor(audios=[chunk_path], descriptions=[description.strip()]).to(device)
                    with torch.inference_mode():
                        result = model.separate(inputs, predict_spans=True, reranking_candidates=1)
                    
                    target_chunks.append(result.target[0].cpu())
                    residual_chunks.append(result.residual[0].cpu())
                finally:
                    if os.path.exists(chunk_path):
                        os.unlink(chunk_path)
            
            progress(0.85, desc="Merging chunks...")
            target_merged = merge_chunks_with_crossfade(target_chunks, sample_rate, OVERLAP_DURATION)
            residual_merged = merge_chunks_with_crossfade(residual_chunks, sample_rate, OVERLAP_DURATION)
            
            progress(0.95, desc="Saving results...")
            target_path = save_audio(target_merged, sample_rate)
            residual_path = save_audio(residual_merged, sample_rate)
            
            progress(1.0, desc="Done!")
            return target_path, residual_path, f"✅ Isolated '{description}' ({num_chunks} chunks processed)."
        
        else:
            # Process short audio directly
            progress(0.3, desc="Processing audio...")
            inputs = processor(audios=[audio_file], descriptions=[description.strip()]).to(device)
            
            progress(0.6, desc="Separating sounds...")
            with torch.inference_mode():
                result = model.separate(inputs, predict_spans=True)
            
            progress(0.9, desc="Saving results...")
            sr = processor.audio_sampling_rate
            target_path = save_audio(result.target[0].unsqueeze(0).cpu(), sr)
            residual_path = save_audio(result.residual[0].unsqueeze(0).cpu(), sr)
            
            return target_path, residual_path, f"✅ Isolated '{description}' successfully."

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, f"❌ Error: {str(e)}"

# ---------------------------------------------------------
# Tab 2: Visual Prompting (Standard Logic)
# ---------------------------------------------------------
@spaces.GPU(duration=180)
def process_visual_prompting(video_file, visual_prompt_text, progress=gr.Progress()):
    if video_file is None:
        return None, None, "❌ Please upload a video."
    if not visual_prompt_text:
        return None, None, "❌ Please describe the visual object."

    load_models()
    
    try:
        progress(0.1, desc="Initializing Video Decoder...")
        decoder = VideoDecoder(video_file)
        
        # Check duration warning
        if len(decoder) > 300: # Approx 10 seconds at 30fps
             print("Warning: Video is long. Visual prompting consumes significant VRAM.")

        frames = decoder[:] # Get all frames
        
        progress(0.2, desc="Generating Masks with SAM3...")
        predictor = get_sam3_predictor()
        
        # Start SAM3 Session
        response = predictor.handle_request({
            "type": "start_session",
            "resource_path": video_file,
        })
        session_id = response["session_id"]
        
        masks = []
        # Generate masks for every frame
        # (Optimization: SAM3 can track, but for simplicity/robustness we prompt per frame or use internal tracking if implemented in `video_predictor` properly. 
        # Here we follow standard snippet logic)
        total_frames = len(decoder)
        for frame_index in range(total_frames):
            if frame_index % 10 == 0:
                progress(0.2 + (frame_index / total_frames) * 0.3, desc=f"Masking frame {frame_index}/{total_frames}...")
            
            response = predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": frame_index,
                "text": visual_prompt_text,
            })
            mask = response["outputs"]["out_binary_masks"]
            
            if mask.shape[0] == 0:
                mask = np.zeros_like(frames[0, [0]], dtype=bool)
            masks.append(mask[:1]) # Take first mask

        # Prepare mask tensor
        final_mask = torch.from_numpy(np.concatenate(masks)).unsqueeze(1)
        
        progress(0.6, desc="Separating Audio with SAM-Audio...")
        # Note: We don't chunk here because visual masks + audio chunking alignment is complex.
        # We process the file as a single block.
        inputs = processor(
            audios=[video_file],
            descriptions=[""], # Empty text description, relying on mask
            masked_videos=processor.mask_videos([frames], [final_mask]),
        ).to(device)

        with torch.inference_mode():
            result = model.separate(inputs)

        progress(0.9, desc="Saving results...")
        sr = processor.audio_sampling_rate
        target_path = save_audio(result.target[0].unsqueeze(0).cpu(), sr)
        residual_path = save_audio(result.residual[0].unsqueeze(0).cpu(), sr)

        return target_path, residual_path, f"✅ Isolated sound for visual object: '{visual_prompt_text}'"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, f"❌ Error: {str(e)}"

# ---------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# **SAM-Audio** 🔊", elem_id="main-title")
    gr.Markdown("Segment and isolate sounds using **Text Descriptions** or **Visual Cues**.")

    with gr.Tabs():
        # ================= TAB 1: Text =================
        with gr.Tab("Text Prompting"):
            gr.Markdown("### Isolate sound using a text description")
            with gr.Row():
                with gr.Column():
                    t1_input = gr.Audio(label="Input Audio/Video", type="filepath")
                    t1_desc = gr.Textbox(label="Description", placeholder="e.g., 'A man speaking', 'Glass breaking'")
                    t1_btn = gr.Button("Separate Audio", variant="primary")
                
                with gr.Column():
                    t1_status = gr.Textbox(label="Status", interactive=False)
                    t1_out_target = gr.Audio(label="Target Audio (Isolated)", type="filepath")
                    t1_out_residual = gr.Audio(label="Residual Audio (Background)", type="filepath")
            
            t1_btn.click(
                fn=process_text_prompting,
                inputs=[t1_input, t1_desc],
                outputs=[t1_out_target, t1_out_residual, t1_status]
            )

        # ================= TAB 2: Visual =================
        with gr.Tab("Visual Prompting"):
            gr.Markdown("### Isolate sound corresponding to a visual object")
            gr.Markdown("**Note:** Visual prompting uses SAM3 to generate masks. Processing long videos may be slow.")
            with gr.Row():
                with gr.Column():
                    t2_input = gr.Video(label="Input Video", format="mp4")
                    t2_prompt = gr.Textbox(label="Visual Object Description", placeholder="e.g., 'The person on the left', 'The red car'")
                    t2_btn = gr.Button("Generate Mask & Separate", variant="primary")
                
                with gr.Column():
                    t2_status = gr.Textbox(label="Status", interactive=False)
                    t2_out_target = gr.Audio(label="Target Audio", type="filepath")
                    t2_out_residual = gr.Audio(label="Residual Audio", type="filepath")

            t2_btn.click(
                fn=process_visual_prompting,
                inputs=[t2_input, t2_prompt],
                outputs=[t2_out_target, t2_out_residual, t2_status]
            )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="indigo",
            neutral_hue="slate",
        ), mcp_server=True, ssr_mode=False)