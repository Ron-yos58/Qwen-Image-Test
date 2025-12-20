import gradio as gr
import torch
import torchaudio
import numpy as np
import os
import tempfile
import spaces

# ---------------------------------------------------------
# Import SAM-Audio
# ---------------------------------------------------------
try:
    from sam_audio import SAMAudio, SAMAudioProcessor
except ImportError as e:
    print(f"Warning: 'sam_audio' library not found. Please install it via git. {e}")

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MODEL_ID = "facebook/sam-audio-large"
DEFAULT_CHUNK_DURATION = 30.0  # Process 30 seconds at a time
OVERLAP_DURATION = 2.0         # 2 seconds overlap for crossfading
MAX_DURATION_WITHOUT_CHUNKING = 45.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# Model Loading
# ---------------------------------------------------------
print(f"Loading {MODEL_ID} on {device}...")
model = None
processor = None

def load_models():
    global model, processor
    if model is None:
        try:
            model = SAMAudio.from_pretrained(MODEL_ID).to(device).eval()
            processor = SAMAudioProcessor.from_pretrained(MODEL_ID)
            print("✅ SAM-Audio loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading SAM-Audio: {e}")

# Load model on startup
load_models()

# ---------------------------------------------------------
# Audio Processing Helpers (Chunking & Merging)
# ---------------------------------------------------------
def load_audio(file_path):
    """Load audio from file."""
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
# Main Processing Logic
# ---------------------------------------------------------
@spaces.GPU
def process_text_prompting(audio_file, description, progress=gr.Progress()):
    if not audio_file:
        return None, None, "❌ Please upload an audio file."
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
# Gradio Interface
# ---------------------------------------------------------
css = """
#main-title h1 {font-size: 2.4em}
"""

with gr.Blocks() as demo:
    
    gr.Markdown("# **SAM-Audio** 🔊", elem_id="main-title")
    gr.Markdown("Segment and isolate specific sounds from audio using text descriptions.")

    with gr.Row():
        with gr.Column(scale=1):
            t1_input = gr.Audio(label="Input Audio", type="filepath", height=250)
            t1_desc = gr.Textbox(label="Text Description", placeholder="e.g., 'A man speaking', 'Glass breaking', 'Bird chirping'")
            t1_btn = gr.Button("Isolate Sound", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            t1_status = gr.Textbox(label="Status", interactive=False)
            t1_out_target = gr.Audio(label="Target Audio (Isolated)", type="filepath")
            t1_out_residual = gr.Audio(label="Residual Audio (Background)", type="filepath")
    
    # Examples
    gr.Examples(
        examples=[
            ["example_audio/dog_bark.wav", "Dog barking"],
            ["example_audio/street.wav", "Car horn"],
        ],
        inputs=[t1_input, t1_desc],
        label="Examples"
    )

    t1_btn.click(
        fn=process_text_prompting,
        inputs=[t1_input, t1_desc],
        outputs=[t1_out_target, t1_out_residual, t1_status]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="indigo",
            neutral_hue="slate",
        ), css=css, mcp_server=True, ssr_mode=False)