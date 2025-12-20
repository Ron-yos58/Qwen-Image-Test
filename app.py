import gradio as gr
import torch
import torchaudio
import numpy as np
import os
import tempfile
import spaces

from typing import Iterable
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

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
        secondary_hue: colors.Color | str = colors.orange_red, # Use the new color
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

try:
    from sam_audio import SAMAudio, SAMAudioProcessor
except ImportError as e:
    print(f"Warning: 'sam_audio' library not found. Please install it to use this app. Error: {e}")

MODEL_ID = "facebook/sam-audio-large"
DEFAULT_CHUNK_DURATION = 30.0
OVERLAP_DURATION = 2.0
MAX_DURATION_WITHOUT_CHUNKING = 30.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading {MODEL_ID} on {device}...")

model = None
processor = None

try:
    model = SAMAudio.from_pretrained(MODEL_ID).to(device).eval()
    processor = SAMAudioProcessor.from_pretrained(MODEL_ID)
    print("✅ SAM-Audio loaded successfully.")
except Exception as e:
    print(f"❌ Error loading SAM-Audio: {e}")

def load_audio(file_path):
    """Load audio from file (supports both audio and video files)."""
    waveform, sample_rate = torchaudio.load(file_path)
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

    if total_samples <= chunk_samples:
        return [waveform]

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

    processed_chunks = []
    for chunk in chunks:
        if chunk.dim() == 1:
            chunk = chunk.unsqueeze(0)
        processed_chunks.append(chunk)

    result = processed_chunks[0]

    for i in range(1, len(processed_chunks)):
        prev_chunk = result
        next_chunk = processed_chunks[i]

        actual_overlap = min(overlap_samples, prev_chunk.shape[1], next_chunk.shape[1])

        if actual_overlap <= 0:
            result = torch.cat([prev_chunk, next_chunk], dim=1)
            continue

        fade_out = torch.linspace(1.0, 0.0, actual_overlap).to(prev_chunk.device)
        fade_in = torch.linspace(0.0, 1.0, actual_overlap).to(next_chunk.device)

        prev_overlap = prev_chunk[:, -actual_overlap:]
        next_overlap = next_chunk[:, :actual_overlap]

        crossfaded = prev_overlap * fade_out + next_overlap * fade_in

        result = torch.cat([
            prev_chunk[:, :-actual_overlap],
            crossfaded,
            next_chunk[:, actual_overlap:]
        ], dim=1)

    return result

def save_audio(tensor, sample_rate):
    """Saves a tensor to a temporary WAV file and returns path."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tensor = tensor.cpu()
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        torchaudio.save(tmp.name, tensor, sample_rate)
        return tmp.name

@spaces.GPU(duration=120)
def process_audio(file_path, text_prompt, chunk_duration_val, progress=gr.Progress()):
    global model, processor

    if model is None or processor is None:
        return None, None, "❌ Model not loaded correctly. Check logs."

    progress(0.05, desc="Checking inputs...")

    if not file_path:
        return None, None, "❌ Please upload an audio or video file."
    if not text_prompt or not text_prompt.strip():
        return None, None, "❌ Please enter a text prompt."

    try:
        progress(0.15, desc="Loading audio...")
        waveform, sample_rate = load_audio(file_path)
        duration = waveform.shape[1] / sample_rate

        c_dur = chunk_duration_val if chunk_duration_val else DEFAULT_CHUNK_DURATION
        use_chunking = duration > MAX_DURATION_WITHOUT_CHUNKING

        if use_chunking:
            progress(0.2, desc=f"Audio is {duration:.1f}s, splitting into chunks...")
            chunks = split_audio_into_chunks(waveform, sample_rate, c_dur, OVERLAP_DURATION)
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

                    target_chunks.append(result.target[0].detach().cpu())
                    residual_chunks.append(result.residual[0].detach().cpu())
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
            return target_path, residual_path, f"✅ Isolated '{text_prompt}' ({num_chunks} chunks)"

        else:
            progress(0.3, desc="Processing audio...")
            inputs = processor(audios=[file_path], descriptions=[text_prompt.strip()]).to(device)

            progress(0.6, desc="Separating sounds...")
            with torch.inference_mode():
                result = model.separate(inputs, predict_spans=False, reranking_candidates=1)

            progress(0.9, desc="Saving results...")
            sr = processor.audio_sampling_rate
            target_path = save_audio(result.target[0].unsqueeze(0).cpu(), sr)
            residual_path = save_audio(result.residual[0].unsqueeze(0).cpu(), sr)

            progress(1.0, desc="Done!")
            return target_path, residual_path, f"✅ Isolated '{text_prompt}'"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, f"❌ Error: {str(e)}"

css = """
#main-title h1 {font-size: 2.4em}
"""

with gr.Blocks() as demo:
    gr.Markdown("# **SAM-Audio-Demo**", elem_id="main-title")
    gr.Markdown("Segment and isolate specific music/sounds from audio files using natural language descriptions, powered by [SAM-Audio-Large](https://huggingface.co/facebook/sam-audio-large).")

    with gr.Column(elem_id="col-container"):
        with gr.Row():
            with gr.Column(scale=1):
                input_file = gr.Audio(label="Input Audio", type="filepath")
                text_prompt = gr.Textbox(label="Sound to Isolate", placeholder="e.g., 'A man speaking', 'Bird chirping'")

                with gr.Accordion("Advanced Settings", open=False):
                    chunk_duration_slider = gr.Slider(
                        minimum=10, maximum=60, value=30, step=5,
                        label="Chunk Duration (seconds)",
                        info="Processing long audio in chunks prevents out-of-memory errors."
                    )

                run_btn = gr.Button("Segment Audio", variant="primary")

            with gr.Column(scale=1):
                output_target = gr.Audio(label="Isolated Sound (Target)", type="filepath")
                output_residual = gr.Audio(label="Background (Residual)", type="filepath")
                status_out = gr.Textbox(label="Status", interactive=False, show_label=True, lines=6)

        gr.Examples(
            examples=[
                ["example_audio/speech.mp3", "Music", 30],
                ["example_audio/song.mp3", "Drum", 30],
                ["example_audio/song2.mp3", "Music", 30],
            ],
            inputs=[input_file, text_prompt, chunk_duration_slider],
            label="Audio Examples"
        )
    
    run_btn.click(
        fn=process_audio,
        inputs=[input_file, text_prompt, chunk_duration_slider],
        outputs=[output_target, output_residual, status_out]
    )

if __name__ == "__main__":
    demo.launch(theme=orange_red_theme, css=css, mcp_server=True, ssr_mode=False)