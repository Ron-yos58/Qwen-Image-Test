import gradio as gr
import torch
import torchaudio
import os
import tempfile
import spaces
from typing import Iterable
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

# ==========================================
# 1. Theme Definition (Orange Red)
# ==========================================
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
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
        )

orange_red_theme = OrangeRedTheme()

# ==========================================
# 2. Model Loading
# ==========================================
try:
    from sam_audio import SAMAudio, SAMAudioProcessor
except ImportError as e:
    print(f"Warning: 'sam_audio' library not found. Error: {e}")

MODEL_ID = "facebook/sam-audio-large"
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

# ==========================================
# 3. Processing Function
# ==========================================
def save_audio(tensor, sample_rate):
    """Saves a tensor to a temporary WAV file and returns path."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tensor = tensor.cpu()
        # torchaudio expects [channels, time]
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        torchaudio.save(tmp.name, tensor, sample_rate)
        return tmp.name

@spaces.GPU(duration=120)
def process_audio(file_path, text_prompt, rerank, progress=gr.Progress()):
    global model, processor

    if model is None or processor is None:
        return None, None, "❌ Model not loaded correctly."

    if not file_path:
        return None, None, "❌ Please upload an audio file."
    if not text_prompt or not text_prompt.strip():
        return None, None, "❌ Please enter a text prompt."

    try:
        progress(0.2, desc="Processing audio...")
        
        # Prepare inputs
        inputs = processor(audios=[file_path], descriptions=[text_prompt.strip()]).to(device)

        progress(0.5, desc="Separating sound...")
        with torch.inference_mode():
            # Run separation
            # Using reranking improves quality but adds latency
            candidates = int(rerank) if rerank else 1
            result = model.separate(inputs, predict_spans=True, reranking_candidates=candidates)

        progress(0.9, desc="Saving results...")
        sr = processor.audio_sampling_rate
        
        # Save Target
        target_path = save_audio(result.target[0], sr)
        
        # Save Residual (Background)
        residual_path = save_audio(result.residual[0], sr)

        progress(1.0, desc="Done!")
        return target_path, residual_path, f"✅ Successfully isolated '{text_prompt}'"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, f"❌ Error: {str(e)}"

# ==========================================
# 4. Gradio Interface
# ==========================================
css = """
#main-title h1 {font-size: 2.4em}
#col-container {max-width: 1000px; margin: 0 auto;}
"""

with gr.Blocks() as demo:
    gr.Markdown("# **SAM-Audio-Demo**", elem_id="main-title")
    gr.Markdown("Segment and isolate specific music/sounds from audio files using natural language descriptions, powered by [SAM-Audio-Large](https://huggingface.co/facebook/sam-audio-large).")

    with gr.Column(elem_id="col-container"):
        with gr.Row():
            # Left Column: Inputs
            with gr.Column(scale=1):
                input_file = gr.Audio(label="Input Audio", type="filepath")
                text_prompt = gr.Textbox(label="Sound to Isolate", placeholder="e.g., 'A man speaking', 'Bird chirping'")
                
                with gr.Accordion("Advanced Settings", open=False):
                    rerank_slider = gr.Slider(
                        minimum=1, maximum=8, value=3, step=1, 
                        label="Reranking Candidates", 
                        info="Higher values improve quality but take longer."
                    )
                
                run_btn = gr.Button("Segment Audio", variant="primary")

            # Right Column: Outputs
            with gr.Column(scale=1):
                output_target = gr.Audio(label="Isolated Sound (Target)", type="filepath")
                output_residual = gr.Audio(label="Background (Residual)", type="filepath")
                status_out = gr.Textbox(label="Status", interactive=False, show_label=True, lines=2)

        # Examples
        gr.Examples(
            examples=[
                ["example_audio/speech.mp3", "Music"],
                ["example_audio/song.mp3", "Drum"],
                ["example_audio/song2.mp3", "Vocals"],
            ],
            inputs=[input_file, text_prompt],
            label="Audio Examples"
        )
    
    run_btn.click(
        fn=process_audio,
        inputs=[input_file, text_prompt, rerank_slider],
        outputs=[output_target, output_residual, status_out]
    )

if __name__ == "__main__":
    demo.launch(theme=orange_red_theme, css=css, mcp_server=True, ssr_mode=False)