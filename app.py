import gradio as gr
import numpy as np
import random
import torch
import spaces
from PIL import Image

# --- Imports ---
from diffusers import FlowMatchEulerDiscreteScheduler
try:
    from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
    from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
    from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3
except ImportError:
    # Fallback/Instruction if custom packages are missing
    raise ImportError("Please ensure the 'qwenimage' package is installed.")

MAX_SEED = np.iinfo(np.int32).max

# --- Configuration & Model Loading ---
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load the Pipeline
pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    transformer=QwenImageTransformer2DModel.from_pretrained(
        "prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19",
        torch_dtype=dtype,
        device_map='cuda'
    ),
    torch_dtype=dtype
).to(device)

# 2. Set Flash Attention 3 (if available)
try:
    pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
    print("Flash Attention 3 Processor set successfully.")
except Exception as e:
    print(f"Warning: Could not set FA3 processor: {e}")

# 3. Adapter Specs (Lighting LoRA)
ADAPTER_SPECS = {
    "Multi-Angle-Lighting": {
        "repo": "dx8152/Qwen-Edit-2509-Multi-Angle-Lighting",
        "weights": "多角度灯光-251116.safetensors",
        "adapter_name": "multi-angle-lighting"
    }
}

# Global state to track currently loaded adapter
CURRENT_LOADED_ADAPTER = None

# --- Logic: Mappings & Prompt Building ---

# Lighting mappings for Azimuth (Horizontal)
# 0 = Front, moving clockwise
LIGHTING_AZIMUTH_MAP = {
    0: "Light source from the Front",
    45: "Light source from the Right Front",
    90: "Light source from the Right",
    135: "Light source from the Right Rear",
    180: "Light source from the Rear",
    225: "Light source from the Left Rear",
    270: "Light source from the Left",
    315: "Light source from the Left Front"
}

def snap_to_nearest_key(value, keys):
    """Finds the nearest key in a list of numbers."""
    return min(keys, key=lambda x: abs(x - value))

def build_lighting_prompt(azimuth: float, elevation: float) -> str:
    """
    Constructs the specific text prompt required by the LoRA.
    Logic:
    1. Prioritize Vertical Extremes (>60° or <-60°)
    2. Fallback to Horizontal Azimuth mappings
    """
    # 1. Vertical Extremes
    if elevation >= 60:
        return "Light source from Above"
    if elevation <= -60:
        return "Light source from Below"
        
    # 2. Horizontal Snap
    keys = list(LIGHTING_AZIMUTH_MAP.keys())
    # Handle the 360 wrap-around for "Front" (0 vs 360)
    # If azimuth is > 337.5, it snaps to 0
    if azimuth > 337.5:
        azimuth = 0
        
    azimuth_snapped = snap_to_nearest_key(azimuth, keys)
    return LIGHTING_AZIMUTH_MAP[azimuth_snapped]

# --- Inference Function ---

@spaces.GPU
def infer_lighting_edit(
    image: Image.Image,
    azimuth: float = 0.0,
    elevation: float = 0.0,
    seed: int = 0,
    randomize_seed: bool = True,
    guidance_scale: float = 5.0,
    num_inference_steps: int = 4,
    height: int = 1024,
    width: int = 1024,
):
    global CURRENT_LOADED_ADAPTER
    
    # 1. Lazy Load Adapter
    spec = ADAPTER_SPECS["Multi-Angle-Lighting"]
    if CURRENT_LOADED_ADAPTER != spec["adapter_name"]:
        print(f"⚙️ Lazy loading adapter: {spec['adapter_name']}...")
        pipe.load_lora_weights(
            spec["repo"],
            weight_name=spec["weights"],
            adapter_name=spec["adapter_name"]
        )
        pipe.set_adapters([spec["adapter_name"]], adapter_weights=[1.0])
        CURRENT_LOADED_ADAPTER = spec["adapter_name"]
    
    # 2. Build Prompt
    prompt = build_lighting_prompt(azimuth, elevation)
    print(f"💡 Generated Prompt: {prompt}")

    # 3. Prepare Inputs
    if image is None:
        raise gr.Error("Please upload an image first.")

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device).manual_seed(seed)

    pil_image = image.convert("RGB")

    # 4. Run Inference
    result = pipe(
        image=[pil_image],
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        generator=generator,
        guidance_scale=guidance_scale,
        num_images_per_prompt=1,
    ).images[0]

    return result, seed, prompt

def update_dimensions_on_upload(image):
    """Resizes image to nearest multiple of 8, max 1024, preserving aspect ratio."""
    if image is None:
        return 1024, 1024
    w, h = image.size
    
    # Constraint: Max dimension 1024
    if w > h:
        new_w = 1024
        new_h = int(new_w * (h / w))
    else:
        new_h = 1024
        new_w = int(new_h * (w / h))
        
    # Constraint: Multiple of 8
    new_w = (new_w // 8) * 8
    new_h = (new_h // 8) * 8
    
    return new_w, new_h

# --- Enhanced 3D Component ---

class LightControl3D(gr.HTML):
    """
    Advanced 3D Light Controller using Three.js.
    Features: Hemisphere guide, Beam visualization, Dynamic color feedback.
    """
    def __init__(self, value=None, imageUrl=None, **kwargs):
        if value is None: value = {"azimuth": 0, "elevation": 0}
        
        # HTML Container
        html_template = """
        <div id="light-control-wrapper" style="width: 100%; height: 500px; position: relative; background: radial-gradient(circle at center, #1a1a1a 0%, #000000 100%); border-radius: 12px; overflow: hidden; border: 1px solid #333; box-shadow: inset 0 0 20px #000;">
            <div id="prompt-badge" style="position: absolute; top: 15px; left: 50%; transform: translateX(-50%); 
                 background: rgba(0,0,0,0.8); border: 1px solid #FFD700; color: #FFD700; 
                 padding: 8px 24px; border-radius: 30px; font-family: monospace; font-weight: bold; font-size: 14px; 
                 z-index: 10; pointer-events: none; transition: all 0.2s ease;">
                 Light Source: Front
            </div>
            
            <div style="position: absolute; bottom: 15px; right: 15px; color: #555; font-size: 10px; font-family: sans-serif; pointer-events: none;">
                Drag to rotate • Scroll to zoom
            </div>
        </div>
        """
        
        # JavaScript Logic
        js_on_load = """
        (() => {
            const wrapper = element.querySelector('#light-control-wrapper');
            const badge = element.querySelector('#prompt-badge');
            
            const initScene = () => {
                if (typeof THREE === 'undefined') { setTimeout(initScene, 100); return; }
                
                // --- 1. Scene & Camera ---
                const scene = new THREE.Scene();
                // No background color set here, letting CSS gradient show through
                
                const camera = new THREE.PerspectiveCamera(45, wrapper.clientWidth / wrapper.clientHeight, 0.1, 1000);
                camera.position.set(4, 3, 4); // Isometric-ish view
                camera.lookAt(0, 0.5, 0);
                
                const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
                renderer.setSize(wrapper.clientWidth, wrapper.clientHeight);
                renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                wrapper.appendChild(renderer.domElement);
                
                // --- 2. Helpers (Grid & Dome) ---
                const CENTER = new THREE.Vector3(0, 0.75, 0);
                const RADIUS = 2.5;
                
                // Floor Grid
                const grid = new THREE.GridHelper(6, 12, 0x444444, 0x111111);
                scene.add(grid);
                
                // Hemisphere Guide (Wireframe Dome)
                const domeGeo = new THREE.SphereGeometry(RADIUS, 16, 8, 0, Math.PI * 2, 0, Math.PI * 0.5);
                const domeMat = new THREE.MeshBasicMaterial({ color: 0x333333, wireframe: true, transparent: true, opacity: 0.15 });
                const dome = new THREE.Mesh(domeGeo, domeMat);
                dome.position.y = CENTER.y - 0.75; // Ground the dome
                scene.add(dome);
                
                // Elevation Rings (Visual guides for 0, 45, 60 degrees)
                const ringMat = new THREE.MeshBasicMaterial({ color: 0x555555, transparent: true, opacity: 0.3, side: THREE.DoubleSide });
                const eqRing = new THREE.Mesh(new THREE.TorusGeometry(RADIUS, 0.01, 8, 64), ringMat);
                eqRing.rotation.x = Math.PI / 2;
                eqRing.position.y = CENTER.y;
                scene.add(eqRing);

                // --- 3. The Subject (Image Plane) ---
                let planeMesh;
                const planeMat = new THREE.MeshBasicMaterial({ color: 0x222222, side: THREE.DoubleSide });
                
                function createPlane(width=1.2, height=1.2) {
                    if(planeMesh) scene.remove(planeMesh);
                    planeMesh = new THREE.Mesh(new THREE.PlaneGeometry(width, height), planeMat);
                    planeMesh.position.copy(CENTER);
                    planeMesh.lookAt(camera.position); // Billboarding slightly? No, fixed upright.
                    planeMesh.rotation.set(0,0,0); // Reset rotation
                    scene.add(planeMesh);
                }
                createPlane();

                // Texture Loader
                function updateTexture(url) {
                    if (!url) {
                        planeMat.map = null; 
                        planeMat.needsUpdate = true; 
                        return;
                    }
                    new THREE.TextureLoader().load(url, (tex) => {
                        planeMat.map = tex;
                        planeMat.needsUpdate = true;
                        // Adjust Aspect Ratio
                        const img = tex.image;
                        if(img && img.width && img.height) {
                            const aspect = img.width / img.height;
                            const size = 1.4; // Max dimension
                            if (aspect > 1) createPlane(size, size/aspect);
                            else createPlane(size*aspect, size);
                        }
                    });
                }
                if (props.imageUrl) updateTexture(props.imageUrl);

                // --- 4. The Light Gizmo (Interactive) ---
                const lightGroup = new THREE.Group();
                scene.add(lightGroup);
                
                // The Orb
                const orb = new THREE.Mesh(
                    new THREE.SphereGeometry(0.2, 32, 32), 
                    new THREE.MeshBasicMaterial({ color: 0xFFD700 })
                );
                
                // The Glow
                const glow = new THREE.Mesh(
                    new THREE.SphereGeometry(0.35, 32, 32),
                    new THREE.MeshBasicMaterial({ color: 0xFFD700, transparent: true, opacity: 0.4 })
                );
                orb.add(glow);
                lightGroup.add(orb);
                
                // The Beam (Cone)
                const beamGeo = new THREE.ConeGeometry(0.4, RADIUS, 32, 1, true);
                beamGeo.translate(0, -RADIUS/2, 0); // Pivot at base
                beamGeo.rotateX(-Math.PI / 2); // Point along Z
                const beamMat = new THREE.MeshBasicMaterial({ 
                    color: 0xFFD700, 
                    transparent: true, 
                    opacity: 0.15, 
                    side: THREE.DoubleSide,
                    depthWrite: false,
                    blending: THREE.AdditiveBlending 
                });
                const beam = new THREE.Mesh(beamGeo, beamMat);
                beam.lookAt(CENTER); // This will need dynamic updating
                lightGroup.add(beam);

                // --- 5. State & Logic ---
                let az = props.value?.azimuth || 0;
                let el = props.value?.elevation || 0;
                
                const AZ_MAP = {
                    0: 'Front', 45: 'Right Front', 90: 'Right', 135: 'Right Rear',
                    180: 'Rear', 225: 'Left Rear', 270: 'Left', 315: 'Left Front'
                };
                
                function getPrompt(a, e) {
                    if (e >= 60) return "Light source from Above";
                    if (e <= -60) return "Light source from Below";
                    // Snap
                    const steps = [0,45,90,135,180,225,270,315];
                    // Handle wrapped 360
                    let normalized = a % 360;
                    if(normalized < 0) normalized += 360;
                    const snapped = steps.reduce((p, c) => Math.abs(c-normalized) < Math.abs(p-normalized) ? c : p);
                    return `Light source from the ${AZ_MAP[snapped]}`;
                }

                function updateGizmo() {
                    const r_az = THREE.MathUtils.degToRad(az);
                    const r_el = THREE.MathUtils.degToRad(el);
                    
                    // Orbit Calculation
                    const x = RADIUS * Math.sin(r_az) * Math.cos(r_el);
                    const y = RADIUS * Math.sin(r_el) + CENTER.y;
                    const z = RADIUS * Math.cos(r_az) * Math.cos(r_el);
                    
                    lightGroup.position.set(x, y, z);
                    lightGroup.lookAt(CENTER); // Points the Beam at center
                    
                    // UI Updates
                    const text = getPrompt(az, el);
                    badge.innerText = text;
                    
                    // Color Logic (Warning for Above/Below)
                    let mainColor = 0xFFD700; // Gold
                    if (el >= 60 || el <= -60) mainColor = 0xFF4500; // OrangeRed
                    
                    orb.material.color.setHex(mainColor);
                    glow.material.color.setHex(mainColor);
                    beam.material.color.setHex(mainColor);
                    badge.style.borderColor = '#' + new THREE.Color(mainColor).getHexString();
                    badge.style.color = '#' + new THREE.Color(mainColor).getHexString();
                }

                // --- 6. Interaction (Drag) ---
                const raycaster = new THREE.Raycaster();
                const mouse = new THREE.Vector2();
                let isDragging = false;
                
                // Invisible Drag Sphere (Larger hit area)
                const dragSphere = new THREE.Mesh(
                    new THREE.SphereGeometry(RADIUS, 32, 16),
                    new THREE.MeshBasicMaterial({ visible: false, side: THREE.DoubleSide })
                );
                dragSphere.position.copy(CENTER);
                scene.add(dragSphere);

                function getMouse(e) {
                    const rect = wrapper.getBoundingClientRect();
                    const clientX = e.clientX || (e.touches ? e.touches[0].clientX : 0);
                    const clientY = e.clientY || (e.touches ? e.touches[0].clientY : 0);
                    return {
                        x: ((clientX - rect.left) / rect.width) * 2 - 1,
                        y: -((clientY - rect.top) / rect.height) * 2 + 1
                    };
                }

                function onDown(e) {
                    const m = getMouse(e);
                    raycaster.setFromCamera(m, camera);
                    // Check if clicked near the light orb
                    const intersects = raycaster.intersectObject(dragSphere);
                    if(intersects.length > 0) {
                        // Check distance to current light pos to prevent jumping if clicked far away
                        if (intersects[0].point.distanceTo(lightGroup.position) < 1.0) {
                            isDragging = true;
                            wrapper.style.cursor = 'none'; // Hide cursor while dragging for immersion
                        }
                    }
                }

                function onMove(e) {
                    if (!isDragging) {
                        // Hover state
                        const m = getMouse(e);
                        raycaster.setFromCamera(m, camera);
                        const hits = raycaster.intersectObject(dragSphere);
                        if (hits.length > 0 && hits[0].point.distanceTo(lightGroup.position) < 0.8) {
                            wrapper.style.cursor = 'pointer';
                        } else {
                            wrapper.style.cursor = 'default';
                        }
                        return;
                    }

                    const m = getMouse(e);
                    raycaster.setFromCamera(m, camera);
                    const intersects = raycaster.intersectObject(dragSphere);
                    
                    if (intersects.length > 0) {
                        const p = intersects[0].point;
                        const rel = new THREE.Vector3().subVectors(p, CENTER);
                        
                        // Convert Cartesian to Spherical (Azimuth/Elevation)
                        let newAz = Math.atan2(rel.x, rel.z) * (180 / Math.PI);
                        if (newAz < 0) newAz += 360;
                        
                        const distXZ = Math.sqrt(rel.x*rel.x + rel.z*rel.z);
                        let newEl = Math.atan2(rel.y, distXZ) * (180 / Math.PI);
                        
                        // Limits
                        newEl = Math.max(-89, Math.min(89, newEl));
                        
                        az = newAz;
                        el = newEl;
                        updateGizmo();
                    }
                }

                function onUp() {
                    if(isDragging) {
                        isDragging = false;
                        wrapper.style.cursor = 'default';
                        // Propagate value back to Gradio
                        props.value = { azimuth: az, elevation: el };
                        trigger('change', props.value);
                    }
                }

                // Event Listeners
                wrapper.addEventListener('mousedown', onDown);
                window.addEventListener('mousemove', onMove);
                window.addEventListener('mouseup', onUp);
                wrapper.addEventListener('touchstart', onDown, {passive: false});
                window.addEventListener('touchmove', onMove, {passive: false});
                window.addEventListener('touchend', onUp);

                // --- 7. Loop & Watchers ---
                updateGizmo(); // Init
                
                function animate() {
                    requestAnimationFrame(animate);
                    // Subtle idle animation for the glow
                    glow.scale.setScalar(1 + Math.sin(Date.now() * 0.003) * 0.1);
                    renderer.render(scene, camera);
                }
                animate();

                // Watch for changes from Python/Sliders
                setInterval(() => {
                    // Texture change
                    if (props.imageUrl && (!planeMat.map || props.imageUrl !== planeMat.map.image.src)) {
                         // handled by dedicated updater usually, but fail-safe
                    }
                    // Value change
                    if (props.value && !isDragging) {
                        if (Math.abs(props.value.azimuth - az) > 0.1 || Math.abs(props.value.elevation - el) > 0.1) {
                            az = props.value.azimuth;
                            el = props.value.elevation;
                            updateGizmo();
                        }
                    }
                }, 100);
                
                // Expose updater
                wrapper._updateTexture = updateTexture;
            };
            initScene();
        })();
        """
        
        super().__init__(
            value=value,
            html_template=html_template,
            js_on_load=js_on_load,
            imageUrl=imageUrl,
            **kwargs
        )

# --- UI Layout ---

css = """
#col-container { max-width: 1200px; margin: 0 auto; }
#3d-container { border: 1px solid #333; border-radius: 12px; overflow: hidden; }
.range-slider { accent-color: #FFD700 !important; }
"""

with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="yellow")) as demo:
    gr.Markdown("""
    # 💡 Qwen Edit 2509 — 3D Lighting Studio
    
    **Interactive Relighting:** Drag the ☀️ Sun in the 3D Viewport to change the lighting direction.
    """)
    
    with gr.Row():
        # --- Left Column: Controls ---
        with gr.Column(scale=5):
            # Input
            image_input = gr.Image(label="Input Image", type="pil", height=320)
            
            gr.Markdown("### 🎮 3D Controller")
            light_controller = LightControl3D(
                value={"azimuth": 0, "elevation": 0},
                elem_id="3d-container"
            )
            
            # Action
            run_btn = gr.Button("✨ Generate Lighting", variant="primary", size="lg")
            
            # Fine Tuning
            with gr.Accordion("🎚️ Fine-Tune & Advanced", open=False):
                with gr.Row():
                    az_slider = gr.Slider(0, 359, value=0, label="Azimuth", step=1)
                    el_slider = gr.Slider(-90, 90, value=0, label="Elevation", step=1)
                
                with gr.Row():
                    seed = gr.Slider(0, MAX_SEED, value=42, label="Seed", step=1)
                    randomize_seed = gr.Checkbox(True, label="Randomize")
                
                with gr.Row():
                    cfg = gr.Slider(1.0, 10.0, value=5.0, label="Guidance (CFG)")
                    steps = gr.Slider(1, 20, value=4, step=1, label="Steps")
                    
                prompt_display = gr.Textbox(label="Actual Prompt sent to Model", interactive=False)

        # --- Right Column: Output ---
        with gr.Column(scale=4):
            result_output = gr.Image(label="Result", height=600)

    # --- wiring ---
    
    # 1. Sync 3D -> Sliders & Text
    def on_3d_change(val):
        az = val.get('azimuth', 0)
        el = val.get('elevation', 0)
        prompt = build_lighting_prompt(az, el)
        return az, el, prompt

    light_controller.change(
        on_3d_change,
        inputs=[light_controller],
        outputs=[az_slider, el_slider, prompt_display]
    )
    
    # 2. Sync Sliders -> 3D & Text
    def on_slider_change(az, el):
        prompt = build_lighting_prompt(az, el)
        return {"azimuth": az, "elevation": el}, prompt
        
    az_slider.change(on_slider_change, inputs=[az_slider, el_slider], outputs=[light_controller, prompt_display])
    el_slider.change(on_slider_change, inputs=[az_slider, el_slider], outputs=[light_controller, prompt_display])

    # 3. Handle Image Upload (Resize + Update 3D Texture)
    def on_upload(img):
        w, h = update_dimensions_on_upload(img)
        if img is None: 
            return w, h, gr.update(imageUrl=None)
        
        # Convert to Base64 for Three.js
        import base64
        from io import BytesIO
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        data_url = f"data:image/png;base64,{img_str}"
        return w, h, gr.update(imageUrl=data_url)

    image_input.upload(
        on_upload,
        inputs=[image_input],
        outputs=[gr.State(), gr.State(), light_controller] # We store W/H in state mostly, or just pass to infer
    ).then(
        # Pass W/H to hidden sliders or just recalc in infer for simplicity
        None, None, None
    )

    # 4. Generate
    def run_inference_wrapper(img, az, el, seed, rand, cfg, steps):
        w, h = update_dimensions_on_upload(img) # Recalc dims here for safety
        res, used_seed, p = infer_lighting_edit(img, az, el, seed, rand, cfg, steps, h, w)
        return res

    run_btn.click(
        run_inference_wrapper,
        inputs=[image_input, az_slider, el_slider, seed, randomize_seed, cfg, steps],
        outputs=[result_output]
    )

if __name__ == "__main__":
    # CDN Load Three.js
    head_js = '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>'
    demo.launch(head=head_js)