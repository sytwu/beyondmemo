#!/usr/bin/env python3
"""
Local Gradio demo for BeyondMemo building age estimation.
Test this locally before deploying to Hugging Face Space.

Usage:
    conda activate yearclip
    python gradio_demo_local.py
    
Then open http://localhost:7860 in your browser.
"""

import gradio as gr
import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
import base64
from pathlib import Path

# Set Gradio temp directory to avoid permission issues
os.environ['GRADIO_TEMP_DIR'] = os.path.expanduser('~/tmp/gradio')
os.makedirs(os.environ['GRADIO_TEMP_DIR'], exist_ok=True)

# Import from scripts
from scripts.visualize_explainability import (
    load_model, 
    predict_with_explainability,
    visualize_explainability
)

# Global variables (avoid reloading model)
runner = None
device = None

# Configuration
MODEL_CONFIG = "configs/building.yaml"
# CHECKPOINT_PATH = "results/building/YearCLIP_114514/ckpts/epoch=21-val_mae_max_metric=39.4220.ckpt"
CHECKPOINT_PATH = "results/building/YearCLIP_114514/ckpts/last.ckpt"
EXAMPLE_DIR = "examples/FI-London/Image"

def initialize_model():
    """Initialize model (only on first run)"""
    global runner, device
    if runner is None:
        print("🔄 Loading BeyondMemo model...")
        runner, device = load_model(MODEL_CONFIG, CHECKPOINT_PATH)
        print("✅ Model loaded successfully!")
    return runner, device


def predict_building_age(image, lon=None, lat=None):
    """
    Predict building age and generate explainability visualization
    
    Args:
        image: PIL Image or numpy array
        lon: Optional longitude (float)
        lat: Optional latitude (float)
    
    Returns:
        HTML formatted text with colored features
    """
    if image is None:
        return "Please upload a building image"
    
    try:
        # Initialize model
        runner, device = initialize_model()
        
        # Prepare location
        location = None
        if lon is not None and lat is not None:
            try:
                location = (float(lon), float(lat))
                print(f"Using location: ({lon}, {lat})")
            except:
                print("Invalid location coordinates, proceeding without location")
                pass
        
        # Run prediction
        print("Running prediction...")
        predicted_year, reasons = predict_with_explainability(
            runner, device, image, location=location
        )
        
        # Import category colors
        from scripts.visualize_explainability import CATEGORY_COLORS
        
        # Build HTML text output with colored features
        html_output = f"<h2>Predicted Year: <span style='color: red; font-weight: bold;'>{int(predicted_year)}</span></h2>\n\n"
        html_output += "<h3>Key Architectural Features (by importance):</h3>\n\n"
        
        for i, reason in enumerate(reasons, 1):
            category = reason['category']
            reason_text = reason['reason']
            score = reason['importance_score']
            color = CATEGORY_COLORS.get(category, '#808080')
            
            # Extract feature description
            if 'is a ' in reason_text:
                prefix = reason_text.split('is a ', 1)[0] + 'is a '
                feature = reason_text.split('is a ', 1)[1]
            elif 'is made of ' in reason_text:
                prefix = reason_text.split('is made of ', 1)[0] + 'is made of '
                feature = reason_text.split('is made of ', 1)[1]
            elif 'features a decorative style of ' in reason_text:
                prefix = reason_text.split('features a decorative style of ', 1)[0] + 'features a decorative style of '
                feature = reason_text.split('features a decorative style of ', 1)[1]
            elif 'is ' in reason_text:
                prefix = reason_text.split('is ', 1)[0] + 'is '
                feature = reason_text.split('is ', 1)[1]
            elif 'has a design of ' in reason_text:
                prefix = reason_text.split('has a design of ', 1)[0] + 'has a design of '
                feature = reason_text.split('has a design of ', 1)[1]
            elif 'has ' in reason_text:
                prefix = reason_text.split('has ', 1)[0] + 'has '
                feature = reason_text.split('has ', 1)[1]
            else:
                prefix = ""
                feature = reason_text
            
            # Remove parentheses content for cleaner display
            if '(' in feature:
                feature = feature.split('(')[0].strip()
            
            html_output += f"<p style='font-size: 16px;'>"
            html_output += f"{i}. <b>[{category}]</b> {prefix}"
            html_output += f"<span style='color: {color}; font-weight: bold;'>{feature}</span>."
            html_output += f"<br><span style='font-size: 14px; color: #666;'>Importance: {score:.4f}</span>"
            html_output += f"</p>\n"
        
        return html_output
    
    except Exception as e:
        import traceback
        error_msg = f"<p style='color: red;'>❌ Error occurred:</p><pre>{str(e)}\n\n{traceback.format_exc()}</pre>"
        print(error_msg)
        return error_msg


# Create Gradio interface
with gr.Blocks(
    title="BeyondMemo: Building Age Estimation",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1600px !important;
    }
    /* Force columns to stretch to same height */
    .equal-height {
        align-items: stretch !important;
    }
    .equal-height > div {
        display: flex;
        flex-direction: column;
    }
    /* Location boxes to fill height equally */
    .location-container {
        display: flex;
        flex-direction: column;
        flex: 1;
        gap: 12px;
    }
    .coord-box {
        flex: 1;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 16px;
        border-radius: 8px;
        background: var(--background-fill-secondary);
    }
    /* Gallery card styling */
    .gallery-item {
        border-radius: 12px !important;
        overflow: hidden !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    .gallery-item:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3) !important;
    }
    .gallery-item img {
        border-radius: 12px !important;
    }
    /* Caption styling */
    .caption {
        font-size: 11px !important;
        padding: 6px 4px !important;
        text-align: center !important;
    }
    """
) as demo:
    gr.Markdown("# 🏛️ BeyondMemo: Building Age Estimation with Explainability (YearCLIP Demo)")
    gr.Markdown(
        "Upload a building image to predict its construction year with AI-powered explanations. "
        "Optionally provide location (longitude, latitude) for improved accuracy."
    )
    
    # Three-column layout: Image | Coordinates | Results
    with gr.Row(elem_classes="equal-height"):
        # Left column: Image upload
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload Image")
            gr.Markdown("*Upload or drag a building photo*")
            input_image = gr.Image(
                type="pil", 
                label="Building Image",
                height=280
            )
            predict_btn = gr.Button(
                "🔮 Predict Building Age",
                variant="primary",
                size="lg"
            )
        
        # Middle column: Location input - matching height with two boxes
        with gr.Column(scale=1):
            gr.Markdown("### 📍 Location (Optional)")
            gr.Markdown("*GPS coordinates can improve accuracy*")
            with gr.Column(elem_classes="location-container"):
                with gr.Group(elem_classes="coord-box"):
                    lon_input = gr.Number(
                        label="🌐 Longitude",
                        value=None,
                        precision=6,
                        info="East-West position (-180 to 180)"
                    )
                with gr.Group(elem_classes="coord-box"):
                    lat_input = gr.Number(
                        label="🌐 Latitude", 
                        value=None,
                        precision=6,
                        info="North-South position (-90 to 90)"
                    )
        
        # Right column: Results
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Prediction Results")
            output_text = gr.HTML(
                label="Explanation",
                value="<p style='color: #888; padding: 20px;'>Upload an image and click 'Predict Building Age' to see results.</p>"
            )
    
    # Example images section - Custom HTML cards with buttons
    gr.Markdown("---")
    gr.Markdown("### 🖼️ Example Images")
    gr.Markdown("*Click any card to load the example*")
    
    # Define all examples with metadata
    all_examples = [
        # FI-London Dataset
        {"path": os.path.join(EXAMPLE_DIR, "28.jpg"), "lon": -0.127, "lat": 51.523, "dataset": "FI-London"},
        {"path": os.path.join(EXAMPLE_DIR, "34.jpg"), "lon": -0.130, "lat": 51.527, "dataset": "FI-London"},
        {"path": os.path.join(EXAMPLE_DIR, "47.jpg"), "lon": -0.130, "lat": 51.524, "dataset": "FI-London"},
        # MapYourCity Dataset
        {"path": "examples/MapYourCity/4cfqi7zewa/street.jpg", "lon": None, "lat": None, "dataset": "MapYourCity"},
        {"path": "examples/MapYourCity/4q6hleeitk/street.jpg", "lon": None, "lat": None, "dataset": "MapYourCity"},
        {"path": "examples/MapYourCity/6uc5rl44wk/street.jpg", "lon": None, "lat": None, "dataset": "MapYourCity"},
        # YearGuessr Dataset
        {"path": "data/BUILDING/174.jpg", "lon": None, "lat": None, "dataset": "YearGuessr"},
        {"path": "data/BUILDING/1990.jpg", "lon": None, "lat": None, "dataset": "YearGuessr"},
        {"path": "data/BUILDING/2474.jpg", "lon": None, "lat": None, "dataset": "YearGuessr"},
    ]
    
    # Create card grid with buttons
    
    def create_example_loader(idx):
        """Create a loader function for specific example index"""
        def loader():
            ex = all_examples[idx]
            img = Image.open(ex["path"])
            return img, ex["lon"], ex["lat"]
        return loader
    
    with gr.Row():
        for idx, ex in enumerate(all_examples):
            with gr.Column(scale=1, min_width=140):
                # Read and encode image
                with open(ex["path"], "rb") as f:
                    img_data = base64.b64encode(f.read()).decode()
                img_ext = ex["path"].split(".")[-1]
                
                lon_str = f"{ex['lon']:.3f}" if ex['lon'] is not None else "N/A"
                lat_str = f"{ex['lat']:.3f}" if ex['lat'] is not None else "N/A"
                
                # Card HTML with image and info table
                card_html = f"""
                <div style="border: 1px solid var(--border-color-primary); border-radius: 12px 12px 0 0; overflow: hidden; background: var(--background-fill-primary);">
                    <img src="data:image/{img_ext};base64,{img_data}" style="width: 100%; height: 100px; object-fit: cover;">
                    <table style="width: 100%; font-size: 12px; border-collapse: collapse;">
                        <tr><td style="color: var(--neutral-400); padding: 1px 4px;">Dataset</td><td style="padding: 1px 4px; font-weight: 500;">{ex['dataset']}</td></tr>
                        <tr><td style="color: var(--neutral-400); padding: 1px 4px;">Lon</td><td style="padding: 1px 4px;">{lon_str}</td></tr>
                        <tr><td style="color: var(--neutral-400); padding: 1px 4px;">Lat</td><td style="padding: 1px 4px;">{lat_str}</td></tr>
                    </table>
                </div>
                """
                gr.HTML(card_html)
                
                # Button to load this example
                btn = gr.Button("Load", size="sm", variant="secondary")
                btn.click(
                    fn=create_example_loader(idx),
                    inputs=None,
                    outputs=[input_image, lon_input, lat_input]
                )
    
    # Connect button to function
    predict_btn.click(
        fn=predict_building_age,
        inputs=[input_image, lon_input, lat_input],
        outputs=output_text
    )


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting BeyondMemo Local Demo")
    print("=" * 60)
    print(f"Model Config: {MODEL_CONFIG}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Example Directory: {EXAMPLE_DIR}")
    print("=" * 60)
    print("\n📝 Note: Model will load on first prediction (may take a moment)\n")
    
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7861,
        show_error=True
    )
