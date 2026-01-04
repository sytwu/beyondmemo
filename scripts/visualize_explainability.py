#!/usr/bin/env python3
"""
Visualize explainability results for BeyondMemo predictions with model inference.

Usage:
    # With pre-computed explainability JSON
    python scripts/visualize_explainability.py \
        --json path/to/explainability.json \
        --image-name 5.jpg \
        --image-dir data/BUILDING \
        --output visualization.png
    
    # With model prediction (no pre-computed JSON needed)
    python scripts/visualize_explainability.py \
        --model-config configs/building.yaml \
        --checkpoint path/to/checkpoint.ckpt \
        --image-name 5.jpg \
        --image-dir data/BUILDING \
        --output visualization.png
    
    # With location information
    python scripts/visualize_explainability.py \
        --model-config configs/building.yaml \
        --checkpoint path/to/checkpoint.ckpt \
        --image-name 5.jpg \
        --image-dir data/BUILDING \
        --location "120.5E,30.2N" \
        --output visualization.png
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import pandas as pd
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from ordinalclip.runner.runner_ssr import Runner
from ordinalclip.runner.utils import get_transforms


CATEGORY_COLORS = {
    'Roof Style': '#FF6B6B',
    'Window Style': '#4ECDC4',
    'Wall Material': '#95E1D3',
    'Decorative Style': '#00D084',
    'Building Height': '#F38181',
    'Number of Floors': '#AA96DA',
    'Ornamentation Level': '#A8E6CF',
    'Color Scheme': '#FFD3B6',
    'Structural Shape': '#FFAAA5',
    'Entrance Design': '#C7CEEA',
    'Balcony Presence': '#B8E994',
    'Column Presence': '#FFB7B2',
    'Façade Symmetry': '#9B59B6',
    'Roof Material': '#3498DB',
    'Construction Period Clues': '#E74C3C'
}


def parse_location(location_str):
    """Parse location string like '120.5E,30.2N' to (lon, lat)."""
    if not location_str:
        return None
    
    parts = location_str.split(',')
    if len(parts) != 2:
        raise ValueError("Location should be in format 'LON,LAT' (e.g., '120.5E,30.2N')")
    
    lon_str, lat_str = parts
    
    if lon_str.endswith('E'):
        lon = float(lon_str[:-1])
    elif lon_str.endswith('W'):
        lon = -float(lon_str[:-1])
    else:
        lon = float(lon_str)
    
    if lat_str.endswith('N'):
        lat = float(lat_str[:-1])
    elif lat_str.endswith('S'):
        lat = -float(lat_str[:-1])
    else:
        lat = float(lat_str)
    
    return (lon, lat)


def load_model(config_path, checkpoint_path):
    """Load the BeyondMemo model from config and checkpoint."""
    cfg = OmegaConf.load(config_path)
    
    runner = Runner.load_from_checkpoint(
        checkpoint_path,
        **OmegaConf.to_container(cfg.runner_cfg),
        explainable=True
    )
    
    runner.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner = runner.to(device)
    
    return runner, device


def predict_with_explainability(runner, device, image_path, location=None, transforms=None):
    """Run model prediction with explainability enabled."""
    img = Image.open(image_path).convert('RGB')
    
    if transforms is None:
        from torchvision import transforms as T
        transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                       std=[0.26862954, 0.26130258, 0.27577711])
        ])
    
    img_tensor = transforms(img).unsqueeze(0).to(device)
    
    location_tensor = None
    if location:
        lon, lat = location
        location_tensor = torch.tensor([[lon, lat]], dtype=torch.float32, device=device)
    
    runner.module.eval()
    runner.module = runner.module.to(device)
    
    if hasattr(runner.module, 'reasoning_features'):
        runner.module.reasoning_features = runner.module.reasoning_features.to(device).float()
    
    if location_tensor is not None:
        original_forward = runner.module.location_encoder.forward
        
        def patched_forward(loc):
            if loc.dim() == 2 and loc.shape[1] == 2:
                lons = loc[:, 0]
                lats = loc[:, 1]
                loc = torch.stack([lats, lons])
            return original_forward(loc)
        
        runner.module.location_encoder.forward = patched_forward
    
    with torch.enable_grad():
        if location_tensor is not None:
            model_output = runner.module(img_tensor, location_tensor, run_type="test", show_importance=True)
        else:
            model_output = runner.module(img_tensor, None, run_type="test", show_importance=True)
        
        logits, regress_age, saliency_logits, saliency_reason = model_output
    
    predicted_year = regress_age.detach().cpu().numpy()[0]
    saliency_reason = saliency_reason.detach().cpu().numpy()[0]
    
    reasoning_prompts = runner.module.reasoning_prompts
    
    category_sizes = [7, 6, 7, 8, 4, 3, 3, 4, 4, 4, 3, 3, 2, 4, 4]
    category_names = [
        'Roof Style', 'Window Style', 'Wall Material', 'Decorative Style',
        'Building Height', 'Number of Floors', 'Ornamentation Level', 'Color Scheme',
        'Structural Shape', 'Entrance Design', 'Balcony Presence', 'Column Presence',
        'Façade Symmetry', 'Roof Material', 'Construction Period Clues'
    ]
    
    idx_to_category = []
    for cat_idx, size in enumerate(category_sizes):
        idx_to_category.extend([cat_idx] * size)
    
    reason_data = []
    for idx, score in enumerate(saliency_reason):
        if idx < len(idx_to_category):
            cat_idx = idx_to_category[idx]
            reason_data.append({
                'index': idx,
                'category': category_names[cat_idx],
                'category_idx': cat_idx,
                'score': float(score),
                'reason': reasoning_prompts[idx]
            })
    
    reason_data.sort(key=lambda x: x['score'], reverse=True)
    
    top_reasons = []
    seen_categories = set()
    for reason in reason_data:
        if reason['category_idx'] not in seen_categories:
            top_reasons.append({
                'category': reason['category'],
                'reason': reason['reason'],
                'importance_score': reason['score']
            })
            seen_categories.add(reason['category_idx'])
            if len(top_reasons) >= 5:
                break
    
    return predicted_year, top_reasons


def load_explainability_data(json_path, image_name):
    """Load explainability data for a specific image."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if image_name not in data:
        raise ValueError(f"Image '{image_name}' not found in explainability JSON")
    
    return data[image_name]


def create_explanation_text(reasons, predicted_year=None, ground_truth_year=None):
    """Create formatted explanation text with color highlighting."""
    lines = []
    
    if predicted_year is not None:
        if ground_truth_year is not None:
            lines.append(f"The estimated built year is {int(predicted_year)} (GT={int(ground_truth_year)}), as the architectural style seems to be:")
        else:
            lines.append(f"The estimated built year is {int(predicted_year)}, as the architectural style seems to be:")
    
    for i, reason_data in enumerate(reasons):
        category = reason_data['category']
        reason_text = reason_data['reason']
        score = reason_data['importance_score']
        
        if 'is a ' in reason_text:
            feature = reason_text.split('is a ', 1)[1]
        elif 'is made of ' in reason_text:
            feature = reason_text.split('is made of ', 1)[1]
        elif 'features a decorative style of ' in reason_text:
            feature = reason_text.split('features a decorative style of ', 1)[1]
        elif 'is ' in reason_text:
            feature = reason_text.split('is ', 1)[1]
        elif 'has a design of ' in reason_text:
            feature = reason_text.split('has a design of ', 1)[1]
        elif 'has ' in reason_text:
            feature = reason_text.split('has ', 1)[1]
        else:
            feature = reason_text
        
        lines.append((category, feature, score))
    
    return lines


def visualize_explainability(image_path, reasons, output_path=None, 
                            predicted_year=None, ground_truth_year=None):
    """Create visualization with image and explanation."""
    img = Image.open(image_path)
    
    # Resize image to square (keeping aspect ratio, center crop if needed)
    width, height = img.size
    size = min(width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    img_square = img.crop((left, top, left + size, top + size))
    
    # Category to connector mapping (pre-defined)
    CATEGORY_CONNECTORS = {
        'Roof Style': 'The roof of the building is a ',
        'Window Style': 'The window of the building is a ',
        'Wall Material': 'The wall of the building is made of ',
        'Decorative Style': 'The building features a decorative style of ',
        'Building Height': 'The height classification of the building is ',
        'Number of Floors': 'The number of floors in the building is ',
        'Ornamentation Level': 'The level of ornamentation is ',
        'Color Scheme': 'The building\'s color scheme is ',
        'Structural Shape': 'The structural shape of the building is ',
        'Entrance Design': 'The entrance of the building has a design of ',
        'Balcony Presence': 'The balcony configuration of the building is ',
        'Column Presence': 'The building has ',
        'Façade Symmetry': 'The building has a façade that is ',
        'Roof Material': 'The material used for the roof is ',
        'Construction Period Clues': 'The architectural period this building resembles is ',
    }
    
    fig = plt.figure(figsize=(20, 8))
    
    # Left: Square image
    ax_img = plt.subplot(1, 2, 1)
    ax_img.imshow(img_square)
    ax_img.axis('off')
    
    # Right: Explanation text
    ax_text = plt.subplot(1, 2, 2)
    ax_text.axis('off')
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(0, 1)
    
    # Build explanation text
    y_position = 0.90
    x_start = 0.02
    line_spacing = 0.10
    
    # Header with predicted year
    if predicted_year is not None:
        header_parts = []
        header_parts.append(("The estimated built year is ", "black", "normal"))
        header_parts.append((f"{int(predicted_year)}", "red", "bold"))
        
        if ground_truth_year is not None:
            header_parts.append((f" (GT={int(ground_truth_year)})", "black", "normal"))
        
        header_parts.append((", as the architectural style seems to be", "black", "normal"))
        
        # Draw header with inline colors
        x_pos = x_start
        for text, color, weight in header_parts:
            t = ax_text.text(x_pos, y_position, text, 
                           fontsize=14, va='top', ha='left',
                           color=color, fontweight=weight,
                           transform=ax_text.transAxes)
            
            # Get text width to position next part
            fig.canvas.draw()
            bbox = t.get_window_extent(renderer=fig.canvas.get_renderer())
            bbox_data = bbox.transformed(ax_text.transAxes.inverted())
            x_pos = bbox_data.x1
        
        y_position -= line_spacing * 1.3
    
    # Add reasoning features - one per line with colored text
    for i, reason_data in enumerate(reasons):
        category = reason_data['category']
        reason_text = reason_data['reason']
        score = reason_data['importance_score']
        color = CATEGORY_COLORS.get(category, '#808080')
        
        # Extract feature description
        if 'is a ' in reason_text:
            feature = reason_text.split('is a ', 1)[1]
        elif 'is made of ' in reason_text:
            feature = reason_text.split('is made of ', 1)[1]
        elif 'features a decorative style of ' in reason_text:
            feature = reason_text.split('features a decorative style of ', 1)[1]
        elif 'is ' in reason_text:
            feature = reason_text.split('is ', 1)[1]
        elif 'has a design of ' in reason_text:
            feature = reason_text.split('has a design of ', 1)[1]
        elif 'has ' in reason_text:
            feature = reason_text.split('has ', 1)[1]
        else:
            feature = reason_text
        
        # Remove parentheses content for cleaner display
        if '(' in feature:
            feature = feature.split('(')[0].strip()
        
        # Get connector for this category
        connector = CATEGORY_CONNECTORS.get(category, 'The building is ')
        
        # Draw bullet point
        ax_text.text(x_start, y_position, "• ", 
                    fontsize=14, va='top', ha='left',
                    color='black', fontweight='normal',
                    transform=ax_text.transAxes)
        
        # Draw connector in black
        x_pos = x_start + 0.03
        t = ax_text.text(x_pos, y_position, connector,
                        fontsize=14, va='top', ha='left',
                        color='black', fontweight='normal',
                        transform=ax_text.transAxes)
        
        fig.canvas.draw()
        bbox = t.get_window_extent(renderer=fig.canvas.get_renderer())
        bbox_data = bbox.transformed(ax_text.transAxes.inverted())
        x_pos = bbox_data.x1
        
        # Draw colored feature text
        t = ax_text.text(x_pos, y_position, feature,
                        fontsize=14, va='top', ha='left',
                        color=color, fontweight='bold',
                        transform=ax_text.transAxes)
        
        # Add period at the end for Column Presence category
        if category == 'Column Presence':
            fig.canvas.draw()
            bbox = t.get_window_extent(renderer=fig.canvas.get_renderer())
            bbox_data = bbox.transformed(ax_text.transAxes.inverted())
            x_pos = bbox_data.x1
            ax_text.text(x_pos, y_position, " used for aesthetic effect.",
                        fontsize=14, va='top', ha='left',
                        color='black', fontweight='normal',
                        transform=ax_text.transAxes)
        else:
            fig.canvas.draw()
            bbox = t.get_window_extent(renderer=fig.canvas.get_renderer())
            bbox_data = bbox.transformed(ax_text.transAxes.inverted())
            x_pos = bbox_data.x1
            ax_text.text(x_pos, y_position, ".",
                        fontsize=14, va='top', ha='left',
                        color='black', fontweight='normal',
                        transform=ax_text.transAxes)
        
        y_position -= line_spacing
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize explainability results for building year estimation'
    )
    
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--json', help='Path to pre-computed explainability JSON file')
    mode_group.add_argument('--model-config', help='Path to model config YAML file')
    
    parser.add_argument('--image-name', required=True, help='Image filename (e.g., 5.jpg)')
    parser.add_argument('--image-dir', required=True, help='Directory containing images')
    
    parser.add_argument('--checkpoint', help='Path to model checkpoint (required if using --model-config)')
    parser.add_argument('--location', help='Location string in format "LON,LAT" (e.g., "120.5E,30.2N")')
    
    parser.add_argument('--ground-truth', type=int, help='Ground truth year (optional)')
    parser.add_argument('--output', help='Output path for visualization (if not specified, will display)')
    
    args = parser.parse_args()
    
    image_path = os.path.join(args.image_dir, args.image_name)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    predicted_year = None
    reasons = None
    
    if args.json:
        reasons = load_explainability_data(args.json, args.image_name)
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when using --model-config")
        
        print("Loading model...")
        runner, device = load_model(args.model_config, args.checkpoint)
        
        location = parse_location(args.location) if args.location else None
        if location:
            print(f"Using location: {location}")
        
        print("Running prediction with explainability...")
        predicted_year, reasons = predict_with_explainability(
            runner, device, image_path, location
        )
        print(f"Predicted year: {int(predicted_year)}")
    
    visualize_explainability(
        image_path=image_path,
        reasons=reasons,
        output_path=args.output,
        predicted_year=predicted_year,
        ground_truth_year=args.ground_truth
    )


if __name__ == '__main__':
    main()
