#!/usr/bin/env python3
"""
Convert MMDetection YOLA checkpoint to pure PyTorch weights.

This script extracts only the YOLA module weights from a full MMDetection
checkpoint, removing the dependency on mmengine for loading.

Usage:
    pip install mmengine  # Temporarily needed for conversion
    python scripts/convert_yola_weights.py models/yola.pth models/yola_converted.pth
"""

import sys
import torch
from pathlib import Path

def convert_checkpoint(input_path, output_path):
    print(f"Loading checkpoint: {input_path}")
    
    # This requires mmengine to be installed
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    print(f"Total keys in checkpoint: {len(state_dict)}")
    
    # Find YOLA-related keys
    yola_keys = {}
    
    # Look for the prefix by finding feat_projector
    prefix = None
    for key in state_dict.keys():
        if 'feat_projector.0.weight' in key:
            prefix = key.replace('feat_projector.0.weight', '')
            print(f"Detected prefix: '{prefix}'")
            break
    
    if prefix is None:
        print("WARNING: Could not detect YOLA prefix. Trying common prefixes...")
        for p in ['iim.', 'model.iim.', 'backbone.iim.']:
            if any(k.startswith(p) for k in state_dict.keys()):
                prefix = p
                print(f"Found prefix: '{prefix}'")
                break
    
    if prefix is None:
        print("ERROR: Could not find YOLA weights in checkpoint!")
        print("First 20 keys:")
        for k in list(state_dict.keys())[:20]:
            print(f"  - {k}")
        return False
    
    # Extract YOLA weights with prefix stripped
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            yola_keys[new_key] = value
    
    print(f"Extracted {len(yola_keys)} YOLA weights")
    
    if len(yola_keys) == 0:
        print("ERROR: No weights extracted!")
        return False
    
    # Print extracted keys
    print("Extracted keys:")
    for k in sorted(yola_keys.keys()):
        print(f"  - {k}")
    
    # Save as pure tensor dict
    torch.save(yola_keys, output_path)
    print(f"\n✅ Saved converted weights to: {output_path}")
    print(f"   File size: {Path(output_path).stat().st_size / 1024:.1f} KB")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/convert_yola_weights.py <input.pth> <output.pth>")
        print("Example: python scripts/convert_yola_weights.py models/yola.pth models/yola_converted.pth")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not Path(input_path).exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)
    
    success = convert_checkpoint(input_path, output_path)
    sys.exit(0 if success else 1)
