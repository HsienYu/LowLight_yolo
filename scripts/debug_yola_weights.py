
import torch
import sys
from pathlib import Path

# Add parent dir to path
sys.path.append(str(Path(__file__).parent))
from yola import YOLAModule

def debug_weights(checkpoint_path):
    print(f"Checking checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("Checkpoint is a dict with 'state_dict' key.")
    else:
        state_dict = checkpoint
        print("Checkpoint is a direct state_dict.")

    print(f"Total keys in checkpoint: {len(state_dict)}")
    
    # Initialize model
    model = YOLAModule()
    model_keys = set(model.state_dict().keys())
    print(f"Total keys in YOLAModule: {len(model_keys)}")
    
    # Simulate loading
    matched_keys = []
    missing_keys = []
    
    prefixes_to_strip = ['iim.', 'model.iim.', 'bbox_head.iim.']
    
    print("\n--- Matching Process ---")
    for key in state_dict.keys():
        clean_key = key
        for prefix in prefixes_to_strip:
            if key.startswith(prefix):
                clean_key = key[len(prefix):]
                break
        
        if clean_key in model_keys:
            matched_keys.append(clean_key)
            # Remove from model_keys to see what's left
            model_keys.remove(clean_key)
            
    print(f"Matched {len(matched_keys)} keys.")
    print(f"Unmatched (Missing) keys in model: {len(model_keys)}")
    
    if len(model_keys) > 0:
        print("\nMISSING KEYS (These weights are staying random!):")
        for k in sorted(model_keys):
            print(f" - {k}")
            
    if len(matched_keys) == 0:
        print("\nCRITICAL: No keys matched! Here are some sample keys from checkpoint:")
        for k in list(state_dict.keys())[:10]:
            print(f" - {k}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/debug_yola_weights.py path/to/yola.pth")
    else:
        debug_weights(sys.argv[1])
