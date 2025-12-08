#!/usr/bin/env python3
"""
Download pretrained Zero-DCE++ weights

This script downloads pretrained weights from the official repository
or provides instructions for manual download.
"""

import urllib.request
import os
from pathlib import Path
import argparse


def download_file(url, destination):
    """Download file with progress bar"""
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\rDownloading: {percent}%", end='', flush=True)
    
    urllib.request.urlretrieve(url, destination, progress_hook)
    print()  # New line after download


def download_zero_dce_weights(output_dir='models'):
    """
    Download Zero-DCE++ pretrained weights
    
    Note: Since official weights may not be directly available via URL,
    this function provides instructions for manual download.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    weights_path = output_dir / 'zero_dce_plus.pth'
    
    if weights_path.exists():
        print(f"✅ Weights already exist at {weights_path}")
        return str(weights_path)
    
    print("=" * 70)
    print("Zero-DCE++ Pretrained Weights Download Instructions")
    print("=" * 70)
    print()
    print("Option 1: Official GitHub Repository")
    print("-" * 70)
    print("1. Visit: https://github.com/Li-Chongyi/Zero-DCE_extension")
    print("2. Navigate to 'Pretrained_model' folder")
    print("3. Download 'Epoch99.pth' or 'Zero-DCE++.pth'")
    print(f"4. Save to: {weights_path.absolute()}")
    print()
    
    print("Option 2: Google Drive (if available)")
    print("-" * 70)
    print("Check the repository README for Google Drive links")
    print()
    
    print("Option 3: Train Your Own")
    print("-" * 70)
    print("Use the provided training script:")
    print("  python scripts/train_zero_dce.py --data data/low_light_images/")
    print()
    
    print("Option 4: Alternative Pre-trained Models")
    print("-" * 70)
    print("Other low-light enhancement models:")
    print("- EnlightenGAN: https://github.com/TAMU-VITA/EnlightenGAN")
    print("- IAT: https://github.com/cuiziteng/Illumination-Adaptive-Transformer")
    print("- SCI: https://github.com/vis-opt-group/SCI")
    print()
    
    print("=" * 70)
    print("After downloading, verify with:")
    print(f"  python scripts/zero_dce.py <test_image.jpg> --model {weights_path}")
    print("=" * 70)
    
    return None


def create_dummy_weights(output_path='models/zero_dce_plus.pth'):
    """
    Create randomly initialized weights for testing
    
    WARNING: These weights are NOT trained and will produce poor results.
    Only use for code testing purposes.
    """
    import torch
    from zero_dce import DCENet
    
    print("⚠️  Creating UNTRAINED weights for testing purposes only")
    print("   Results will be poor - download pretrained weights for real use")
    
    model = DCENet()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), output_path)
    print(f"✅ Created dummy weights at {output_path}")
    print("   Remember to replace with pretrained weights!")
    
    return str(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download Zero-DCE++ pretrained weights'
    )
    parser.add_argument(
        '-o', '--output',
        default='models',
        help='Output directory for weights'
    )
    parser.add_argument(
        '--create-dummy',
        action='store_true',
        help='Create untrained weights for testing (NOT recommended)'
    )
    
    args = parser.parse_args()
    
    if args.create_dummy:
        create_dummy_weights(f"{args.output}/zero_dce_plus.pth")
    else:
        download_zero_dce_weights(args.output)
