#!/usr/bin/env python3
"""
Zero-DCE++ (Zero-Reference Deep Curve Estimation)
Implementation for low-light image enhancement

Paper: "Learning to Enhance Low-Light Image via Zero-Reference Deep Curve Estimation"
GitHub: https://github.com/Li-Chongyi/Zero-DCE_extension
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path


class DCENet(nn.Module):
    """
    Zero-DCE++ Network Architecture
    
    Enhanced Deep Curve Estimation network with 8 iteration layers.
    Estimates pixel-wise light enhancement curves without reference images.
    """
    
    def __init__(self, scale_factor=1, in_channels=3):
        super(DCENet, self).__init__()
        self.scale_factor = scale_factor
        
        # Official Zero-DCE++ architecture
        # Uses standard convolutions with residual connections
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(64, 24, kernel_size=3, stride=1, padding=1)
        
        # Activation
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        """
        Forward pass with residual connections (official Zero-DCE++ architecture)
        
        Args:
            x: Input image tensor [B, 3, H, W], range [0, 1]
            
        Returns:
            enhanced: Enhanced image [B, 3, H, W]
            A_list: List of curve parameter maps for each iteration
        """
        # Feature extraction with residual connections
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        
        # Concatenate for residual connections
        x5 = self.relu(self.conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], 1)))
        
        # Predict curve parameters
        x7 = self.tanh(self.conv7(torch.cat([x1, x6], 1)))  # [-1, 1]
        
        # Split into 8 iterations, each with 3 channel parameters
        A_list = torch.split(x7, 3, dim=1)
        
        # Apply iterative enhancement
        enhanced = x
        for A in A_list:
            enhanced = enhanced + A * (torch.pow(enhanced, 2) - enhanced)
        
        return enhanced, A_list


class ZeroDCEEnhancer:
    """
    High-level interface for Zero-DCE++ image enhancement
    
    Usage:
        enhancer = ZeroDCEEnhancer(model_path='models/zero_dce_plus.pth')
        enhanced_image = enhancer.enhance(image)
    """
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize enhancer
        
        Args:
            model_path: Path to pretrained model weights
            device: 'cuda', 'mps', or 'cpu' (auto-detect if None)
        """
        self.device = self._setup_device(device)
        self.model = DCENet().to(self.device)
        
        if model_path and Path(model_path).exists():
            self.load_weights(model_path)
            print(f"✅ Loaded Zero-DCE++ weights from {model_path}")
        else:
            print("⚠️  No pretrained weights loaded - using random initialization")
            print("   Download weights with: python scripts/download_zero_dce_weights.py")
        
        self.model.eval()
    
    def _setup_device(self, device):
        """Auto-detect best available device"""
        if device and device != 'auto':
            return torch.device(device)
        
        # Auto-detect or device is None or 'auto'
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def load_weights(self, model_path):
        """Load pretrained model weights"""
        try:
            # Try with weights_only=True first (PyTorch 2.6+ default)
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        except Exception:
            # Fallback to weights_only=False for older checkpoints
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Handle key mismatch: official weights use 'e_conv' prefix, our model uses 'conv'
        if 'e_conv1.weight' in state_dict:
            new_state_dict = {}
            for key, value in state_dict.items():
                # Remove 'e_' prefix from keys
                new_key = key.replace('e_conv', 'conv')
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        
        self.model.load_state_dict(state_dict)
    
    def enhance(self, image, return_numpy=True):
        """
        Enhance a single image
        
        Args:
            image: Input image (numpy BGR uint8 or torch tensor)
            return_numpy: Return numpy array (True) or torch tensor (False)
            
        Returns:
            Enhanced image in same format as input
        """
        # Convert to torch tensor if needed
        if isinstance(image, np.ndarray):
            input_image = self._numpy_to_torch(image)
        else:
            input_image = image
        
        # Inference
        with torch.no_grad():
            enhanced, _ = self.model(input_image)
        
        # Convert back to numpy if needed
        if return_numpy:
            return self._torch_to_numpy(enhanced)
        return enhanced
    
    def enhance_batch(self, images):
        """
        Enhance multiple images
        
        Args:
            images: List of numpy images or batched torch tensor
            
        Returns:
            List of enhanced images (same format as input)
        """
        if isinstance(images, list):
            return [self.enhance(img) for img in images]
        else:
            # Batched tensor
            with torch.no_grad():
                enhanced, _ = self.model(images)
            return enhanced
    
    def _numpy_to_torch(self, image):
        """Convert OpenCV BGR uint8 image to torch tensor"""
        # BGR -> RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # uint8 [0, 255] -> float32 [0, 1]
        image_float = image_rgb.astype(np.float32) / 255.0
        
        # HWC -> CHW
        image_tensor = torch.from_numpy(image_float).permute(2, 0, 1)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor
    
    def _torch_to_numpy(self, tensor):
        """Convert torch tensor to OpenCV BGR uint8 image"""
        # Remove batch dimension
        tensor = tensor.squeeze(0)
        
        # CHW -> HWC
        image = tensor.permute(1, 2, 0).cpu().numpy()
        
        # Clip to [0, 1]
        image = np.clip(image, 0, 1)
        
        # float32 [0, 1] -> uint8 [0, 255]
        image = (image * 255).astype(np.uint8)
        
        # RGB -> BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image


def compare_enhancement_methods(image_path, output_dir='results/enhancement_comparison'):
    """
    Compare different enhancement methods on a single image
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save comparison results
    """
    import matplotlib.pyplot as plt
    from preprocessing import apply_clahe
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Original
    original = image.copy()
    
    # CLAHE
    clahe_enhanced = apply_clahe(image, clip_limit=2.0, tile_size=8)
    
    # Zero-DCE++
    enhancer = ZeroDCEEnhancer()
    zero_dce_enhanced = enhancer.enhance(image)
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    images_to_show = [original, clahe_enhanced, zero_dce_enhanced]
    titles = ['Original', 'CLAHE', 'Zero-DCE++']
    
    for ax, img, title in zip(axes, images_to_show, titles):
        # Convert BGR to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save comparison
    output_path = output_dir / f"{Path(image_path).stem}_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved comparison to {output_path}")
    
    # Save individual enhanced images
    cv2.imwrite(str(output_dir / f"{Path(image_path).stem}_clahe.jpg"), clahe_enhanced)
    cv2.imwrite(str(output_dir / f"{Path(image_path).stem}_zerodce.jpg"), zero_dce_enhanced)
    
    plt.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Zero-DCE++ Image Enhancement')
    parser.add_argument('input', help='Input image or directory')
    parser.add_argument('-o', '--output', default='results/zero_dce',
                        help='Output directory')
    parser.add_argument('--model', default='models/zero_dce_plus.pth',
                        help='Path to pretrained model')
    parser.add_argument('--compare', action='store_true',
                        help='Compare with CLAHE')
    parser.add_argument('--device', choices=['cuda', 'mps', 'cpu'],
                        help='Device to use (auto-detect if not specified)')
    
    args = parser.parse_args()
    
    # Initialize enhancer
    enhancer = ZeroDCEEnhancer(model_path=args.model, device=args.device)
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_file():
        # Single image
        if args.compare:
            compare_enhancement_methods(str(input_path), output_dir)
        else:
            image = cv2.imread(str(input_path))
            enhanced = enhancer.enhance(image)
            output_path = output_dir / f"{input_path.stem}_enhanced.jpg"
            cv2.imwrite(str(output_path), enhanced)
            print(f"✅ Saved to {output_path}")
    
    elif input_path.is_dir():
        # Batch processing
        image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
        
        from tqdm import tqdm
        for img_path in tqdm(image_files, desc='Enhancing images'):
            image = cv2.imread(str(img_path))
            enhanced = enhancer.enhance(image)
            output_path = output_dir / f"{img_path.stem}_enhanced.jpg"
            cv2.imwrite(str(output_path), enhanced)
        
        print(f"✅ Processed {len(image_files)} images to {output_dir}")
    
    else:
        print(f"❌ Invalid input path: {input_path}")
