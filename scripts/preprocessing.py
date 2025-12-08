"""
CLAHE (Contrast Limited Adaptive Histogram Equalization) Preprocessing
For low-light image enhancement before YOLO detection.

CLAHE improves local contrast and enhances details in low-light images
without amplifying noise as much as global histogram equalization.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple
import argparse


class CLAHEPreprocessor:
    """
    CLAHE image preprocessing for low-light enhancement.
    
    Parameters:
    -----------
    clip_limit : float
        Threshold for contrast limiting (default: 2.0)
        Higher values = more contrast enhancement
        Recommended range: 1.0-4.0
    
    tile_grid_size : tuple
        Size of grid for histogram equalization (default: (8, 8))
        Smaller tiles = more local adaptation but potentially more noise
        Recommended: (8, 8) for general use, (4, 4) for more aggressive
    
    apply_to_rgb : bool
        If True, applies CLAHE to RGB channels separately
        If False, applies only to luminance (recommended for natural look)
    """
    
    def __init__(
        self, 
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        apply_to_rgb: bool = False
    ):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.apply_to_rgb = apply_to_rgb
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE preprocessing to an image.
        
        Parameters:
        -----------
        image : np.ndarray
            Input image (BGR format from cv2.imread)
        
        Returns:
        --------
        np.ndarray
            Enhanced image in BGR format
        """
        if self.apply_to_rgb:
            # Apply CLAHE to each channel separately
            channels = cv2.split(image)
            channels_enhanced = [self.clahe.apply(ch) for ch in channels]
            return cv2.merge(channels_enhanced)
        else:
            # Apply CLAHE to luminance channel only (more natural)
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            l_enhanced = self.clahe.apply(l)
            
            # Merge and convert back to BGR
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    def process_file(
        self, 
        input_path: Union[str, Path],
        output_path: Union[str, Path] = None,
        show_comparison: bool = False
    ) -> np.ndarray:
        """
        Process a single image file.
        
        Parameters:
        -----------
        input_path : str or Path
            Path to input image
        output_path : str or Path, optional
            Path to save enhanced image (if None, returns array only)
        show_comparison : bool
            If True, displays before/after comparison
        
        Returns:
        --------
        np.ndarray
            Enhanced image
        """
        # Read image
        image = cv2.imread(str(input_path))
        if image is None:
            raise ValueError(f"Could not read image: {input_path}")
        
        # Process
        enhanced = self.process(image)
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), enhanced)
            print(f"Saved enhanced image to: {output_path}")
        
        # Show comparison if requested
        if show_comparison:
            self.show_comparison(image, enhanced)
        
        return enhanced
    
    def process_batch(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        extensions: Tuple[str] = ('.jpg', '.jpeg', '.png', '.bmp')
    ):
        """
        Process all images in a directory.
        
        Parameters:
        -----------
        input_dir : str or Path
            Directory containing input images
        output_dir : str or Path
            Directory to save enhanced images
        extensions : tuple
            Image file extensions to process
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_files = []
        for ext in extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return
        
        print(f"Processing {len(image_files)} images...")
        for img_path in image_files:
            output_path = output_dir / img_path.name
            try:
                self.process_file(img_path, output_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        print(f"Batch processing complete. Results in: {output_dir}")
    
    @staticmethod
    def show_comparison(original: np.ndarray, enhanced: np.ndarray):
        """Display side-by-side comparison of original and enhanced images."""
        import matplotlib.pyplot as plt
        
        # Convert BGR to RGB for display
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(original_rgb)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(enhanced_rgb)
        axes[1].set_title('CLAHE Enhanced')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: int = 8,
    apply_to_rgb: bool = False
) -> np.ndarray:
    """
    Convenience function to apply CLAHE to an image.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image (BGR format)
    clip_limit : float
        CLAHE clip limit (default: 2.0)
    tile_size : int
        CLAHE tile grid size (default: 8)
    apply_to_rgb : bool
        Apply to RGB channels or luminance only (default: False)
    
    Returns:
    --------
    np.ndarray
        Enhanced image in BGR format
    
    Example:
    --------
    >>> image = cv2.imread('dark_image.jpg')
    >>> enhanced = apply_clahe(image, clip_limit=2.0, tile_size=8)
    >>> cv2.imwrite('enhanced.jpg', enhanced)
    """
    preprocessor = CLAHEPreprocessor(
        clip_limit=clip_limit,
        tile_grid_size=(tile_size, tile_size),
        apply_to_rgb=apply_to_rgb
    )
    return preprocessor.process(image)


def main():
    """Command-line interface for CLAHE preprocessing."""
    parser = argparse.ArgumentParser(
        description='Apply CLAHE preprocessing to low-light images'
    )
    parser.add_argument(
        'input',
        type=str,
        help='Input image file or directory'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file or directory',
        required=True
    )
    parser.add_argument(
        '-c', '--clip-limit',
        type=float,
        default=2.0,
        help='CLAHE clip limit (default: 2.0, range: 1.0-4.0)'
    )
    parser.add_argument(
        '-t', '--tile-size',
        type=int,
        default=8,
        help='CLAHE tile grid size (default: 8, creates 8x8 grid)'
    )
    parser.add_argument(
        '--rgb',
        action='store_true',
        help='Apply CLAHE to RGB channels (default: luminance only)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show before/after comparison (single image only)'
    )
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = CLAHEPreprocessor(
        clip_limit=args.clip_limit,
        tile_grid_size=(args.tile_size, args.tile_size),
        apply_to_rgb=args.rgb
    )
    
    # Process
    input_path = Path(args.input)
    if input_path.is_file():
        # Single file
        preprocessor.process_file(
            input_path,
            args.output,
            show_comparison=args.show
        )
    elif input_path.is_dir():
        # Batch processing
        preprocessor.process_batch(input_path, args.output)
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
