import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ReflectedConvolution(nn.Module):
    def __init__(self, kernel_nums=8, kernel_size=3):
        super(ReflectedConvolution, self).__init__()
        self.kernel_nums = kernel_nums
        self.kernel_size = kernel_size
        self.rg_bn = nn.BatchNorm2d(kernel_nums)
        self.gb_bn = nn.BatchNorm2d(kernel_nums)
        self.rb_bn = nn.BatchNorm2d(kernel_nums)
        self.filter = torch.nn.Parameter(torch.randn(self.kernel_nums, 1, self.kernel_size, self.kernel_size))
        
        self.init_weights()

    def init_weights(self):
        torch.nn.init.kaiming_normal_(self.filter)
        torch.nn.init.constant_(self.rg_bn.weight, 0.01)
        torch.nn.init.constant_(self.rg_bn.bias, 0)
        torch.nn.init.constant_(self.gb_bn.weight, 0.01)
        torch.nn.init.constant_(self.gb_bn.bias, 0)
        torch.nn.init.constant_(self.rb_bn.weight, 0.01)
        torch.nn.init.constant_(self.rb_bn.bias, 0)

    def mean_constraint(self, kernel):
        bs, cin, kw, kh = kernel.shape
        kernel_mean = torch.mean(kernel.view(bs, -1), dim=1, keepdim=True)
        kernel = (kernel.view(bs, -1) - kernel_mean)
        return kernel.view(bs, cin, kw, kh)

    def forward(self, img):
        # Expecting img in [0, 1]
        zeroMasks = torch.zeros_like(img)
        zeroMasks[img == 0] = 1
        log_img = torch.log(img + 1e-7)

        red_chan = log_img[:, 0, :, :].unsqueeze(1)
        green_chan = log_img[:, 1, :, :].unsqueeze(1)
        blue_chan = log_img[:, 2, :, :].unsqueeze(1)
        
        normalized_filter = self.mean_constraint(self.filter)

        # Red-Green
        filt_r1 = F.conv2d(red_chan, weight=normalized_filter, padding=self.kernel_size//2)
        filt_g1 = F.conv2d(green_chan, weight=-normalized_filter, padding=self.kernel_size//2)
        filt_rg = self.rg_bn(filt_r1 + filt_g1)

        # Green-Blue
        filt_g2 = F.conv2d(green_chan, weight=normalized_filter, padding=self.kernel_size//2)
        filt_b1 = F.conv2d(blue_chan, weight=-normalized_filter, padding=self.kernel_size//2)
        filt_gb = self.gb_bn(filt_g2 + filt_b1)

        # Red-Blue
        filt_r2 = F.conv2d(red_chan, weight=normalized_filter, padding=self.kernel_size//2)
        filt_b2 = F.conv2d(blue_chan, weight=-normalized_filter, padding=self.kernel_size//2)
        filt_rb = self.rb_bn(filt_r2 + filt_b2)

        rg = torch.where(zeroMasks[:, 0:1, ...].expand(-1, self.kernel_nums, -1, -1)==1, 0, filt_rg)
        gb = torch.where(zeroMasks[:, 1:2, ...].expand(-1, self.kernel_nums, -1, -1)==1, 0, filt_gb)
        rb = torch.where(zeroMasks[:, 2:3, ...].expand(-1, self.kernel_nums, -1, -1)==1, 0, filt_rb)
        
        out = torch.cat([rg, gb, rb], dim=1)
        return out

class YOLAModule(nn.Module):
    """
    YOLA: You Only Look Around - Low-light Enhancement Module
    Based on NeurIPS 2024 paper: https://github.com/MingboHong/YOLA
    """
    def __init__(self, kernel_nums=8, kernel_size=3):
        super(YOLAModule, self).__init__()
        
        # Projects 3-channel input to 24 feature channels
        self.feat_projector = nn.Sequential(
            nn.Conv2d(3, 24, 3, 1, 1, groups=1), 
            nn.BatchNorm2d(24),
            nn.LeakyReLU()
        )
        
        # Fuses concatenated features (24 from projector + 24 from iim = 48) -> 3 output channels
        # With groups=2, weight shape is [out, in/groups, H, W] = [32, 24, 3, 3]
        self.fuse_net = nn.Sequential(
            nn.Conv2d(48, 32, 3, 1, 1, groups=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 3, 3, 1, 1, groups=1)
        )
        
        # Illumination Invariant Module (Reflected Convolution)
        self.iim = ReflectedConvolution(kernel_nums, kernel_size)

    def forward(self, x):
        """
        Forward pass - matches official IIBlock implementation.
        Args:
            x (Tensor): Input image batch (B, 3, H, W) in range [0, 1]
        Returns:
            Tensor: Enhanced image (B, 3, H, W) in range [0, 1]
        """
        # 1. Extract illumination invariant features (24 channels from 3*8 kernel_nums)
        feat_ii = self.iim(x)
        
        # 2. Project original features (24 channels)
        feats = self.feat_projector(x)
        
        # 3. Concatenate (24 + 24 = 48 channels)
        feats_ = torch.cat((feats, feat_ii), dim=1)
        
        # 4. Fuse to generate enhanced image
        x_out = self.fuse_net(feats_)
        
        # 5. Clamp output to valid range
        x_out = torch.clamp(x_out, 0, 1)
        
        return x_out

    def load_weights_from_mmdet(self, state_dict):
        """
        Helper to load weights from the official MMDetection checkpoint.
        The official checkpoint has keys like 'iim.filter', 'feat_projector.0.weight', etc.
        """
        own_state = self.state_dict()
        for name, param in state_dict.items():
            # Filter out keys that don't belong to this module
            # In official repo, this module is usually 'model.iim' or similar.
            # We assume the state_dict passed here is already filtered or has matching keys.
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                own_state[name].copy_(param)

import cv2
from pathlib import Path

class YOLAEnhancer:
    """
    High-level interface for YOLA image enhancement
    """
    def __init__(self, model_path=None, device=None):
        self.device = self._setup_device(device)
        self.model = YOLAModule().to(self.device)
        
        if model_path:
            # Convert to absolute path
            model_path = Path(model_path)
            if not model_path.is_absolute():
                # Try relative to script directory first
                script_dir = Path(__file__).parent.parent
                abs_path = script_dir / model_path
                if abs_path.exists():
                    model_path = abs_path
            
            if model_path.exists():
                self.load_weights(str(model_path))
            else:
                print(f"⚠️  YOLA weights file not found: {model_path}")
                print(f"   Checked absolute path: {model_path.absolute()}")
                print("   Using random initialization")
        else:
            print("⚠️  No YOLA weights path provided - using random initialization")
        
        self.model.eval()

    def _setup_device(self, device):
        if device and device != 'auto':
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    def load_weights(self, model_path):
        try:
            # Load checkpoint
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Handle MMDetection checkpoint structure
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Check if keys match directly (converted weights)
            own_keys = set(self.model.state_dict().keys())
            checkpoint_keys = set(state_dict.keys())
            
            if own_keys == checkpoint_keys:
                # Direct match - use simple loading
                self.model.load_state_dict(state_dict, strict=True)
                print(f"✅ YOLA: Successfully loaded all {len(own_keys)} weights.")
                return
            
            # Try to find and strip prefix for MMDetection checkpoints
            detected_prefix = ""
            known_key_suffix = "feat_projector.0.weight"
            
            for key in state_dict.keys():
                if key.endswith(known_key_suffix):
                    detected_prefix = key[:-len(known_key_suffix)]
                    print(f"✅ Auto-detected prefix: '{detected_prefix}'")
                    break
            
            # Build new state dict with stripped prefix
            new_state_dict = {}
            for key, value in state_dict.items():
                clean_key = key
                if detected_prefix and key.startswith(detected_prefix):
                    clean_key = key[len(detected_prefix):]
                
                if clean_key in own_keys:
                    new_state_dict[clean_key] = value
            
            if len(new_state_dict) > 0:
                self.model.load_state_dict(new_state_dict, strict=False)
                print(f"✅ YOLA: Successfully loaded {len(new_state_dict)}/{len(own_keys)} weights.")
            else:
                print("❌ YOLA: No matching weights found in checkpoint!")
                print(f"   Checkpoint keys (first 5): {list(state_dict.keys())[:5]}")
                
        except Exception as e:
            print(f"Error loading weights: {e}")

    def enhance(self, image, return_numpy=True):
        if isinstance(image, np.ndarray):
            input_image = self._numpy_to_torch(image)
        else:
            input_image = image
            
        with torch.no_grad():
            enhanced = self.model(input_image)
            
        if return_numpy:
            return self._torch_to_numpy(enhanced)
        return enhanced

    def _numpy_to_torch(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_float = image_rgb.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_float).permute(2, 0, 1)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        return image_tensor

    def _torch_to_numpy(self, tensor):
        tensor = tensor.squeeze(0)
        image = tensor.permute(1, 2, 0).cpu().numpy()
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
