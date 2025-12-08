#!/bin/bash
# Setup environment for Zero-DCE++ system
# Handles PyTorch 2.6 compatibility issues

# Set PYTORCH environment variables
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TORCH_WEIGHTS_ONLY=0

echo "✅ Environment configured for Zero-DCE++ system"
echo "   - MPS fallback enabled"
echo "   - PyTorch 2.6 weights compatibility mode enabled"
echo ""
echo "Run your commands now:"
echo "  python test_zero_dce.py"
echo "  python scripts/hybrid_detector.py <image.jpg> --mode adaptive"
