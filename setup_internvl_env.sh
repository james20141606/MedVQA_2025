#!/bin/bash

# Setup script for InternVL3.5 conda environment
# This script creates a new conda environment with the correct dependencies

echo "🚀 Setting up InternVL3.5 conda environment..."

# Create new conda environment
echo "📦 Creating conda environment 'internvl'..."
conda create -n internvl python=3.10 -y

# Activate the environment
echo "🔧 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate internvl

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA support..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install transformers from source (required for InternVL3.5)
echo "🔄 Installing transformers from source..."
pip install git+https://github.com/huggingface/transformers.git

# Install other required packages
echo "📚 Installing additional dependencies..."
pip install accelerate
pip install safetensors
pip install pillow
pip install sentencepiece
pip install timm
# Install optional but recommended packages
echo "✨ Installing optional packages..."
pip install einops
pip install flash-attn --no-build-isolation

# Verify installation

echo ""
echo "🎉 InternVL3.5 environment setup complete!"
echo ""
echo "To activate the environment:"
echo "  conda activate internvl"
echo ""
echo "To test the environment:"
echo "  python -c \"from transformers import AutoModelForCausalLM; print('✅ Transformers working!')\""
echo ""
echo "To deactivate:"
echo "  conda deactivate"
