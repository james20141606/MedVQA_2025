#!/bin/bash

# Simplified setup script for InternVL3.5 conda environment (HPC version)
# This script creates a new conda environment with minimal but sufficient dependencies

echo "🚀 Setting up InternVL3.5 conda environment (HPC version)..."

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

# Install essential packages
echo "📚 Installing essential dependencies..."
pip install accelerate
pip install safetensors
pip install pillow

# Verify installation
echo "✅ Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "🎉 InternVL3.5 environment setup complete!"
echo ""
echo "To activate the environment:"
echo "  conda activate internvl"
echo ""
echo "To test InternVL3.5:"
echo "  python -c \"from transformers import AutoModelForCausalLM; print('✅ Ready for InternVL3.5!')\""
