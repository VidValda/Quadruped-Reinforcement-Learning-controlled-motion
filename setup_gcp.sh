#!/bin/bash
# Setup script for Google Cloud Platform GPU instance (Debian/Ubuntu compatible)
# Run this script on your GCP instance after creating it

set -e

echo "=========================================="
echo "Spot RL GCP GPU Setup Script"
echo "=========================================="

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
   echo "Please do not run as root. Run as your user account."
   exit 1
fi

# Detect OS
if [ -f /etc/debian_version ]; then
    OS="debian"
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    echo "Detected Debian with Python $PYTHON_VERSION"
else
    OS="ubuntu"
    PYTHON_VERSION="3.10"
    echo "Detected Ubuntu"
fi

# Update system packages
echo "Updating system packages..."
sudo apt-get update

# Install Python and dependencies (Debian-compatible)
if [ "$OS" == "debian" ]; then
    sudo apt-get install -y python3 python3-venv python3-pip git screen tmux wget curl
    PYTHON_CMD="python3"
else
    sudo apt-get install -y python3.10 python3.10-venv python3-pip git screen tmux wget curl
    PYTHON_CMD="python3.10"
fi

# Check NVIDIA GPU
echo "Checking NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1)
    echo "Detected CUDA Version: $CUDA_VERSION"
else
    echo "WARNING: nvidia-smi not found. GPU drivers may not be installed."
    echo "Please install NVIDIA drivers first."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    CUDA_VERSION="12"
fi

# Create project directory
PROJECT_DIR="$HOME/spot-rl"
echo "Creating project directory: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Create virtual environment
echo "Creating Python virtual environment..."
$PYTHON_CMD -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install JAX with CUDA support
echo "Installing JAX with CUDA support..."
if [ "$CUDA_VERSION" == "11" ]; then
    pip install --upgrade "jax[cuda11]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
else
    # Default to CUDA 12
    pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
fi

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
echo "Installing other dependencies..."
pip install gymnasium mujoco-mjx stable-baselines3[extra] robot_descriptions tensorboard

# Verify installations
echo ""
echo "=========================================="
echo "Verifying installations..."
echo "=========================================="

echo "JAX devices:"
python3 -c "import jax; print('  ', jax.devices())" || echo "  ERROR: JAX not working"

echo "PyTorch CUDA:"
python3 -c "import torch; print('  Available:', torch.cuda.is_available()); print('  Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" || echo "  ERROR: PyTorch not working"

echo "MuJoCo MJX:"
python3 -c "import mujoco.mjx as mjx; print('  OK')" || echo "  ERROR: MJX not working"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  cd $PROJECT_DIR"
echo "  source venv/bin/activate"
echo ""
echo "To start training, run:"
echo "  python train_colab_mjx.py --total_timesteps 30000000 --model_name ppo_spot_mjx_gcp --device cuda"
echo ""
echo "To run in background with screen:"
echo "  screen -S training"
echo "  source venv/bin/activate"
echo "  python train_colab_mjx.py ..."
echo "  # Press Ctrl+A, then D to detach"
echo ""