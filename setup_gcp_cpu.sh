#!/bin/bash
# Setup script for Google Cloud Platform CPU-only instance (optimized for maximum parallelization)
# Run this script on your GCP instance after creating it

set -e

# Set non-interactive mode to avoid prompts during installation
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a

echo "=========================================="
echo "Spot RL GCP CPU-Only Setup Script"
echo "Optimized for maximum parallel environments"
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
sudo DEBIAN_FRONTEND=noninteractive apt-get update

# Install Python and dependencies (Debian-compatible)
# Also install graphics libraries needed for MuJoCo (OpenGL)
if [ "$OS" == "debian" ]; then
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
        -o Dpkg::Options::="--force-confold" \
        python3 python3-venv python3-pip git screen tmux wget curl htop \
        libgl1-mesa-glx libglib2.0-0 libgomp1 libegl1-mesa libxrandr2 libxss1 libxcursor1 \
        libxcomposite1 libasound2 libxi6 libxtst6
    PYTHON_CMD="python3"
else
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
        -o Dpkg::Options::="--force-confold" \
        python3.10 python3.10-venv python3-pip git screen tmux wget curl htop \
        libgl1-mesa-glx libglib2.0-0 libgomp1 libegl1-mesa libxrandr2 libxss1 libxcursor1 \
        libxcomposite1 libasound2 libxi6 libxtst6
    PYTHON_CMD="python3.10"
fi

# Get CPU count for optimization
CPU_COUNT=$(nproc)
echo "Detected $CPU_COUNT CPU cores"

# Create project directory
PROJECT_DIR="$HOME/spot-rl"
echo "Creating project directory: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Create virtual environment
echo "Creating Python virtual environment..."
$PYTHON_CMD -m venv venv
source venv/bin/activate

# Set up environment variables for headless rendering (MuJoCo)
echo "Setting up environment variables for headless rendering..."
export MUJOCO_GL=egl
export DISPLAY=:99
# Add to bashrc for persistence
if ! grep -q "MUJOCO_GL=egl" ~/.bashrc 2>/dev/null; then
    echo "" >> ~/.bashrc
    echo "# MuJoCo headless rendering" >> ~/.bashrc
    echo "export MUJOCO_GL=egl" >> ~/.bashrc
    echo "export DISPLAY=:99" >> ~/.bashrc
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU-only version - much smaller and faster to install)
echo "Installing PyTorch (CPU-only)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies (no JAX/MJX needed for CPU-only)
echo "Installing other dependencies..."
pip install gymnasium mujoco stable-baselines3[extra] robot_descriptions tensorboard

# Verify installations
echo ""
echo "=========================================="
echo "Verifying installations..."
echo "=========================================="

echo "Python version:"
python3 --version

echo "PyTorch:"
python3 -c "import torch; print('  Version:', torch.__version__); print('  CPU available: True')" || echo "  ERROR: PyTorch not working"

echo "MuJoCo:"
python3 -c "import mujoco; print('  OK')" || echo "  ERROR: MuJoCo not working"

echo "Stable-Baselines3:"
python3 -c "import stable_baselines3; print('  OK')" || echo "  ERROR: Stable-Baselines3 not working"

echo ""
echo "=========================================="
echo "CPU Optimization Recommendations"
echo "=========================================="
echo ""
echo "Your instance has $CPU_COUNT CPU cores."
echo ""
echo "Recommended training command:"
echo "  python train_colab.py \\"
echo "    --total_timesteps 30000000 \\"
echo "    --model_name ppo_spot_gcp_cpu \\"
echo "    --output_dir ./models \\"
echo "    --tensorboard_log ./logs \\"
echo "    --device cpu \\"
echo "    --num_envs $CPU_COUNT \\"
echo "    --n_steps 4096 \\"
echo "    --batch_size 256 \\"
echo "    --n_epochs 4"
echo ""
echo "Or for maximum speed (if you have enough RAM):"
echo "  python train_colab.py \\"
echo "    --total_timesteps 30000000 \\"
echo "    --model_name ppo_spot_gcp_cpu \\"
echo "    --output_dir ./models \\"
echo "    --tensorboard_log ./logs \\"
echo "    --device cpu \\"
echo "    --num_envs $((CPU_COUNT * 2)) \\"
echo "    --n_steps 4096 \\"
echo "    --batch_size 256 \\"
echo "    --n_epochs 4"
echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  cd $PROJECT_DIR"
echo "  source venv/bin/activate"
echo ""
echo "To start training, run the command above."
echo ""
echo "To run in background with screen:"
echo "  screen -S training"
echo "  source venv/bin/activate"
echo "  python train_colab.py ..."
echo "  # Press Ctrl+A, then D to detach"
echo ""
echo "To monitor CPU usage:"
echo "  htop"
echo ""

