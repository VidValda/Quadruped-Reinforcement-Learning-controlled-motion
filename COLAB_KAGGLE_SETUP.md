# Training on Google Colab / Kaggle with GPU

This guide explains how to train the Spot robot reinforcement learning model on Google Colab or Kaggle with GPU acceleration.

**Two versions available:**

- `train_colab.py` - Standard version using regular MuJoCo (CPU simulation)
- `train_colab_mjx.py` - **Recommended** - Uses MuJoCo MJX for GPU-accelerated environment simulation (much faster!)

## Quick Start

### Google Colab

#### Option 1: MJX (GPU-accelerated environments) - RECOMMENDED ⚡

1. **Open a new Colab notebook** and enable GPU:

   - Runtime → Change runtime type → Hardware accelerator: GPU

2. **Install dependencies**:

   ```python
   !pip install gymnasium mujoco[mjx] jax[cuda12] pybullet stable-baselines3[extra] torch robot_descriptions
   ```

3. **Upload the script** or clone the repository:

   ```python
   # Option 1: Upload train_colab_mjx.py to Colab
   # Option 2: Clone repository
   !git clone <your-repo-url>
   %cd <repo-directory>
   ```

4. **Run training with MJX**:

   ```python
   !python train_colab_mjx.py --total_timesteps 10000000 --model_name ppo_spot_mjx --tensorboard_log ./logs
   ```

5. **Save to Google Drive** (optional):

   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   !python train_colab_mjx.py \
     --total_timesteps 10000000 \
     --model_name ppo_spot_mjx \
     --output_dir /content/drive/MyDrive/spot_models \
     --tensorboard_log /content/drive/MyDrive/spot_logs
   ```

#### Option 2: Standard MuJoCo (CPU environments)

1. **Open a new Colab notebook** and enable GPU:

   - Runtime → Change runtime type → Hardware accelerator: GPU

2. **Install dependencies**:

   ```python
   !pip install gymnasium mujoco pybullet stable-baselines3[extra] torch robot_descriptions
   ```

3. **Upload the script** or clone the repository:

   ```python
   # Option 1: Upload train_colab.py to Colab
   # Option 2: Clone repository
   !git clone <your-repo-url>
   %cd <repo-directory>
   ```

4. **Run training**:
   ```python
   !python train_colab.py --total_timesteps 10000000 --model_name ppo_spot_colab --tensorboard_log ./logs
   ```

### Kaggle

#### Option 1: MJX (GPU-accelerated) - RECOMMENDED ⚡

1. **Create a new notebook** and enable GPU:

   - Settings → Accelerator: GPU

2. **Add data source** (if needed):

   - Upload `train_colab_mjx.py` as a dataset or add it directly to the notebook

3. **Install dependencies** in a code cell:

   ```python
   !pip install gymnasium mujoco[mjx] jax[cuda12] pybullet stable-baselines3[extra] torch robot_descriptions
   ```

4. **Run training**:
   ```python
   !python train_colab_mjx.py --total_timesteps 10000000 --model_name ppo_spot_mjx
   ```

#### Option 2: Standard MuJoCo

1. **Create a new notebook** and enable GPU:

   - Settings → Accelerator: GPU

2. **Add data source** (if needed):

   - Upload `train_colab.py` as a dataset or add it directly to the notebook

3. **Install dependencies** in a code cell:

   ```python
   !pip install gymnasium mujoco pybullet stable-baselines3[extra] torch robot_descriptions
   ```

4. **Run training**:
   ```python
   !python train_colab.py --total_timesteps 10000000 --model_name ppo_spot_kaggle
   ```

## Command Line Arguments

```
--total_timesteps    Total number of timesteps to train (default: 30,000,000)
--model_name         Name for saved model files (default: ppo_spot_colab)
--output_dir         Directory to save models and stats (default: ./models)
--tensorboard_log    Directory for tensorboard logs (default: None)
--num_envs           Number of parallel environments (default: CPU count)
--device             Device: 'cuda', 'cpu', or 'auto' (default: auto-detect)
--n_steps            Number of steps per update (default: 2048)
--batch_size         Batch size (default: 64)
--n_epochs           Number of epochs per update (default: 10)
--learning_rate      Learning rate (default: 3e-4)
```

## Example Usage

### Basic training with default settings:

```bash
python train_colab.py
```

### Custom training configuration:

```bash
python train_colab.py \
  --total_timesteps 5000000 \
  --model_name my_spot_model \
  --n_steps 4096 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --num_envs 8
```

### Training with TensorBoard logging:

```bash
python train_colab.py \
  --total_timesteps 10000000 \
  --tensorboard_log ./tensorboard_logs \
  --model_name ppo_spot_v1
```

## GPU Detection

The script automatically detects GPU availability:

- If CUDA is available, it uses GPU (`cuda`)
- Otherwise, it falls back to CPU

You can manually specify the device:

```bash
python train_colab.py --device cuda  # Force GPU
python train_colab.py --device cpu   # Force CPU
```

## Output Files

After training, the script saves:

- `{model_name}.zip` - The trained PPO model
- `{model_name}_stats.pkl` - VecNormalize statistics for the environment

## MJX vs Standard MuJoCo

**MJX (train_colab_mjx.py)** - **RECOMMENDED for speed** ⚡:

- ✅ GPU-accelerated environment simulation (10-100x faster)
- ✅ Environments run on GPU via JAX
- ✅ Significantly reduces training time, especially with many parallel environments
- ✅ Better utilization of GPU resources
- ⚠️ Requires JAX and CUDA setup
- ⚠️ Slightly more complex installation

**Standard MuJoCo (train_colab.py)**:

- ✅ Simpler installation
- ✅ More stable/compatible
- ❌ Environment simulation runs on CPU (bottleneck)
- ❌ Slower training, especially with many environments

**Recommendation:** Use `train_colab_mjx.py` for faster training, especially with many parallel environments. The environment bottleneck is the main performance issue, and MJX solves this by running simulations on GPU.

## Tips for Colab/Kaggle

1. **Session Timeouts**: Colab free tier has session timeouts. Save checkpoints frequently or use Colab Pro.

2. **Memory Management**:

   - Reduce `num_envs` if you run out of memory
   - Reduce `batch_size` if needed
   - MJX uses GPU memory for environments, monitor with `!nvidia-smi`

3. **Progress Monitoring**:

   - Use TensorBoard to monitor training:
     ```python
     # In Colab
     %load_ext tensorboard
     %tensorboard --logdir ./logs
     ```

4. **Saving Models**:

   - Always save to Google Drive in Colab to persist models
   - In Kaggle, download models before session ends

5. **Resume Training**:

   - The script doesn't support resuming yet, but you can load a saved model and continue training with Stable-Baselines3 API

6. **MJX Performance**:
   - MJX environments automatically use GPU via JAX
   - PPO model training can also use GPU (separate from MJX)
   - Monitor GPU usage: `!nvidia-smi` in Colab
   - For best performance, use more parallel environments with MJX

## Troubleshooting

### Import Errors

If you get import errors, make sure all dependencies are installed:

```bash
pip install gymnasium mujoco pybullet stable-baselines3[extra] torch robot_descriptions
```

### GPU Not Detected

- Check that GPU is enabled in runtime settings
- For PPO: `!python -c "import torch; print(torch.cuda.is_available())"`
- For MJX: `!python -c "import jax; print(jax.devices())"` - should show GPU devices
- MJX requires JAX with CUDA: `pip install jax[cuda12]` or `jax[cuda11]` depending on CUDA version
- If JAX doesn't detect GPU, try: `pip install --upgrade jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`

### Out of Memory

- Reduce `num_envs` (fewer parallel environments)
- Reduce `batch_size`
- Reduce `n_steps`

### Model Loading Issues

- Ensure `robot_descriptions` package is installed
- The script will fall back to default model if XML loading fails

## Performance Tips

1. **Use more parallel environments** for faster training (if memory allows)
2. **Larger batch sizes** can improve stability (if memory allows)
3. **GPU is most beneficial** for large batch sizes and many environments
4. **Monitor GPU usage**: Use `nvidia-smi` or Colab's GPU monitor
