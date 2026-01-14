# Google Cloud Platform CPU-Only Setup Guide

This guide shows how to set up a **CPU-only** GCP instance optimized for maximum parallel training. This is **cheaper and often faster** than GPU instances for MLP policies.

## Why CPU-Only?

- ✅ **Cheaper**: ~$0.48/hour vs ~$2.67/hour (5x cheaper!)
- ✅ **Faster for MLP policies**: CPU is actually faster than GPU for small networks
- ✅ **Better parallelization**: Can use all CPU cores for environments
- ✅ **No GPU overhead**: No data transfer delays

## Step 1: Create CPU-Only Instance

### Using Google Cloud Console

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **Compute Engine** → **VM instances**
3. Click **Create Instance**
4. Configure:
   - **Name**: `spot-rl-cpu` (or your preferred name)
   - **Machine type**:
     - **Recommended**: `n1-highcpu-16` (16 vCPUs, 14.4 GB RAM)
     - **Maximum**: `n1-highcpu-32` (32 vCPUs, 28.8 GB RAM)
     - **Budget**: `n1-highcpu-8` (8 vCPUs, 7.2 GB RAM)
   - **Boot disk**:
     - OS: **Debian 12** or **Ubuntu 22.04 LTS**
     - Disk size: 50 GB minimum (100 GB recommended)
     - Disk type: Standard Persistent Disk or SSD
   - **Firewall**: Check "Allow HTTP traffic" and "Allow HTTPS traffic"
   - **NO GPU NEEDED** - Skip GPU section entirely
5. Click **Create**

### Using gcloud CLI

```bash
# Set your project ID
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Create high-CPU instance (16 cores)
gcloud compute instances create spot-rl-cpu \
  --zone=us-central1-c \
  --machine-type=n1-highcpu-16 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd
```

## Step 2: SSH into Instance

```bash
gcloud compute ssh spot-rl-cpu-max --zone=us-central1-c
```

## Step 3: Run Setup Script

```bash
# Upload the setup script first (from local machine)
# Or copy-paste the script content

# Make executable and run
chmod +x setup_gcp_cpu.sh
./setup_gcp_cpu.sh
```

## Step 4: Upload Training Script

From your local machine:

```bash
gcloud compute scp train_colab.py spot-rl-cpu:~/spot-rl/ --zone=us-central1-c
```

## Step 5: Start Training

SSH into instance and run:

```bash
cd ~/spot-rl
source venv/bin/activate

# Get CPU count
CPU_COUNT=$(nproc)
echo "Using $CPU_COUNT parallel environments"

# Start training
python train_colab.py \
  --total_timesteps 30000000 \
  --model_name ppo_spot_gcp_cpu \
  --output_dir ./models \
  --tensorboard_log ./logs \
  --device cpu \
  --num_envs $CPU_COUNT \
  --n_steps 4096 \
  --batch_size 256 \
  --n_epochs 4 \
  --learning_rate 3e-4
```

## Step 6: Download Generated Models to Local PC

After training completes, download the generated model files (.zip and .pkl) from the GCP instance to your local machine.

### Download Model Files

From your **local machine** (not SSH'd into the instance):

```bash
# Download all model files (.zip) from the models directory
gcloud compute scp spot-rl-cpu:~/spot-rl/models/*.zip ./models/ --zone=us-central1-c

# Download all stats files (.pkl) from the models directory
gcloud compute scp spot-rl-cpu:~/spot-rl/models/*.pkl ./models/ --zone=us-central1-c
```

### Download Specific Model

If you want to download a specific model (e.g., `ppo_spot_gcp_cpu`):

```bash
# Download specific model file
gcloud compute scp spot-rl-cpu-max:/home/david/Quadruped-Reinforcement-Learning-controlled-motion/models/ppo_spot_gcp_cpu.zip ./models/ --zone=us-central1-c
# Download corresponding stats file
gcloud compute scp spot-rl-cpu-max:/home/david/Quadruped-Reinforcement-Learning-controlled-motion/models/ppo_spot_gcp_cpu_stats.pkl ./models/ --zone=us-central1-c
```

### Download Entire Models Directory

To download the entire models directory (useful if you have multiple models):

```bash
# Download entire models directory recursively
gcloud compute scp --recurse spot-rl-cpu:~/spot-rl/models ./models --zone=us-central1-c
```

### Verify Downloads

After downloading, verify the files are on your local machine:

```bash
# List downloaded files
ls -lh models/

# You should see files like:
# - ppo_spot_gcp_cpu.zip (model file)
# - ppo_spot_gcp_cpu_stats.pkl (normalization stats)
```

**Note:** Make sure the local `./models/` directory exists before downloading, or the files will be saved in your current directory.

## Extra: stop and restart the instance

```bash
# Start
gcloud compute instances start spot-rl-cpu-max --zone=us-central1-c
```

```bash
gcloud compute ssh spot-rl-cpu-max --zone=us-central1-c
```

```bash
# Stop
gcloud compute instances stop spot-rl-cpu-max --zone=us-central1-c
```

## Optimal Configurations

### Configuration 1: Balanced (16 cores)

```bash
python train_colab.py \
  --total_timesteps 30000000 \
  --model_name ppo_spot_gcp_cpu \
  --output_dir ./models \
  --tensorboard_log ./logs \
  --device cpu \
  --num_envs 16 \
  --n_steps 4096 \
  --batch_size 256 \
  --n_epochs 4
```

**Instance**: n1-highcpu-16  
**Cost**: ~$0.48/hour  
**Speed**: ~10-20x faster than single environment

### Configuration 2: Maximum (32 cores)

```bash
python train_colab.py \
  --total_timesteps 30000000 \
  --model_name ppo_spot_gcp_cpu \
  --output_dir ./models \
  --tensorboard_log ./logs \
  --device cpu \
  --num_envs 32 \
  --n_steps 8192 \
  --batch_size 512 \
  --n_epochs 4
```

**Instance**: n1-highcpu-32  
**Cost**: ~$0.96/hour  
**Speed**: ~30-50x faster than single environment

### Configuration 3: Budget (8 cores)

```bash
python train_colab.py \
  --total_timesteps 30000000 \
  --model_name ppo_spot_gcp_cpu \
  --output_dir ./models \
  --tensorboard_log ./logs \
  --device cpu \
  --num_envs 8 \
  --n_steps 4096 \
  --batch_size 128 \
  --n_epochs 4
```

**Instance**: n1-highcpu-8  
**Cost**: ~$0.24/hour  
**Speed**: ~5-10x faster than single environment

### Configuration 4: Maximum Performance (Newer Machine Types)

Para **máximo procesamiento CPU**, las siguientes series ofrecen mejor rendimiento que N1:

#### 🚀 **C2 (Intel Cascade Lake) - ALTO RENDIMIENTO CON 60 CORES**

```bash
# c2-standard-60: 60 vCPUs, 240 GB RAM
gcloud compute instances create spot-rl-cpu-max \
  --zone=us-central1-c \
  --machine-type=c2-standard-60 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd
```

**Ventajas:**

- ✅ **60 vCPUs** - perfecto para 60 entornos paralelos
- ✅ **240 GB RAM** - suficiente para entrenamiento masivo
- ✅ **Intel Cascade Lake** - alto rendimiento consistente
- ✅ **Optimizado para AVX512** - aprovecha instrucciones vectoriales avanzadas

**Configuración de entrenamiento optimizada para 60 robots:**

```bash
# Con 60 robots: 60 * 1024 pasos = 61,440 muestras por iteración
python train_colab.py \
  --total_timesteps 30000000 \
  --model_name ppo_spot_gcp_c2 \
  --output_dir ./models \
  --tensorboard_log ./logs \
  --device cpu \
  --num_envs 60 \
  --n_steps 1024 \
  --batch_size 4096 \
  --n_epochs 10 \
  --learning_rate 5e-4
```

**Parámetros explicados:**

- `n_steps=1024`: Pasos que da CADA robot antes de actualizar (60 robots × 1024 = 61,440 muestras/iteración)
- `batch_size=4096`: Procesa 4096 muestras a la vez, aprovechando AVX512 de la CPU
- `n_epochs=10`: 10 épocas por actualización (mantiene balance sin sobre-entrenar)
- `learning_rate=5e-4`: LR ligeramente más alto que el default (3e-4) para batches grandes

**Opciones recomendadas C4D:**

- `c4d-highcpu-64`: 64 vCPUs, 512 GB RAM (equilibrado)
- `c4d-highcpu-96`: 96 vCPUs, 768 GB RAM (recomendado para máximo)
- `c4d-highcpu-128`: 128 vCPUs, 1,024 GB RAM (muy potente)
- `c4d-highcpu-192`: 192 vCPUs, 1,536 GB RAM (extremo)

#### ⚡ **C4 (Intel Emerald Rapids) - ALTO RENDIMIENTO**

```bash
# Máximo: 288 vCPUs, 2,232 GB RAM
# Ejemplo: c4-highcpu-64 (64 vCPUs, 512 GB RAM)
gcloud compute instances create spot-rl-cpu-c4 \
  --zone=us-central1-c \
  --machine-type=c4-highcpu-64 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB
```

**Ventajas:**

- ✅ **Hasta 288 vCPUs**
- ✅ **Intel Emerald Rapids** (última generación Intel)
- ✅ **Rendimiento consistente**

#### 💰 **N2D (AMD Milan) - MEJOR RELACIÓN PRECIO/RENDIMIENTO**

```bash
# Máximo: 224 vCPUs, 896 GB RAM
# Ejemplo: n2d-highcpu-64 (64 vCPUs, 64 GB RAM)
gcloud compute instances create spot-rl-cpu-n2d \
  --zone=us-central1-c \
  --machine-type=n2d-highcpu-64 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB
```

**Ventajas:**

- ✅ **Hasta 224 vCPUs**
- ✅ **AMD Milan** (buen rendimiento)
- ✅ **"Balanced price & performance"** - mejor relación costo/rendimiento
- ✅ **Más económico que C4/C4D**

**Opciones recomendadas N2D:**

- `n2d-highcpu-32`: 32 vCPUs, 32 GB RAM
- `n2d-highcpu-64`: 64 vCPUs, 64 GB RAM (recomendado)
- `n2d-highcpu-96`: 96 vCPUs, 96 GB RAM
- `n2d-highcpu-128`: 128 vCPUs, 128 GB RAM

#### 📊 **Comparación de Series para CPU-Only**

| Serie   | vCPUs Máx | CPU Platform         | RAM Máx  | Características               | Mejor Para                            |
| ------- | --------- | -------------------- | -------- | ----------------------------- | ------------------------------------- |
| **C4D** | **384**   | AMD Turin            | 3,072 GB | Consistently high performance | **Máximo rendimiento absoluto**       |
| **C4**  | 288       | Intel Emerald Rapids | 2,232 GB | Consistently high performance | Alto rendimiento Intel                |
| **C3D** | 360       | AMD Genoa            | 2,880 GB | Consistently high performance | Alto rendimiento (anterior gen)       |
| **C2**  | 60        | Intel Cascade Lake   | 240 GB   | High performance, AVX512      | **60 entornos paralelos optimizado**  |
| **N2D** | 224       | AMD Milan            | 896 GB   | Balanced price & performance  | **Mejor relación precio/rendimiento** |
| **N4D** | 96        | AMD Turin            | 768 GB   | Flexible & cost-optimized     | Equilibrado moderno                   |
| **N1**  | 96        | Intel Haswell        | 624 GB   | Balanced (legacy)             | Presupuesto limitado                  |

**Recomendación final:**

- **60 entornos paralelos optimizado**: **C2** (c2-standard-60) - **RECOMENDADO para entrenamiento con 60 robots**
- **Máximo rendimiento**: **C4D** (c4d-highcpu-96 o superior)
- **Mejor relación precio/rendimiento**: **N2D** (n2d-highcpu-64)
- **Presupuesto limitado**: **N1** (n1-highcpu-32)

**Nota sobre memoria:** Cada entorno paralelo usa ~500MB-1GB RAM. Para 60 entornos necesitas ~30-60 GB RAM mínimo. La máquina c2-standard-60 tiene 240 GB RAM, más que suficiente para esta configuración.

## Running in Background

### Using screen

```bash
# Start screen session
screen -S training

# Activate environment and start training
cd ~/spot-rl
source venv/bin/activate
python train_colab.py --device cpu --num_envs 16 ...

# Detach: Ctrl+A, then D
# Reattach: screen -r training
```

### Using nohup

```bash
cd ~/spot-rl
source venv/bin/activate
nohup python train_colab.py \
  --device cpu \
  --num_envs 16 \
  --total_timesteps 30000000 \
  > training.log 2>&1 &

# Monitor
tail -f training.log
```

## Monitoring

### Check CPU Usage

```bash
# Real-time CPU monitoring
htop

# Or
top
```

### Check Training Progress

```bash
# View logs
tail -f training.log

# Check model files
ls -lh models/
```

### View TensorBoard Logs

#### Option A: Port Forwarding (Recommended)

Desde tu máquina local:

```bash
# Primero, matar cualquier proceso de TensorBoard existente
gcloud compute ssh spot-rl-cpu --zone=us-central1-a \
  --command="pkill -f tensorboard || kill \$(lsof -ti:6006) 2>/dev/null || true"

# Luego ejecutar TensorBoard con port forwarding
gcloud compute ssh spot-rl-cpu --zone=us-central1-a \
  --ssh-flag="-L 6006:localhost:6006" \
  --command="cd ~/spot-rl && source venv/bin/activate && tensorboard --logdir=./logs --host=0.0.0.0 --port=6006"
```

Luego abre `http://localhost:6006` en tu navegador.

**Si el puerto 6006 está ocupado:**

```bash
# Opción 1: Matar el proceso que usa el puerto
gcloud compute ssh spot-rl-cpu --zone=us-central1-a \
  --command="pkill -f tensorboard; kill \$(lsof -ti:6006) 2>/dev/null || true"

# Opción 2: Usar un puerto diferente (ej: 6007)
gcloud compute ssh spot-rl-cpu --zone=us-central1-a \
  --ssh-flag="-L 6007:localhost:6007" \
  --command="cd ~/spot-rl && source venv/bin/activate && tensorboard --logdir=./logs --host=0.0.0.0 --port=6007"
```

#### Option B: Run TensorBoard on Instance in Background

```bash
# Conectarse a la instancia
gcloud compute ssh spot-rl-cpu --zone=us-central1-a

# En la instancia, ejecutar TensorBoard en background
cd ~/spot-rl
source venv/bin/activate
pkill -f tensorboard || true
nohup tensorboard --logdir=./logs --host=0.0.0.0 --port=6006 > /tmp/tensorboard.log 2>&1 &

# Verificar que está corriendo
tail -f /tmp/tensorboard.log

# Salir de la instancia (Ctrl+D)

# Desde tu máquina local, hacer port forwarding
gcloud compute ssh spot-rl-cpu --zone=us-central1-a --ssh-flag="-L 6006:localhost:6006" -N
```

## Cost Comparison

### Series N1 (Legacy - Más Económica)

| Instance Type        | Cores | RAM     | Cost/hour\* | Best For                 |
| -------------------- | ----- | ------- | ----------- | ------------------------ |
| n1-highcpu-8         | 8     | 7.2 GB  | ~$0.24      | Budget training          |
| n1-highcpu-16        | 16    | 14.4 GB | ~$0.48      | **Recomendado (budget)** |
| n1-highcpu-32        | 32    | 28.8 GB | ~$0.96      | Máximo N1                |
| n1-standard-8 + V100 | 8     | 30 GB   | ~$2.67      | GPU needed (not for MLP) |

### Series Modernas (Máximo Rendimiento)

| Instance Type       | Cores | RAM      | Características                   | Best For                              |
| ------------------- | ----- | -------- | --------------------------------- | ------------------------------------- |
| **c2-standard-60**  | 60    | 240 GB   | Intel Cascade Lake, AVX512        | **60 entornos paralelos optimizado**  |
| **c4d-highcpu-64**  | 64    | 512 GB   | AMD Turin, consistently high perf | **Máximo rendimiento (recomendado)**  |
| **c4d-highcpu-96**  | 96    | 768 GB   | AMD Turin, consistently high perf | **Máximo rendimiento extremo**        |
| **c4d-highcpu-128** | 128   | 1,024 GB | AMD Turin, consistently high perf | Rendimiento extremo                   |
| **c4-highcpu-64**   | 64    | 512 GB   | Intel Emerald Rapids              | Alto rendimiento Intel                |
| **n2d-highcpu-64**  | 64    | 64 GB    | AMD Milan, balanced price/perf    | **Mejor relación precio/rendimiento** |
| **n2d-highcpu-96**  | 96    | 96 GB    | AMD Milan, balanced price/perf    | Equilibrado potente                   |

\*Los precios exactos varían por región y pueden cambiar. Consulta [GCP Pricing Calculator](https://cloud.google.com/products/calculator) para precios actualizados.

**Savings**: CPU-only es **5-10x más barato** que instancias GPU!

**Recomendación por caso de uso:**

- **60 entornos paralelos optimizado**: `c2-standard-60` (60 vCPUs, 240 GB RAM, AVX512) - **RECOMENDADO para entrenamiento con 60 robots**
- **Máximo rendimiento absoluto**: `c4d-highcpu-96` o superior (hasta 384 vCPUs)
- **Mejor relación precio/rendimiento**: `n2d-highcpu-64` (64 vCPUs, buen precio)
- **Presupuesto limitado**: `n1-highcpu-32` (32 vCPUs, más económico)

## Performance Tips

1. **Match num_envs to CPU cores**: Use `--num_envs $(nproc)` or slightly less
2. **Monitor CPU usage**: Should be 80-100% with many environments
3. **Use larger batches**: `--batch_size 256` or `512` if memory allows
4. **Reduce n_epochs**: `--n_epochs 4` is faster than `10`
5. **Increase n_steps**: `--n_steps 4096` or `8192` for less frequent updates

## Troubleshooting

### Out of Memory

- Reduce `--num_envs` (each uses ~500MB-1GB)
- Reduce `--batch_size`
- Use smaller instance type

### ImportError: libGL.so.1: cannot open shared object file

Este error ocurre cuando faltan las bibliotecas gráficas necesarias para MuJoCo. **El script de setup ya las instala automáticamente**, pero si encuentras este error:

**Solución rápida:**

```bash
# Instalar bibliotecas gráficas necesarias
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libgomp1 libegl1-mesa \
    libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

# Configurar variables de entorno para headless rendering
export MUJOCO_GL=egl
export DISPLAY=:99

# Agregar a ~/.bashrc para persistencia
echo "" >> ~/.bashrc
echo "# MuJoCo headless rendering" >> ~/.bashrc
echo "export MUJOCO_GL=egl" >> ~/.bashrc
echo "export DISPLAY=:99" >> ~/.bashrc

# Recargar configuración
source ~/.bashrc
```

**Nota:** El script `setup_gcp_cpu.sh` ya instala estas dependencias automáticamente. Si ves este error, probablemente necesitas ejecutar el script de setup nuevamente o instalar las dependencias manualmente como se muestra arriba.

### CPU Not Fully Utilized

- Increase `--num_envs` to match CPU cores
- Check with `htop` - should see all cores busy

### Slow Training

- Make sure you're using `--device cpu` (not cuda)
- Use more parallel environments
- Increase batch size if memory allows

### TensorBoard: Port 6006 Already in Use

Si ves el error `TensorBoard could not bind to port 6006, it was already in use`:

**Solución rápida:**

```bash
# Matar el proceso que usa el puerto 6006
gcloud compute ssh spot-rl-cpu --zone=us-central1-a \
  --command="pkill -f tensorboard; kill \$(lsof -ti:6006) 2>/dev/null || true"

# Luego ejecutar TensorBoard nuevamente
gcloud compute ssh spot-rl-cpu --zone=us-central1-a \
  --ssh-flag="-L 6006:localhost:6006" \
  --command="cd ~/spot-rl && source venv/bin/activate && tensorboard --logdir=./logs --host=0.0.0.0 --port=6006"
```

**Alternativa: Usar un puerto diferente**

```bash
# Usar puerto 6007
gcloud compute ssh spot-rl-cpu --zone=us-central1-a \
  --ssh-flag="-L 6007:localhost:6007" \
  --command="cd ~/spot-rl && source venv/bin/activate && tensorboard --logdir=./logs --host=0.0.0.0 --port=6007"
```

Luego abre `http://localhost:6007` en tu navegador.

## Quick Start Commands

```bash
# 1. Create instance
gcloud compute instances create spot-rl-cpu \
  --zone=us-central1-c \
  --machine-type=n1-highcpu-16 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB

# 2. SSH in
gcloud compute ssh spot-rl-cpu --zone=us-central1-c

# 3. Run setup
chmod +x setup_gcp_cpu.sh && ./setup_gcp_cpu.sh

# 4. Upload script (from local)
gcloud compute scp train_colab.py spot-rl-cpu:~/spot-rl/ --zone=us-central1-c

# 5. Train (on instance)
cd ~/spot-rl && source venv/bin/activate
python train_colab.py --device cpu --num_envs 16 --total_timesteps 30000000
```

## Expected Training Times

**30M timesteps with different CPU configs:**

| Config            | Time Estimate | Cost Estimate |
| ----------------- | ------------- | ------------- |
| 8 envs, 8 cores   | ~25-40 hours  | ~$6-10        |
| 16 envs, 16 cores | ~12-20 hours  | ~$6-10        |
| 32 envs, 32 cores | ~6-10 hours   | ~$6-10        |

**Much cheaper than GPU instances!**
