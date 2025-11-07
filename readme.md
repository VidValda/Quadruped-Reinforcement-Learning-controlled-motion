# Create and activate a virtual environment (optional but recommended)

python -m venv rl_env
source rl_env/bin/activate # On Windows, use `rl_env\Scripts\activate`

# Install the core libraries

Upgrade pip and install from requirements.txt:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Notes:

- Use the active virtual environment when running these commands.
- On Windows PowerShell, activate with `.\rl_env\Scripts\Activate.ps1` and run the same commands.

# Code

Study and try to run advanced_training_normalized.py

# Evaluate

tensorboard --logdir=./spot_tensorboard_advanced
