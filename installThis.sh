#!/bin/bash

# Set variabel direktori instalasi Miniconda
MINICONDA_DIR="$HOME/miniconda3"

# Download Miniconda (jika belum ada)
if [ ! -f "$HOME/Miniconda3-latest-Linux-x86_64.sh" ]; then
  echo "Downloading Miniconda..."
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$HOME/Miniconda3-latest-Linux-x86_64.sh"
fi

# Install Miniconda di home directory
echo "Installing Miniconda to $MINICONDA_DIR..."
bash "$HOME/Miniconda3-latest-Linux-x86_64.sh" -b -p "$MINICONDA_DIR"

# Tambahkan conda ke shell (jika belum ada di .bashrc)
if ! grep -q 'conda shell.bash hook' "$HOME/.bashrc"; then
  echo 'eval "$($HOME/miniconda3/bin/conda shell.bash hook)"' >> "$HOME/.bashrc"
fi

# Muat conda ke shell sekarang
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# Update conda
echo "Updating conda..."
conda update -n base -c defaults conda -y

# Buat environment baru untuk RecBole
echo "Creating conda environment pyEnv37 with Python 3.7..."
conda create -n pyEnv37 python=3.7 -y

# Aktifkan environment
echo "Activating conda environment pyEnv37..."
conda activate pyEnv37

# Install Jupyter dan RecBole
echo "Installing Jupyter Notebook and RecBole..."
pip install notebook
python -m ipykernel install --user --name pyEnv37 --display-name "pyEnv37"
pip install recbole==1.0.1

echo "✅ Installation completed successfully!"

