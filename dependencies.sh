#!/bin/bash

ENV_DIR="rob_env"

if ! command -v python3.11 &> /dev/null; then
  echo "Error: Python 3.11 no está instalado. Instálalo con 'brew install python@3.11' en macOS."
  exit 1
fi

if [ -d "$ENV_DIR" ]; then
  echo "The environment already exists."
  exit 0
fi

echo "Creating Python 3.11 environment: $ENV_DIR"
python3.11 -m venv "$ENV_DIR"

source "$ENV_DIR/bin/activate"

echo "Installing dependencies for $ENV_DIR"
pip install --upgrade pip

# Librerías del simulador y RL
pip install robobopy robobosim gymnasium "stable-baselines3[extra]" pygame

# Para gráficas y análisis de resultados
pip install matplotlib seaborn pandas tensorboard

echo "Environment created successfully"

grep -qxF "$ENV_DIR/" .gitignore || echo "$ENV_DIR/" >> .gitignore
