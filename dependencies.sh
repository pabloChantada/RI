#!/bin/bash

# Definir el nombre del entorno virtual
ENV_DIR="rob_env"

# Verificar si Python 3.11 está instalado
if ! command -v python3.11 &> /dev/null; then
  echo "Error: Python 3.11 no está instalado. Instálalo con 'brew install python@3.11' en macOS."
  exit 1
fi

# Verificar si el entorno virtual ya existe
if [ -d "$ENV_DIR" ]; then
  echo "The environment already exists."
  exit 0
fi

echo "Creating Python 3.11 environment: $ENV_DIR"
python3.11 -m venv "$ENV_DIR"

# Activar el entorno virtual
source "$ENV_DIR/bin/activate"

# Instalar dependencias necesarias
echo "Installing dependencies for $ENV_DIR"
pip install --upgrade pip
pip install robobopy robobosim gymnasium stable-baselines3 pygame

echo "Environment created successfully"

# Añadir el entorno al .gitignore si no está presente
grep -qxF "$ENV_DIR/" .gitignore || echo "$ENV_DIR/" >> .gitignore

