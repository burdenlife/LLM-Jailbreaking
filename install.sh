#!usrbinenv bash
set -e  # exit immediately on error

ENV_NAME=fyp
PYTHON_VERSION=3.11

echo === Loading Miniforge ===
module purge
module load miniforge3

echo === Creating conda environment $ENV_NAME ===
if conda env list  grep -q $ENV_NAME; then
    echo Environment $ENV_NAME already exists. Skipping creation.
else
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
fi

echo === Activating environment ===
source activate $ENV_NAME

echo === Upgrading pip ===
pip install --upgrade pip

echo === Installing Python packages ===
pip install 
    torch 
    transformers 
    accelerate 
    pandas 
    numpy 
    tqdm 
    scikit-learn

echo === Optional HuggingFace login (skip if already set via env var) ===
# huggingface-cli login

echo === Installation complete ===
echo Activate later with module load miniforge3 && source activate $ENV_NAME
