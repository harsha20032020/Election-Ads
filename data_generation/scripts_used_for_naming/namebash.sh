#!/bin/bash
#SBATCH -n 4
#SBATCH --mem-per-cpu=4096
#SBATCH --time=72:00:00
#SBATCH --mincpus=3
#SBATCH --gres=gpu:1
#SBATCH --mail-user=nemani.v@research.iiit.ac.in
#SBATCH --mail-type=ALL
module add cuda/8.0
module add cudnn/7-cuda-8.0
export CUDA_VISIBLE_DEVICES=0

# Initialize conda environment
eval "$(conda shell.bash hook)"
conda activate namer

# Check if conda environment is activated
if [ "$CONDA_DEFAULT_ENV" == "namer" ]; then
    echo "Successfully activated conda environment: $CONDA_DEFAULT_ENV"
else
    echo "Failed to activate conda environment"
    exit 1
fi
cd ElectionSpend/IndicTrans2/huggingface_inference
conda env list
echo "Running Files"
python3 names_from_trans.py chunk_1.csv 