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
echo "Activating virtualenv"
conda activate itv2_hf
cd ElectionSpend/IndicTrans2/huggingface_inference
echo "Running Files"
python3 trans.py 
