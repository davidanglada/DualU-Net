#!/bin/bash
#SBATCH -p gpi.compute
#SBATCH --time 20:00:00
#SBATCH -c 1
#SBATCH --mem 40G
#SBATCH --gres gpu:1,gpumem:16G

source ../../../../venv/phd2/bin/activate

python3 -m torch.distributed.launch --use-env --nproc-per-node=1 --master-port=29561 inference.py --config ./configs/config_inference.yaml
