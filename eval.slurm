#!/bin/bash
#SBATCH -p gpi.compute
#SBATCH --time 20:00:00
#SBATCH -c 1
#SBATCH --mem 40G
#SBATCH --gres gpu:1,gpumem:24G

source ../../../venv/phd2/bin/activate

python3 -m torch.distributed.launch --use-env --nproc-per-node=1 --master-port=29560 eval.py --config ./config_eval.yaml
