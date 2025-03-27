#!/bin/bash
#SBATCH -p gpi.compute
#SBATCH --time 20:00:00
#SBATCH -c 4
#SBATCH --mem 40G
#SBATCH --gres gpu:2,gpumem:24G

source ../../../../venv/phd2/bin/activate

python3 -m torch.distributed.launch --use-env --nproc-per-node=2 --master-port=29510 train.py --config ./configs/config_train.yaml