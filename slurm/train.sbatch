#!/bin/bash
#SBATCH --job-name=ddp-train
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --output=ddp-train-%j.out

module load cuda/12.1 || true

if [ -z "$SLURM_GPUS" ] || [ "$SLURM_GPUS" -eq 0 ]; then
  echo "No GPU assigned; exiting."
  exit 0
fi

export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_P2P_LEVEL=NVL
# export NCCL_IB_HCA=mlx5

srun python -m torch.distributed.run --nproc_per_node=$SLURM_GPUS train/main.py
