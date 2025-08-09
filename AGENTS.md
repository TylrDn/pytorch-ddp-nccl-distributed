You are scaffolding a complete repo named pytorch-ddp-nccl-distributed for DDP training with NCCL.

Objectives
PyTorch DDP example (CIFAR10 or synthetic) that runs single/multi-node.

Docker image for training; K8s Job (and optional Kubeflow PyTorchJob comment).

Slurm sbatch script.

Sensible NCCL env defaults + tuning notes.

Makefile + tests + CI + pre-commit; Apache-2.0 license.

Create this structure
arduino
Copy
Edit
README.md
LICENSE
.gitignore
.pre-commit-config.yaml
.github/workflows/build.yaml
Makefile
docker/
  Dockerfile.train
train/
  main.py
  ddp_launcher.py
  dataset.py
k8s/
  job.yaml
  configmap.env
  pytorchjob.yaml   # commented optional
slurm/
  train.sbatch
tests/
  test_dataset.py
ACCEPTANCE.md
File requirements (high level)
Dockerfile.train: CUDA + cuDNN + PyTorch pinned; creates non-root user; sets NCCL_DEBUG=INFO.

ddp_launcher.py: spawns per-GPU processes; reads RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT; handles SIGTERM for preemption.

main.py: DDP init, tiny model (ResNet18 or MLP), mixed precision optional; logs throughput/epoch time.

dataset.py: synthetic or CIFAR10 fallback; unit test covers basic shape.

k8s/job.yaml: N workers + 1 master (or all identical with role env); includes NCCL envs (NCCL_SOCKET_IFNAME, NCCL_P2P_LEVEL, NCCL_IB_HCA commented); resource requests/limits for GPU.

slurm/train.sbatch: srun python -m torch.distributed.run --nproc_per_node=$SLURM_GPUS ....

Makefile: build, push, run-k8s, run-slurm, lint, clean.

README.md: quickstart (Docker build), K8s and Slurm runs, NCCL tuning cheat sheet, topology notes, troubleshooting.

build.yaml: build & push on tag; run flake8/black + pytest.

Acceptance checks
Image builds and unit tests pass.

K8s job runs and prints synchronized step logs.

Slurm sbatch completes on a cluster (or prints friendly “no GPU” message).

CI and pre-commit pass.

Output format
Use:

pgsql
Copy
Edit
=== path/to/file ===
<contents>
