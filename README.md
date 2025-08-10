# pytorch-ddp-nccl-distributed

Turn-key PyTorch DistributedDataParallel (DDP) with NCCL for single- and multi-node GPU training. Includes Docker image, K8s Job and optional Kubeflow PyTorchJob, Slurm sbatch, sensible NCCL env defaults, Makefile, and CI.

## Quickstart

### Build the training image

```
make build
```

### Run on Kubernetes

```
make run-k8s
```

This creates the NCCL config map and launches a two-pod Job.

### Run on Slurm

```
make run-slurm
```

If no GPU is available the script prints a friendly message and exits.

## NCCL tuning cheat sheet

- `NCCL_SOCKET_IFNAME` – network interface (e.g., `eth0`)
- `NCCL_P2P_LEVEL` – peer-to-peer level (`NVL` for NVLink)
- `# NCCL_IB_HCA` – uncomment to specify Infiniband device
- `NCCL_DEBUG=INFO` – verbose logging

## Topology & troubleshooting

Adjust env vars for multi-homed nodes or different fabrics. Ensure `MASTER_ADDR` and `MASTER_PORT` are reachable. For hangs, enable `NCCL_DEBUG=INFO` and verify network reachability.
