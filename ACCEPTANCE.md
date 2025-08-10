# Acceptance

- `docker build -t ghcr.io/your-org/ddp-train:latest -f docker/Dockerfile.train .`
- `pytest`
- `make run-k8s` prints synchronized step logs
- `make run-slurm` finishes or prints "No GPU assigned" message
- `pre-commit run --all-files`
