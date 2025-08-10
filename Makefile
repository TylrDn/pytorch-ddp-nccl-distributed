IMAGE ?= ghcr.io/your-org/ddp-train:latest

.PHONY: build push run-k8s run-slurm lint test clean

build:
	docker build -t $(IMAGE) -f docker/Dockerfile.train .

push:
	docker push $(IMAGE)

run-k8s:
	kubectl create configmap nccl-env --from-env-file=k8s/configmap.env --dry-run=client -o yaml | kubectl apply -f -
	kubectl apply -f k8s/job.yaml

run-slurm:
	sbatch slurm/train.sbatch || echo "sbatch not available"

lint:
	pre-commit run --all-files

test:
	pytest

clean:
	docker rmi $(IMAGE) || true
