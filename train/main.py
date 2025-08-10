import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import dataset


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDP Training Example")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument(
        "--data-path",
        default=os.environ.get("DATA_PATH", "/tmp/data"),
        help="Dataset path (env: DATA_PATH)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)

    model = TinyModel().to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss().to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

    dataset_obj = dataset.get_dataset(args.data_path)
    sampler = DistributedSampler(dataset_obj)
    loader = DataLoader(dataset_obj, batch_size=args.batch_size, sampler=sampler)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        epoch_start = time.time()
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        dist.barrier()
        epoch_time = time.time() - epoch_start
        num_images = len(loader.dataset)
        world_size = dist.get_world_size()
        if dist.get_rank() == 0:
            global_num_images = num_images * world_size
            throughput = global_num_images / epoch_time
            print(
                f"Epoch {epoch} | Global throughput: {throughput:.2f} img/s | {epoch_time:.2f}s"
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
