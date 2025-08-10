import argparse
import os
import signal
import subprocess
import sys

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spawn per-GPU workers")
    parser.add_argument("script", help="Training script to run")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_gpus = torch.cuda.device_count()
    base_rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", n_gpus))
    processes = []

    def handle_sigterm(signum, frame):  # noqa: ARG001
        for p in processes:
            p.send_signal(signum)

    signal.signal(signal.SIGTERM, handle_sigterm)

    for local_rank in range(n_gpus):
        env = os.environ.copy()
        env.update(
            {
                "LOCAL_RANK": str(local_rank),
                "RANK": str(base_rank + local_rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": os.environ.get("MASTER_ADDR", "127.0.0.1"),
                "MASTER_PORT": os.environ.get("MASTER_PORT", "29500"),
            }
        )
        cmd = [sys.executable, args.script] + list(args.script_args)
        processes.append(subprocess.Popen(cmd, env=env))

    codes = [p.wait() for p in processes]
    sys.exit(max(codes))


if __name__ == "__main__":
    main()
