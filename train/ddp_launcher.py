import argparse
import os
import signal
import subprocess
import sys
import time

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
    env_world_size = os.environ.get("WORLD_SIZE")
    if n_gpus == 0 and env_world_size is None:
        raise RuntimeError("No GPUs found and WORLD_SIZE not specified")
    world_size = int(env_world_size) if env_world_size else n_gpus
    processes: list[subprocess.Popen] = []

    def handle_sigterm(signum, frame):  # noqa: ARG001
        timeout = 10
        for p in processes:
            p.send_signal(signum)

        start = time.time()
        while True:
            alive = [p for p in processes if p.poll() is None]
            if not alive or (time.time() - start) > timeout:
                break
            time.sleep(0.5)

        for p in processes:
            if p.poll() is None:
                p.terminate()

    signal.signal(signal.SIGTERM, handle_sigterm)

    n_procs = n_gpus if n_gpus > 0 else world_size
    for local_rank in range(n_procs):
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
        script_path = os.path.abspath(args.script)
        if not os.path.exists(script_path):
            raise FileNotFoundError(script_path)
        cmd = [sys.executable, script_path] + list(args.script_args)
        processes.append(subprocess.Popen(cmd, env=env, shell=False))

    codes = [p.wait() for p in processes]
    for idx, code in enumerate(codes):
        print(f"Process {idx} exited with code {code}")
    sys.exit(max(codes))


if __name__ == "__main__":
    main()
