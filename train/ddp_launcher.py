import argparse
import os
import signal
import socket
import subprocess
import sys
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spawn per-GPU workers")
    parser.add_argument("script", help="Training script to run")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    parser.add_argument(
        "--sigterm-timeout",
        type=float,
        default=float(os.environ.get("SIGTERM_TIMEOUT", "10")),
        help="Seconds to wait after SIGTERM before forcing termination (env: SIGTERM_TIMEOUT)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_gpus = torch.cuda.device_count()
    base_rank = int(os.environ.get("RANK", "0"))
    env_world_size = os.environ.get("WORLD_SIZE")
    if n_gpus == 0 and env_world_size is None:
        raise RuntimeError("No GPUs found and WORLD_SIZE not specified")
    world_size = int(env_world_size) if env_world_size else n_gpus

    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.environ.get("MASTER_PORT", "29500"))

    # Fail fast if the master is unreachable or the port is already in use
    try:
        with socket.create_connection((master_addr, master_port), timeout=2):
            pass
    except OSError:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2)
            try:
                sock.bind((master_addr, master_port))
            except OSError as exc:  # pragma: no cover - network dependent
                raise RuntimeError(
                    f"MASTER_ADDR {master_addr}:{master_port} unreachable or port unavailable"
                ) from exc

    processes: list[subprocess.Popen] = []

    def handle_sigterm(signum, frame):  # noqa: ARG001
        for p in processes:
            p.send_signal(signum)

        deadline = time.time() + args.sigterm_timeout
        for p in processes:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                pass

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
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": str(master_port),
            }
        )
        script_path = os.path.realpath(args.script)
        if not os.path.isfile(script_path):
            raise FileNotFoundError(script_path)
        cmd = [sys.executable, script_path, *args.script_args]
        # Pass a list with shell=False to avoid shell injection
        processes.append(subprocess.Popen(cmd, env=env, shell=False))

    codes = [p.wait() for p in processes]
    for idx, code in enumerate(codes):
        print(f"Process {idx} exited with code {code}")
    sys.exit(max(codes))


if __name__ == "__main__":
    main()
