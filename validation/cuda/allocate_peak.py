"""Controlled CUDA allocation and direct readings; never imports a collector."""

import argparse
import json
import os
from pathlib import Path


def allocate_and_read(size_bytes: int) -> dict:
    import torch

    if size_bytes <= 0:
        raise ValueError("Allocation size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This fixture requires NVIDIA CUDA")
    tensor = torch.empty(size_bytes, dtype=torch.uint8, device="cuda:0")

    def readings() -> dict:
        return {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "current_allocated_bytes": torch.cuda.memory_allocated(0),
        }

    while_live = readings()
    del tensor
    after_free = readings()
    return {
        "pid": os.getpid(),
        "device": "cuda:0",
        "requested_bytes": size_bytes,
        "allocator_backend": torch.cuda.memory.get_allocator_backend(),
        "while_live": while_live,
        "after_free": after_free,
    }


def write_reference(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    reference = allocate_and_read(args.bytes)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(reference, stream, allow_nan=False, indent=2)
        stream.write("\n")
    # No subsequent CUDA activity before the bootstrap's final query.


if __name__ == "__main__":
    write_reference()
