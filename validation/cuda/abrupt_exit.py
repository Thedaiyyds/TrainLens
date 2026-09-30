"""Allocate CUDA memory, then bypass Python finalization deliberately."""

import argparse
import os


def main(argv: list[str] | None = None) -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bytes", type=int, required=True)
    args = parser.parse_args(argv)
    if args.bytes <= 0:
        raise ValueError("Allocation size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This fixture requires NVIDIA CUDA")
    tensor = torch.empty(args.bytes, dtype=torch.uint8, device="cuda:0")
    # Keep the allocation alive until the fatal exit; there is no reference peak.
    if tensor.numel() != args.bytes:
        raise RuntimeError("Unexpected allocation size")
    os._exit(7)


if __name__ == "__main__":
    main()
