"""Efficient and correctly measured PyTorch training loop for homework task 1."""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def prepare_data(
    num_samples: int = 10_000,
    input_dim: int = 128,
    seed: int = 42,
) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(num_samples, input_dim, generator=generator)
    targets = torch.randint(0, 2, (num_samples,), generator=generator)
    return TensorDataset(features, targets)


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _make_model(input_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 2),
    )


def train(
    *,
    device: str | torch.device | None = None,
    num_samples: int = 10_000,
    input_dim: int = 128,
    batch_size: int = 256,
    num_workers: int = 2,
    warmup_batches: int = 5,
    max_timed_batches: int = 100,
) -> dict[str, Any]:
    """Train for one epoch and return honest loss/timing metrics.

    On CUDA, timings are collected with events without synchronizing the stream
    inside the loop. CPU/MPS use synchronized wall-clock timing as a portable
    fallback; the CUDA path is the optimized path required by the assignment.
    """

    device = torch.device(device) if device is not None else get_default_device()
    use_cuda_pipeline = device.type == "cuda"

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": True,
        # Pinned host memory is what makes non_blocking CUDA copies asynchronous.
        "pin_memory": use_cuda_pipeline,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)

    dataloader = DataLoader(
        prepare_data(num_samples=num_samples, input_dim=input_dim),
        **loader_kwargs,
    )
    model = _make_model(input_dim).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Accumulate only detached values. Keeping `loss` itself in a Python list
    # would retain every autograd graph and eventually cause OOM.
    loss_sum = torch.zeros((), device=device)
    seen_samples = 0

    # CUDA events measure work on the CUDA stream correctly. time.time() around
    # asynchronous kernels measures launch overhead rather than execution time.
    forward_event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    backward_event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    forward_times_ms: list[float] = []
    backward_times_ms: list[float] = []

    for batch_index, (data, target) in enumerate(dataloader):
        data = data.to(device, non_blocking=use_cuda_pipeline)
        target = target.to(device, non_blocking=use_cuda_pipeline)

        # Generate noise directly on the accelerator. The original code first
        # allocated it on CPU and then performed an extra blocking transfer.
        data.add_(torch.randn_like(data))
        optimizer.zero_grad(set_to_none=True)

        should_time = (
            batch_index >= warmup_batches
            and batch_index < warmup_batches + max_timed_batches
        )

        if use_cuda_pipeline and should_time:
            forward_start = torch.cuda.Event(enable_timing=True)
            forward_end = torch.cuda.Event(enable_timing=True)
            forward_start.record()
        elif should_time:
            synchronize(device)
            forward_started_at = time.perf_counter()

        output = model(data)
        loss = criterion(output, target)

        if use_cuda_pipeline and should_time:
            forward_end.record()
            forward_event_pairs.append((forward_start, forward_end))
            backward_start = torch.cuda.Event(enable_timing=True)
            backward_end = torch.cuda.Event(enable_timing=True)
            backward_start.record()
        elif should_time:
            synchronize(device)
            forward_times_ms.append((time.perf_counter() - forward_started_at) * 1_000)
            synchronize(device)
            backward_started_at = time.perf_counter()

        loss.backward()

        if use_cuda_pipeline and should_time:
            backward_end.record()
            backward_event_pairs.append((backward_start, backward_end))
        elif should_time:
            synchronize(device)
            backward_times_ms.append((time.perf_counter() - backward_started_at) * 1_000)

        optimizer.step()

        current_batch_size = target.size(0)
        with torch.no_grad():
            # Weight by the real batch size: the final partial batch must not
            # count as much as a full batch in the epoch average.
            loss_sum.add_(loss.detach() * current_batch_size)
        seen_samples += current_batch_size

        # No loss.item(), print(), or torch.cuda.empty_cache() here: all three
        # serialize/slow the asynchronous pipeline when used every iteration.

    synchronize(device)

    if use_cuda_pipeline:
        forward_times_ms = [start.elapsed_time(end) for start, end in forward_event_pairs]
        backward_times_ms = [start.elapsed_time(end) for start, end in backward_event_pairs]

    metrics = {
        "device": str(device),
        "average_loss": (loss_sum / seen_samples).item(),
        "average_forward_ms": statistics.fmean(forward_times_ms),
        "average_backward_ms": statistics.fmean(backward_times_ms),
        "timed_batches": len(forward_times_ms),
        "seen_samples": seen_samples,
    }
    print(
        f"Epoch finished on {metrics['device']}: "
        f"loss={metrics['average_loss']:.6f}, "
        f"forward={metrics['average_forward_ms']:.3f} ms, "
        f"backward={metrics['average_backward_ms']:.3f} ms "
        f"({metrics['timed_batches']} batches after warm-up)"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="cuda, mps, or cpu")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--timed-batches", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        device=args.device,
        num_samples=args.samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        warmup_batches=args.warmup_batches,
        max_timed_batches=args.timed_batches,
    )
