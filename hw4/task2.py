"""Triton LayerNorm forward/backward kernels, correctness tests, and benchmark.

Requires an NVIDIA GPU, CUDA-enabled PyTorch, and Triton >= 3.6.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def get_autotune_configs() -> list[triton.Config]:
    # Return fresh objects: forward and backward keep independent autotune caches.
    return [
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ]


@triton.autotune(configs=get_autotune_configs(), key=["N_BUCKET"])
@triton.jit
def _layernorm_forward_kernel(
    X,
    WEIGHT,
    BIAS,
    Y,
    MEAN,
    RSTD,
    N: tl.constexpr,
    N_BUCKET: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One Triton program processes one row of shape [N]."""

    row = tl.program_id(axis=0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < N
    row_offsets = row * N + columns

    x = tl.load(X + row_offsets, mask=mask, other=0.0).to(tl.float32)
    x_for_reduction = tl.where(mask, x, 0.0)
    mean = tl.sum(x_for_reduction, axis=0) / N

    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(variance + EPS)

    weight = tl.load(WEIGHT + columns, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(BIAS + columns, mask=mask, other=0.0).to(tl.float32)
    output = centered * rstd * weight + bias

    tl.store(Y + row_offsets, output, mask=mask)
    tl.store(MEAN + row, mean)
    tl.store(RSTD + row, rstd)


@triton.autotune(
    configs=get_autotune_configs(),
    key=["N_BUCKET"],
    # Autotuning executes the kernel several times. These outputs are
    # accumulators, so they must be zeroed before every candidate config.
    reset_to_zero=["PARTIAL_DW", "PARTIAL_DB", "LOCKS"],
)
@triton.jit
def _layernorm_backward_kernel(
    DY,
    X,
    WEIGHT,
    MEAN,
    RSTD,
    DX,
    PARTIAL_DW,
    PARTIAL_DB,
    LOCKS,
    N: tl.constexpr,
    N_BUCKET: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute dx and grouped partial sums for dweight/dbias."""

    row = tl.program_id(axis=0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < N
    row_offsets = row * N + columns

    x = tl.load(X + row_offsets, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + row_offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(WEIGHT + columns, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(MEAN + row)
    rstd = tl.load(RSTD + row)

    x_hat = tl.where(mask, (x - mean) * rstd, 0.0)
    weighted_dy = tl.where(mask, weight * dy, 0.0)
    c1 = tl.sum(x_hat * weighted_dy, axis=0) / N
    c2 = tl.sum(weighted_dy, axis=0) / N
    dx = (weighted_dy - x_hat * c1 - c2) * rstd
    tl.store(DX + row_offsets, dx, mask=mask)

    # Rows are distributed among a bounded number of accumulation buffers.
    # A lock protects each buffer; a second kernel reduces the buffers.
    group_id = row % GROUP_SIZE_M
    lock_pointer = LOCKS + group_id
    count_pointer = LOCKS + GROUP_SIZE_M + group_id
    partial_offsets = group_id * N + columns

    partial_dw = dy * x_hat
    partial_db = dy

    while tl.atomic_cas(lock_pointer, 0, 1) == 1:
        pass

    count = tl.load(count_pointer)
    if count == 0:
        tl.atomic_xchg(count_pointer, 1)
    else:
        partial_dw += tl.load(
            PARTIAL_DW + partial_offsets, mask=mask, other=0.0
        )
        partial_db += tl.load(
            PARTIAL_DB + partial_offsets, mask=mask, other=0.0
        )

    tl.store(PARTIAL_DW + partial_offsets, partial_dw, mask=mask)
    tl.store(PARTIAL_DB + partial_offsets, partial_db, mask=mask)
    tl.debug_barrier()
    tl.atomic_xchg(lock_pointer, 0)


@triton.jit
def _reduce_weight_bias_gradients_kernel(
    PARTIAL_DW,
    PARTIAL_DB,
    DW,
    DB,
    NUM_GROUPS,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    column_block = tl.program_id(axis=0)
    columns = column_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    accumulated_dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32)
    accumulated_db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32)

    for group_start in range(0, NUM_GROUPS, BLOCK_SIZE_M):
        groups = group_start + tl.arange(0, BLOCK_SIZE_M)
        mask = (groups[:, None] < NUM_GROUPS) & (columns[None, :] < N)
        offsets = groups[:, None] * N + columns[None, :]
        accumulated_dw += tl.load(PARTIAL_DW + offsets, mask=mask, other=0.0)
        accumulated_db += tl.load(PARTIAL_DB + offsets, mask=mask, other=0.0)

    tl.store(
        DW + columns,
        tl.sum(accumulated_dw, axis=0),
        mask=columns < N,
    )
    tl.store(
        DB + columns,
        tl.sum(accumulated_db, axis=0),
        mask=columns < N,
    )


def _check_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> None:
    if not x.is_cuda or not weight.is_cuda or not bias.is_cuda:
        raise ValueError("Triton LayerNorm requires CUDA tensors")
    if x.device != weight.device or x.device != bias.device:
        raise ValueError("x, weight, and bias must be on the same CUDA device")
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    if weight.ndim != 1 or bias.ndim != 1:
        raise ValueError("weight and bias must be one-dimensional")
    if x.shape[-1] != weight.numel() or weight.shape != bias.shape:
        raise ValueError("weight and bias must match x.shape[-1]")


def _launch_metadata(x: torch.Tensor) -> tuple[int, int, int, int]:
    n = x.shape[-1]
    m = x.numel() // n
    block_size = triton.next_power_of_2(n)
    max_fused_size = 65_536 // x.element_size()
    if block_size > max_fused_size:
        raise RuntimeError(
            f"Feature dimension {n} is too large for the fused kernel "
            f"(maximum block size for {x.dtype}: {max_fused_size})"
        )
    n_bucket = int(math.log2(block_size))
    return m, n, block_size, n_bucket


def layernorm_forward_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return output, per-row mean, and per-row reciprocal std."""

    _check_inputs(x, weight, bias)
    x_2d = x.contiguous().view(-1, x.shape[-1])
    weight = weight.contiguous()
    bias = bias.contiguous()
    m, n, block_size, n_bucket = _launch_metadata(x_2d)

    output = torch.empty_like(x_2d)
    mean = torch.empty(m, device=x.device, dtype=torch.float32)
    rstd = torch.empty(m, device=x.device, dtype=torch.float32)

    _layernorm_forward_kernel[(m,)](
        x_2d,
        weight,
        bias,
        output,
        mean,
        rstd,
        N=n,
        N_BUCKET=n_bucket,
        EPS=eps,
        BLOCK_SIZE=block_size,
    )
    return output.view_as(x), mean, rstd


def _group_size_for_rows(n: int) -> int:
    if n <= 1_024:
        return 256
    if n <= 4_096:
        return 128
    if n <= 8_192:
        return 96
    return 64


def layernorm_backward_triton(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return gradients for x, weight, and bias."""

    if not grad_output.is_cuda:
        raise ValueError("grad_output must be a CUDA tensor")
    grad_output_2d = grad_output.contiguous().view(-1, x.shape[-1])
    x_2d = x.contiguous().view(-1, x.shape[-1])
    m, n, block_size, n_bucket = _launch_metadata(x_2d)

    group_size = min(_group_size_for_rows(n), m)
    grad_input = torch.empty_like(x_2d)
    # Float32 accumulation makes dweight/dbias stable for fp16/bf16 inputs.
    partial_dw = torch.zeros(
        (group_size, n), device=x.device, dtype=torch.float32
    )
    partial_db = torch.zeros_like(partial_dw)
    locks = torch.zeros(2 * group_size, device=x.device, dtype=torch.int32)
    grad_weight_fp32 = torch.empty(n, device=x.device, dtype=torch.float32)
    grad_bias_fp32 = torch.empty(n, device=x.device, dtype=torch.float32)

    _layernorm_backward_kernel[(m,)](
        grad_output_2d,
        x_2d,
        weight,
        mean,
        rstd,
        grad_input,
        partial_dw,
        partial_db,
        locks,
        N=n,
        N_BUCKET=n_bucket,
        GROUP_SIZE_M=group_size,
        BLOCK_SIZE=block_size,
    )

    reduction_grid = (triton.cdiv(n, 128),)
    _reduce_weight_bias_gradients_kernel[reduction_grid](
        partial_dw,
        partial_db,
        grad_weight_fp32,
        grad_bias_fp32,
        group_size,
        n,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=128,
        num_warps=4,
    )

    return (
        grad_input.view_as(x),
        grad_weight_fp32.to(weight.dtype),
        grad_bias_fp32.to(weight.dtype),
    )


class _LayerNormTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        x_contiguous = x.contiguous()
        output, mean, rstd = layernorm_forward_triton(
            x_contiguous, weight, bias, eps
        )
        ctx.save_for_backward(x_contiguous, weight, mean, rstd)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, mean, rstd = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = layernorm_backward_triton(
            grad_output, x, weight, mean, rstd
        )
        return grad_input, grad_weight, grad_bias, None


def layer_norm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    return _LayerNormTriton.apply(x, weight, bias, eps)


def layernorm_forward_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, unbiased=False, keepdim=True)
    rstd = torch.rsqrt(variance + eps)
    x_hat = (x - mean) * rstd
    return x_hat * weight + bias


def test_layer_norm() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Correctness tests require an NVIDIA CUDA GPU")

    torch.manual_seed(42)
    test_cases = [
        ((128, 256), torch.float32),
        ((64, 1_024), torch.float32),
        ((32, 2_048), torch.float16),
        ((4, 7, 1_536), torch.float16),
    ]

    for shape, dtype in test_cases:
        x_triton = torch.randn(
            shape, device="cuda", dtype=dtype, requires_grad=True
        )
        weight_triton = torch.randn(
            shape[-1], device="cuda", dtype=dtype, requires_grad=True
        )
        bias_triton = torch.randn(
            shape[-1], device="cuda", dtype=dtype, requires_grad=True
        )
        x_torch = x_triton.detach().clone().requires_grad_(True)
        weight_torch = weight_triton.detach().clone().requires_grad_(True)
        bias_torch = bias_triton.detach().clone().requires_grad_(True)

        output_triton = layer_norm_triton(
            x_triton, weight_triton, bias_triton
        )
        output_torch = F.layer_norm(
            x_torch, (shape[-1],), weight_torch, bias_torch
        )

        tolerance = 1e-3 if dtype == torch.float32 else 2e-2
        torch.testing.assert_close(
            output_triton,
            output_torch,
            atol=tolerance,
            rtol=tolerance,
        )

        grad_output = torch.randn_like(output_triton)
        output_triton.backward(grad_output)
        output_torch.backward(grad_output)

        for name, actual, expected in (
            ("dx", x_triton.grad, x_torch.grad),
            ("dweight", weight_triton.grad, weight_torch.grad),
            ("dbias", bias_triton.grad, bias_torch.grad),
        ):
            torch.testing.assert_close(
                actual,
                expected,
                atol=tolerance,
                rtol=tolerance,
                msg=lambda message: f"{shape}, {dtype}, {name}: {message}",
            )

        print(f"PASS shape={shape}, dtype={dtype}")


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[256, 512, 1_024, 2_048, 4_096, 8_192],
            line_arg="provider",
            line_vals=["triton", "torch"],
            line_names=["Triton", "PyTorch"],
            styles=[("blue", "-"), ("green", "--")],
            ylabel="Time (ms), lower is better",
            plot_name="layernorm-forward",
            args={"M": 4_096, "dtype": torch.float16, "mode": "forward"},
        ),
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[256, 512, 1_024, 2_048, 4_096, 8_192],
            line_arg="provider",
            line_vals=["triton", "torch"],
            line_names=["Triton", "PyTorch"],
            styles=[("blue", "-"), ("green", "--")],
            ylabel="Time (ms), lower is better",
            plot_name="layernorm-backward",
            args={"M": 4_096, "dtype": torch.float16, "mode": "backward"},
        ),
    ]
)
def benchmark_layer_norm(M, N, dtype, provider, mode):
    x = torch.randn((M, N), device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(N, device="cuda", dtype=dtype, requires_grad=True)
    bias = torch.randn(N, device="cuda", dtype=dtype, requires_grad=True)
    grad_output = torch.randn_like(x)

    if provider == "triton":
        forward = lambda: layer_norm_triton(x, weight, bias)
    else:
        forward = lambda: F.layer_norm(x, (N,), weight, bias)

    quantiles = [0.5, 0.2, 0.8]
    if mode == "forward":
        median_ms, min_ms, max_ms = triton.testing.do_bench(
            forward, quantiles=quantiles
        )
    else:
        output = forward()
        median_ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: output.backward(grad_output, retain_graph=True),
            quantiles=quantiles,
            grad_to_none=[x, weight, bias],
        )

    return median_ms, min_ms, max_ms


def run_benchmark(save_path: str | Path = "benchmark_results") -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Benchmark requires an NVIDIA CUDA GPU")
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    benchmark_layer_norm.run(
        save_path=str(save_path),
        show_plots=False,
        print_data=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["test", "benchmark", "all"],
        nargs="?",
        default="all",
    )
    parser.add_argument("--save-path", default="benchmark_results")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command in {"test", "all"}:
        test_layer_norm()
    if arguments.command in {"benchmark", "all"}:
        run_benchmark(arguments.save_path)
