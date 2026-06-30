# Homework: efficient PyTorch loop and Triton LayerNorm

## Task 1

`task1.py` fixes all three classes of issues from the assignment:

- memory: no autograd tensors are retained in history;
- asynchronous CUDA pipeline: pinned DataLoader memory, non-blocking transfers,
  accelerator-side noise generation, `zero_grad(set_to_none=True)`, no per-batch
  `item()`, `print()`, synchronization, or `empty_cache()`;
- honest metrics: sample-weighted epoch loss, warm-up exclusion, and CUDA Event
  timing instead of wall-clock timing around asynchronous kernel launches.

Run on CUDA:

```bash
python task1.py --device cuda
```

The CPU/MPS fallback exists for portability and basic validation, but the optimized
path is the CUDA path.

## Task 2

`task2.py` contains:

- a Triton LayerNorm forward kernel;
- a Triton backward kernel for `dx` plus grouped accumulation of `dweight` and
  `dbias`;
- a second reduction kernel for the parameter gradients;
- autotuning over warp configurations, including safe reset of buffers modified
  by backward candidates;
- `torch.autograd.Function` integration;
- forward/backward correctness checks with `torch.testing.assert_close`;
- forward and backward benchmarks against `torch.nn.functional.layer_norm`.

Requirements: Linux, NVIDIA GPU, CUDA-enabled PyTorch, Triton 3.6+.

```bash
python task2.py test
python task2.py benchmark --save-path benchmark_results
# or both:
python task2.py all --save-path benchmark_results
```

The benchmark prints a table and saves `layernorm-forward.png` and
`layernorm-backward.png`. Real benchmark numbers are intentionally not included
here because this solution was prepared on a machine without an NVIDIA GPU; they
must be measured on the target CUDA hardware.

References:

- [Assignment repository](https://github.com/mrapplexz/hw-misis-26/tree/master/homework)
- [Official Triton LayerNorm tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
- [Triton autotune API](https://triton-lang.org/main/python-api/generated/triton.autotune.html)
