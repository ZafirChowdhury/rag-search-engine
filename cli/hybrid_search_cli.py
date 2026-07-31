import argparse

from lib.hybrid_search import min_max_normalization

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # normalize
    normalize_parser = subparsers.add_parser("normalize", help="Normalize a list of values using min-max normalization")
    normalize_parser.add_argument("numbers", type=float, nargs="*", help="List of arguments you want to normalize")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            n_scores = min_max_normalization(args.numbers)
            if len(n_scores) != 0:
                for score in n_scores:
                    print(f"* {score:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
