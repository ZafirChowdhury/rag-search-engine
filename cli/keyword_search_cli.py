import argparse

from lib.keyword_search import build_command, get_idf, get_tf_helper, search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser("tf", help="Lookup term frequency")
    search_parser.add_argument(
        "doc_id", type=int, help="Document id of the term frequency"
    )
    search_parser.add_argument("term", type=str, help="Term you want to search")

    search_parser = subparsers.add_parser(
        "idf", help="Get Inverse Document Frequency of a term"
    )
    search_parser.add_argument("idf_term", help="Term that you want to find the idf of")

    args = parser.parse_args()

    match args.command:
        case "idf":
            print(f"{get_idf(args.idf_term):.2f}")
        case "tf":
            print(get_tf_helper(args.doc_id, args.term))
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
