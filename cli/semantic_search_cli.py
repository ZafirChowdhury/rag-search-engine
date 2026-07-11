import argparse

from lib.semantic_search_command import verify_model_command, embed_text_command, verify_embeddings_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # verify
    subparsers.add_parser("verify", help="Verify if the model is loaded properly")

    # embed_text
    embed_text_parser = subparsers.add_parser("embed_text", help="Generate an embedding for a single text")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    # build/verify embed
    subparsers.add_parser(
            "verify_embeddings", help="Verify embeddings for the movie dataset"
        )

    args = parser.parse_args()

    match args.command:
        case "verify_embeddings":
            verify_embeddings_command()
        case "embed_text":
            embed_text_command(args.text)
        case "verify":
            verify_model_command()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
