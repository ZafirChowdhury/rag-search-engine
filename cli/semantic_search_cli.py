import argparse

from lib.semantic_search_command import (
    embed_query_text_command,
    embed_text_command,
    verify_embeddings_command,
    verify_model_command,
    search_command,
    chunk_command,
)

from lib.search_utils import DEFAULT_SEARCH_LIMIT, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

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

    # embed_query
    embed_query_parser = subparsers.add_parser("embed_query", help="generate embeding for user query")
    embed_query_parser.add_argument("query", help="query to be embeded")

    # search
    search_parser = subparsers.add_parser("search", help="Semantic search")
    search_parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )
    search_parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Maximum number of results to return")

    # chunk
    chunk_parser = subparsers.add_parser(
        "chunk",
        help="Split text into chunks"
    )
    chunk_parser.add_argument(
        "text",
        type=str,
        help="Text to chunk"
    )
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Maximum chunk size"
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Overlap size"
    )

    args = parser.parse_args()

    match args.command:
        case "chunk":
            chunk_command(args.text, args.chunk_size, args.overlap)
        case "search":
            search_command(args.query, args.limit)
        case "embed_query":
            embed_query_text_command(args.query)
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
