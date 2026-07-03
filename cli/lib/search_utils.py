import json
import os
from typing import TypedDict


class Movie(TypedDict):
    id: int
    title: str
    description: str


DEFAULT_SEARCH_LIMIT = 5

# __file__ -> current file
# os.path.dirname() -> drops the file name
# os.path.dirname() -> os.path.dirname() -> os.path.dirname() -> project root -> cli -> lib
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")


def load_movies() -> list[Movie]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]
