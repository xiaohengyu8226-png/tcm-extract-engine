"""Explicit migration defaults to dry run. --apply writes backed-up artifacts."""

import argparse
import json
from pathlib import Path

from tcm_extract_engine.core.migration import migrate_book


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("books", nargs="+", type=Path, help="Book workspace directories")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    for book in args.books:
        print(json.dumps(migrate_book(book, dry_run=not args.apply), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
