"""Run the published colour-retrieval example with local model and artifact files."""
import argparse
import json
import sys

from latentwire import _fixed_reader
from latentwire._reader_errors import ReaderError


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ReaderError("invalid_inputs")


def main(argv=None):
    parser = _Parser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifacts", required=True)
    try:
        args = parser.parse_args(argv)
        value = _fixed_reader.run_example(args.model, args.artifacts)
    except _fixed_reader._OWNER_ERRORS as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
