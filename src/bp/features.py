"""PPG (and optional ECG) features per window, each a median over beats.

Owner: Model track. Target: Phase 2. Formats: docs/data-contract.md, section 3.
Placeholder from Phase 0 setup; replace the body of main().
"""

import sys


def main(argv: list[str] | None = None) -> int:
    print("bp.features is not built yet (Model track, Phase 2).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
