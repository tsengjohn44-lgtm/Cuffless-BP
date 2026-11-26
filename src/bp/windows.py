"""Cut records into 10 s windows, apply quality filters, label SBP/DBP.

Owner: Data + baselines track. Target: Phase 1. Formats: docs/data-contract.md, section 1.
Placeholder from Phase 0 setup; replace the body of main().
"""

import sys


def main(argv: list[str] | None = None) -> int:
    print("bp.windows is not built yet (Data + baselines track, Phase 1).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
