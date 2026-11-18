"""Command-line entry point for the momentum persistence ablation."""

import argparse
from pathlib import Path
from os import environ

# Limit the cost of repeated small matrix products. Respect an explicit setting.
environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pandas as pd

from baskets import build_panel, load_inputs
from evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Build basket features, outcomes, and coverage.")
    build.add_argument("--prices", required=True, type=Path)
    build.add_argument("--volume", required=True, type=Path)
    build.add_argument("--start", type=pd.Timestamp, help="Earliest formation date to emit.")
    build.add_argument("--through", type=pd.Timestamp, help="Last source date to read, inclusive.")
    score = commands.add_parser("evaluate", help="Tune on walk-forward validation and report results.")
    score.add_argument("--holdout", action="store_true", help="Explicitly open the final holdout.")
    for command in (build, score):
        command.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    if args.command == "build":
        prices, volumes = load_inputs(args.prices, args.volume)
        panel, coverage = build_panel(prices, volumes, args.start, args.through)
        args.output.mkdir(parents=True, exist_ok=True)
        panel.to_csv(args.output / "basket_panel.csv", index=False)
        coverage.to_csv(args.output / "coverage.csv", index=False)
        print(f"Wrote {len(panel):,} basket rows and {len(coverage):,} session coverage rows.")
    else:
        summary = evaluate(
            args.output / "basket_panel.csv",
            args.output / "coverage.csv",
            args.output,
            include_holdout=args.holdout,
        )
        print(f"Selected lambda: baseline={summary['lambda_base']}, "
              f"with persistence={summary['lambda_persistence']}.")
        print("Wrote holdout and validation results." if args.holdout else "Wrote validation results.")


if __name__ == "__main__":
    main()
