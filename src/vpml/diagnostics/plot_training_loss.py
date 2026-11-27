"""Regenerate an interface-flux training-loss figure from saved metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from vpml.visualization.training import save_training_loss_plot_from_metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    output = save_training_loss_plot_from_metrics(args.metrics, args.output)
    print(f"Saved training-loss plot to {output}")


if __name__ == "__main__":
    main()
