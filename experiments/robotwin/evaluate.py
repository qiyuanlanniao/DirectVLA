"""Launch the public RoboTwin clean50 evaluation workflow."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to a released TurboVLA checkpoint")
    parser.add_argument(
        "tasks",
        nargs="*",
        help="Optional RoboTwin task names; omit to evaluate all clean50 tasks",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(
        os.environ.get("TURBOVLA_REPO_ROOT", Path(__file__).resolve().parents[2])
    ).resolve()
    script = repo_root / "scripts" / "robotwin" / "evaluate.sh"
    if not script.is_file():
        raise FileNotFoundError(f"RoboTwin evaluation script not found: {script}")
    subprocess.run(
        ["bash", str(script), args.checkpoint, *args.tasks],
        cwd=repo_root,
        check=True,
    )


if __name__ == "__main__":
    main()
