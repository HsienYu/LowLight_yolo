#!/usr/bin/env python3
"""Download / setup helper for Zero-DCE++ weights.

This repo expects the Zero-DCE++ weights at:
  models/zero_dce_plus.pth

Because large weights are typically not committed to git, this script:
- Prints the recommended manual download steps
- Optionally creates a dummy (random) weights file for smoke-testing the pipeline

Usage:
  python scripts/download_zero_dce_weights.py
  python scripts/download_zero_dce_weights.py --create-dummy
  python scripts/download_zero_dce_weights.py --output models/zero_dce_plus.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path


def repo_root() -> Path:
    # scripts/ is one level under repo root
    return Path(__file__).resolve().parent.parent


def default_output_path() -> Path:
    return repo_root() / "models" / "zero_dce_plus.pth"


def print_instructions(out_path: Path) -> None:
    print("Zero-DCE++ weights setup")
    print("=" * 60)
    print(f"Expected path: {out_path}")
    print(f"Exists: {out_path.exists()}")
    print()
    print("Recommended: official pretrained weights (manual)")
    print("1) Visit: https://github.com/Li-Chongyi/Zero-DCE_extension")
    print("2) Download: Pretrained_model/Epoch99.pth")
    print(f"3) Save as: {out_path}")
    print()
    print("If you want to only smoke-test the pipeline (quality will be poor):")
    print("  python scripts/download_zero_dce_weights.py --create-dummy")


def create_dummy_weights(out_path: Path) -> None:
    # Import lazily so this script can still print instructions even if torch isn't installed.
    from zero_dce import DCENet  # type: ignore
    import torch  # type: ignore

    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = DCENet()
    state_dict = model.state_dict()  # random init

    torch.save(state_dict, out_path)
    print("\n✅ Created dummy Zero-DCE++ weights (random init)")
    print(f"Saved to: {out_path}")
    print("NOTE: This is only for testing that the pipeline runs. Detection quality will be poor.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Zero-DCE++ weights setup helper")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for weights (default: models/zero_dce_plus.pth)",
    )
    parser.add_argument(
        "--create-dummy",
        action="store_true",
        help="Create a dummy (random) weights file for testing",
    )

    args = parser.parse_args()

    out_path = Path(args.output) if args.output else default_output_path()

    print_instructions(out_path)

    if args.create_dummy:
        try:
            create_dummy_weights(out_path)
        except Exception as e:
            print("\n❌ Failed to create dummy weights:")
            print(e)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
