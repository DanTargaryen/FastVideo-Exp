from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_REPO_ID = "YuryyyLee/world-renderer-controlnet-union"
DEFAULT_LOCAL_DIR = "/vePFS-buaa/linming/workspace/worldrender/world-renderer-controlnet-union"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face repository snapshot for local experiments."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Model or dataset repo id to download (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--local-dir",
        default=DEFAULT_LOCAL_DIR,
        help=f"Destination directory (default: {DEFAULT_LOCAL_DIR})",
    )
    parser.add_argument(
        "--repo-type",
        choices=("model", "dataset", "space"),
        default="model",
        help="Hugging Face repository type.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. If omitted, use HF_TOKEN or HUGGINGFACE_HUB_TOKEN.",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=None,
        help="Optional glob of files to include. Can be passed multiple times.",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=None,
        help="Optional glob of files to skip. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from huggingface_hub import login, snapshot_download

    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    if token:
        login(token=token)

    local_dir = Path(args.local_dir).expanduser()
    local_dir.parent.mkdir(parents=True, exist_ok=True)

    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=args.allow_pattern,
        ignore_patterns=args.ignore_pattern,
        token=token,
    )
    print("✅ Downloaded to:", path)


if __name__ == "__main__":
    main()
