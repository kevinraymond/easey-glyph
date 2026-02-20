#!/usr/bin/env python3
"""Push EASEy-GLYPH models to HuggingFace Hub.

Uploads all safetensors, JSON sidecars, README, and sample images from the
models/ directory to a HuggingFace repository.

Usage:
    uv run scripts/push_to_hf.py --dry-run        # preview files + sizes
    uv run scripts/push_to_hf.py --create-repo     # create repo + upload
    uv run scripts/push_to_hf.py                   # upload to existing repo
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_REPO_ID = "kjraym/easey-glyph"
MODELS_DIR = Path("models")

ALLOW_PATTERNS = [
    "*.safetensors",
    "*.json",
    "README.md",
    "images/*.png",
]


def get_upload_files(models_dir: Path) -> list[Path]:
    """Collect all files matching upload patterns."""
    files = []
    files.extend(models_dir.glob("*.safetensors"))
    files.extend(models_dir.glob("*.json"))
    readme = models_dir / "README.md"
    if readme.exists():
        files.append(readme)
    files.extend(models_dir.glob("images/*.png"))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Push models to HuggingFace Hub")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID,
                        help=f"HuggingFace repo ID (default: {DEFAULT_REPO_ID})")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files and sizes without uploading")
    parser.add_argument("--create-repo", action="store_true",
                        help="Create the repo if it doesn't exist")
    args = parser.parse_args()

    if not MODELS_DIR.exists():
        print(f"Error: {MODELS_DIR}/ not found. Run export_safetensors.py first.")
        return

    # List files
    files = get_upload_files(MODELS_DIR)
    total_bytes = sum(f.stat().st_size for f in files)

    print(f"Repository: {args.repo_id}")
    print(f"Files: {len(files)}")
    print(f"Total size: {total_bytes / 1e9:.2f} GB\n")

    for f in files:
        size = f.stat().st_size
        rel = f.relative_to(MODELS_DIR)
        if size > 1e6:
            print(f"  {rel} ({size / 1e6:.1f} MB)")
        else:
            print(f"  {rel} ({size / 1e3:.1f} KB)")

    if args.dry_run:
        print("\n--dry-run: no upload performed.")
        return

    # Authenticate
    api = HfApi()
    try:
        user = api.whoami()
        print(f"\nAuthenticated as: {user['name']}")
    except Exception as e:
        print(f"\nError: Not authenticated with HuggingFace Hub.")
        print(f"Run `huggingface-cli login` first.\n{e}")
        return

    # Create repo if requested
    if args.create_repo:
        api.create_repo(repo_id=args.repo_id, exist_ok=True, repo_type="model")
        print(f"Repo {args.repo_id} ready.")

    # Upload
    print(f"\nUploading {len(files)} files to {args.repo_id}...")
    api.upload_folder(
        folder_path=str(MODELS_DIR),
        repo_id=args.repo_id,
        repo_type="model",
        allow_patterns=ALLOW_PATTERNS,
        commit_message=f"Upload all EASEy-GLYPH models (7 variants x 3 models + images)",
    )

    print(f"\nDone! Visit: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
