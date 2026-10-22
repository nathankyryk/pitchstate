"""Download and unzip the SoccerNet Game State Reconstruction (GSR) dataset.

Usage:
    uv run python scripts/download_gsr.py
    uv run python scripts/download_gsr.py --splits valid --out data/SoccerNetGS
"""

import argparse
import zipfile
from pathlib import Path

from SoccerNet.Downloader import SoccerNetDownloader

MARKER = ".extracted"  # written only once a whole archive is unpacked


def extract(archive: Path) -> None:
    """Unpack a split. A folder without the marker was interrupted mid-unpack (a stopped pod,
    say), so it is unpacked again rather than trusted."""
    target = archive.with_suffix("")
    if (target / MARKER).exists():
        print(f"skip {archive.name}: already extracted")
        return
    with zipfile.ZipFile(archive) as zf:
        # Some archives already contain a top-level folder named after the split.
        nested = all(name.startswith(f"{target.name}/") for name in zf.namelist())
        dest = archive.parent if nested else target
        print(f"extracting {archive.name} -> {dest}")
        zf.extractall(dest)
    (target / MARKER).write_text("ok\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SoccerNet GSR data.")
    parser.add_argument("--out", type=Path, default=Path("data/SoccerNetGS"))
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    SoccerNetDownloader(LocalDirectory=str(args.out)).downloadDataTask(
        task="gamestate-2024", split=args.splits
    )
    for archive in sorted(args.out.rglob("*.zip")):
        extract(archive)


if __name__ == "__main__":
    main()
