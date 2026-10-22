"""Download and unzip the SoccerNet Game State Reconstruction (GSR) dataset.

Usage:
    uv run python scripts/download_gsr.py
    uv run python scripts/download_gsr.py --splits valid
    uv run python scripts/download_gsr.py --zips /workspace/datasets/SoccerNetGS --out data/SoccerNetGS

--zips is where the archives are downloaded and kept; --out is where they are unpacked
(default: next to the zips). On the pod the zips stay on the network volume and the
~123,000 frames go to the fast local disk.
"""

import argparse
import zipfile
from pathlib import Path

from SoccerNet.Downloader import SoccerNetDownloader

TASK = "gamestate-2024"
MARKER = ".extracted"  # written only once a whole archive is unpacked


def extract(archive: Path, target: Path) -> None:
    """Unpack one split into `target`. A folder without the marker was interrupted
    mid-unpack (a stopped pod, say), so it is unpacked again rather than trusted."""
    if (target / MARKER).exists():
        print(f"skip {archive.name}: already extracted to {target}")
        return
    with zipfile.ZipFile(archive) as zf:
        # Some archives already contain a top-level folder named after the split.
        nested = all(name.startswith(f"{target.name}/") for name in zf.namelist())
        dest = target.parent if nested else target
        dest.mkdir(parents=True, exist_ok=True)
        print(f"extracting {archive} -> {target}")
        zf.extractall(dest)
    (target / MARKER).write_text("ok\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SoccerNet GSR data.")
    parser.add_argument("--zips", type=Path, default=Path("data/SoccerNetGS"), help="where archives are kept")
    parser.add_argument("--out", type=Path, default=None, help="where to unpack (default: same as --zips)")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    args = parser.parse_args()
    out = args.out or args.zips

    targets = {split: out / TASK / split for split in args.splits}
    archives = {split: args.zips / TASK / f"{split}.zip" for split in args.splits}
    needed = [s for s in args.splits if not (targets[s] / MARKER).exists()]
    missing = [s for s in needed if not archives[s].exists()]
    if missing:
        args.zips.mkdir(parents=True, exist_ok=True)
        SoccerNetDownloader(LocalDirectory=str(args.zips)).downloadDataTask(task=TASK, split=missing)
    for split in needed:
        extract(archives[split], targets[split])


if __name__ == "__main__":
    main()
