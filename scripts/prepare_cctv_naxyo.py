"""Download and verify Roboflow Universe `dataset-uutxr/cctv-naxyo` (COCO format) cleanly using `ROBOFLOW_API_KEY`."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download cctv-naxyo (COCO) into datasets/cctv-naxyo/.")
    parser.add_argument("--version", type=int, default=2, choices=[1, 2], help="2 = no augment (3,737 imgs, default), 1 = augment (9,857 imgs)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "datasets" / "cctv-naxyo")
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("ROBOFLOW_API_KEY environment variable is required to fetch Roboflow export metadata.")

    meta_url = f"https://api.roboflow.com/dataset-uutxr/cctv-naxyo/{args.version}/coco?api_key={api_key}"
    with urllib.request.urlopen(urllib.request.Request(meta_url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30) as resp:
        meta = json.loads(resp.read().decode("utf-8"))

    link = meta["export"]["link"]
    size_mb = meta["export"].get("size", 0.0)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = args.out_dir.parent / f"cctv-naxyo_v{args.version}_coco.zip"

    print(f"[ROBOFLOW] Downloading dataset-uutxr/cctv-naxyo v{args.version} (~{size_mb:.1f} MiB)...")
    with urllib.request.urlopen(urllib.request.Request(link, headers={"User-Agent": "Mozilla/5.0"}), timeout=240) as r, open(zip_path, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(args.out_dir)
    zip_path.unlink(missing_ok=True)
    print(f"[ROBOFLOW] Extracted successfully into {args.out_dir}")


if __name__ == "__main__":
    main()
