"""Dataset preparation and distribution-matching script for Team_A.

Supports:
  --calib-only  Fast fetch of exact disjoint Calib_200 images (default) into calibration_data/.
  --calib-400   Fetch full Calib_400 pool referenced by datasets/dataset_manifest.json.
  --full        Reproduce full 7,500-image test suite (val2017 + matched 2,500 train2017) + Calib_400 + COCO GT JSON.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import sys
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dst: Path, retries: int = 3) -> bool:
    """Download a single URL atomically if absent or empty."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return True
    for attempt in range(retries):
        try:
            tmp = dst.with_suffix(dst.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(dst)
            return True
        except Exception:
            time.sleep(0.3 * (attempt + 1))
    return False


def populate_calibration_from_manifest(
    manifest_path: Path,
    calib_dir: Path,
    include_400: bool = False,
    workers: int = 24,
) -> list[Path]:
    """Download exact disjoint calibration images listed in `dataset_manifest.json`."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [
        r for r in manifest["calib_records"]
        if include_400 or r.get("in_calib200", False)
    ]
    calib_dir.mkdir(parents=True, exist_ok=True)

    desired = {r["file_name"] for r in records}
    for old_f in calib_dir.glob("*.jpg"):
        if old_f.name not in desired:
            old_f.unlink()

    jobs = [
        (f"http://images.cocodataset.org/train2017/{r['file_name']}", calib_dir / r["file_name"])
        for r in records
    ]
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        ok_flags = list(ex.map(lambda j: download_file(j[0], j[1]), jobs))

    failed = [jobs[i][1].name for i, ok in enumerate(ok_flags) if not ok]
    if failed:
        raise RuntimeError(f"Failed to download {len(failed)} calibration images: {failed[:5]}")

    paths = sorted(calib_dir.glob("*.jpg"))
    bad = [p.name for p in paths if cv2.imread(str(p)) is None]
    if bad:
        raise RuntimeError(f"Unreadable JPEG files in {calib_dir}: {bad[:5]}")

    print(f"[CALIB] Verified {len(paths)} disjoint images in {calib_dir} ({time.perf_counter() - t0:.2f}s)")
    return paths


def compute_distribution_stats(coco_dict: dict, img_id_subset: set[int] | None = None) -> dict:
    """Compute pos/neg balance, COCO S/M/L box ratio, mean persons/img, and co-object rates."""
    imgs = (
        coco_dict["images"]
        if img_id_subset is None
        else [im for im in coco_dict["images"] if im["id"] in img_id_subset]
    )
    id_set = {im["id"] for im in imgs}
    small = med = large = 0
    co_cats: Counter[int] = Counter()
    imgs_with_person: set[int] = set()

    for ann in coco_dict["annotations"]:
        iid = ann["image_id"]
        if iid not in id_set:
            continue
        if ann["category_id"] == 1 and ann.get("iscrowd", 0) == 0:
            imgs_with_person.add(iid)
            ar = ann["area"]
            if ar < 1024:
                small += 1
            elif ar < 9216:
                med += 1
            else:
                large += 1

    for ann in coco_dict["annotations"]:
        if ann["image_id"] in imgs_with_person and ann["category_id"] != 1:
            co_cats[ann["category_id"]] += 1

    total_boxes = max(1, small + med + large)
    n_img = max(1, len(imgs))
    n_pos = len(imgs_with_person)
    cat_names = {c["id"]: c["name"] for c in coco_dict["categories"]}
    return {
        "n_img": len(imgs),
        "pos_pct": round(100.0 * n_pos / n_img, 1),
        "neg_pct": round(100.0 * (len(imgs) - n_pos) / n_img, 1),
        "boxes": small + med + large,
        "small_pct": round(100.0 * small / total_boxes, 1),
        "med_pct": round(100.0 * med / total_boxes, 1),
        "large_pct": round(100.0 * large / total_boxes, 1),
        "mean_person_per_pos": round((small + med + large) / max(1, n_pos), 2),
        "top_co_objects": [(cat_names[k], round(v / max(1, n_pos), 3)) for k, v in co_cats.most_common(6)],
    }


def _allocate_quotas(strata_counts: dict, total_n: int) -> dict:
    """Hamilton / largest-remainder proportional allocation across strata."""
    pop_total = sum(strata_counts.values())
    exact = {k: (v / pop_total) * total_n for k, v in strata_counts.items()}
    alloc = {k: int(exact[k]) for k in strata_counts}
    rem = total_n - sum(alloc.values())
    for k in sorted(strata_counts.keys(), key=lambda x: exact[x] - alloc[x], reverse=True)[:rem]:
        alloc[k] += 1
    return alloc


def build_full_datasets(data_dir: Path, calib_dir: Path, seed: int = 20261117, workers: int = 24) -> None:
    """Reconstruct full 7,500-image curated test set + 400-image disjoint calibration pool from COCO 2017."""
    coco_dir = data_dir / "coco"
    ann_dir = coco_dir / "annotations"
    val_dir = coco_dir / "val2017"
    train_sub_dir = coco_dir / "train2017_curated"
    calib_400_dir = data_dir / "calibration_matched"
    for d in (coco_dir, train_sub_dir, calib_400_dir, calib_dir):
        d.mkdir(parents=True, exist_ok=True)

    ann_zip = coco_dir / "annotations_trainval2017.zip"
    if not download_file("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", ann_zip):
        raise RuntimeError("Failed to download annotations_trainval2017.zip")
    if not (ann_dir / "instances_val2017.json").exists() or not (ann_dir / "instances_train2017.json").exists():
        with zipfile.ZipFile(ann_zip, "r") as zf:
            zf.extractall(coco_dir)

    val_zip = coco_dir / "val2017.zip"
    if not download_file("http://images.cocodataset.org/zips/val2017.zip", val_zip):
        raise RuntimeError("Failed to download val2017.zip")
    if not val_dir.exists() or len(list(val_dir.glob("*.jpg"))) < 5000:
        with zipfile.ZipFile(val_zip, "r") as zf:
            zf.extractall(coco_dir)

    val_coco_raw = json.loads((ann_dir / "instances_val2017.json").read_text(encoding="utf-8"))
    train_coco_raw = json.loads((ann_dir / "instances_train2017.json").read_text(encoding="utf-8"))

    train_img_map = {im["id"]: im for im in train_coco_raw["images"]}
    train_person_anns: dict[int, list] = {iid: [] for iid in train_img_map}
    train_other_cats: dict[int, set[int]] = {iid: set() for iid in train_img_map}
    train_crowd_imgs: set[int] = set()

    for ann in train_coco_raw["annotations"]:
        iid = ann["image_id"]
        if ann["category_id"] == 1:
            if ann.get("iscrowd", 0) == 1:
                train_crowd_imgs.add(iid)
            elif ann.get("area", 0) > 0:
                train_person_anns[iid].append(ann)
        else:
            train_other_cats[iid].add(ann["category_id"])

    INDOOR_CATS = {62, 63, 65, 67, 72, 84}
    VEHICLE_CATS = {2, 3, 4, 6, 8}
    strata: dict[tuple, list[int]] = {}
    for iid, im_info in train_img_map.items():
        if im_info.get("width", 0) < 128 or im_info.get("height", 0) < 128:
            continue
        anns = train_person_anns[iid]
        oc = train_other_cats[iid]
        ctx = "indoor" if (oc & INDOOR_CATS) else ("vehicle" if (oc & VEHICLE_CATS) else "other")
        if not anns:
            if iid in train_crowd_imgs:
                continue
            key = ("neg", 0, "none", ctx)
        else:
            n = len(anns)
            cnt_bin = "1" if n == 1 else ("2_3" if n <= 3 else ("4_7" if n <= 7 else "8p"))
            med_ar = float(np.median([a["area"] for a in anns]))
            sc_bin = "S" if med_ar < 1024 else ("M" if med_ar < 9216 else "L")
            key = ("pos", cnt_bin, sc_bin, ctx)
        strata.setdefault(key, []).append(iid)

    rng = np.random.default_rng(seed)
    for k in strata:
        arr = np.array(sorted(strata[k]), dtype=np.int64)
        rng.shuffle(arr)
        strata[k] = arr.tolist()

    strata_counts = {k: len(v) for k, v in strata.items()}
    test_b_quotas = _allocate_quotas(strata_counts, 2500)
    test_b_ids: list[int] = []
    test_b_stratum: dict[int, str] = {}
    for k, q in sorted(test_b_quotas.items()):
        chosen = strata[k][:q]
        test_b_ids.extend(chosen)
        for iid in chosen:
            test_b_stratum[iid] = f"{k[0]}_{k[1]}_{k[2]}_{k[3]}"
    test_b_set = set(test_b_ids)

    rem_strata = {k: [iid for iid in v if iid not in test_b_set] for k, v in strata.items()}
    rem_counts = {k: len(v) for k, v in rem_strata.items()}
    c200_q = _allocate_quotas(rem_counts, 200)
    c400_q = _allocate_quotas(rem_counts, 400)

    calib_200_ids: list[int] = []
    calib_extra_ids: list[int] = []
    calib_stratum: dict[int, str] = {}
    for k in sorted(rem_strata.keys()):
        q200 = c200_q[k]
        q400 = max(q200, c400_q[k])
        c200 = rem_strata[k][:q200]
        cextra = rem_strata[k][q200:q400]
        calib_200_ids.extend(c200)
        calib_extra_ids.extend(cextra)
        for iid in c200 + cextra:
            calib_stratum[iid] = f"{k[0]}_{k[1]}_{k[2]}_{k[3]}"

    calib_400_ids = calib_200_ids + calib_extra_ids[: 400 - len(calib_200_ids)]
    assert len(test_b_ids) == 2500 and len(calib_200_ids) == 200 and len(calib_400_ids) == 400
    assert len(test_b_set & set(calib_400_ids)) == 0

    jobs = [
        (f"http://images.cocodataset.org/train2017/{train_img_map[iid]['file_name']}", train_sub_dir / train_img_map[iid]["file_name"])
        for iid in test_b_ids
    ] + [
        (f"http://images.cocodataset.org/train2017/{train_img_map[iid]['file_name']}", calib_400_dir / train_img_map[iid]["file_name"])
        for iid in calib_400_ids
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        ok_flags = list(ex.map(lambda j: download_file(j[0], j[1]), jobs))
    if not all(ok_flags):
        raise RuntimeError("Some COCO train2017 images failed to download.")

    test_records = [
        {
            "image_id": im["id"],
            "file_name": im["file_name"],
            "path": str((val_dir / im["file_name"]).relative_to(REPO_ROOT).as_posix()),
            "slice": "val2017_official",
            "bucket": "val2017",
            "width": im["width"],
            "height": im["height"],
        }
        for im in val_coco_raw["images"]
    ] + [
        {
            "image_id": iid,
            "file_name": train_img_map[iid]["file_name"],
            "path": str((train_sub_dir / train_img_map[iid]["file_name"]).relative_to(REPO_ROOT).as_posix()),
            "slice": "matched_train_2500",
            "bucket": test_b_stratum[iid],
            "width": train_img_map[iid]["width"],
            "height": train_img_map[iid]["height"],
        }
        for iid in test_b_ids
    ]

    calib_200_set = set(calib_200_ids)
    calib_records = [
        {
            "image_id": iid,
            "file_name": train_img_map[iid]["file_name"],
            "path": str((calib_400_dir / train_img_map[iid]["file_name"]).relative_to(REPO_ROOT).as_posix()),
            "in_calib200": iid in calib_200_set,
            "bucket": calib_stratum[iid],
            "width": train_img_map[iid]["width"],
            "height": train_img_map[iid]["height"],
        }
        for iid in calib_400_ids
    ]

    for rec in calib_records:
        if rec["in_calib200"]:
            shutil.copy2(REPO_ROOT / rec["path"], calib_dir / rec["file_name"])

    def _audit_record(rec: dict) -> tuple[bool, str]:
        p = REPO_ROOT / rec["path"]
        img = cv2.imread(str(p))
        ok = img is not None and img.ndim == 3
        return ok, sha256_file(p) if ok else ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        t_chk = list(ex.map(_audit_record, test_records))
        c_chk = list(ex.map(_audit_record, calib_records))

    assert all(ok for ok, _ in t_chk) and all(ok for ok, _ in c_chk)
    overlap = {h for _, h in t_chk} & {h for _, h in c_chk}
    assert len(overlap) == 0

    val_person_anns = [a for a in val_coco_raw["annotations"] if a["category_id"] == 1]
    train_person_b = [a for a in train_coco_raw["annotations"] if a["category_id"] == 1 and a["image_id"] in test_b_set]
    base_id = max((a["id"] for a in val_person_anns), default=0) + 1_000_000
    reindexed = [dict(a, id=base_id + i) for i, a in enumerate(train_person_b)]
    gt_json = {
        "images": [{"id": r["image_id"], "file_name": r["file_name"], "width": r["width"], "height": r["height"]} for r in test_records],
        "annotations": val_person_anns + reindexed,
        "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
    }
    (data_dir / "test_7500_person_gt.json").write_text(json.dumps(gt_json), encoding="utf-8")
    manifest = {
        "test_count": len(test_records),
        "calib_count": len(calib_records),
        "calib200_count": len(calib_200_set),
        "sha256_overlap": len(overlap),
        "test_records": test_records,
        "calib_records": calib_records,
    }
    (data_dir / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    print("[DATASET] Rebuilt 7,500 test + 400 calib images with SHA-256 overlap = 0.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare calibration_data/ or full 7,500-image benchmark datasets.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "datasets")
    parser.add_argument("--calib-dir", type=Path, default=REPO_ROOT / "calibration_data")
    parser.add_argument("--calib-only", action="store_true", help="Populate calibration_data/ (200 images) from manifest.")
    parser.add_argument("--calib-400", action="store_true", help="Populate 400 calibration images from manifest.")
    parser.add_argument("--full", action="store_true", help="Download val2017 + reconstruct full 7,500 test + 400 calib sets.")
    args = parser.parse_args()

    manifest_path = args.data_dir / "dataset_manifest.json"
    if args.full or not manifest_path.exists():
        build_full_datasets(args.data_dir, args.calib_dir)
    else:
        populate_calibration_from_manifest(manifest_path, args.calib_dir, include_400=args.calib_400)


if __name__ == "__main__":
    main()
