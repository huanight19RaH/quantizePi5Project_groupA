# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "matplotlib==3.11.2",
#     "numpy==2.5.5",
#     "onnx==1.23.2",
#     "onnxconverter-common==1.16.0",
#     "onnxscript==0.7.2",
#     "onnxslim==0.1.99",
#     "opencv-python==5.0.0.93",
#     "pycocotools==2.0.11",
#     "torch==2.15.0",
#     "ultralytics==8.4.215",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def setup_env():
    import marimo as mo
    import concurrent.futures
    import hashlib
    import json
    import os
    import re
    import time
    import urllib.request
    import zipfile
    from pathlib import Path

    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import onnx
    import onnxruntime as ort
    import pandas as pd
    import torch
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from ultralytics import YOLO

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    cv2.setNumThreads(1)

    WORK_DIR = Path("/marimo/yolo26_quant_workspace")
    DATA_DIR = WORK_DIR / "datasets"
    MODELS_DIR = WORK_DIR / "models"
    RESULTS_DIR = WORK_DIR / "results"
    for _d in (WORK_DIR, DATA_DIR, MODELS_DIR, RESULTS_DIR):
        _d.mkdir(parents=True, exist_ok=True)

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-only"

    mo.md(
        f'''
        ### Workspace & Environment Ready
        - **PyTorch:** `{torch.__version__}` | **Device:** `{DEVICE}` (`{GPU_NAME}`)
        - **ONNX:** `{onnx.__version__}` | **ONNX Runtime:** `{ort.__version__}` (`{ort.get_available_providers()}`)
        - **CPU Cores:** `{os.cpu_count()}` | **OpenCV:** `{cv2.__version__}`
        - **Workspace:** `{WORK_DIR}`
        '''
    )
    return (
        COCO,
        COCOeval,
        DATA_DIR,
        DEVICE,
        MODELS_DIR,
        Path,
        RESULTS_DIR,
        YOLO,
        concurrent,
        cv2,
        hashlib,
        json,
        mo,
        np,
        onnx,
        ort,
        pd,
        plt,
        time,
        torch,
        urllib,
        zipfile,
    )


@app.cell
def build_curated_datasets(
    COCO,
    DATA_DIR,
    Path,
    concurrent,
    cv2,
    hashlib,
    json,
    mo,
    np,
    pd,
    time,
    urllib,
    zipfile,
):
    # Cell 1: Distribution-Matched Curated 7,500-image Test Suite & Disjoint Stratified Calibration Pool
    COCO_DIR = DATA_DIR / "coco"
    ANN_DIR = COCO_DIR / "annotations"
    VAL_DIR = COCO_DIR / "val2017"
    TRAIN_SUB_DIR = COCO_DIR / "train2017_curated"
    CALIB_DIR = DATA_DIR / "calibration_matched"
    MANIFEST_PATH = DATA_DIR / "dataset_manifest.json"
    TEST_GT_JSON = DATA_DIR / "test_7500_person_gt.json"

    def _download_file(url: str, dst: Path) -> Path:
        if dst.exists() and dst.stat().st_size > 0:
            return dst
        tmp = dst.with_suffix(dst.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dst)
        return dst

    def _sha256(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while ch := f.read(65536):
                h.update(ch)
        return h.hexdigest()

    def _compute_dist_stats(coco_d: dict, img_id_subset: set = None) -> dict:
        from collections import Counter
        imgs = coco_d["images"] if img_id_subset is None else [im for im in coco_d["images"] if im["id"] in img_id_subset]
        id_set = {im["id"] for im in imgs}
        s = m = l = 0
        co_cats = Counter()
        imgs_with_person = set()
        for ann in coco_d["annotations"]:
            iid = ann["image_id"]
            if iid not in id_set:
                continue
            if ann["category_id"] == 1 and ann.get("iscrowd", 0) == 0:
                imgs_with_person.add(iid)
                ar = ann["area"]
                if ar < 1024:
                    s += 1
                elif ar < 9216:
                    m += 1
                else:
                    l += 1
        for ann in coco_d["annotations"]:
            if ann["image_id"] in imgs_with_person and ann["category_id"] != 1:
                co_cats[ann["category_id"]] += 1
        tot = max(1, s + m + l)
        n_img = max(1, len(imgs))
        n_pos = len(imgs_with_person)
        cat_names = {c["id"]: c["name"] for c in coco_d["categories"]}
        return {
            "n_img": len(imgs),
            "pos_pct": round(100.0 * n_pos / n_img, 1),
            "neg_pct": round(100.0 * (len(imgs) - n_pos) / n_img, 1),
            "boxes": s + m + l,
            "small_pct": round(100.0 * s / tot, 1),
            "med_pct": round(100.0 * m / tot, 1),
            "large_pct": round(100.0 * l / tot, 1),
            "mean_person_per_pos": round((s + m + l) / max(1, n_pos), 2),
            "top_co_objects": [(cat_names[k], round(v / max(1, n_pos), 3)) for k, v in co_cats.most_common(6)],
        }

    def _build_curated_datasets():
        for d in (COCO_DIR, TRAIN_SUB_DIR, CALIB_DIR):
            d.mkdir(parents=True, exist_ok=True)

        ann_zip = _download_file(
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            COCO_DIR / "annotations_trainval2017.zip",
        )
        if not (ANN_DIR / "instances_val2017.json").exists() or not (ANN_DIR / "instances_train2017.json").exists():
            with zipfile.ZipFile(ann_zip, "r") as zf:
                zf.extractall(COCO_DIR)

        val_zip = _download_file("http://images.cocodataset.org/zips/val2017.zip", COCO_DIR / "val2017.zip")
        if not VAL_DIR.exists() or len(list(VAL_DIR.glob("*.jpg"))) < 5000:
            with zipfile.ZipFile(val_zip, "r") as zf:
                zf.extractall(COCO_DIR)

        with open(ANN_DIR / "instances_val2017.json", "r") as f:
            val_coco_raw = json.load(f)
        with open(ANN_DIR / "instances_train2017.json", "r") as f:
            train_coco_raw = json.load(f)

        # Index train2017 person boxes & co-occurring context per image
        train_img_map = {im["id"]: im for im in train_coco_raw["images"]}
        train_person_anns = {iid: [] for iid in train_img_map}
        train_other_cats = {iid: set() for iid in train_img_map}
        train_crowd_imgs = set()

        for ann in train_coco_raw["annotations"]:
            iid = ann["image_id"]
            if ann["category_id"] == 1:
                if ann.get("iscrowd", 0) == 1:
                    train_crowd_imgs.add(iid)
                elif ann.get("area", 0) > 0:
                    train_person_anns[iid].append(ann)
            else:
                train_other_cats[iid].add(ann["category_id"])

        # Assign joint stratum key: (pos/neg) x (count_bin) x (scale_profile) x (context)
        # Context: indoor_furniture (chair=62, dining_table=67, couch=63, bed=65), vehicle (car=3, bus=6, truck=8, moto=4, bike=2), or other
        INDOOR_CATS = {62, 63, 65, 67, 72, 84}
        VEHICLE_CATS = {2, 3, 4, 6, 8}

        strata = {}
        eligible_ids = []
        for iid, im_info in train_img_map.items():
            if im_info.get("width", 0) < 128 or im_info.get("height", 0) < 128:
                continue
            anns = train_person_anns[iid]
            oc = train_other_cats[iid]
            ctx = "indoor" if (oc & INDOOR_CATS) else ("vehicle" if (oc & VEHICLE_CATS) else "other")

            if len(anns) == 0:
                if iid in train_crowd_imgs:
                    continue  # skip ambiguous crowd-only negatives
                key = ("neg", 0, "none", ctx)
            else:
                n = len(anns)
                cnt_bin = "1" if n == 1 else ("2_3" if n <= 3 else ("4_7" if n <= 7 else "8p"))
                # Scale profile by median person box area so S/M/L proportions mirror COCO naturally
                med_ar = float(np.median([a["area"] for a in anns]))
                sc_bin = "S" if med_ar < 1024 else ("M" if med_ar < 9216 else "L")
                key = ("pos", cnt_bin, sc_bin, ctx)

            strata.setdefault(key, []).append(iid)
            eligible_ids.append(iid)

        rng = np.random.default_rng(20261117)
        for k in strata:
            arr = np.array(sorted(strata[k]), dtype=np.int64)
            rng.shuffle(arr)
            strata[k] = arr.tolist()

        # Exact proportional stratified sampling (Hamilton / largest-remainder allocation)
        def _allocate_quotas(strata_counts: dict, total_n: int) -> dict:
            pop_total = sum(strata_counts.values())
            exact = {k: (v / pop_total) * total_n for k, v in strata_counts.items()}
            alloc = {k: int(exact[k]) for k in strata_counts}
            rem = total_n - sum(alloc.values())
            by_frac = sorted(strata_counts.keys(), key=lambda k: exact[k] - alloc[k], reverse=True)
            for k in by_frac[:rem]:
                alloc[k] += 1
            return alloc

        strata_counts = {k: len(v) for k, v in strata.items()}
        # 1) Allocate 2,500 proportional train2017 images for Test Slice B
        test_b_quotas = _allocate_quotas(strata_counts, 2500)
        test_b_ids = []
        test_b_stratum = {}
        for k, q in sorted(test_b_quotas.items()):
            chosen = strata[k][:q]
            test_b_ids.extend(chosen)
            for iid in chosen:
                test_b_stratum[iid] = f"{k[0]}_{k[1]}_{k[2]}_{k[3]}"
        test_b_set = set(test_b_ids)
        assert len(test_b_ids) == 2500

        # 2) Allocate 400 STRICTLY DISJOINT proportional train2017 images for Calibration
        #    (First 200 images allocated proportionally inside the 400 serve as CALIB_200!)
        rem_strata = {k: [iid for iid in v if iid not in test_b_set] for k, v in strata.items()}
        rem_counts = {k: len(v) for k, v in rem_strata.items()}
        calib_200_quotas = _allocate_quotas(rem_counts, 200)
        calib_400_quotas = _allocate_quotas(rem_counts, 400)

        calib_200_ids = []
        calib_extra_ids = []
        calib_stratum = {}
        for k in sorted(rem_strata.keys()):
            q200 = calib_200_quotas[k]
            q400 = max(q200, calib_400_quotas[k])
            c200 = rem_strata[k][:q200]
            cextra = rem_strata[k][q200:q400]
            calib_200_ids.extend(c200)
            calib_extra_ids.extend(cextra)
            for iid in c200 + cextra:
                calib_stratum[iid] = f"{k[0]}_{k[1]}_{k[2]}_{k[3]}"

        # Ensure exact 200 + 200 = 400
        calib_400_ids = calib_200_ids + calib_extra_ids[:400 - len(calib_200_ids)]
        assert len(calib_200_ids) == 200 and len(calib_400_ids) == 400
        assert len(test_b_set & set(calib_400_ids)) == 0, "LEAKAGE detected!"

        # Clean stale downloaded train/calib files if replacing from previous run
        desired_train = {train_img_map[iid]["file_name"] for iid in test_b_ids}
        desired_calib = {train_img_map[iid]["file_name"] for iid in calib_400_ids}
        for old_p in TRAIN_SUB_DIR.glob("*.jpg"):
            if old_p.name not in desired_train:
                old_p.unlink()
        for old_p in CALIB_DIR.glob("*.jpg"):
            if old_p.name not in desired_calib:
                old_p.unlink()

        download_jobs = []
        for iid in test_b_ids:
            fn = train_img_map[iid]["file_name"]
            download_jobs.append((f"http://images.cocodataset.org/train2017/{fn}", TRAIN_SUB_DIR / fn))
        for iid in calib_400_ids:
            fn = train_img_map[iid]["file_name"]
            download_jobs.append((f"http://images.cocodataset.org/train2017/{fn}", CALIB_DIR / fn))

        def _fetch_one(job):
            url, dst = job
            if dst.exists() and dst.stat().st_size > 0:
                return dst.name, True
            for attempt in range(3):
                try:
                    tmp = dst.with_suffix(".part")
                    urllib.request.urlretrieve(url, tmp)
                    tmp.replace(dst)
                    return dst.name, True
                except Exception:
                    time.sleep(0.3 * (attempt + 1))
            return dst.name, False

        t_dl = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
            dl_results = list(ex.map(_fetch_one, download_jobs))
        failed_dl = [fn for fn, ok in dl_results if not ok]
        assert not failed_dl, f"Failed downloads: {failed_dl[:10]}"
        print(f"Prepared {len(download_jobs)} distribution-matched images in {time.perf_counter() - t_dl:.1f}s")

        test_records = []
        for im in val_coco_raw["images"]:
            p = VAL_DIR / im["file_name"]
            test_records.append({
                "image_id": im["id"],
                "file_name": im["file_name"],
                "path": str(p),
                "slice": "val2017_official",
                "bucket": "val2017",
                "width": im["width"],
                "height": im["height"],
            })
        for iid in test_b_ids:
            im = train_img_map[iid]
            p = TRAIN_SUB_DIR / im["file_name"]
            test_records.append({
                "image_id": iid,
                "file_name": im["file_name"],
                "path": str(p),
                "slice": "matched_train_2500",
                "bucket": test_b_stratum[iid],
                "width": im["width"],
                "height": im["height"],
            })

        calib_200_set = set(calib_200_ids)
        calib_records = []
        for iid in calib_400_ids:
            im = train_img_map[iid]
            p = CALIB_DIR / im["file_name"]
            calib_records.append({
                "image_id": iid,
                "file_name": im["file_name"],
                "path": str(p),
                "in_calib200": iid in calib_200_set,
                "bucket": calib_stratum[iid],
                "width": im["width"],
                "height": im["height"],
            })

        def _check_img(rec):
            p = Path(rec["path"])
            img = cv2.imread(str(p))
            ok = img is not None and img.ndim == 3
            return rec["image_id"], ok, _sha256(p) if ok else ""

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
            test_chk = list(ex.map(_check_img, test_records))
            calib_chk = list(ex.map(_check_img, calib_records))

        assert all(ok for _, ok, _ in test_chk), "Corrupt image in test set!"
        assert all(ok for _, ok, _ in calib_chk), "Corrupt image in calib set!"
        overlap_hashes = {h for _, _, h in test_chk} & {h for _, _, h in calib_chk}
        assert len(overlap_hashes) == 0, f"SHA-256 overlap: {len(overlap_hashes)}"

        # Build unified COCO GT JSON
        val_person_anns = [a for a in val_coco_raw["annotations"] if a["category_id"] == 1]
        train_b_person_anns = [
            a for a in train_coco_raw["annotations"]
            if a["category_id"] == 1 and a["image_id"] in test_b_set
        ]
        max_val_ann_id = max((a["id"] for a in val_person_anns), default=0) + 1_000_000
        reindexed_train_anns = []
        for idx, a in enumerate(train_b_person_anns):
            na = dict(a)
            na["id"] = max_val_ann_id + idx
            reindexed_train_anns.append(na)

        gt_json = {
            "images": [{"id": r["image_id"], "file_name": r["file_name"], "width": r["width"], "height": r["height"]} for r in test_records],
            "annotations": val_person_anns + reindexed_train_anns,
            "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
        }
        with open(TEST_GT_JSON, "w") as f:
            json.dump(gt_json, f)

        manifest = {
            "test_count": len(test_records),
            "calib_count": len(calib_records),
            "calib200_count": sum(1 for r in calib_records if r["in_calib200"]),
            "sha256_overlap": len(overlap_hashes),
            "test_records": test_records,
            "calib_records": calib_records,
        }
        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f)

        coco_gt = COCO(str(TEST_GT_JSON))
        dist_audit = {
            "COCO_train2017_natural": _compute_dist_stats(train_coco_raw),
            "COCO_val2017_natural": _compute_dist_stats(val_coco_raw),
            "Test_SliceB_2500_matched": _compute_dist_stats(train_coco_raw, test_b_set),
            "Test_Full_7500": _compute_dist_stats(gt_json),
            "Calib_200_matched": _compute_dist_stats(train_coco_raw, calib_200_set),
            "Calib_400_matched": _compute_dist_stats(train_coco_raw, set(calib_400_ids)),
        }
        print("=== DISTRIBUTION ALIGNMENT AUDIT ===")
        for k, v in dist_audit.items():
            print(f"  {k:26s}: {json.dumps(v)}")

        return manifest, coco_gt, dist_audit

    DATASET_MANIFEST, COCO_GT, DATASET_DIST_AUDIT = _build_curated_datasets()
    DATASET_SUMMARY = {
        "test_total": DATASET_MANIFEST["test_count"],
        "calib_400_total": DATASET_MANIFEST["calib_count"],
        "calib_200_total": DATASET_MANIFEST["calib200_count"],
        "sha256_overlap": DATASET_MANIFEST["sha256_overlap"],
        "full_7500_stats": DATASET_DIST_AUDIT["Test_Full_7500"],
    }
    _df_audit = pd.DataFrame([
        {
            "Split": k,
            "Images": v["n_img"],
            "Pos %": v["pos_pct"],
            "Neg %": v["neg_pct"],
            "Person Boxes": v["boxes"],
            "Small %": v["small_pct"],
            "Med %": v["med_pct"],
            "Large %": v["large_pct"],
            "Mean Person/Pos": v["mean_person_per_pos"],
            "Top Co-Objects": ", ".join(f"{n}({r})" for n, r in v["top_co_objects"][:4]),
        }
        for k, v in DATASET_DIST_AUDIT.items()
    ])
    mo.vstack([
        mo.md("### Distribution-Matched Dataset & Disjoint Calibration Audit (`SHA-256 Overlap = 0`)"),
        mo.ui.table(_df_audit),
    ])
    return COCO_GT, DATASET_MANIFEST


@app.cell
def prepack_and_eval_helpers(
    COCO_GT,
    COCOeval,
    DATASET_MANIFEST,
    concurrent,
    cv2,
    mo,
    np,
    time,
):
    # Cell 2: RAM Pre-pack Cache (Zero Accuracy Loss Optimization) & Unified COCO/Drift Evaluation Engine
    IMGSZ = 416
    CONF_COCO = 0.001
    IOU_COCO = 0.65
    MAX_DET = 300
    CONF_OP = 0.25
    IOU_OP = 0.45

    def letterbox_uint8_rgb(img_bgr: np.ndarray, imgsz: int = IMGSZ):
        oh, ow = img_bgr.shape[:2]
        scale = min(imgsz / ow, imgsz / oh)
        nw, nh = int(round(ow * scale)), int(round(oh * scale))
        px, py = (imgsz - nw) // 2, (imgsz - nh) // 2
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        canvas[py:py + nh, px:px + nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return rgb, (ow, oh, float(scale), int(px), int(py))

    def _prepack_all():
        t0 = time.perf_counter()
        test_recs = DATASET_MANIFEST["test_records"]
        calib_recs = DATASET_MANIFEST["calib_records"]

        test_buf = np.empty((len(test_recs), IMGSZ, IMGSZ, 3), dtype=np.uint8)
        test_meta = [None] * len(test_recs)

        def _proc_test(idx_rec):
            idx, rec = idx_rec
            img = cv2.imread(rec["path"], cv2.IMREAD_COLOR)
            rgb, meta_t = letterbox_uint8_rgb(img, IMGSZ)
            test_buf[idx] = rgb
            test_meta[idx] = {
                "idx": idx,
                "image_id": rec["image_id"],
                "slice": rec["slice"],
                "bucket": rec["bucket"],
                "ow": meta_t[0],
                "oh": meta_t[1],
                "scale": meta_t[2],
                "px": meta_t[3],
                "py": meta_t[4],
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
            list(ex.map(_proc_test, enumerate(test_recs)))

        calib_buf = np.empty((len(calib_recs), IMGSZ, IMGSZ, 3), dtype=np.uint8)
        def _proc_calib(idx_rec):
            idx, rec = idx_rec
            img = cv2.imread(rec["path"], cv2.IMREAD_COLOR)
            rgb, _ = letterbox_uint8_rgb(img, IMGSZ)
            calib_buf[idx] = rgb

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
            list(ex.map(_proc_calib, enumerate(calib_recs)))

        dt = time.perf_counter() - t0
        ram_gb = (test_buf.nbytes + calib_buf.nbytes) / (1024 ** 3)
        print(f"Pre-packed {len(test_recs)} test + {len(calib_recs)} calib images into RAM ({ram_gb:.2f} GB) in {dt:.2f}s")
        return test_buf, test_meta, calib_buf

    TEST_RGB_NHWC, TEST_META, CALIB_RGB_NHWC = _prepack_all()

    # Deterministic 256-image stratified probe slice for raw tensor drift monitoring
    def _select_probe_indices():
        rng = np.random.default_rng(777)
        by_bucket = {}
        for m in TEST_META:
            by_bucket.setdefault(m["bucket"], []).append(m["idx"])
        chosen = []
        for b, idxs in sorted(by_bucket.items()):
            arr = np.array(idxs, dtype=np.int64)
            rng.shuffle(arr)
            take = 40 if b != "val2017" else 96
            chosen.extend(arr[:take].tolist())
        return sorted(chosen[:256])

    PROBE_INDICES = _select_probe_indices()

    def decode_raw_to_boxes(raw_5xN: np.ndarray, meta: dict, conf_th: float = CONF_COCO, iou_th: float = IOU_COCO, max_det: int = MAX_DET):
        # raw_5xN: shape (5, 3549) or (84, 3549) where channel 4 is person score
        scores_all = raw_5xN[4]
        idx = np.flatnonzero(scores_all >= conf_th)
        if idx.size == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        if idx.size > 1200:
            sel = np.argpartition(scores_all[idx], -1200)[-1200:]
            idx = idx[sel]
        sc = scores_all[idx].astype(np.float32)
        cx = raw_5xN[0, idx]
        cy = raw_5xN[1, idx]
        bw = raw_5xN[2, idx]
        bh = raw_5xN[3, idx]
        x1 = cx - 0.5 * bw
        y1 = cy - 0.5 * bh
        x2 = cx + 0.5 * bw
        y2 = cy + 0.5 * bh
        # Restore to original image coordinates
        inv_s = 1.0 / meta["scale"]
        px, py = meta["px"], meta["py"]
        ow, oh = meta["ow"], meta["oh"]
        ox1 = np.clip((x1 - px) * inv_s, 0.0, float(ow))
        oy1 = np.clip((y1 - py) * inv_s, 0.0, float(oh))
        ox2 = np.clip((x2 - px) * inv_s, 0.0, float(ow))
        oy2 = np.clip((y2 - py) * inv_s, 0.0, float(oh))
        boxes_xyxy = np.stack([ox1, oy1, ox2, oy2], axis=1).astype(np.float32)
        boxes_xywh = np.stack([ox1, oy1, np.maximum(0.0, ox2 - ox1), np.maximum(0.0, oy2 - oy1)], axis=1)
        keep = cv2.dnn.NMSBoxes(boxes_xywh, sc, float(conf_th), float(iou_th))
        if len(keep) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        keep_idx = np.asarray(keep, dtype=np.int64).reshape(-1)
        # Sort descending by score & cap max_det
        order = np.argsort(-sc[keep_idx])[:max_det]
        keep_idx = keep_idx[order]
        return boxes_xyxy[keep_idx], sc[keep_idx]

    def _box_iou_matrix(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
        if len(b1) == 0 or len(b2) == 0:
            return np.zeros((len(b1), len(b2)), dtype=np.float32)
        ix1 = np.maximum(b1[:, None, 0], b2[None, :, 0])
        iy1 = np.maximum(b1[:, None, 1], b2[None, :, 1])
        ix2 = np.minimum(b1[:, None, 2], b2[None, :, 2])
        iy2 = np.minimum(b1[:, None, 3], b2[None, :, 3])
        iw = np.maximum(0.0, ix2 - ix1)
        ih = np.maximum(0.0, iy2 - iy1)
        inter = iw * ih
        a1 = np.maximum(0.0, (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1]))[:, None]
        a2 = np.maximum(0.0, (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1]))[None, :]
        return inter / np.maximum(a1 + a2 - inter, 1e-6)

    def _run_cocoeval_subset(coco_gt_obj, coco_results, img_ids):
        if not coco_results:
            return {k: 0.0 for k in ("mAP50_95", "mAP50", "mAP75", "AP_small", "AP_medium", "AP_large", "AR_100")}
        dt = coco_gt_obj.loadRes(coco_results)
        ev = COCOeval(coco_gt_obj, dt, "bbox")
        ev.params.catIds = [1]
        ev.params.imgIds = img_ids
        ev.evaluate()
        ev.accumulate()
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            ev.summarize()
        s = ev.stats
        return {
            "mAP50_95": round(float(s[0]), 5),
            "mAP50": round(float(s[1]), 5),
            "mAP75": round(float(s[2]), 5),
            "AP_small": round(float(s[3]), 5),
            "AP_medium": round(float(s[4]), 5),
            "AP_large": round(float(s[5]), 5),
            "AR_100": round(float(s[8]), 5),
        }

    # Prebuild non-crowd GT dict for fast Operating-Point P/R/F1 & FP-on-Negatives
    _GT_OP_BOXES = {}
    for m in TEST_META:
        iid = m["image_id"]
        anns = [a for a in COCO_GT.imgToAnns[iid] if a.get("iscrowd", 0) == 0]
        if anns:
            arr = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns], dtype=np.float32)
        else:
            arr = np.zeros((0, 4), dtype=np.float32)
        _GT_OP_BOXES[iid] = arr

    def compute_detailed_metrics(name: str, per_img_boxes: list, per_img_scores: list, wall_s: float, ref_boxes: list = None, probe_raw: np.ndarray = None, ref_probe_raw: np.ndarray = None):
        coco_results = []
        op_preds_boxes = []
        op_preds_scores = []

        tp = fp = fn = 0
        neg_fp = 0
        neg_imgs_with_fp = 0
        total_op_preds = 0
        matched_ious_vs_ref = []
        matched_score_diffs_vs_ref = []

        for i, meta in enumerate(TEST_META):
            iid = meta["image_id"]
            bx = per_img_boxes[i]
            sc = per_img_scores[i]
            for j in range(len(sc)):
                x1, y1, x2, y2 = float(bx[j, 0]), float(bx[j, 1]), float(bx[j, 2]), float(bx[j, 3])
                coco_results.append({
                    "image_id": int(iid),
                    "category_id": 1,
                    "bbox": [round(x1, 2), round(y1, 2), round(max(0.0, x2 - x1), 2), round(max(0.0, y2 - y1), 2)],
                    "score": round(float(sc[j]), 5),
                })

            # Operating-point slice (conf >= CONF_OP, IOU_OP NMS)
            mask_op = sc >= CONF_OP
            if np.any(mask_op):
                bx_op = bx[mask_op]
                sc_op = sc[mask_op]
                xywh_op = np.stack([bx_op[:, 0], bx_op[:, 1], np.maximum(0.0, bx_op[:, 2] - bx_op[:, 0]), np.maximum(0.0, bx_op[:, 3] - bx_op[:, 1])], axis=1)
                k2 = cv2.dnn.NMSBoxes(xywh_op, sc_op, float(CONF_OP), float(IOU_OP))
                if len(k2) > 0:
                    k2 = np.asarray(k2, dtype=np.int64).reshape(-1)
                    bx_op, sc_op = bx_op[k2], sc_op[k2]
                else:
                    bx_op = np.zeros((0, 4), dtype=np.float32)
                    sc_op = np.zeros((0,), dtype=np.float32)
            else:
                bx_op = np.zeros((0, 4), dtype=np.float32)
                sc_op = np.zeros((0,), dtype=np.float32)

            op_preds_boxes.append(bx_op)
            op_preds_scores.append(sc_op)
            total_op_preds += len(sc_op)

            gt_b = _GT_OP_BOXES[iid]
            if len(gt_b) == 0:
                if len(sc_op) > 0:
                    neg_fp += len(sc_op)
                    neg_imgs_with_fp += 1
                fp += len(sc_op)
            else:
                if len(sc_op) == 0:
                    fn += len(gt_b)
                else:
                    ious = _box_iou_matrix(bx_op, gt_b)
                    used_gt = set()
                    for pi in range(len(sc_op)):
                        best_g = int(np.argmax(ious[pi]))
                        if ious[pi, best_g] >= 0.50 and best_g not in used_gt:
                            used_gt.add(best_g)
                            tp += 1
                        else:
                            fp += 1
                    fn += len(gt_b) - len(used_gt)

            if ref_boxes is not None and len(ref_boxes[i]) > 0 and len(bx_op) > 0:
                rmat = _box_iou_matrix(bx_op, ref_boxes[i])
                best_ious = rmat.max(axis=1)
                matched_ious_vs_ref.extend(best_ious[best_ious > 0.1].tolist())

        all_ids = [m["image_id"] for m in TEST_META]
        val_ids = [m["image_id"] for m in TEST_META if m["slice"] == "val2017_official"]
        hard_ids = [m["image_id"] for m in TEST_META if m["slice"] == "matched_train_2500"]

        coco_all = _run_cocoeval_subset(COCO_GT, coco_results, all_ids)
        coco_val = _run_cocoeval_subset(COCO_GT, coco_results, val_ids)
        coco_hard = _run_cocoeval_subset(COCO_GT, coco_results, hard_ids)

        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)

        raw_cos = None
        raw_box_mae = None
        raw_score_mae = None
        if probe_raw is not None and ref_probe_raw is not None:
            a = probe_raw.reshape(len(PROBE_INDICES), -1).astype(np.float64)
            b = ref_probe_raw.reshape(len(PROBE_INDICES), -1).astype(np.float64)
            dots = np.sum(a * b, axis=1)
            norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
            raw_cos = round(float(np.mean(dots / norms)), 6)
            raw_box_mae = round(float(np.mean(np.abs(probe_raw[:, :4, :] - ref_probe_raw[:, :4, :]))), 5)
            raw_score_mae = round(float(np.mean(np.abs(probe_raw[:, 4, :] - ref_probe_raw[:, 4, :]))), 6)

        row = {
            "variant": name,
            "eval_time_s": round(wall_s, 2),
            "throughput_img_s": round(len(TEST_META) / max(1e-6, wall_s), 1),
            "mAP50_95": coco_all["mAP50_95"],
            "mAP50": coco_all["mAP50"],
            "mAP75": coco_all["mAP75"],
            "AP_small": coco_all["AP_small"],
            "AP_medium": coco_all["AP_medium"],
            "AP_large": coco_all["AP_large"],
            "AR_100": coco_all["AR_100"],
            "val2017_mAP50": coco_val["mAP50"],
            "val2017_mAP50_95": coco_val["mAP50_95"],
            "train2500_mAP50": coco_hard["mAP50"],
            "train2500_mAP50_95": coco_hard["mAP50_95"],
            "op_precision": round(prec, 4),
            "op_recall": round(rec, 4),
            "op_f1": round(f1, 4),
            "op_preds": total_op_preds,
            "neg_fp_boxes": neg_fp,
            "neg_fp_imgs": neg_imgs_with_fp,
            "mean_iou_vs_fp32": round(float(np.mean(matched_ious_vs_ref)), 4) if matched_ious_vs_ref else (1.0 if ref_boxes is None else 0.0),
            "probe_raw_cos": raw_cos if raw_cos is not None else 1.0,
            "probe_box_mae": raw_box_mae if raw_box_mae is not None else 0.0,
            "probe_score_mae": raw_score_mae if raw_score_mae is not None else 0.0,
        }
        return row, op_preds_boxes, op_preds_scores

    mo.md(f"### Cache & Evaluation Engine Ready: `TEST_RGB_NHWC={TEST_RGB_NHWC.shape}`, `CALIB_RGB_NHWC={CALIB_RGB_NHWC.shape}`, `PROBE={len(PROBE_INDICES)}`")
    return (
        CALIB_RGB_NHWC,
        IMGSZ,
        PROBE_INDICES,
        TEST_META,
        TEST_RGB_NHWC,
        compute_detailed_metrics,
        decode_raw_to_boxes,
    )


@app.cell
def baseline_fp32_export_and_eval(
    DEVICE,
    IMGSZ,
    MODELS_DIR,
    PROBE_INDICES,
    Path,
    TEST_META,
    TEST_RGB_NHWC,
    YOLO,
    compute_detailed_metrics,
    concurrent,
    decode_raw_to_boxes,
    mo,
    np,
    onnx,
    ort,
    pd,
    time,
    torch,
):
    # Cell 3: Stage 1 - Export FP32 (80-cls & 1-cls Person-Pruned), Verify Bit-Exact Parity & Evaluate All 7,500 Test Images
    import torch.nn as nn

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    def export_fp32_models():
        pt_path = MODELS_DIR / "yolo26n.pt"
        if not pt_path.exists():
            YOLO(str(pt_path))  # auto-downloads yolo26n.pt to pt_path

        # 1) 80-class FP32 ONNX
        y80 = YOLO(str(pt_path))
        exp80 = Path(y80.export(format="onnx", imgsz=IMGSZ, opset=17, simplify=True, dynamic=False))
        p80 = MODELS_DIR / "yolo26n_80cls_fp32.onnx"
        exp80.replace(p80)

        # 2) Bit-exact 1-class ("person") pruned FP32 ONNX
        y1 = YOLO(str(pt_path))
        head = y1.model.model[-1]
        branches = [head.cv3] + ([head.one2one_cv3] if hasattr(head, "one2one_cv3") else [])
        for branch in branches:
            for seq in branch:
                old_conv = seq[-1]
                new_conv = nn.Conv2d(
                    old_conv.in_channels,
                    1,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None,
                )
                with torch.no_grad():
                    new_conv.weight.copy_(old_conv.weight[:1])
                    if old_conv.bias is not None:
                        new_conv.bias.copy_(old_conv.bias[:1])
                seq[-1] = new_conv
        head.nc = 1
        head.no = head.nc + head.reg_max * 4
        y1.model.nc = 1
        y1.model.names = {0: "person"}

        exp1 = Path(y1.export(format="onnx", imgsz=IMGSZ, opset=17, simplify=True, dynamic=False))
        p1 = MODELS_DIR / "model_fp32.onnx"
        exp1.replace(p1)
        onnx.checker.check_model(onnx.load(str(p1)))

        # Verify exact numerical parity across 16 real test images
        x_np = TEST_RGB_NHWC[:16].transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        pt_model = YOLO(str(pt_path)).model.fuse().to(DEVICE).eval()
        with torch.no_grad():
            pt_out = pt_model(torch.from_numpy(x_np).to(DEVICE))[0].cpu().numpy()

        s80 = ort.InferenceSession(str(p80), providers=["CPUExecutionProvider"])
        s1 = ort.InferenceSession(str(p1), providers=["CPUExecutionProvider"])
        o80 = np.concatenate([s80.run(None, {s80.get_inputs()[0].name: x_np[i:i+1]})[0] for i in range(16)], axis=0)
        o1 = np.concatenate([s1.run(None, {s1.get_inputs()[0].name: x_np[i:i+1]})[0] for i in range(16)], axis=0)

        diff_pt_80 = float(np.max(np.abs(pt_out[:, :5] - o80[:, :5])))
        diff_80_1 = float(np.max(np.abs(o80[:, :5] - o1)))
        diff_pt_1 = float(np.max(np.abs(pt_out[:, :5] - o1)))
        print(f"[PARITY] MaxAbsDiff(PyTorch vs ONNX80)={diff_pt_80:.2e} | MaxAbsDiff(ONNX80 vs ONNX1)={diff_80_1:.2e} | MaxAbsDiff(PyTorch vs ONNX1)={diff_pt_1:.2e}")
        assert diff_80_1 < 1e-4, f"1-class pruning mismatch: {diff_80_1}"
        return pt_path, p80, p1, {"pt_vs_80": diff_pt_80, "80_vs_1": diff_80_1, "pt_vs_1": diff_pt_1}

    def run_pytorch_gpu_all(pt_path: Path, batch_size: int = 64):
        pt_model = YOLO(str(pt_path)).model.fuse().to(DEVICE).eval()
        n = len(TEST_META)
        per_boxes = [None] * n
        per_scores = [None] * n
        probe_map = {idx: i for i, idx in enumerate(PROBE_INDICES)}
        probe_raw = np.empty((len(PROBE_INDICES), 5, 3549), dtype=np.float32)

        # Warmup
        with torch.no_grad():
            _ = pt_model(torch.zeros((1, 3, IMGSZ, IMGSZ), device=DEVICE, dtype=torch.float32))
        if DEVICE.startswith("cuda"):
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(n, start + batch_size)
                batch_u8 = torch.from_numpy(TEST_RGB_NHWC[start:end]).to(DEVICE, non_blocking=True)
                inp = batch_u8.permute(0, 3, 1, 2).contiguous().float().mul_(1.0 / 255.0)
                out = pt_model(inp)[0][:, :5, :].cpu().numpy()
                for bi in range(end - start):
                    gidx = start + bi
                    raw_5 = out[bi]
                    if gidx in probe_map:
                        probe_raw[probe_map[gidx]] = raw_5
                    bx, sc = decode_raw_to_boxes(raw_5, TEST_META[gidx])
                    per_boxes[gidx] = bx
                    per_scores[gidx] = sc
        if DEVICE.startswith("cuda"):
            torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0
        return per_boxes, per_scores, wall_s, probe_raw

    def run_onnx_parallel_all(onnx_path: Path, num_workers: int = 16, providers=None):
        if providers is None:
            providers = ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        sessions = [ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers) for _ in range(num_workers)]
        in_name = sessions[0].get_inputs()[0].name
        in_type = sessions[0].get_inputs()[0].type
        is_fp16_io = "float16" in in_type

        n = len(TEST_META)
        per_boxes = [None] * n
        per_scores = [None] * n
        probe_map = {idx: i for i, idx in enumerate(PROBE_INDICES)}
        probe_raw = np.empty((len(PROBE_INDICES), 5, 3549), dtype=np.float32)

        t0 = time.perf_counter()
        def _worker(wid: int):
            sess = sessions[wid]
            dtype_np = np.float16 if is_fp16_io else np.float32
            scale_val = np.float32(1.0 / 255.0)
            for idx in range(wid, n, num_workers):
                rgb = TEST_RGB_NHWC[idx]
                inp = (rgb.transpose(2, 0, 1).astype(np.float32) * scale_val).astype(dtype_np)[None, ...]
                out = sess.run(None, {in_name: inp})[0][0][:5].astype(np.float32)
                if idx in probe_map:
                    probe_raw[probe_map[idx]] = out
                bx, sc = decode_raw_to_boxes(out, TEST_META[idx])
                per_boxes[idx] = bx
                per_scores[idx] = sc

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
            list(ex.map(_worker, range(num_workers)))
        wall_s = time.perf_counter() - t0
        return per_boxes, per_scores, wall_s, probe_raw

    PT_PATH, ONNX_80_PATH, ONNX_FP32_PATH, PARITY_DIFFS = export_fp32_models()

    _pt_bx, _pt_sc, _pt_t, PT_PROBE_RAW = run_pytorch_gpu_all(PT_PATH)
    _m_pt, _op_pt_bx, _op_pt_sc = compute_detailed_metrics("PyTorch_FP32_GPU", _pt_bx, _pt_sc, _pt_t, probe_raw=PT_PROBE_RAW)

    _o80_bx, _o80_sc, _o80_t, _o80_probe = run_onnx_parallel_all(ONNX_80_PATH, num_workers=16)
    _m_80, _, _ = compute_detailed_metrics("ONNX_FP32_80cls", _o80_bx, _o80_sc, _o80_t, probe_raw=_o80_probe)

    _o1_bx, _o1_sc, _o1_t, FP32_PROBE_RAW = run_onnx_parallel_all(ONNX_FP32_PATH, num_workers=16)
    _m_1, FP32_OP_BOXES, FP32_OP_SCORES = compute_detailed_metrics("ONNX_FP32_1cls_Baseline", _o1_bx, _o1_sc, _o1_t, probe_raw=FP32_PROBE_RAW)

    BASELINE_METRICS = [_m_pt, _m_80, _m_1]
    _df_base = pd.DataFrame(BASELINE_METRICS)
    print("=== STAGE 1: BASELINE TEST METRICS (7,500 IMAGES BEFORE QUANTIZATION) ===")
    print(_df_base.to_string(index=False))
    mo.vstack([
        mo.md(
            f"""
            ### Stage 1 Complete: Baseline Evaluation Before Quantization (7,500 Test Images)
            - **1-Class Pruning Verification:** `MaxAbsDiff(ONNX_80cls vs ONNX_1cls) = {PARITY_DIFFS['80_vs_1']:.2e}`, `MaxAbsDiff(PyTorch_FP32 vs ONNX_1cls) = {PARITY_DIFFS['pt_vs_1']:.2e}`
            - **Model Size:** `yolo26n_80cls_fp32.onnx` = `{ONNX_80_PATH.stat().st_size/(1024*1024):.2f} MB` ? `model_fp32.onnx` (1-class) = `{ONNX_FP32_PATH.stat().st_size/(1024*1024):.2f} MB`
            """
        ),
        mo.ui.table(_df_base),
    ])
    return (
        BASELINE_METRICS,
        FP32_OP_BOXES,
        FP32_PROBE_RAW,
        ONNX_FP32_PATH,
        run_onnx_parallel_all,
    )


@app.cell
def quant_format_baselines(
    BASELINE_METRICS,
    CALIB_RGB_NHWC,
    DATASET_MANIFEST,
    FP32_OP_BOXES,
    FP32_PROBE_RAW,
    MODELS_DIR,
    ONNX_FP32_PATH,
    Path,
    RESULTS_DIR,
    compute_detailed_metrics,
    json,
    mo,
    np,
    onnx,
    pd,
    run_onnx_parallel_all,
):
    # Cell 4: Format Conversion Baselines (FP16, INT8 Dynamic, INT8 Static Naive) on 7,500 Test Images
    from onnxruntime.transformers import float16 as ort_float16
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )
    from onnxruntime.quantization.preprocess import quant_pre_process

    PRE_FP32_PATH = MODELS_DIR / "model_fp32_pre.onnx"
    if not PRE_FP32_PATH.exists():
        quant_pre_process(str(ONNX_FP32_PATH), str(PRE_FP32_PATH))

    CALIB_200_IDX = [i for i, r in enumerate(DATASET_MANIFEST["calib_records"]) if r["in_calib200"]]
    CALIB_400_IDX = list(range(len(DATASET_MANIFEST["calib_records"])))

    class CachedCalibrationReader(CalibrationDataReader):
        def __init__(self, rgb_nhwc: np.ndarray, indices: list, input_name: str = "images"):
            self.rgb_nhwc = rgb_nhwc
            self.indices = list(indices)
            self.input_name = input_name
            self.pos = 0

        def get_next(self):
            if self.pos >= len(self.indices):
                return None
            idx = self.indices[self.pos]
            self.pos += 1
            blob = (self.rgb_nhwc[idx].transpose(2, 0, 1).astype(np.float32) / 255.0)[None, ...]
            return {self.input_name: blob}

        def rewind(self):
            self.pos = 0

    def _ensure_stage2a_models():
        fp16_p = MODELS_DIR / "model_fp16.onnx"
        if not fp16_p.exists():
            m_f16 = ort_float16.convert_float_to_float16(onnx.load(str(ONNX_FP32_PATH)), keep_io_types=True)
            onnx.save(m_f16, str(fp16_p))

        dyn_p = MODELS_DIR / "model_int8_dynamic.onnx"
        if not dyn_p.exists():
            m_pre = onnx.load(str(PRE_FP32_PATH))
            dyn_excl = [
                n.name for n in m_pre.graph.node
                if n.name.startswith("/model.23/") or n.op_type in ("Resize", "Upsample", "Softmax")
            ]
            quantize_dynamic(
                model_input=str(PRE_FP32_PATH),
                model_output=str(dyn_p),
                weight_type=QuantType.QUInt8,
                nodes_to_exclude=dyn_excl,
            )

        naive_p = MODELS_DIR / "model_int8_static_naive.onnx"
        if not naive_p.exists():
            quantize_static(
                model_input=str(PRE_FP32_PATH),
                model_output=str(naive_p),
                calibration_data_reader=CachedCalibrationReader(CALIB_RGB_NHWC, CALIB_200_IDX),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                per_channel=False,
                calibrate_method=CalibrationMethod.MinMax,
                nodes_to_exclude=[],
            )
        return fp16_p, dyn_p, naive_p

    FP16_PATH, INT8_DYN_PATH, INT8_NAIVE_PATH = _ensure_stage2a_models()

    def eval_quant_variant(label: str, path: Path):
        bx, sc, dt, pr = run_onnx_parallel_all(path, num_workers=16)
        row, _, _ = compute_detailed_metrics(
            label, bx, sc, dt, ref_boxes=FP32_OP_BOXES, probe_raw=pr, ref_probe_raw=FP32_PROBE_RAW
        )
        row["size_mb"] = round(path.stat().st_size / (1024 * 1024), 2)
        row["delta_mAP50_vs_fp32"] = round(row["mAP50"] - BASELINE_METRICS[-1]["mAP50"], 5)
        row["delta_mAP50_95_vs_fp32"] = round(row["mAP50_95"] - BASELINE_METRICS[-1]["mAP50_95"], 5)
        row["delta_AP_small_vs_fp32"] = round(row["AP_small"] - BASELINE_METRICS[-1]["AP_small"], 5)
        return row

    STAGE2A_METRICS = [
        {
            **BASELINE_METRICS[-1],
            "size_mb": round(ONNX_FP32_PATH.stat().st_size / (1024 * 1024), 2),
            "delta_mAP50_vs_fp32": 0.0,
            "delta_mAP50_95_vs_fp32": 0.0,
            "delta_AP_small_vs_fp32": 0.0,
        },
        eval_quant_variant("ONNX_FP16", FP16_PATH),
        eval_quant_variant("ONNX_INT8_Dynamic", INT8_DYN_PATH),
        eval_quant_variant("ONNX_INT8_Static_Naive", INT8_NAIVE_PATH),
    ]

    with open(RESULTS_DIR / "stage2a_metrics.json", "w") as f:
        json.dump(STAGE2A_METRICS, f, indent=2)

    mo.vstack([
        mo.md("### Stage 2A: Format Baselines Before Optimization (7,500 Test Images)"),
        mo.ui.table(pd.DataFrame(STAGE2A_METRICS)),
    ])
    return (
        CALIB_200_IDX,
        CachedCalibrationReader,
        CalibrationMethod,
        PRE_FP32_PATH,
        QuantFormat,
        QuantType,
        STAGE2A_METRICS,
        eval_quant_variant,
        quantize_static,
    )


@app.cell
def block_sensitivity_audit(
    CALIB_200_IDX,
    CALIB_RGB_NHWC,
    CachedCalibrationReader,
    CalibrationMethod,
    FP32_PROBE_RAW,
    MODELS_DIR,
    PRE_FP32_PATH,
    PROBE_INDICES,
    QuantFormat,
    QuantType,
    RESULTS_DIR,
    TEST_RGB_NHWC,
    mo,
    np,
    onnx,
    ort,
    pd,
    plt,
    quantize_static,
    time,
):
    # Cell 5: Per-Block INT8 Sensitivity Audit (Track Model Layer-by-Layer Degradation via Calibration Cache)
    def run_block_sensitivity_audit():
        t0 = time.perf_counter()
        cache_path = MODELS_DIR / "calib_cache_minmax_200.json"
        if not cache_path.exists():
            _tmp_c = MODELS_DIR / "_tmp_cache_init.onnx"
            quantize_static(
                model_input=str(PRE_FP32_PATH),
                model_output=str(_tmp_c),
                calibration_data_reader=CachedCalibrationReader(CALIB_RGB_NHWC, CALIB_200_IDX),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                per_channel=True,
                calibrate_method=CalibrationMethod.MinMax,
                nodes_to_exclude=[],
                calibration_cache_path=str(cache_path),
            )
            if _tmp_c.exists():
                _tmp_c.unlink()

        m_pre = onnx.load(str(PRE_FP32_PATH))
        block_groups = {}
        for n in m_pre.graph.node:
            if n.name.startswith("/model.23/"):
                if "/cv2." in n.name:
                    g = "/model.23/cv2 (Box Tower)"
                elif "/cv3." in n.name:
                    g = "/model.23/cv3 (Cls Tower)"
                else:
                    g = "/model.23/decode_tail"
            elif "/attn/" in n.name:
                prefix = "/".join(n.name.split("/")[:2])
                g = f"{prefix}/attn (C2PSA)"
            else:
                parts = n.name.split("/")
                g = f"/{parts[1]}" if len(parts) > 1 and parts[1].startswith("model.") else "/other"
            block_groups.setdefault(g, []).append(n.name)

        probe_sub = PROBE_INDICES[:48]
        ref_sub = FP32_PROBE_RAW[:48]
        all_quant_ops = [n.name for n in m_pre.graph.node]
        tmp_onnx = MODELS_DIR / "_tmp_sens_blk.onnx"
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.inter_op_num_threads = 1

        rows = []
        for gname, gnodes in sorted(block_groups.items()):
            excl = [nm for nm in all_quant_ops if nm not in set(gnodes)]
            try:
                quantize_static(
                    model_input=str(PRE_FP32_PATH),
                    model_output=str(tmp_onnx),
                    calibration_data_reader=None,
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8,
                    per_channel=True,
                    calibrate_method=CalibrationMethod.MinMax,
                    nodes_to_exclude=excl,
                    calibration_cache_path=str(cache_path),
                )
                sess = ort.InferenceSession(str(tmp_onnx), sess_options=so, providers=["CPUExecutionProvider"])
                iname = sess.get_inputs()[0].name
                preds = np.stack([
                    sess.run(None, {iname: (TEST_RGB_NHWC[idx].transpose(2, 0, 1).astype(np.float32) / 255.0)[None, ...]})[0][0]
                    for idx in probe_sub
                ], axis=0)
                a = preds.reshape(len(probe_sub), -1).astype(np.float64)
                b = ref_sub.reshape(len(probe_sub), -1).astype(np.float64)
                cos = float(np.mean(np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)))
                box_mae = float(np.mean(np.abs(preds[:, :4, :] - ref_sub[:, :4, :])))
                score_mae = float(np.mean(np.abs(preds[:, 4, :] - ref_sub[:, 4, :])))
                rows.append({
                    "block": gname,
                    "nodes": len(gnodes),
                    "cos_sim": round(cos, 6),
                    "cos_drop_x1e4": round((1.0 - cos) * 10000.0, 3),
                    "box_mae_px": round(box_mae, 4),
                    "score_mae": round(score_mae, 6),
                })
            except Exception as e:
                rows.append({"block": gname, "nodes": len(gnodes), "cos_sim": 0.0, "cos_drop_x1e4": 9999.0, "box_mae_px": 999.0, "score_mae": 1.0, "err": str(e)[:60]})
        if tmp_onnx.exists():
            tmp_onnx.unlink()

        df_s = pd.DataFrame(rows).sort_values("box_mae_px", ascending=False).reset_index(drop=True)
        print(f"Completed block sensitivity audit ({len(rows)} blocks) in {time.perf_counter() - t0:.1f}s")
        return df_s

    SENSITIVITY_DF = run_block_sensitivity_audit()
    SENSITIVITY_DF.to_csv(RESULTS_DIR / "block_sensitivity_audit.csv", index=False)
    print(SENSITIVITY_DF.to_string(index=False))

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    _top = SENSITIVITY_DF.head(15).iloc[::-1]
    _ax1.barh(_top["block"], _top["box_mae_px"], color="#e11d48")
    _ax1.set_xlabel("Box Coordinate MAE (px vs FP32)")
    _ax1.set_title("Single-Block Quantization: Box Drift Sensitivity")
    _top_s = SENSITIVITY_DF.sort_values("score_mae", ascending=False).head(15).iloc[::-1]
    _ax2.barh(_top_s["block"], _top_s["score_mae"], color="#2563eb")
    _ax2.set_xlabel("Person Score MAE (vs FP32)")
    _ax2.set_title("Single-Block Quantization: Confidence Score Sensitivity")
    plt.tight_layout()

    mo.vstack([
        mo.md("### Stage 2B: Per-Block INT8 Sensitivity Tracking (`YOLO26n`)"),
        mo.ui.table(SENSITIVITY_DF),
        _fig,
    ])
    return


@app.cell
def int8_calib_method_search(
    CALIB_200_IDX,
    CALIB_RGB_NHWC,
    CachedCalibrationReader,
    CalibrationMethod,
    MODELS_DIR,
    PRE_FP32_PATH,
    Path,
    QuantFormat,
    QuantType,
    RESULTS_DIR,
    eval_quant_variant,
    json,
    mo,
    onnx,
    pd,
    quantize_static,
    time,
):
    # Cell 6: Stage 3A - Optimize Calibration Method (MinMax vs Entropy vs Percentile) under Mixed-Precision
    def get_std_mix_exclude(onnx_path: Path):
        m = onnx.load(str(onnx_path))
        excl = set()
        for n in m.graph.node:
            nm = n.name
            if nm.startswith("/model.0/") or "/attn/" in nm or n.op_type in ("Resize", "Upsample", "Softmax") or nm.startswith("/model.23/"):
                excl.add(nm)
        return sorted(excl)

    STD_MIX_EXCL = get_std_mix_exclude(PRE_FP32_PATH)
    print(f"STD_MIX_EXCL count = {len(STD_MIX_EXCL)} nodes kept in FP32")

    def build_and_eval_int8(label: str, filename: str, calib_idx: list, method, exclude_nodes: list, cache_name: str, extra_opts: dict = None, act_type=QuantType.QUInt8):
        out_p = MODELS_DIR / filename
        cache_p = MODELS_DIR / cache_name
        t_q = time.perf_counter()
        quantize_static(
            model_input=str(PRE_FP32_PATH),
            model_output=str(out_p),
            calibration_data_reader=None if cache_p.exists() else CachedCalibrationReader(CALIB_RGB_NHWC, calib_idx),
            quant_format=QuantFormat.QDQ,
            activation_type=act_type,
            weight_type=QuantType.QInt8,
            per_channel=True,
            calibrate_method=method,
            nodes_to_exclude=exclude_nodes,
            extra_options=extra_opts or {},
            calibration_cache_path=str(cache_p),
        )
        q_s = time.perf_counter() - t_q
        row = eval_quant_variant(label, out_p)
        row["quant_time_s"] = round(q_s, 2)
        row["excluded_nodes"] = len(exclude_nodes)
        print(f"[{label}] size={row['size_mb']}MB | mAP50={row['mAP50']} (d={row['delta_mAP50_vs_fp32']:+.5f}) | mAP50-95={row['mAP50_95']} (d={row['delta_mAP50_95_vs_fp32']:+.5f}) | AP_s={row['AP_small']} | IoU={row['mean_iou_vs_fp32']} | cos={row['probe_raw_cos']} | q_t={q_s:.1f}s")
        return row

    STAGE3A_METRICS = [
        build_and_eval_int8(
            "INT8_StdMix_MinMax_C200",
            "model_int8_stdmix_minmax_c200.onnx",
            CALIB_200_IDX,
            CalibrationMethod.MinMax,
            STD_MIX_EXCL,
            "calib_cache_minmax_200.json",
        ),
        build_and_eval_int8(
            "INT8_StdMix_Entropy_C200",
            "model_int8_stdmix_entropy_c200.onnx",
            CALIB_200_IDX,
            CalibrationMethod.Entropy,
            STD_MIX_EXCL,
            "calib_cache_entropy_200.json",
        ),
        build_and_eval_int8(
            "INT8_StdMix_Percentile9999_C200",
            "model_int8_stdmix_pct9999_c200.onnx",
            CALIB_200_IDX,
            CalibrationMethod.Percentile,
            STD_MIX_EXCL,
            "calib_cache_pct9999_200.json",
            extra_opts={"CalibPercentile": 99.99, "CalibTensorRangeSymmetric": False},
        ),
    ]

    with open(RESULTS_DIR / "stage3a_calib_method_metrics.json", "w") as _f:
        json.dump(STAGE3A_METRICS, _f, indent=2)

    mo.vstack([
        mo.md("### Stage 3A: Calibration Algorithm Comparison on 7,500 Test Images (`MinMax` vs `Entropy` vs `Percentile 99.99%`)"),
        mo.ui.table(pd.DataFrame(STAGE3A_METRICS)),
    ])
    return STAGE3A_METRICS, build_and_eval_int8


@app.cell
def int8_sensitivity_optimization(
    CALIB_200_IDX,
    CalibrationMethod,
    PRE_FP32_PATH,
    Path,
    RESULTS_DIR,
    build_and_eval_int8,
    json,
    mo,
    onnx,
    pd,
):
    # Cell 7: Stage 3B - Sensitivity-Guided Node Protection Optimization (7,500 Test Images, Percentile 99.99%)
    def _get_custom_excludes(onnx_path: Path, extra_prefixes: list, include_stem_attn: bool = True):
        m = onnx.load(str(onnx_path))
        excl = set()
        for n in m.graph.node:
            nm = n.name
            if n.op_type in ("Resize", "Upsample", "Softmax") or nm.startswith("/model.23/"):
                excl.add(nm)
            if include_stem_attn and (nm.startswith("/model.0/") or "/attn/" in nm):
                excl.add(nm)
            for pfx in extra_prefixes:
                if nm.startswith(pfx):
                    excl.add(nm)
        return sorted(excl)

    EXCL_HEAD_ONLY = _get_custom_excludes(PRE_FP32_PATH, [], include_stem_attn=False)
    EXCL_PLUS_M1 = _get_custom_excludes(PRE_FP32_PATH, ["/model.1/"], include_stem_attn=True)
    EXCL_PLUS_M1M2 = _get_custom_excludes(PRE_FP32_PATH, ["/model.1/", "/model.2/"], include_stem_attn=True)

    _pct_opts = {"CalibPercentile": 99.99, "CalibTensorRangeSymmetric": False}

    STAGE3B_METRICS = [
        build_and_eval_int8(
            "INT8_HeadOnly_Pct9999_C200",
            "model_int8_headonly_pct9999_c200.onnx",
            CALIB_200_IDX,
            CalibrationMethod.Percentile,
            EXCL_HEAD_ONLY,
            "calib_cache_pct9999_200.json",
            extra_opts=_pct_opts,
        ),
        build_and_eval_int8(
            "INT8_SensPlusM1_Pct9999_C200",
            "model_int8_sens_m1_pct9999_c200.onnx",
            CALIB_200_IDX,
            CalibrationMethod.Percentile,
            EXCL_PLUS_M1,
            "calib_cache_pct9999_200.json",
            extra_opts=_pct_opts,
        ),
        build_and_eval_int8(
            "INT8_SensPlusM1M2_Pct9999_C200",
            "model_int8_sens_m1m2_pct9999_c200.onnx",
            CALIB_200_IDX,
            CalibrationMethod.Percentile,
            EXCL_PLUS_M1M2,
            "calib_cache_pct9999_200.json",
            extra_opts=_pct_opts,
        ),
    ]

    with open(RESULTS_DIR / "stage3b_sensitivity_metrics.json", "w") as _f:
        json.dump(STAGE3B_METRICS, _f, indent=2)

    mo.vstack([
        mo.md("### Stage 3B: Sensitivity-Guided Mixed-Precision Optimization (7,500 Test Images)"),
        mo.ui.table(pd.DataFrame(STAGE3B_METRICS)),
    ])
    return EXCL_PLUS_M1M2, STAGE3B_METRICS


@app.cell
def int8_percentile_tuning(
    CALIB_200_IDX,
    CalibrationMethod,
    EXCL_PLUS_M1M2,
    PRE_FP32_PATH,
    Path,
    RESULTS_DIR,
    build_and_eval_int8,
    json,
    mo,
    onnx,
    pd,
):
    # Cell 8: Stage 3C - Test Keeping C2PSA in INT8 + Tuning Percentile 99.995% (7,500 Test Images)
    def _make_excl_no_attn(onnx_path: Path, extra_prefixes: list):
        m = onnx.load(str(onnx_path))
        excl = set()
        for n in m.graph.node:
            nm = n.name
            if n.op_type in ("Resize", "Upsample", "Softmax") or nm.startswith("/model.23/"):
                excl.add(nm)
            for pfx in extra_prefixes:
                if nm.startswith(pfx):
                    excl.add(nm)
        return sorted(excl)

    EXCL_HEAD_M0M1M2 = _make_excl_no_attn(PRE_FP32_PATH, ["/model.0/", "/model.1/", "/model.2/"])

    STAGE3C_METRICS = [
        build_and_eval_int8(
            "INT8_HeadM0M1M2_Pct9999_C200",
            "model_int8_head_m012_pct9999_c200.onnx",
            CALIB_200_IDX,
            CalibrationMethod.Percentile,
            EXCL_HEAD_M0M1M2,
            "calib_cache_pct9999_200.json",
            extra_opts={"CalibPercentile": 99.99, "CalibTensorRangeSymmetric": False},
        ),
        build_and_eval_int8(
            "INT8_SensPlusM1M2_Pct99995_C200",
            "model_int8_sens_m1m2_pct99995_c200.onnx",
            CALIB_200_IDX,
            CalibrationMethod.Percentile,
            EXCL_PLUS_M1M2,
            "calib_cache_pct99995_200.json",
            extra_opts={"CalibPercentile": 99.995, "CalibTensorRangeSymmetric": False},
        ),
    ]

    with open(RESULTS_DIR / "stage3c_metrics.json", "w") as _f:
        json.dump(STAGE3C_METRICS, _f, indent=2)

    mo.vstack([
        mo.md("### Stage 3C: INT8 Coverage & Percentile Tail Refinement (7,500 Test Images)"),
        mo.ui.table(pd.DataFrame(STAGE3C_METRICS)),
    ])
    return (STAGE3C_METRICS,)


@app.cell
def final_ablation_and_master_report(
    BASELINE_METRICS,
    CALIB_200_IDX,
    CalibrationMethod,
    MODELS_DIR,
    PRE_FP32_PATH,
    Path,
    RESULTS_DIR,
    STAGE2A_METRICS,
    STAGE3A_METRICS,
    STAGE3B_METRICS,
    STAGE3C_METRICS,
    build_and_eval_int8,
    json,
    mo,
    onnx,
    ort,
    pd,
    plt,
):
    # Cell 9: Final INT8 Optimization Variants, Fused Op Coverage Audit & Master 7,500-Image Comparison Report
    def _excl_set(onnx_path: Path, extra_prefixes: list, include_stem_attn: bool):
        m = onnx.load(str(onnx_path))
        excl = set()
        for n in m.graph.node:
            nm = n.name
            if n.op_type in ("Resize", "Upsample", "Softmax") or nm.startswith("/model.23/"):
                excl.add(nm)
            if include_stem_attn and (nm.startswith("/model.0/") or "/attn/" in nm):
                excl.add(nm)
            for pfx in extra_prefixes:
                if nm.startswith(pfx):
                    excl.add(nm)
        return sorted(excl)

    _excl_head_m012 = _excl_set(PRE_FP32_PATH, ["/model.0/", "/model.1/", "/model.2/"], include_stem_attn=False)
    _excl_sens_top4 = _excl_set(PRE_FP32_PATH, ["/model.1/", "/model.2/", "/model.16/", "/model.4/"], include_stem_attn=True)
    _pct99995 = {"CalibPercentile": 99.995, "CalibTensorRangeSymmetric": False}

    STAGE3D_METRICS = [
        build_and_eval_int8(
            "INT8_HeadM0M1M2_Pct99995_C200",
            "model_int8_head_m012_pct99995_c200.onnx",
            CALIB_200_IDX,
            CalibrationMethod.Percentile,
            _excl_head_m012,
            "calib_cache_pct99995_200.json",
            extra_opts=_pct99995,
        ),
        build_and_eval_int8(
            "INT8_SensM1M2M16M4_Pct99995_C200",
            "model_int8_sens_m12_16_4_pct99995_c200.onnx",
            CALIB_200_IDX,
            CalibrationMethod.Percentile,
            _excl_sens_top4,
            "calib_cache_pct99995_200.json",
            extra_opts=_pct99995,
        ),
    ]

    _MODEL_FILES = {
        "ONNX_FP32_1cls_Baseline": "model_fp32.onnx",
        "ONNX_FP16": "model_fp16.onnx",
        "ONNX_INT8_Dynamic": "model_int8_dynamic.onnx",
        "ONNX_INT8_Static_Naive": "model_int8_static_naive.onnx",
        "INT8_StdMix_MinMax_C200": "model_int8_stdmix_minmax_c200.onnx",
        "INT8_StdMix_Entropy_C200": "model_int8_stdmix_entropy_c200.onnx",
        "INT8_StdMix_Percentile9999_C200": "model_int8_stdmix_pct9999_c200.onnx",
        "INT8_HeadOnly_Pct9999_C200": "model_int8_headonly_pct9999_c200.onnx",
        "INT8_SensPlusM1_Pct9999_C200": "model_int8_sens_m1_pct9999_c200.onnx",
        "INT8_SensPlusM1M2_Pct9999_C200": "model_int8_sens_m1m2_pct9999_c200.onnx",
        "INT8_HeadM0M1M2_Pct9999_C200": "model_int8_head_m012_pct9999_c200.onnx",
        "INT8_SensPlusM1M2_Pct99995_C200": "model_int8_sens_m1m2_pct99995_c200.onnx",
        "INT8_HeadM0M1M2_Pct99995_C200": "model_int8_head_m012_pct99995_c200.onnx",
        "INT8_SensM1M2M16M4_Pct99995_C200": "model_int8_sens_m12_16_4_pct99995_c200.onnx",
    }

    def _audit_fused_ops(fname: str):
        from collections import Counter
        p = MODELS_DIR / fname
        opt_p = MODELS_DIR / f"_tmp_opt_{fname}"
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.optimized_model_filepath = str(opt_p)
        _ = ort.InferenceSession(str(p), sess_options=so, providers=["CPUExecutionProvider"])
        mf = onnx.load(str(opt_p))
        c = Counter(n.op_type for n in mf.graph.node)
        opt_p.unlink()
        qconv = c.get("QLinearConv", 0) + c.get("ConvInteger", 0)
        fconv = c.get("Conv", 0)
        return qconv, fconv, c.get("QLinearMatMul", 0) + c.get("MatMulInteger", 0), c.get("QuantizeLinear", 0), c.get("DequantizeLinear", 0)

    ALL_METRICS = (
        [{"variant": BASELINE_METRICS[0]["variant"], **BASELINE_METRICS[0], "size_mb": round((MODELS_DIR / "yolo26n.pt").stat().st_size / (1024 * 1024), 2), "delta_mAP50_vs_fp32": 0.0, "delta_mAP50_95_vs_fp32": 0.0, "delta_AP_small_vs_fp32": 0.0}]
        + STAGE2A_METRICS
        + STAGE3A_METRICS
        + STAGE3B_METRICS
        + STAGE3C_METRICS
        + STAGE3D_METRICS
    )

    for row in ALL_METRICS:
        vn = row["variant"]
        if vn in _MODEL_FILES:
            qc, fc, qm, ql, dq = _audit_fused_ops(_MODEL_FILES[vn])
            row["int8_convs"] = qc
            row["fp32_convs"] = fc
            row["int8_conv_pct"] = round(100.0 * qc / max(1, qc + fc), 1)
            row["int8_matmuls"] = qm
            row["boundary_q_dq"] = f"{ql}/{dq}"
        else:
            row["int8_convs"] = 0
            row["fp32_convs"] = 102
            row["int8_conv_pct"] = 0.0
            row["int8_matmuls"] = 0
            row["boundary_q_dq"] = "0/0"

    MASTER_DF = pd.DataFrame(ALL_METRICS)
    MASTER_DF.to_csv(RESULTS_DIR / "full_7500_ablation_report.csv", index=False)
    with open(RESULTS_DIR / "full_7500_ablation_report.json", "w") as _f:
        json.dump(ALL_METRICS, _f, indent=2)

    _int8_cands = [r for r in ALL_METRICS if "INT8" in r["variant"] and "Naive" not in r["variant"] and "Dynamic" not in r["variant"] and r["int8_conv_pct"] >= 60.0]
    BEST_INT8_ROW = max(_int8_cands, key=lambda r: (r["mAP50"] + r["mAP50_95"], r["int8_conv_pct"]))
    import shutil
    shutil.copy2(MODELS_DIR / _MODEL_FILES[BEST_INT8_ROW["variant"]], MODELS_DIR / "model_int8_static_best.onnx")

    _cols_show = [
        "variant", "size_mb", "int8_convs", "fp32_convs", "int8_conv_pct",
        "mAP50", "delta_mAP50_vs_fp32", "mAP50_95", "delta_mAP50_95_vs_fp32",
        "AP_small", "AP_medium", "AP_large", "op_precision", "op_recall", "op_f1",
        "neg_fp_boxes", "mean_iou_vs_fp32", "probe_raw_cos", "probe_box_mae"
    ]
    print("=== MASTER 7,500-IMAGE ABLATION TABLE ===")
    print(MASTER_DF[_cols_show].to_string(index=False))
    print("---")
    print("BEST_INT8_ROW:", json.dumps(BEST_INT8_ROW, indent=2))

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    _sub = MASTER_DF[MASTER_DF["mAP50"] > 0.5].copy()
    _ax1.scatter(_sub["size_mb"], _sub["mAP50"], s=90, c=_sub["int8_conv_pct"], cmap="viridis")
    for _, _r in _sub.iterrows():
        _ax1.annotate(_r["variant"].replace("INT8_", "").replace("ONNX_", ""), (_r["size_mb"], _r["mAP50"]), fontsize=7, xytext=(4, 2), textcoords="offset points")
    _ax1.set_xlabel("Model Size (MB)")
    _ax1.set_ylabel("COCO mAP@50 (7,500 Test Images)")
    _ax1.set_title("Size vs mAP@50 Pareto Frontier")
    _ax1.grid(True, alpha=0.3)

    _ax2.scatter(_sub["int8_conv_pct"], _sub["mAP50_95"], s=90, c=_sub["mean_iou_vs_fp32"], cmap="plasma")
    for _, _r in _sub.iterrows():
        _ax2.annotate(_r["variant"].replace("INT8_", "").replace("ONNX_", ""), (_r["int8_conv_pct"], _r["mAP50_95"]), fontsize=7, xytext=(4, 2), textcoords="offset points")
    _ax2.set_xlabel("INT8 Quantized Conv Ratio (%)")
    _ax2.set_ylabel("COCO mAP@50:95 (7,500 Test Images)")
    _ax2.set_title("INT8 Coverage (%) vs Localization Precision (mAP@50:95)")
    _ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    mo.vstack([
        mo.md(
            f"""
            ### Master Report: 7,500-Image Distribution-Matched Evaluation & INT8 Optimization
            - **Best INT8 Variant:** `{BEST_INT8_ROW['variant']}` (`{BEST_INT8_ROW['size_mb']} MB`, `{BEST_INT8_ROW['int8_convs']}/102` Convs = `{BEST_INT8_ROW['int8_conv_pct']}%` INT8)
            - **Accuracy vs FP32 Baseline:** `mAP@50 = {BEST_INT8_ROW['mAP50']}` (`{BEST_INT8_ROW['delta_mAP50_vs_fp32']:+.5f}`), `mAP@50:95 = {BEST_INT8_ROW['mAP50_95']}` (`{BEST_INT8_ROW['delta_mAP50_95_vs_fp32']:+.5f}`), `AP_small = {BEST_INT8_ROW['AP_small']}` (`{BEST_INT8_ROW['delta_AP_small_vs_fp32']:+.5f}`), `Box IoU vs FP32 = {BEST_INT8_ROW['mean_iou_vs_fp32']}`
            """
        ),
        mo.ui.table(MASTER_DF[_cols_show]),
        _fig,
    ])
    return


if __name__ == "__main__":
    app.run()
