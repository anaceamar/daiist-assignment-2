"""
Assignment 2: training pipeline. Varroa mite detection at the hive entrance.

    uv run python main.py train

Start to finish, no manual steps:

1. Download the dataset version from Roboflow Universe (key via api_key.py).
2. Describe it: images and instances per class per split, how many images
   contain a mite, object size relative to the image.
3. Finetune yolo11n from the COCO-pretrained checkpoint and copy the best
   weights into weights/.
4. Evaluate on the test split, per class, against two baselines: the COCO
   model zero-shot and a trivial "predict nothing" model.
5. Turn detections into an image-level decision (does this image show a mite?)
   and cost it: sweep the threshold, pick the cheapest single threshold and the
   cheapest pass / review / fail band ON VALIDATION, then report them on test.
6. Save everything app.py needs into artifacts/.

Every number marked ASSUMPTION is a business assumption explained in
REPORT.md. Change it and rerun: only steps 5-6 depend on it.
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "datasets" / "varroa"))
RUNS = ROOT / "runs"
WEIGHTS = ROOT / "weights"
ARTIFACTS = ROOT / "artifacts"
WEIGHTS_FILE = WEIGHTS / "varroa_yolo11n.pt"

# ---------------------------------------------------------------- dataset ---
WORKSPACE = "varroa-j8231"
PROJECT = "varroa3k"
VERSION = 1
POSITIVE = "varroa"          # the class whose presence triggers an alert

# --------------------------------------------------------------- training ---
BASE_MODEL = os.environ.get("BASE_MODEL", "yolo11n.pt")
EPOCHS = int(os.environ.get("EPOCHS", 60))
IMGSZ = int(os.environ.get("IMGSZ", 640))
BATCH = int(os.environ.get("BATCH", 16))
PATIENCE = 15
SEED = 0

# ------------------------------------------- business model (ASSUMPTIONS) ---
# One "case" = one image from a camera at a hive entrance.
COST_FN = 60.0      # infested hive not flagged: treatment delayed, colony at risk
COST_FP = 10.0      # healthy hive flagged: beekeeper visit + mite count for nothing
COST_TP = 13.0      # infested hive flagged: the treatment itself (needed anyway)
COST_TN = 0.0
COST_REVIEW = 0.15  # a person looks at one image for ~20 s at ~25 EUR/h
BASE_RATE = 0.05    # share of images in production that really show a mite
HUMAN_RECALL = 0.95       # a reviewer still misses some mites
HUMAN_SPECIFICITY = 0.98  # ... and raises some false alarms
IMAGES_PER_DAY = 2000     # e.g. 400 hives x 5 images a day
REVIEW_SECONDS = 20
REVIEW_HOURS_PER_PERSON = 6   # focused review hours in a working day
REVIEWERS_AVAILABLE = 1

THRESHOLDS = np.round(np.linspace(0, 1, 101), 2)
PRED_CONF = 0.01    # keep low-confidence boxes so every threshold can be swept
IOU_MATCH = 0.5


# =========================================================== 1. download ===

def download_dataset() -> Path:
    yaml_path = DATA_DIR / "data.yaml"
    if yaml_path.exists():
        print(f"dataset already at {DATA_DIR}")
        return yaml_path
    from roboflow import Roboflow
    from api_key import load_api_key

    rf = Roboflow(api_key=load_api_key())
    rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
        "yolov11", location=str(DATA_DIR), overwrite=True)
    return yaml_path


def normalise_yaml(yaml_path: Path) -> tuple[Path, list[str]]:
    """Rewrite data.yaml with absolute paths and make sure a test split exists."""
    cfg = yaml.safe_load(yaml_path.read_text())
    names = cfg["names"]
    names = list(names.values()) if isinstance(names, dict) else list(names)
    splits = {s: DATA_DIR / s / "images" for s in ("train", "valid", "test")}
    if not splits["test"].exists():
        raise SystemExit(
            "This dataset version has no test split. Pick a version that has one "
            "(train/valid/test) and update VERSION.")
    fixed = {"path": str(DATA_DIR), "train": "train/images", "val": "valid/images",
             "test": "test/images", "nc": len(names), "names": names}
    out = DATA_DIR / "data_abs.yaml"
    out.write_text(yaml.safe_dump(fixed, sort_keys=False))
    return out, names


# ============================================================ 2. describe ===

def read_labels(split: str) -> dict[str, np.ndarray]:
    """image stem -> array of [cls, cx, cy, w, h] (normalised)."""
    out = {}
    img_dir = DATA_DIR / split / "images"
    lab_dir = DATA_DIR / split / "labels"
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        lab = lab_dir / f"{img.stem}.txt"
        rows = []
        if lab.exists():
            for line in lab.read_text().split("\n"):
                parts = line.split()
                if len(parts) == 5:
                    rows.append([float(p) for p in parts])
                elif len(parts) > 5:   # polygon: turn into its bounding box
                    xs = [float(p) for p in parts[1::2]]
                    ys = [float(p) for p in parts[2::2]]
                    rows.append([float(parts[0]), (min(xs) + max(xs)) / 2,
                                 (min(ys) + max(ys)) / 2, max(xs) - min(xs),
                                 max(ys) - min(ys)])
        out[str(img)] = np.array(rows).reshape(-1, 5)
    return out


def describe(names: list[str], pos_id: int) -> dict:  # noqa: C901
    stats = {"classes": names, "positive_class": POSITIVE, "splits": {}}
    for split in ("train", "valid", "test"):
        labels = read_labels(split)
        all_boxes = np.vstack([b for b in labels.values() if len(b)] or [np.zeros((0, 5))])
        per_class = {n: int((all_boxes[:, 0] == i).sum()) for i, n in enumerate(names)}
        area = all_boxes[:, 3] * all_boxes[:, 4] if len(all_boxes) else np.array([])
        size = {}
        for i, n in enumerate(names):
            a = area[all_boxes[:, 0] == i] if len(all_boxes) else np.array([])
            size[n] = {"median_area_pct": round(float(np.median(a) * 100), 3) if len(a) else None,
                       "median_side_pct": round(float(np.median(np.sqrt(a)) * 100), 2) if len(a) else None}
        with_pos = sum(1 for b in labels.values() if (b[:, 0] == pos_id).any())
        stats["splits"][split] = {
            "images": len(labels),
            "images_without_labels": sum(1 for b in labels.values() if len(b) == 0),
            "instances": per_class,
            "images_with_positive": with_pos,
            "share_images_with_positive": round(with_pos / max(len(labels), 1), 3),
            "object_size": size,
        }
    return stats


# ============================================================== 3. train ===

def pick_device() -> str:
    if os.environ.get("DEVICE"):
        return os.environ["DEVICE"]
    import torch
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def finetune(data_yaml: Path) -> Path:
    from ultralytics import YOLO

    device = pick_device()
    print(f"training {BASE_MODEL} for up to {EPOCHS} epochs on {device}")
    model = YOLO(BASE_MODEL)
    model.train(data=str(data_yaml), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
                patience=PATIENCE, seed=SEED, deterministic=True, device=device,
                project=str(RUNS), name="finetune", exist_ok=True, plots=False,
                verbose=False)
    best = RUNS / "finetune" / "weights" / "best.pt"
    WEIGHTS.mkdir(exist_ok=True)
    shutil.copy(best, WEIGHTS_FILE)
    return WEIGHTS_FILE


# ============================================== 4. evaluation helpers =====

def xywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    return np.column_stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                            b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a, b = xywh_to_xyxy(a), xywh_to_xyxy(b)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def predict_split(weights: str, split: str) -> list[dict]:
    """Run a model over one split and keep every box above PRED_CONF."""
    from ultralytics import YOLO

    labels = read_labels(split)
    model = YOLO(weights)
    records = []
    paths = list(labels)
    for i in range(0, len(paths), 32):
        chunk = paths[i:i + 32]
        for path, r in zip(chunk, model.predict(chunk, conf=PRED_CONF, imgsz=IMGSZ,
                                                verbose=False)):
            b = r.boxes
            records.append({
                "image": path,
                "gt": labels[path].tolist(),
                "pred_cls": b.cls.cpu().numpy().astype(int).tolist(),
                "pred_conf": [round(float(c), 4) for c in b.conf.cpu().numpy()],
                "pred_xywh": [[round(float(v), 5) for v in row] for row in b.xywhn.cpu().numpy()],
            })
    return records


def image_scores(records: list[dict], gt_pos: int,
                 score_cls: int | None) -> tuple[np.ndarray, np.ndarray]:
    """y = image has a label of class gt_pos; score = max confidence of a box
    of class score_cls.

    score_cls=None (COCO zero-shot): COCO has no mite class, so the best it can
    do is say "something is there" -> score = max confidence of ANY box.
    """
    y, s = [], []
    for r in records:
        gt = np.array(r["gt"]).reshape(-1, 5)
        y.append(int((gt[:, 0] == gt_pos).any()))
        conf = np.array(r["pred_conf"])
        cls = np.array(r["pred_cls"])
        keep = conf if score_cls is None else conf[cls == score_cls]
        s.append(float(keep.max()) if len(keep) else 0.0)
    return np.array(y), np.array(s)


def object_confusion(records: list[dict], n_cls: int, conf: float) -> np.ndarray:
    """Rows = true class (+ background), cols = predicted class (+ background)."""
    cm = np.zeros((n_cls + 1, n_cls + 1), dtype=int)
    bg = n_cls
    for r in records:
        gt = np.array(r["gt"]).reshape(-1, 5)
        keep = np.array(r["pred_conf"]) >= conf
        pcls = np.array(r["pred_cls"], dtype=int)[keep] if len(r["pred_cls"]) else np.array([], int)
        pbox = np.array(r["pred_xywh"]).reshape(-1, 4)[keep] if len(r["pred_xywh"]) else np.zeros((0, 4))
        ious = iou_matrix(gt[:, 1:], pbox)
        used_g, used_p = set(), set()
        pairs = sorted(((ious[g, p], g, p) for g in range(len(gt)) for p in range(len(pbox))
                        if ious[g, p] >= IOU_MATCH), reverse=True)
        for _, g, p in pairs:
            if g in used_g or p in used_p:
                continue
            used_g.add(g); used_p.add(p)
            cm[int(gt[g, 0]), pcls[p]] += 1
        for g in range(len(gt)):
            if g not in used_g:
                cm[int(gt[g, 0]), bg] += 1
        for p in range(len(pbox)):
            if p not in used_p:
                cm[bg, pcls[p]] += 1
    return cm


def auc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(greater + 0.5 * ties)


# ================================================== 5. cost of a decision ===

def outcome_shares(y: np.ndarray, s: np.ndarray, lo: float, hi: float) -> dict:
    """Shares of positives / negatives sent to pass, review and fail.

    score >= hi -> fail (alert);  score < lo -> pass;  otherwise -> review.
    lo == hi means no review band: the model decides alone.
    """
    out = {}
    for name, mask in (("pos", y == 1), ("neg", y == 0)):
        v = s[mask]
        n = max(len(v), 1)
        fail = (v >= hi).sum() / n
        passed = (v < lo).sum() / n
        out[name] = {"fail": fail, "review": 1 - fail - passed, "pass": passed}
    return out


def default_costs() -> dict:
    return {"fn": COST_FN, "fp": COST_FP, "tp": COST_TP, "tn": COST_TN,
            "review": COST_REVIEW, "base_rate": BASE_RATE,
            "human_recall": HUMAN_RECALL, "human_specificity": HUMAN_SPECIFICITY}


def cost_per_1000(shares: dict, c: dict | None = None) -> dict:
    """Expected cost per 1,000 production images at the assumed base rate.

    The test set's own mix of positives is NOT used: shares are rates within
    positives and within negatives, re-weighted to the production base rate.
    """
    c = c or default_costs()
    base_rate = c["base_rate"]
    p, n = shares["pos"], shares["neg"]
    pos_review = c["review"] + c["human_recall"] * c["tp"] + (1 - c["human_recall"]) * c["fn"]
    neg_review = (c["review"] + (1 - c["human_specificity"]) * c["fp"]
                  + c["human_specificity"] * c["tn"])
    pos_cost = p["fail"] * c["tp"] + p["pass"] * c["fn"] + p["review"] * pos_review
    neg_cost = n["fail"] * c["fp"] + n["pass"] * c["tn"] + n["review"] * neg_review
    review_share = base_rate * p["review"] + (1 - base_rate) * n["review"]
    per_case = base_rate * pos_cost + (1 - base_rate) * neg_cost
    return {"cost_per_1000": 1000 * per_case, "review_share": review_share}


def staffing(review_share: float) -> dict:
    cases = review_share * IMAGES_PER_DAY
    hours = cases * REVIEW_SECONDS / 3600
    return {"review_cases_per_day": round(cases, 1), "review_hours_per_day": round(hours, 2),
            "people_needed": round(hours / REVIEW_HOURS_PER_PERSON, 2)}


def single_threshold_sweep(y, s) -> np.ndarray:
    return np.array([cost_per_1000(outcome_shares(y, s, t, t))["cost_per_1000"]
                     for t in THRESHOLDS])


def best_band(y, s) -> dict:
    """Cheapest (lo, hi) whose review load fits the reviewers available."""
    capacity_share = (REVIEWERS_AVAILABLE * REVIEW_HOURS_PER_PERSON * 3600
                      / REVIEW_SECONDS / IMAGES_PER_DAY)
    best = None
    for i, lo in enumerate(THRESHOLDS):
        for hi in THRESHOLDS[i:]:
            c = cost_per_1000(outcome_shares(y, s, lo, hi))
            if c["review_share"] > capacity_share + 1e-9:
                continue
            if best is None or c["cost_per_1000"] < best["cost_per_1000"] - 1e-9:
                best = {"lo": float(lo), "hi": float(hi), **c}
    best["capacity_share"] = capacity_share
    return best


# =================================================================== main ===

def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)

    data_yaml, names = normalise_yaml(download_dataset())
    pos_matches = [i for i, n in enumerate(names) if POSITIVE.lower() in n.lower()]
    if not pos_matches:
        raise SystemExit(f"No class containing '{POSITIVE}' in {names}")
    pos_id = pos_matches[0]
    print(f"classes: {names}; positive = {names[pos_id]} (id {pos_id})")

    stats = describe(names, pos_id)
    (ARTIFACTS / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    for split, d in stats["splits"].items():
        print(f"{split:5s} images={d['images']:5d} instances={d['instances']} "
              f"images with {POSITIVE}={d['share_images_with_positive']:.0%}")

    weights = finetune(data_yaml)

    # -- object-level metrics on test (Ultralytics' own evaluator) --------
    from ultralytics import YOLO
    m = YOLO(str(weights)).val(data=str(data_yaml), split="test", imgsz=IMGSZ,
                               conf=0.001, plots=False, verbose=False,
                               project=str(RUNS), name="test_eval", exist_ok=True)
    per_class = []
    for k, c in enumerate(m.box.ap_class_index):
        per_class.append({"class": names[int(c)],
                          "precision": round(float(m.box.p[k]), 4),
                          "recall": round(float(m.box.r[k]), 4),
                          "ap50": round(float(m.box.ap50[k]), 4),
                          "ap50_95": round(float(m.box.ap[k]), 4)})
    overall = {"map50": round(float(m.box.map50), 4), "map50_95": round(float(m.box.map), 4)}
    print(pd.DataFrame(per_class).to_string(index=False))
    print(overall)

    # -- raw predictions, kept so the app can recompute anything ----------
    preds = {"val": predict_split(str(weights), "valid"),
             "test": predict_split(str(weights), "test")}
    coco_test = predict_split(BASE_MODEL, "test")
    for split, recs in preds.items():
        (ARTIFACTS / f"predictions_{split}.json").write_text(json.dumps(recs))
    (ARTIFACTS / "predictions_test_coco.json").write_text(json.dumps(coco_test))

    # -- image-level decision -------------------------------------------
    y_val, s_val = image_scores(preds["val"], pos_id, pos_id)
    y_test, s_test = image_scores(preds["test"], pos_id, pos_id)
    _, s_coco = image_scores(coco_test, pos_id, None)
    s_nothing = np.zeros_like(s_test)

    sweep = pd.DataFrame({
        "threshold": THRESHOLDS,
        "cost_val": single_threshold_sweep(y_val, s_val),
        "cost_test": single_threshold_sweep(y_test, s_test),
        "cost_test_coco": single_threshold_sweep(y_test, s_coco),
    })
    sweep.to_csv(ARTIFACTS / "cost_sweep.csv", index=False)

    # Thresholds are chosen on validation, then applied unchanged to test.
    t_alone = float(THRESHOLDS[int(np.argmin(sweep["cost_val"]))])
    band = best_band(y_val, s_val)
    t_coco = float(THRESHOLDS[int(np.argmin(sweep["cost_test_coco"]))])

    def test_result(lo, hi, scores=s_test):
        sh = outcome_shares(y_test, scores, lo, hi)
        c = cost_per_1000(sh)
        return {"lo": lo, "hi": hi, **c, **staffing(c["review_share"]),
                "recall_pos": sh["pos"]["fail"], "false_alarm_rate": sh["neg"]["fail"]}

    alternatives = {
        "do_nothing": test_result(1.01, 1.01, s_nothing) | {"note": "every image passes"},
        "person_checks_everything": test_result(0.0, 1.01) | {"note": "every image reviewed"},
        "model_alone": test_result(t_alone, t_alone),
        "model_plus_review": test_result(band["lo"], band["hi"]),
        "coco_zero_shot_alone": test_result(t_coco, t_coco, s_coco)
                                | {"note": "threshold picked on test: an upper bound"},
    }
    for k, v in alternatives.items():
        print(f"{k:26s} cost/1000={v['cost_per_1000']:8.1f} "
              f"review={v['review_share']:.1%} people={v['people_needed']}")

    image_level = {
        "test_images": int(len(y_test)), "test_positive_images": int(y_test.sum()),
        "auc_finetuned": round(auc(y_test, s_test), 4),
        "auc_coco_zero_shot": round(auc(y_test, s_coco), 4),
        "auc_predict_nothing": 0.5,
    }

    metrics = {
        "classes": names, "positive_class": names[pos_id], "positive_id": pos_id,
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION},
        "training": {"base_model": BASE_MODEL, "epochs": EPOCHS, "imgsz": IMGSZ,
                     "batch": BATCH, "device": pick_device()},
        "per_class_test": per_class, "overall_test": overall,
        "baselines_object_level": {
            "coco_zero_shot": {"map50": 0.0, "note": "COCO has no bee or varroa class, "
                               "so no prediction can match a label"},
            "predict_nothing": {"map50": 0.0, "recall": 0.0},
        },
        "image_level_test": image_level,
        "costs": {"fn": COST_FN, "fp": COST_FP, "tp": COST_TP, "tn": COST_TN,
                  "review": COST_REVIEW, "base_rate": BASE_RATE,
                  "human_recall": HUMAN_RECALL, "human_specificity": HUMAN_SPECIFICITY,
                  "images_per_day": IMAGES_PER_DAY, "review_seconds": REVIEW_SECONDS,
                  "review_hours_per_person": REVIEW_HOURS_PER_PERSON,
                  "reviewers_available": REVIEWERS_AVAILABLE},
        "chosen": {"single_threshold": t_alone, "band_lo": band["lo"], "band_hi": band["hi"],
                   "chosen_on": "validation", "review_capacity_share": band["capacity_share"]},
        "alternatives_test": alternatives,
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"\nchosen on validation: model alone t={t_alone}; band [{band['lo']}, {band['hi']})")
    print(f"saved weights to {WEIGHTS_FILE.relative_to(ROOT)} and artifacts to artifacts/")


if __name__ == "__main__":
    main()
