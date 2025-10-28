#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple YOLOv8 training script (Python API) — training + validation only.

Expected layout (relative to this file):
  ./train_simple.py
  ./DataSet/
      data.yaml  (if present, used as-is; else auto-generated)
      train/images, train/labels
      valid/images, valid/labels   # or val/images, val/labels
"""

from pathlib import Path
import argparse

# Optional dependencies
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

try:
    import torch  # type: ignore
except Exception:
    torch = None

from ultralytics import YOLO


def log(msg: str):
    print(f"[train_simple] {msg}", flush=True)


def valid_key(ds_root: Path) -> str:
    """Return subdir used for validation images: 'valid/images' if exists, otherwise 'val/images'."""
    return "valid/images" if (ds_root / "valid" / "images").exists() else "val/images"


def ensure_yaml(ds_root: Path) -> Path:
    """Use an existing data.yaml if present; otherwise create dataset.auto.yaml."""
    for cand in (ds_root / "data.yaml", ds_root / "dataset.yaml"):
        if cand.exists():
            log(f"Found dataset yaml: {cand}")
            return cand

    # Basic structure checks
    train_images = ds_root / "train" / "images"
    v_images = ds_root / "valid" / "images"
    alt_v_images = ds_root / "val" / "images"
    if not train_images.exists() or not (v_images.exists() or alt_v_images.exists()):
        raise FileNotFoundError(
            f"Expected YOLO structure under {ds_root}:\n"
            f"  train/images, train/labels, and valid|val/images, valid|val/labels"
        )

    # Rough class count inference from train/labels
    nc = 1
    labels_dir = ds_root / "train" / "labels"
    if labels_dir.exists():
        max_cls = -1
        for t in list(labels_dir.rglob("*.txt"))[:500]:
            try:
                for line in t.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    cid = int(float(line.split()[0]))
                    if cid > max_cls:
                        max_cls = cid
            except Exception:
                continue
        if max_cls >= 0:
            nc = max_cls + 1

    names = {i: f"class_{i}" for i in range(nc)}
    data = {
        "path": str(ds_root.resolve()),
        "train": "train/images",
        "val": valid_key(ds_root),
        "names": names,
    }

    out = ds_root / "dataset.auto.yaml"
    if yaml is not None:
        with open(out, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    else:
        lines = [f'path: {data["path"]}', f'train: {data["train"]}', f'val: {data["val"]}', "names:"]
        for k, v in names.items():
            lines.append(f"  {k}: {v}")
        out.write_text("\n".join(lines), encoding="utf-8")

    log(f"Auto-generated dataset yaml: {out}")
    return out


def choose_device(user_device: str | None) -> str:
    """Prefer CUDA:0 when available; otherwise CPU."""
    if user_device:
        return user_device
    try:
        if (torch is not None) and torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="DataSet", type=str, help="Dataset root (train/, valid|val/)")
    p.add_argument("--model", default="yolov8n.pt", type=str, help="Ultralytics model or weights")
    p.add_argument("--epochs", default=100, type=int)
    p.add_argument("--imgsz", default=640, type=int)
    p.add_argument("--batch", default=16, type=int)
    p.add_argument("--device", default=None, type=str, help='e.g. "cpu", "0", or "0,1"')
    p.add_argument("--name", default="exp", type=str, help="Run name under runs/detect")
    p.add_argument("--project", default="runs", type=str, help="Project directory for outputs")
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    ds_root = (root / args.dataset).resolve()
    data_yaml = ensure_yaml(ds_root)

    device = choose_device(args.device)
    log(f"Using device={device}")

    model = YOLO(args.model)

    # Train
    log(f"Training {args.model} for {args.epochs} epochs (imgsz={args.imgsz}, batch={args.batch})")
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    # Validate best.pt if present, else validate current
    save_dir = Path(args.project) / "detect" / args.name / "weights"
    best = save_dir / "best.pt"
    if best.exists():
        log(f"Validating best weights: {best}")
        YOLO(str(best)).val()
    else:
        log("best.pt not found; validating current model in memory.")
        model.val()

    log("Done.")


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())
