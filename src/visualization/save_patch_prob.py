# # save qq prob per slide as npz. loads saved prototypes. run once per dataset (e.g. FA_PT_5x, FA_PT_10x, FA_PT_2.5x) and per fold (if applicable)

# import os
# import argparse
# import numpy as np
# import h5py
# import torch

# # If get_panther_encoder lives elsewhere, adjust this import accordingly.
# from .prototype_visualization_utils import get_panther_encoder


# SPLIT_NAMES_DEFAULT = {"train", "val", "test"}


# def build_panther_encoder(in_dim, p, proto_path, config_dir, model_config, out_type):
#     """
#     Construct the PANTHER encoder once, set to eval mode.
#     """
#     encoder = get_panther_encoder(
#         in_dim=in_dim,
#         p=p,
#         proto_path=proto_path,
#         config_dir=config_dir,
#         model_config=model_config,
#         out_type=out_type,
#     )
#     encoder.eval()
#     return encoder


# def infer_split_from_path(path, split_names=SPLIT_NAMES_DEFAULT):
#     parts = os.path.normpath(path).split(os.sep)
#     for p in reversed(parts):
#         if p.lower() in split_names:
#             return p.lower()
#     return None  # if not found


# def find_h5_files(root_dir):
#     for dirpath, _, filenames in os.walk(root_dir):
#         for fn in filenames:
#             if fn.lower().endswith(".h5"):
#                 yield os.path.join(dirpath, fn)


# def process_h5(h5_path, out_root, encoder, split_names=SPLIT_NAMES_DEFAULT):
#     """
#     Read coords + features from a slide .h5, run PANTHER to get patch probabilities,
#     and save a per-slide .npz with (qq, coords, mask).
#     """
#     slide_id = os.path.splitext(os.path.basename(h5_path))[0]
#     split = infer_split_from_path(h5_path, split_names) or "unspecified"

#     out_dir = os.path.join(out_root, split)
#     os.makedirs(out_dir, exist_ok=True)
#     out_path = os.path.join(out_dir, f"{slide_id}.npz")

#     # Load inputs
#     with h5py.File(h5_path, "r") as h5:
#         coords = h5["coords"][:]
#         feats = np.asarray(h5["features"][:], dtype=np.float32)

#     # To torch
#     feats_t = torch.from_numpy(feats)  # [N, d]
#     try:
#         enc_device = next(encoder.parameters()).device
#     except StopIteration:
#         enc_device = torch.device("cpu")
#     if enc_device.type == "cuda":
#         feats_t = feats_t.cuda()

#     # Run encoder → per-patch responsibilities
#     with torch.inference_mode():
#         info = encoder.representation(feats_t.unsqueeze(0))  # [1, N, d]
#         qqs = info["qq"]  # [1, N, K, 1]

#     # Squeeze to [N, K]
#     qq = qqs[0, :, :, 0].detach().cpu().numpy().astype(np.float32)

#     # Coords + mask
#     coords = coords.astype(np.int32)
#     mask = np.ones((qq.shape[0],), dtype=bool)  # update if you use padding elsewhere

#     # Save
#     np.savez_compressed(out_path, qq=qq, coords=coords, mask=mask)
#     return out_path, split, slide_id, qq.shape


# def main():
#     parser = argparse.ArgumentParser(
#         description="Dump per-patch prototype probabilities (qq) to .npz per slide."
#     )
#     parser.add_argument("--h5_root", required=True,
#                         help="Root directory containing slide .h5 files (optionally under train/val/test).")
#     parser.add_argument("--out_root", required=True,
#                         help="Output root directory for .npz files.")
#     parser.add_argument("--splits", default="train,val,test",
#                         help="Comma-separated split names to recognize in paths.")
#     # Encoder args
#     parser.add_argument("--in_dim", type=int, required=True)
#     parser.add_argument("--n_proto", type=int, required=True)
#     parser.add_argument("--proto_path", type=str, required=True)
#     parser.add_argument("--config_dir", type=str, required=True)
#     parser.add_argument("--model_config", type=str, default="PANTHER_fa_pt")
#     parser.add_argument("--out_type", type=str, default="allcat")
#     args = parser.parse_args()

#     split_names = {s.strip().lower() for s in args.splits.split(",") if s.strip()}

#     # Build encoder once
#     encoder = build_panther_encoder(
#         in_dim=args.in_dim,
#         p=args.n_proto,
#         proto_path=args.proto_path,
#         config_dir=args.config_dir,
#         model_config=args.model_config,
#         out_type=args.out_type,
#     )

#     h5_list = list(find_h5_files(args.h5_root))
#     print(f"Found {len(h5_list)} H5 files under {args.h5_root}")

#     processed = 0
#     for h5_path in h5_list:
#         try:
#             out_path, split, slide_id, shape = process_h5(h5_path, args.out_root, encoder, split_names)
#             print(f"[OK] {split}/{slide_id}: qq shape {shape} -> {out_path}")
#             processed += 1
#         except Exception as e:
#             print(f"[ERROR] {h5_path}: {e}")

#     print(f"Done. Wrote {processed} slide npz files to {args.out_root}.")


# if __name__ == "__main__":
#     main()

# ------------------ revised script
#!/usr/bin/env python3
"""
dump_patch_probs_from_csvsplits.py

Reads slide splits from a directory (train.csv/val.csv/test.csv with columns: slide_id,label),
finds the corresponding slide .h5 feature files under --h5_root (even if the paths don't include split names),
runs the PANTHER encoder to compute per-patch prototype probabilities (qq),
and saves each slide's result directly into:
    <out_root>/<split>/<slide_id>.npz
where the .npz contains keys: qq (N,K), coords (N,2), mask (N,).

Usage (example):
  python dump_patch_probs_from_csvsplits.py ^
    --h5_root C:/.../features/5x/feats_h5 ^
    --split_dir C:/.../src/splits/cross-val/FA_PT_k=0 ^
    --out_root C:/.../qq_outputs/FA_PT_k=0/5x ^
    --in_dim 1024 --n_proto 16 ^
    --proto_path C:/.../checkpoints/FA_PT_k=0/5x/prototypes.pt ^
    --config_dir C:/.../src/configs ^
    --model_config PANTHER_fa_pt ^
    --out_type allcat

Notes:
- We index ALL .h5 files under --h5_root once, then match by slide_id (case/whitespace/underscore/hyphen tolerant).
- If your .h5 filenames exactly match slide_id + '.h5', pass --h5_exact_names to skip indexing and speed things up.
"""

import os
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import h5py
import torch
from typing import Optional, Dict, Tuple, List


# If get_panther_encoder lives elsewhere, adjust this import accordingly.
from .prototype_visualization_utils import get_panther_encoder

DEFAULT_SPLITS = ("train", "val", "test")


def canon(s: str) -> str:
    """Canonicalize a slide_id or filename stem for tolerant matching."""
    return re.sub(r"[\s_-]+", "", s.strip().upper())

def load_csv_split(split_csv: Path) -> List[str]:
    """Read a CSV with columns: slide_id,label and return list of slide_id strings."""
    if not split_csv.exists():
        return []
    df = pd.read_csv(split_csv, dtype=str)
    assert {"slide_id", "label"}.issubset(df.columns), f"Missing slide_id,label in {split_csv}"
    # Keep original formatting of slide_id for saving; also a canonical key for matching
    slide_ids = df["slide_id"].astype(str).tolist()
    return slide_ids

def build_slide_lists(split_dir: Path, splits: Tuple[str, ...]) -> Dict[str, List[str]]:
    """Load split CSVs from split_dir and return {split: [slide_id,...]}."""
    out: dict[str, list[str]] = {}
    for sp in splits:
        csv_path = split_dir / f"{sp}.csv"
        out[sp] = load_csv_split(csv_path)
    return out

def index_h5(h5_root: Path) -> Dict[str, Path]:
    """
    Recursively index all .h5 under h5_root by canonical filename stem -> Path.
    If duplicates occur, first one wins; we warn only in verbose scenarios.
    """
    idx: dict[str, Path] = {}
    for p in h5_root.rglob("*.h5"):
        key = canon(p.stem)
        idx.setdefault(key, p)
    return idx


def resolve_h5_path(slide_id: str, h5_root: Path, idx: Optional[Dict[str, Path]], h5_exact_names: bool) -> Optional[Path]:
    """
    Resolve the .h5 path for slide_id.
    If h5_exact_names=True, require an exact file at h5_root/<slide_id>.h5.
    Else, use the prebuilt index idx (canonical match).
    """
    if h5_exact_names:
        p = h5_root / f"{slide_id}.h5"
        return p if p.exists() else None
    if idx is None:
        raise ValueError("Internal error: idx is None but h5_exact_names=False")
    return idx.get(canon(slide_id))


def build_panther_encoder(in_dim, p, proto_path, config_dir, model_config, out_type):
    encoder = get_panther_encoder(
        in_dim=in_dim,
        p=p,
        proto_path=proto_path,
        config_dir=config_dir,
        model_config=model_config,
        out_type=out_type,
    )
    encoder.eval()
    return encoder


def process_one_slide(h5_path: Path, encoder: torch.nn.Module) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load coords+features from slide .h5, run encoder, return (qq, coords, mask).
    qq -> (N,K) float32
    coords -> (N,2) int32
    mask -> (N,) bool (all True unless you have padding)
    """
    with h5py.File(h5_path, "r") as h5:
        coords = h5["coords"][:].astype(np.int32)
        feats = np.asarray(h5["features"][:], dtype=np.float32)  # (N,D)

    feats_t = torch.from_numpy(feats)  # (N,D)
    try:
        enc_device = next(encoder.parameters()).device
    except StopIteration:
        enc_device = torch.device("cpu")

    if enc_device.type == "cuda":
        feats_t = feats_t.cuda(non_blocking=True)

    with torch.inference_mode():
        # expects [B,N,D]; returns info with ["qq"] shaped [B,N,K,1]
        info = encoder.representation(feats_t.unsqueeze(0))
        qqs = info["qq"]

    qq = qqs[0, :, :, 0].detach().cpu().numpy().astype(np.float32)  # (N,K)
    mask = np.ones((qq.shape[0],), dtype=bool)
    return qq, coords, mask


def main():
    ap = argparse.ArgumentParser(description="Save per-patch prototype probabilities (qq) into split folders from CSV split dir.")
    ap.add_argument("--h5_root", required=True, help="Root directory containing slide .h5 files (no split names inside).")
    ap.add_argument("--split_dir", required=True, help="Directory containing train.csv/val.csv/test.csv (slide_id,label).")
    ap.add_argument("--out_root", required=True, help="Output root; saves to <out_root>/<split>/<slide_id>.npz")
    ap.add_argument("--splits", default="train,val,test", help="Comma-separated split names to read in split_dir. Default: train,val,test")
    ap.add_argument("--h5_exact_names", action="store_true",
                    help="If set, require files at <h5_root>/<slide_id>.h5 exactly. Otherwise, index recursively and match by filename stem.")
    # Encoder args
    ap.add_argument("--in_dim", type=int, required=True)
    ap.add_argument("--n_proto", type=int, required=True)
    ap.add_argument("--proto_path", type=str, required=True)
    ap.add_argument("--config_dir", type=str, required=True)
    ap.add_argument("--model_config", type=str, default="PANTHER_fa_pt")
    ap.add_argument("--out_type", type=str, default="allcat")
    args = ap.parse_args()

    h5_root = Path(args.h5_root)
    split_dir = Path(args.split_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    split_names = tuple(s.strip().lower() for s in args.splits.split(",") if s.strip())

    # Load split CSVs
    slides_by_split = build_slide_lists(split_dir, split_names)
    total_listed = sum(len(v) for v in slides_by_split.values())
    print(f"Loaded split CSVs from {split_dir}: { {k: len(v) for k,v in slides_by_split.items()} } (total {total_listed} slides)")

    # Build H5 index if needed
    idx = None
    if not args.h5_exact_names:
        print(f"Indexing .h5 files under {h5_root} ...")
        idx = index_h5(h5_root)
        print(f"Indexed {len(idx)} .h5 files.")

    # Build encoder once
    print("Loading PANTHER encoder ...")
    encoder = build_panther_encoder(
        in_dim=args.in_dim,
        p=args.n_proto,
        proto_path=args.proto_path,
        config_dir=args.config_dir,
        model_config=args.model_config,
        out_type=args.out_type,
    )
    try:
        device = next(encoder.parameters()).device
        print(f"Encoder device: {device}")
    except StopIteration:
        print("Encoder has no parameters; assuming CPU.")

    # Process slides split-by-split
    missing, done = [], 0
    for sp, slide_ids in slides_by_split.items():
        sp_out = out_root / sp
        sp_out.mkdir(parents=True, exist_ok=True)

        for slide_id in slide_ids:
            h5_path = resolve_h5_path(slide_id, h5_root, idx, args.h5_exact_names)
            if h5_path is None:
                print(f"[MISS] {slide_id}: no matching .h5 under {h5_root}")
                missing.append(slide_id)
                continue

            out_path = sp_out / f"{slide_id}.npz"  # preserve original slide_id formatting
            if out_path.exists():
                # Skip if already done
                print(f"[SKIP] {sp}/{slide_id} (exists)")
                continue

            try:
                qq, coords, mask = process_one_slide(h5_path, encoder)
                np.savez_compressed(out_path, qq=qq, coords=coords, mask=mask)
                print(f"[OK] {sp}/{slide_id}: qq={qq.shape} -> {out_path}")
                done += 1
            except Exception as e:
                print(f"[ERROR] {sp}/{slide_id} from {h5_path}: {e}")

    print(f"\nDone. Wrote {done} npz files to {out_root}. Missing: {len(missing)}")
    if missing:
        print("First few missing:", missing[:10])


if __name__ == "__main__":
    main()
