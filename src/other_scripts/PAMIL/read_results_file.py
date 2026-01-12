#  read results pkl file and convert to csv
# import pickle
# import pandas as pd

# pkl_path = "results/fa_pt_run1_s1/split_0_results.pkl"

# with open(pkl_path, "rb") as f:
#     results = pickle.load(f)

# rows = []
# for slide_id, info in results.items():
#     row = {
#         "slide_id": slide_id,
#         "true_label": int(info["label"]),
#     }
#     probs = info["prob"]
#     for i, p in enumerate(probs):
#         row[f"prob_class{i}"] = p
#     row["pred_class"] = probs.argmax()
#     rows.append(row)

# df = pd.DataFrame(rows)
# print(df.head())
# df.to_csv("results/fa_pt_run1_s1/split_0_results.csv", index=False)

# -------------------------
# # read prototype ckpt file
# import torch

# ckpt_path = "results/fa_pt_run2_s1/global_proto_330_s_0.ckpt"
# ckpt = torch.load(ckpt_path, map_location="cpu")

# print(type(ckpt))
# if isinstance(ckpt, dict):
#     print("Top-level keys:", list(ckpt.keys())[:20])
#     # if nested state dict:
#     state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
#     print("State keys (sample):", [k for k in list(state.keys())[:30]])
# else:
#     # some repos save raw state_dict
#     state = ckpt

# ----------------
# # read prototype matches and save to csv (empty csv)
# import os, torch, numpy as np, pandas as pd

# ckpt_path = "results/fa_pt_run2_s1/global_proto_330_s_0.ckpt"
# out_csv   = "results/fa_pt_run2_s1/global_proto_matches.csv"

# ckpt = torch.load(ckpt_path, map_location="cpu")

# patients_id = ckpt["patients_id"]  # could be list[str] or list[list[str]]
# coords      = ckpt["coords"]       # could be array/list or list of arrays

# rows = []

# def _coerce_coords(arr):
#     arr = np.asarray(arr)
#     if arr.ndim == 1 and arr.size == 2:  # (x,y)
#         return [arr.tolist()]
#     if arr.ndim == 2 and arr.shape[1] == 2:  # (N,2)
#         return arr.tolist()
#     return []

# # Support both “one match per prototype” and “top-k per prototype” formats.
# if isinstance(patients_id, (list, tuple)) and len(patients_id) > 0:
#     # case A: list[str] & coords (N,2) => one match per proto
#     if isinstance(patients_id[0], str):
#         coords_list = _coerce_coords(coords)
#         for pid, (x, y) in zip(patients_id, coords_list):
#             rows.append({"proto_id": len(rows), "slide_id": pid, "x": int(x), "y": int(y)})
#     else:
#         # case B: list[list[str]] & list[(K,2)] => top-k per proto
#         for p, (pids, c) in enumerate(zip(patients_id, coords)):
#             for pid, (x, y) in zip(pids, _coerce_coords(c)):
#                 rows.append({"proto_id": p, "slide_id": pid, "x": int(x), "y": int(y)})

# df = pd.DataFrame(rows)
# os.makedirs(os.path.dirname(out_csv), exist_ok=True)
# df.to_csv(out_csv, index=False)
# print("Wrote:", out_csv)
# print(df.head())

# ---------------------------
# # read features from h5 and match to prototypes, save to csv for one slide
# import h5py, numpy as np, pandas as pd
# from numpy.linalg import norm

# proto_path = "datasets_proto/fa_pt_16_0/train_instance_feats_proto.npy"
# h5_path    = "/home/jovyan/Documents/HistoDataset/uni_features_2.5x/feats_h5/FA 57B.h5"  # ← put a real slide

# P = np.load(proto_path).astype(np.float32)      # (K, 1024)
# P = P / (norm(P, axis=1, keepdims=True) + 1e-8)

# with h5py.File(h5_path, "r") as f:
#     X = np.asarray(f["features"], dtype=np.float32)        # (N, 1024)
#     coords = np.asarray(f.get("coords", np.empty((0,2), np.int32)))
# Xn = X / (norm(X, axis=1, keepdims=True) + 1e-8)

# # cosine similarity matrix (N x K)
# S = Xn @ P.T
# best_id = S.argmax(axis=1)
# best_sim = S[np.arange(S.shape[0]), best_id]

# df = pd.DataFrame({
#     "x": coords[:,0] if coords.size else np.arange(X.shape[0]),
#     "y": coords[:,1] if coords.size else np.zeros(X.shape[0], dtype=int),
#     "best_proto": best_id,
#     "best_sim": best_sim
# })
# out_csv = "results/fa_pt_run2_s1/FA 57B_proto_heatmap.csv"   # replace SLIDE_ID
# df.to_csv(out_csv, index=False)
# print("Wrote", out_csv, "rows:", len(df))

# -------------------------
# # read features from h5 and match to prototypes, save top-k exemplars per proto to csv
# import os, h5py, numpy as np, pandas as pd
# from numpy.linalg import norm

# H5_DIR     = "/home/jovyan/Documents/HistoDataset/uni_features_5x/feats_h5"
# PROTO_NPY  = "datasets_proto/fa_pt_16_0/train_instance_feats_proto.npy"
# TOPK       = 5

# P = np.load(PROTO_NPY).astype(np.float32)          # (K, 1024)
# P = P / (norm(P, axis=1, keepdims=True) + 1e-8)
# K = P.shape[0]

# # per-proto min-heaps via arrays (store many and trim)
# records = [[] for _ in range(K)]

# for fn in os.listdir(H5_DIR):
#     if not fn.endswith(".h5"): continue
#     slide = os.path.splitext(fn)[0]
#     path  = os.path.join(H5_DIR, fn)
#     with h5py.File(path, "r") as f:
#         X = np.asarray(f["features"], dtype=np.float32)
#         coords = np.asarray(f.get("coords", np.empty((0,2), np.int32)))
#     Xn = X / (norm(X, axis=1, keepdims=True) + 1e-8)  # (N, 1024)
#     S = Xn @ P.T                                      # (N, K)
#     for p in range(K):
#         sim = S[:, p]
#         idx = np.argpartition(-sim, TOPK)[:TOPK]      # top-k indices (unsorted)
#         for i in idx:
#             r = {
#                 "proto_id": p,
#                 "slide_id": slide,
#                 "patch_idx": int(i),
#                 "sim": float(sim[i])
#             }
#             if coords.size:
#                 r["x"], r["y"] = int(coords[i,0]), int(coords[i,1])
#             records[p].append(r)

# # keep true top-k per proto
# rows = []
# for p in range(K):
#     rows_p = sorted(records[p], key=lambda r: -r["sim"])[:TOPK]
#     rows.extend(rows_p)

# df = pd.DataFrame(rows)
# os.makedirs("results/fa_pt_5x_run1_s1", exist_ok=True)
# df.to_csv("results/fa_pt_5x_run1_s1/topk_exemplars_per_proto.csv", index=False)
# print("Wrote results/fa_pt_5x_run1_s1/topk_exemplars_per_proto.csv")

# ----------------
#!/usr/bin/env python3
# make_proto_grids_optionA.py
# Build top-k exemplar grids from saved .npy patches using a MultiIndex (x,y)

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Dict

# -----------------------------
# Utilities
# -----------------------------

def build_slide_index(slides_root: str, mag: Optional[str]):
    """
    Returns a dict: slide_id -> absolute slide_dir.

    It tries these layouts (in order):
      1) slides_root/<MAG>/<CLASS>/<SLIDE_ID>/
      2) slides_root/<CLASS>/<SLIDE_ID>/
      3) slides_root/<SLIDE_ID>/

    If 'mag' is provided (e.g., '2.5x', '5x', '10x', '20x', '40x'), restrict search
    to that subdir in layout (1). Otherwise it will look across all first-level subdirs.
    """
    idx = {}

    def _index_layout_1():
        # slides_root/<MAG>/<CLASS>/<SLIDE_ID>/
        mag_dirs = []
        if mag:
            cand = os.path.join(slides_root, mag)
            if os.path.isdir(cand):
                mag_dirs = [cand]
        if not mag_dirs:
            # Any first-level dir could be a mag folder (2.5x, 5x, 10x, 20x, 40x)
            mag_dirs = [d for d in glob.glob(os.path.join(slides_root, "*")) if os.path.isdir(d)]

        for mdir in mag_dirs:
            for cls_dir in glob.glob(os.path.join(mdir, "*")):
                if not os.path.isdir(cls_dir):
                    continue
                for slide_dir in glob.glob(os.path.join(cls_dir, "*")):
                    if not os.path.isdir(slide_dir):
                        continue
                    slide_id = os.path.basename(slide_dir)
                    idx.setdefault(slide_id, slide_dir)

    def _index_layout_2():
        # slides_root/<CLASS>/<SLIDE_ID>/
        for cls_dir in glob.glob(os.path.join(slides_root, "*")):
            if not os.path.isdir(cls_dir):
                continue
            for slide_dir in glob.glob(os.path.join(cls_dir, "*")):
                if not os.path.isdir(slide_dir):
                    continue
                slide_id = os.path.basename(slide_dir)
                idx.setdefault(slide_id, slide_dir)

    def _index_layout_3():
        # slides_root/<SLIDE_ID>/
        for slide_dir in glob.glob(os.path.join(slides_root, "*")):
            if not os.path.isdir(slide_dir):
                continue
            slide_id = os.path.basename(slide_dir)
            idx.setdefault(slide_id, slide_dir)

    _index_layout_1()
    if not idx:
        _index_layout_2()
    if not idx:
        _index_layout_3()
    return idx


def find_index_csv(slide_dir: str, explicit_name: Optional[str]):
    """
    Return a path to a CSV file inside slide_dir that has columns ['patch_file','x','y'].
    If explicit_name is provided, it must exist in slide_dir.
    Delimiter is auto-detected.
    """
    if explicit_name:
        cand = os.path.join(slide_dir, explicit_name)
        if not os.path.isfile(cand):
            raise FileNotFoundError(f"Index CSV not found: {cand}")
        _ = _read_index_csv(cand)  # will validate cols
        return cand

    # autodetect any CSV with needed columns
    for path in glob.glob(os.path.join(slide_dir, "*.csv")):
        try:
            _ = _read_index_csv(path)
        except Exception:
            continue
        return path

    raise FileNotFoundError(f"No CSV with columns [patch_file,x,y] found in {slide_dir}")


def _read_index_csv(path: str) -> pd.DataFrame:
    """
    Read the per-slide index CSV robustly, clean columns, enforce dtypes,
    deduplicate (x,y), and set a MultiIndex on ['x','y'].

    Returns a DataFrame with index ['x','y'] and a 'patch_file' column.
    """
    # auto-detect delimiter, handle commas and tabs
    df = pd.read_csv(path, sep=None, engine="python")

    # drop any "Unnamed: *" artifact columns
    for c in list(df.columns):
        if isinstance(c, str) and c.lower().startswith("unnamed"):
            df.drop(columns=[c], inplace=True)

    # strip whitespace from column names
    df.columns = df.columns.str.strip()

    must = {"patch_file", "x", "y"}
    if not must.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {must}. Found: {list(df.columns)}")

    # clean types
    df = df.dropna(subset=["patch_file", "x", "y"])
    df = df.astype({"x": int, "y": int, "patch_file": str})

    # deduplicate keys if needed
    if df.duplicated(subset=["x", "y"]).any():
        print(f"[warn] duplicate (x,y) rows in {os.path.basename(path)}; keeping the first per key.")
        df = df.drop_duplicates(subset=["x", "y"], keep="first")

    # MultiIndex on (x,y)
    df = df.set_index(["x", "y"]).sort_index()

    # sanity: ensure unique index
    if not df.index.is_unique:
        df = df[~df.index.duplicated(keep="first")]

    return df


def npy_to_pil(path: str) -> Image.Image:
    """
    Load .npy patch and convert to PIL.Image (uint8 RGB).
    Accepts arrays of shapes: (H,W), (H,W,3), (3,H,W).
    """
    arr = np.load(path)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[0] in (1, 3) and (arr.shape[2] not in (1, 3)):
        # channel-first -> channel-last
        arr = np.moveaxis(arr, 0, -1)

    # normalize to 0..255 if needed
    if arr.dtype != np.uint8:
        a_min, a_max = float(arr.min()), float(arr.max())
        if a_max > a_min:
            arr = (255.0 * (arr - a_min) / (a_max - a_min)).clip(0, 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)

    # ensure 3 channels
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] > 3:
        arr = arr[:, :, :3]

    return Image.fromarray(arr)


def make_grid(images, captions, cols=5, pad=8, cap_h=28):
    """
    Create a tiled grid with small text captions under each patch.
    """
    if not images:
        raise ValueError("No images to grid.")
    W, H = images[0].size
    imgs = [im.resize((W, H)) if im.size != (W, H) else im for im in images]
    rows = (len(imgs) + cols - 1) // cols
    grid_w = cols * W + (cols + 1) * pad
    grid_h = rows * (H + cap_h) + (rows + 1) * pad
    grid = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for idx, (im, cap) in enumerate(zip(imgs, captions)):
        r = idx // cols
        c = idx % cols
        x0 = pad + c * (W + pad)
        y0 = pad + r * (H + cap_h + pad)
        grid.paste(im, (x0, y0))
        draw.text((x0, y0 + H + 2), cap, fill=(0, 0, 0), font=font)
    return grid


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Make prototype exemplar grids & per-patch PNGs from saved .npy tiles (MultiIndex version)"
    )
    ap.add_argument("--exemplars_csv", required=True,
                    help="CSV with columns: proto_id,slide_id,patch_idx(optional),sim,x,y")
    ap.add_argument("--slides_root", required=True,
                    help="Root containing slide patches (supports layouts like <mag>/<class>/<slide_id>/ etc.)")
    ap.add_argument("--out_dir", required=True,
                    help="Output dir (PNG grids + individual PNGs + summary CSV)")
    ap.add_argument("--k", type=int, default=5, help="Top-k exemplars per prototype")
    ap.add_argument("--cols", type=int, default=5, help="Columns in the grid")
    ap.add_argument("--index_csv_name", default=None,
                    help="Explicit per-slide CSV name (e.g., patch_map.csv). If omitted, autodetect.")
    ap.add_argument("--mag", default=None,
                    help="Restrict search to a magnification folder (e.g., '2.5x', '5x', '10x', '20x', '40x').")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load exemplars and sort by similarity descending (so head(k) is top-k)
    df = pd.read_csv(args.exemplars_csv)
    need = {"proto_id", "slide_id", "sim", "x", "y"}
    if not need.issubset(df.columns):
        raise ValueError(f"{args.exemplars_csv} must contain columns {need}")
    df = df.sort_values(["proto_id", "sim"], ascending=[True, False]).reset_index(drop=True)

    # Build slide_id -> slide_dir index
    slide_index = build_slide_index(args.slides_root, args.mag)

    # Warn about missing slide dirs
    missing = sorted(set(df["slide_id"]) - set(slide_index.keys()))
    if missing:
        print(f"[warn] Could not locate {len(missing)} slide dirs. First few: {missing[:8]}")

    # Cache for per-slide patch index (MultiIndex on ['x','y'])
    idx_cache: Dict[str, pd.DataFrame] = {}

    summary_rows = []
    for pid, dfp in df.groupby("proto_id"):
        top = dfp.head(args.k)
        images, captions = [], []

        # subdir per prototype for individual PNGs
        indiv_dir = os.path.join(args.out_dir, f"proto_{int(pid)}")
        # os.makedirs(indiv_dir, exist_ok=True)

        for _, row in top.iterrows():
            slide_id = str(row["slide_id"])
            slide_dir = slide_index.get(slide_id)
            if not slide_dir:
                print(f"[skip] slide_id not found under slides_root: {slide_id}")
                continue

            # Load per-slide index csv (once per slide)
            if slide_dir not in idx_cache:
                idx_csv = find_index_csv(slide_dir, args.index_csv_name)
                df_idx = _read_index_csv(idx_csv)  # MultiIndex on ['x','y']
                idx_cache[slide_dir] = df_idx

            df_idx = idx_cache[slide_dir]
            key = (int(row["x"]), int(row["y"]))

            # Try MultiIndex access first
            try:
                patch_file = df_idx.at[key, "patch_file"]  # scalar if index is unique
                patch_path = os.path.join(slide_dir, patch_file)
            except KeyError:
                # Fallback: filename pattern x{X}_y{Y}
                patt = os.path.join(slide_dir, f"*x{key[0]}_y{key[1]}*.npy")
                matches = glob.glob(patt)
                if matches:
                    patch_path = matches[0]
                    patch_file = os.path.relpath(patch_path, slide_dir)
                else:
                    print(f"[skip] patch (x={key[0]}, y={key[1]}) not found in {slide_dir}")
                    continue

            # Load the patch as image
            try:
                im = npy_to_pil(patch_path)
            except Exception as e:
                print(f"[warn] failed to load {patch_path}: {e}")
                continue

            # Save individual PNG
            clean_slide = slide_id.replace(" ", "_")
            out_patch_png = os.path.join(
                indiv_dir,
                f"{clean_slide}_x{key[0]}_y{key[1]}_sim{float(row['sim']):.3f}.png"
            )
            # im.save(out_patch_png, format="PNG")

            images.append(im)
            captions.append(f"{slide_id} | sim={float(row['sim']):.3f}\n(x{key[0]},y{key[1]})")
            summary_rows.append({
                "proto_id": int(pid),
                "slide_id": slide_id,
                "x": key[0],
                "y": key[1],
                "sim": float(row["sim"]),
                "patch_file": os.path.join(os.path.relpath(slide_dir, args.slides_root), patch_file),
                "png_file": os.path.relpath(out_patch_png, args.out_dir),
            })

        # Save grid per prototype
        if images:
            grid = make_grid(images, captions, cols=args.cols)
            out_grid = os.path.join(args.out_dir, f"proto_{int(pid)}_top{args.k}.png")
            grid.save(out_grid, format="PNG")
            print(f"[saved] {out_grid}")
        else:
            print(f"[info] proto {pid}: no images gathered.")

    # Save summary CSV
    out_csv = os.path.join(args.out_dir, f"exemplars_used_top{args.k}.csv")
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")


if __name__ == "__main__":
    main()

