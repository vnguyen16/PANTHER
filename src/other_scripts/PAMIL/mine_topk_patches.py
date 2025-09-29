#!/usr/bin/env python3
# viz_meta_winners_multi_mag.py
# Load pushed winners from META and render patches from multiple MAG dirs.

import os, glob, math
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# ---- EDIT THESE ----
PROTO_NPY   = r"C:\Users\Vivian\Documents\PANTHER\PANTHER\src\other_scripts\PAMIL\10x_prototypes_final"
META        = r"C:\Users\Vivian\Documents\PANTHER\PANTHER\src\other_scripts\PAMIL\10x_global_proto_330_s_0"
SLIDES_ROOT = r"C:\Users\Vivian\Documents\CONCH\patches_tiled\patches_10x"
MAGS        = []      # try these in order; or set to [] to search all subdirs under SLIDES_ROOT
OUT_DIR     = r"C:\Users\Vivian\Documents\PANTHER\PANTHER\src\other_scripts\PAMIL\results\fa_pt_10x_run1"
COLS        = 5
# --------------------

os.makedirs(OUT_DIR, exist_ok=True)

# Load proto count and META winners
P = np.load(PROTO_NPY)
K = int(P.shape[0])
meta = torch.load(META, map_location="cpu")
patients = meta.get("patients_id", [])
coords   = meta.get("coords", [])

def npy_to_rgb(path):
    arr = np.load(path)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    elif arr.ndim == 3 and arr.shape[0] in (1,3) and (arr.shape[2] not in (1,3)):
        arr = np.moveaxis(arr, 0, -1)
    if arr.dtype != np.uint8:
        a_min, a_max = float(arr.min()), float(arr.max())
        if a_max > a_min:
            arr = (255*(arr - a_min)/(a_max - a_min)).clip(0,255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] > 3:
        arr = arr[:, :, :3]
    return Image.fromarray(arr)

def list_mags(root):
    if MAGS: return MAGS
    return [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

images, captions = [], []
found_count = 0

for i in range(min(K, len(patients))):
    slide_id = str(patients[i]).strip()
    if not slide_id:
        print(f"[skip] proto {i}: empty slide_id"); continue
    x, y = map(int, coords[i])
    cls = slide_id.split()[0]
    got_any = False

    for mag in list_mags(SLIDES_ROOT):
        slide_dir = os.path.join(SLIDES_ROOT, mag, cls, slide_id)
        if not os.path.isdir(slide_dir):
            continue

        patt = os.path.join(slide_dir, f"*x{x}_y{y}*.npy")
        files = glob.glob(patt)
        if not files:
            continue

        img = npy_to_rgb(files[0])

        # save individual image (include MAG for clarity)
        clean_slide = slide_id.replace(" ", "_")
        out_png = os.path.join(OUT_DIR, f"proto_{i:02d}_{mag}_{clean_slide}_x{x}_y{y}.png")
        img.save(out_png, format="PNG")

        images.append(img)
        captions.append(f"p{i} | {mag} | {slide_id}\n({x},{y})")
        got_any = True
        found_count += 1
        # if you only want the first MAG that matches, uncomment the next line:
        # break

    if not got_any:
        print(f"[miss] proto {i}: no patch found for ({slide_id}, x={x}, y={y}) in any MAG")

print(f"[info] saved {found_count} individual PNGs to {OUT_DIR}")

# Build a simple grid (mixed across protos/mags in the order found)
if images:
    W, H = images[0].size
    pad, cap_h = 8, 28
    rows = (len(images) + COLS - 1) // COLS
    grid_w = COLS * W + (COLS + 1) * pad
    grid_h = rows * (H + cap_h) + (rows + 1) * pad
    grid = Image.new("RGB", (grid_w, grid_h), color=(255,255,255))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for idx, (im, cap) in enumerate(zip(images, captions)):
        r = idx // COLS
        c = idx % COLS
        x0 = pad + c * (W + pad)
        y0 = pad + r * (H + cap_h + pad)
        grid.paste(im, (x0, y0))
        draw.text((x0, y0 + H + 2), cap, fill=(0,0,0), font=font)

    grid_path = os.path.join(OUT_DIR, "proto_winners_grid_multi.png")
    grid.save(grid_path, format="PNG")
    print("[saved]", grid_path)
else:
    print("[info] no images to grid.")
