
# inspect_npz_allk.py
import os, argparse, numpy as np, pandas as pd

def load_npz(npz_path):
    d = np.load(npz_path, allow_pickle=False)
    return d["qq"], d["coords"], d["mask"]

def stats_for_slide(npz_path, renorm=False, nearhard=0.99, atol=1e-4):
    slide_id = os.path.splitext(os.path.basename(npz_path))[0]
    qq, coords, mask = load_npz(npz_path)

    assert qq.ndim == 2
    N, K = qq.shape
    assert coords.shape[0] == N and mask.shape[0] == N

    # keep only real patches (if mask used)
    qq_m = qq[mask]
    coords_m = coords[mask]
    Nm = qq_m.shape[0]

    # row-sum sanity
    row_sums = qq_m.sum(axis=1)
    row_sum_ok = np.allclose(row_sums, 1.0, atol=atol)
    row_sum_min, row_sum_max = float(row_sums.min()), float(row_sums.max())

    # optional renorm (for clean stats)
    if renorm:
        qq_m = qq_m / (row_sums[:, None] + 1e-12)
        row_sums = qq_m.sum(axis=1)
        row_sum_ok = np.allclose(row_sums, 1.0, atol=1e-6)
        row_sum_min, row_sum_max = float(row_sums.min()), float(row_sums.max())

    # entropy (natural log)
    eps = 1e-12
    ent = -(qq_m * np.log(qq_m + eps)).sum(axis=1)
    ent_mean, ent_std = float(ent.mean()), float(ent.std())

    # near-hard
    mx = qq_m.max(axis=1)
    nh_count = int((mx > nearhard).sum())
    nh_frac = float(nh_count / max(1, Nm))

    # usage/avgprob for ALL K
    top = qq_m.argmax(axis=1)
    counts = np.bincount(top, minlength=K)
    usage = counts / counts.sum()
    avgprob = qq_m.mean(axis=0)

    rec = {
        "slide_id": slide_id,
        "N_patches": int(Nm),
        "K_protos": int(K),
        "row_sum_ok": bool(row_sum_ok),
        "row_sum_min": row_sum_min,
        "row_sum_max": row_sum_max,
        "entropy_mean": ent_mean,
        "entropy_std": ent_std,
        "nearhard_thresh": float(nearhard),
        "nearhard_count": nh_count,
        "nearhard_frac": nh_frac,
    }
    for k in range(K):
        rec[f"usage_p{k}"]   = float(usage[k])
        rec[f"avgprob_p{k}"] = float(avgprob[k])

    # top proto details
    p_star = int(usage.argmax())
    idx = np.argsort(-qq_m[:, p_star])[:5]
    rec["top_proto"] = p_star
    rec["top5_probs"]  = ";".join(f"{qq_m[i, p_star]:.4f}" for i in idx)
    rec["top5_coords"] = ";".join(f"{coords_m[i,0]},{coords_m[i,1]}" for i in idx)
    return rec

def main():
    ap = argparse.ArgumentParser(description="Inspect per-slide .npz files and emit all-K usage/avgprob.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--npz")
    g.add_argument("--dir")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--nearhard", type=float, default=0.99, help="threshold for near-hard count")
    ap.add_argument("--renorm", action="store_true", help="renormalize rows before stats")
    args = ap.parse_args()

    records = []
    if args.npz:
        records.append(stats_for_slide(args.npz, args.renorm, args.nearhard))
    else:
        for root, _, files in os.walk(args.dir):
            for f in files:
                if f.lower().endswith(".npz"):
                    p = os.path.join(root, f)
                    try:
                        records.append(stats_for_slide(p, args.renorm, args.nearhard))
                    except Exception as e:
                        print(f"[ERROR] {p}: {e}")

    df = pd.DataFrame.from_records(records)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df)} rows -> {args.out_csv}")

if __name__ == "__main__":
    main()
