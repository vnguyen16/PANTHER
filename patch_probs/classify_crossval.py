#!/usr/bin/env python3
# cv_majority_vote_runner.py
# Cross-val on patch-prob npz with slide-level majority/soft voting.
# Works with NPZ layout: <NPZ_ROOT>/<MAG>/k=<k>/{train,val,test}

import argparse, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             classification_report, balanced_accuracy_score)

# ---------------- Utils ----------------

def canon(s: str) -> str:
    return re.sub(r"[\s_-]+", "", str(s).strip().upper())

def load_npz(p: Path):
    d = np.load(p, allow_pickle=False)
    return d["qq"], d["coords"], d["mask"]

def slide_vector_from_qq(qq: np.ndarray, mask: Optional[np.ndarray] = None, renorm: bool = True) -> np.ndarray:
    if mask is None:
        mask = np.ones((qq.shape[0],), dtype=bool)
    qq = qq[mask]
    if renorm:
        qq = qq / (qq.sum(axis=1, keepdims=True) + 1e-12)
    return qq.mean(axis=0).astype(np.float32)

def index_npz_split_dir(npz_split_dir: Path) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    for p in npz_split_dir.glob("*.npz"):
        idx.setdefault(canon(p.stem), p)
    return idx

def build_split_from_csv(npz_split_dir: Path, csv_path: Path,
                         label_map: Dict[str,int], renorm_rows: bool = True
                         ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if not csv_path.exists():
        return np.empty((0,0), np.float32), np.array([], int), []
    df = pd.read_csv(csv_path, dtype=str)
    if not {"slide_id","label"}.issubset(df.columns):
        raise ValueError(f"CSV must have columns slide_id,label: {csv_path}")
    df["slide_id"] = df["slide_id"].astype(str).str.strip()
    df["label"]    = df["label"].astype(str).str.strip().map(label_map)

    idx = index_npz_split_dir(npz_split_dir)
    X, y, ids = [], [], []
    missing_npz, missing_lab = [], []
    for sid, lab in df[["slide_id","label"]].itertuples(index=False):
        if pd.isna(lab): missing_lab.append(sid); continue
        p = idx.get(canon(sid))
        if p is None: missing_npz.append(sid); continue
        qq, _, mask = load_npz(p)
        X.append(slide_vector_from_qq(qq, mask, renorm_rows))
        y.append(int(lab)); ids.append(sid)
    if missing_npz: print(f"[WARN] {len(missing_npz)} CSV slides missing NPZ in '{npz_split_dir}': {missing_npz[:5]}{' ...' if len(missing_npz)>5 else ''}")
    if missing_lab: print(f"[WARN] {len(missing_lab)} slides missing valid label in '{csv_path.name}': {missing_lab[:5]}{' ...' if len(missing_lab)>5 else ''}")
    if not X: return np.empty((0,0), np.float32), np.array([], int), []
    return np.vstack(X).astype(np.float32), np.asarray(y, int), ids

def rows_from_csv(npz_split_dir: Path, csv_path: Path, label_map: Dict[str,int]) -> List[Tuple[str,int,Path]]:
    idx = {p.stem: p for p in npz_split_dir.glob("*.npz")}
    df = pd.read_csv(csv_path, dtype=str)
    if not {"slide_id","label"}.issubset(df.columns):
        raise ValueError(f"CSV must have columns slide_id,label: {csv_path}")
    df["slide_id"] = df["slide_id"].str.strip()
    df["label"]    = df["label"].str.strip().map(label_map)
    rows, missing = [], []
    for sid, lab in df[["slide_id","label"]].itertuples(index=False):
        p = idx.get(sid)
        if p is None: missing.append(sid); continue
        rows.append((sid, int(lab), p))
    if missing: print(f"[WARN] {len(missing)} missing NPZ in {npz_split_dir.name}, e.g. {missing[:5]}")
    return rows

def build_patches(rows: List[Tuple[str,int,Path]], renorm_rows: bool = True
                  ) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
    X, y, slide_ids, patch_idx = [], [], [], []
    for sid, lab, p in rows:
        qq, _, mask = load_npz(p)
        q = qq[mask]
        if renorm_rows:
            q = q / (q.sum(axis=1, keepdims=True) + 1e-12)
        X.append(q.astype(np.float32))
        y.append(np.full(q.shape[0], lab, dtype=int))
        slide_ids += [sid]*q.shape[0]
        patch_idx += list(range(q.shape[0]))
    if not X: return np.empty((0,0), np.float32), np.array([], int), [], []
    return np.vstack(X), np.concatenate(y), slide_ids, patch_idx

# --------------- Metrics ---------------

def patch_metrics(model, X, y, tau=0.5, name="PATCH"):
    p = model.predict_proba(X)[:, 1]
    yhat = (p >= tau).astype(int)
    acc = accuracy_score(y, yhat)
    try: auc = roc_auc_score(y, p)
    except ValueError: auc = float("nan")
    cm  = confusion_matrix(y, yhat)
    rep = classification_report(y, yhat, target_names=["FA","PT"], digits=3)
    bal = balanced_accuracy_score(y, yhat) if len(np.unique(y))==2 else float("nan")
    print(f"\n{name} — acc: {acc:.3f} | auc: {auc:.3f} | bAcc: {bal:.3f}")
    print(cm, "\n", rep)
    return p, yhat, {"acc":acc, "auc":auc, "bacc":bal, "cm":cm, "report":rep}

def agg_metrics(model, X, y_slide_by_patch, slide_ids_by_patch, tau=0.5, mode="majority"):
    p = model.predict_proba(X)[:, 1]
    df = pd.DataFrame({"slide": slide_ids_by_patch, "prob": p, "y": y_slide_by_patch})
    g = df.groupby("slide", sort=False)
    soft     = g["prob"].mean().to_numpy()
    # voteFrac = g.apply(lambda q: (q["prob"] >= tau).mean()).to_numpy()
    voteFrac = (
    df.assign(_v=(df["prob"] >= tau).astype(float))
      .groupby("slide", sort=False)["_v"]
      .mean()
      .to_numpy()
)
    slide_y  = g["y"].first().to_numpy()
    if mode == "majority":
        slide_pred = (voteFrac >= 0.5).astype(int); auc_scores = voteFrac
    elif mode == "soft":
        slide_pred = (soft >= tau).astype(int);     auc_scores = soft
    else:
        raise ValueError("mode must be 'majority' or 'soft'")
    acc = accuracy_score(slide_y, slide_pred)
    try: auc = roc_auc_score(slide_y, auc_scores)
    except ValueError: auc = float("nan")
    cm  = confusion_matrix(slide_y, slide_pred)
    # rep = classification_report(slide_y, slide_pred, target_names=["FA","PT"], digits=3)
    rep_dict = classification_report(slide_y, slide_pred, target_names=["FA","PT"], digits=3, output_dict=True)
    rep_str  = classification_report(slide_y, slide_pred, target_names=["FA","PT"], digits=3)
    extras = {"slide_ids": list(g.size().index), "slide_y": slide_y, "soft": soft, "voteFrac": voteFrac}
    return acc, auc, cm, rep_str, extras, rep_dict

def tune_tau_on_val(model, X_va, y_va, sid_va, mode="majority"):
    ts = np.linspace(0, 1, 201)
    best, best_tau = -1.0, 0.5
    for t in ts:
        # acc, auc, cm, rep, _ = agg_metrics(model, X_va, y_va, sid_va, tau=t, mode=mode)
        acc, auc, cm, rep_str, extras, rep_dict = agg_metrics(model, X_va, y_va, sid_va, tau=t, mode=mode)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            tpr = tp/(tp+fn+1e-9); tnr = tn/(tn+fp+1e-9)
            bal = 0.5*(tpr+tnr)
        else:
            bal = float("nan")
        if bal > best:
            best, best_tau = bal, t
    return best_tau, best

# --------- Fold orchestration ---------

def load_fold_data(npz_root_k: Path, split_dir_k: Path,
                   label_map: Dict[str,int], renorm_rows: bool):
    train_dir = npz_root_k / "train"
    val_dir   = npz_root_k / "val"
    test_dir  = npz_root_k / "test"
    csv_train = split_dir_k / "train.csv"
    csv_val   = split_dir_k / "val.csv"
    csv_test  = split_dir_k / "test.csv"
    # Build patch-level sets
    X_tr, y_tr, sid_tr, _ = build_patches(rows_from_csv(train_dir, csv_train, label_map), renorm_rows)
    X_va, y_va, sid_va, _ = build_patches(rows_from_csv(val_dir,   csv_val,   label_map), renorm_rows)
    X_te, y_te, sid_te, _ = build_patches(rows_from_csv(test_dir,  csv_test,  label_map), renorm_rows)
    return (X_tr, y_tr, sid_tr, X_va, y_va, sid_va, X_te, y_te, sid_te)

def clf_factory(max_iter=2000, solver="liblinear", class_weight="balanced", random_state=0):
    return LogisticRegression(max_iter=max_iter, solver=solver,
                              class_weight=class_weight, random_state=random_state)

def run_one_fold(npz_root_k: Path, split_dir_k: Path, label_map: Dict[str,int],
                 renorm_rows: bool, vote_mode: str, clf_kwargs: Dict):
    X_tr, y_tr, sid_tr, X_va, y_va, sid_va, X_te, y_te, sid_te = load_fold_data(
        npz_root_k, split_dir_k, label_map, renorm_rows
    )
    print(f"[Fold] TRAIN patches: {X_tr.shape}, VAL: {X_va.shape}, TEST: {X_te.shape}")
    counts_tr = Counter(sid_tr)
    w_tr = np.array([1.0 / counts_tr[s] for s in sid_tr], dtype=np.float32) if len(sid_tr) else None
    clf = clf_factory(**clf_kwargs)
    clf.fit(X_tr, y_tr, sample_weight=w_tr)
    tau, best_bal = tune_tau_on_val(clf, X_va, y_va, sid_va, mode=vote_mode)
    print(f"[Fold] τ={tau:.3f} (VAL bAcc={best_bal:.3f}, mode={vote_mode})")
    print("\nVAL (slide-level):")
    # acc_v, auc_v, cm_v, rep_v, _ = agg_metrics(clf, X_va, y_va, sid_va, tau=tau, mode=vote_mode)
    acc_v, auc_v, cm_v, rep_v_str, extras_v, rep_v_dict = agg_metrics(clf, X_va, y_va, sid_va, tau=tau, mode=vote_mode)
    print(f"acc={acc_v:.3f} | auc={auc_v:.3f}\n{cm_v}\n{rep_v_str}")
    print("\nTEST (slide-level):")
    # acc_t, auc_t, cm_t, rep_t, extras_t = agg_metrics(clf, X_te, y_te, sid_te, tau=tau, mode=vote_mode)
    acc_t, auc_t, cm_t, rep_t_str, extras_t, rep_t_dict = agg_metrics(clf, X_te, y_te, sid_te, tau=tau, mode=vote_mode)

    print(f"acc={acc_t:.3f} | auc={auc_t:.3f}\n{cm_t}\n{rep_t_str}")
    _ = patch_metrics(clf, X_tr, y_tr, tau=0.5, name="TRAIN (patch)")
    _ = patch_metrics(clf, X_va, y_va, tau=0.5, name="VAL (patch)")
    _ = patch_metrics(clf, X_te, y_te, tau=0.5, name="TEST (patch)")
    return {
    "tau": tau,
    "val": {"acc": acc_v, "auc": auc_v, "cm": cm_v,
            "report": rep_v_str, "report_dict": rep_v_dict},
    "test": {"acc": acc_t, "auc": auc_t, "cm": cm_t,
             "report": rep_t_str, "report_dict": rep_t_dict},
    "test_extras": extras_t
}

    # return {"tau": tau, "val": {"acc":acc_v,"auc":auc_v,"cm":cm_v,"report":rep_v_str},
    #         "test": {"acc":acc_t,"auc":auc_t,"cm":cm_t,"report":rep_t_str},
    #         "test_extras": extras_t}

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="CV majority/soft voting on patch-prob npz.")
    ap.add_argument("--npz_root_tpl", required=True,
        help=r"Template to per-fold NPZ root (use {mag} and {k}); must contain train/ val/ test/ subfolders. "
             r"Ex: C:/.../patch_probs/cross-val/{mag}/k={k}")
    ap.add_argument("--mag", required=True, help="Magnification string to plug into {mag} (e.g., 2.5x, 5x, 10x).")
    ap.add_argument("--split_dir_tpl", required=True,
        help=r"Template to per-fold CSV dir (use {k}); Ex: C:/.../src/splits/cross-val/FA_PT_k={k}")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--label_map", default="FA:0,PT:1")
    ap.add_argument("--renorm_rows", action="store_true")
    ap.add_argument("--vote_mode", choices=["majority","soft"], default="majority")
    ap.add_argument("--max_iter", type=int, default=2000)
    ap.add_argument("--solver", default="liblinear")
    ap.add_argument("--random_state", type=int, default=0)
    ap.add_argument("--no_class_weight", action="store_true")
    ap.add_argument("--out_dir", type=str,
        help="Optional directory to save per-fold and summary metrics.")

    args = ap.parse_args()

    # Parse label map
    lm: Dict[str,int] = {}
    for pair in args.label_map.split(","):
        k_, v_ = pair.split(":")
        lm[k_.strip()] = int(v_)

    clf_kwargs = dict(
        max_iter=args.max_iter,
        solver=args.solver,
        class_weight=None if args.no_class_weight else "balanced",
        random_state=args.random_state,
    )

    fold_results = []
    for k in range(args.k):
        npz_root_k  = Path(args.npz_root_tpl.format(mag=args.mag, k=k))
        split_dir_k = Path(args.split_dir_tpl.format(k=k))
        print("\n" + "="*80)
        print(f"[RUN] Fold {k}\n  npz_root:  {npz_root_k}\n  split_dir: {split_dir_k}")
        print("="*80)
        if not (npz_root_k / "train").exists():
            print(f"[WARN] {npz_root_k} missing train/val/test; skipping."); continue
        if not (split_dir_k / "train.csv").exists():
            print(f"[WARN] {split_dir_k} missing CSVs; skipping."); continue
        res = run_one_fold(npz_root_k, split_dir_k, lm, args.renorm_rows, args.vote_mode, clf_kwargs)
        res["fold"] = k
        fold_results.append(res)

    if not fold_results:
        print("No folds ran. Check your templates/paths.")
        return

    accs = [r["test"]["acc"] for r in fold_results]
    aucs = [r["test"]["auc"] for r in fold_results]
    print("\n=== Cross-validated (per-fold) ===")
    print(f"Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"ROC-AUC:  {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

    all_slide_y, all_soft, all_vote = [], [], []
    for r in fold_results:
        e = r["test_extras"]
        all_slide_y.append(e["slide_y"])
        all_soft.append(e["soft"])
        all_vote.append(e["voteFrac"])
    all_slide_y = np.concatenate(all_slide_y)
    all_soft    = np.concatenate(all_soft)
    all_vote    = np.concatenate(all_vote)

    pooled_auc_soft  = roc_auc_score(all_slide_y, all_soft)  if len(np.unique(all_slide_y))==2 else float("nan")
    pooled_auc_major = roc_auc_score(all_slide_y, all_vote)  if len(np.unique(all_slide_y))==2 else float("nan")
    print("\n=== Pooled across folds (slide-level) ===")
    print(f"Pooled ROC-AUC (soft mean prob):    {pooled_auc_soft:.3f}")
    print(f"Pooled ROC-AUC (majority voteFrac): {pooled_auc_major:.3f}")

    # --- Summary & Save ---
    # Aggregate metrics
    accs = [r["test"]["acc"] for r in fold_results]
    aucs = [r["test"]["auc"] for r in fold_results]

    # Pull weighted-avg precision/recall/F1 per fold from classification_report dicts
    test_prec = []
    test_rec  = []
    test_f1   = []
    for r in fold_results:
        wa = r["test"]["report_dict"]["weighted avg"]  # precision/recall/f1-score/support
        test_prec.append(wa["precision"])
        test_rec.append(wa["recall"])
        test_f1.append(wa["f1-score"])

    summary = {
        "per_fold": fold_results,  # full details (includes cm + reports)
        "mean_acc": float(np.mean(accs)),
        "std_acc":  float(np.std(accs)),
        "mean_auc": float(np.mean(aucs)),
        "std_auc":  float(np.std(aucs)),
        "mean_precision": float(np.mean(test_prec)),
        "std_precision":  float(np.std(test_prec)),
        "mean_recall":    float(np.mean(test_rec)),
        "std_recall":     float(np.std(test_rec)),
        "mean_f1":        float(np.mean(test_f1)),
        "std_f1":         float(np.std(test_f1)),
    }

    # Pooled metrics
    all_slide_y = np.concatenate([r["test_extras"]["slide_y"] for r in fold_results])
    all_soft    = np.concatenate([r["test_extras"]["soft"]     for r in fold_results])
    all_vote    = np.concatenate([r["test_extras"]["voteFrac"] for r in fold_results])
    summary["pooled_auc_soft"]  = float(roc_auc_score(all_slide_y, all_soft))  if len(np.unique(all_slide_y))==2 else float("nan")
    summary["pooled_auc_major"] = float(roc_auc_score(all_slide_y, all_vote))  if len(np.unique(all_slide_y))==2 else float("nan")

    print("\n=== Cross-validated summary ===")
    print(f"Accuracy:  {summary['mean_acc']:.3f} ± {summary['std_acc']:.3f}")
    print(f"ROC-AUC:   {summary['mean_auc']:.3f} ± {summary['std_auc']:.3f}")
    print(f"Precision: {summary['mean_precision']:.3f} ± {summary['std_precision']:.3f}")
    print(f"Recall:    {summary['mean_recall']:.3f} ± {summary['std_recall']:.3f}")
    print(f"F1:        {summary['mean_f1']:.3f} ± {summary['std_f1']:.3f}")
    print(f"Pooled AUC (soft):  {summary['pooled_auc_soft']:.3f}")
    print(f"Pooled AUC (major): {summary['pooled_auc_major']:.3f}")

    import json
    # import numpy as np

    def json_safe(obj):
        """Recursively convert numpy/pandas-ish objects to JSON-safe Python types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.generic,)):  # e.g., np.float32, np.int64
            return obj.item()
        if isinstance(obj, dict):
            return {k: json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [json_safe(v) for v in obj]
        return obj

    # --- SAVE ---
    if args.out_dir:
        out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        # Full JSON (sanitized)
        with open(out_dir / f"results_{args.mag}.json", "w") as f:
            json.dump(json_safe(summary), f, indent=2)
        # Per-fold CSV (already plain types)
        rows = []
        for r in fold_results:
            vwa = r["val"]["report_dict"]["weighted avg"]
            twa = r["test"]["report_dict"]["weighted avg"]
            rows.append({
                "fold": r["fold"], "tau": r["tau"],
                "val_acc": r["val"]["acc"], "val_auc": r["val"]["auc"],
                "val_precision": vwa["precision"], "val_recall": vwa["recall"], "val_f1": vwa["f1-score"],
                "test_acc": r["test"]["acc"], "test_auc": r["test"]["auc"],
                "test_precision": twa["precision"], "test_recall": twa["recall"], "test_f1": twa["f1-score"],
            })
        pd.DataFrame(rows).to_csv(out_dir / f"per_fold_{args.mag}.csv", index=False)
        print(f"[SAVED] Metrics written to {out_dir}")

    # # --- SAVE ---
    # if args.out_dir:
    #     out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    #     import json

    #     # Full JSON summary (includes per-fold dicts)
    #     with open(out_dir / f"results_{args.mag}.json", "w") as f:
    #         json.dump(summary, f, indent=2)

    #     # Per-fold CSV with val/test + precision/recall/F1 (weighted avg)
    #     rows = []
    #     for r in fold_results:
    #         vwa = r["val"]["report_dict"]["weighted avg"]
    #         twa = r["test"]["report_dict"]["weighted avg"]
    #         rows.append({
    #             "fold": r["fold"], "tau": r["tau"],
    #             "val_acc": r["val"]["acc"], "val_auc": r["val"]["auc"],
    #             "val_precision": vwa["precision"], "val_recall": vwa["recall"], "val_f1": vwa["f1-score"],
    #             "test_acc": r["test"]["acc"], "test_auc": r["test"]["auc"],
    #             "test_precision": twa["precision"], "test_recall": twa["recall"], "test_f1": twa["f1-score"],
    #         })
    #     pd.DataFrame(rows).to_csv(out_dir / f"per_fold_{args.mag}.csv", index=False)
    #     print(f"[SAVED] Metrics written to {out_dir}")


if __name__ == "__main__":
    main()
