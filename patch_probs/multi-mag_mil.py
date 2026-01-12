# ============================
# QQ-only Multi-mag Attention MIL (robust to misalignment)
# ============================
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Tuple, List

from sklearn.neighbors import KDTree
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

import torch
import torch.nn as nn
import torch.optim as optim

# ---------- CONFIG ----------
MAGS = ["2p5x", "5x", "10x"]

ROOTS = {
    "2p5x": Path(r"C:\Users\Vivian\Documents\PANTHER\PANTHER\patch_probs\qq_2.5x_norm2\unspecified"),
    "5x":   Path(r"C:\Users\Vivian\Documents\PANTHER\PANTHER\patch_probs\qq_5x_norm2\unspecified"),
    "10x":  Path(r"C:\Users\Vivian\Documents\PANTHER\PANTHER\patch_probs\qq_10x_norm2\unspecified"),
}

SPLITS = {
    "train": pd.read_csv(r"C:\Users\Vivian\Documents\PANTHER\PANTHER\src\splits\FA_PT_5x_norm2_k=0\train.csv", dtype=str),
    "val":   pd.read_csv(r"C:\Users\Vivian\Documents\PANTHER\PANTHER\src\splits\FA_PT_5x_norm2_k=0\val.csv",   dtype=str),
    "test":  pd.read_csv(r"C:\Users\Vivian\Documents\PANTHER\PANTHER\src\splits\FA_PT_5x_norm2_k=0\test.csv",  dtype=str),
}

# --- Normalize labels ONCE after reading CSVs ---
# ---- Normalize labels ONCE after reading CSVs; accept FA/PT or 0/1 ----
def normalize_split(df):
    df = df.copy()

    # choose label column
    for cand in ["label", "Class", "class", "y", "target"]:
        if cand in df.columns:
            label_col = cand
            break
    else:
        raise ValueError(f"No label column found in {list(df.columns)}")

    if "slide_id" not in df.columns:
        raise ValueError("'slide_id' missing from CSV")

    # canonicalize
    slide = df["slide_id"].astype(str).str.strip()
    raw   = df[label_col].astype(str).str.strip().str.upper()

    # map FA/PT/0/1 → {0,1}
    map_all = {"FA": 0, "PT": 1, "0": 0, "1": 1}
    mapped = raw.map(map_all)

    # report & drop unmapped
    bad = mapped.isna()
    if bad.any():
        print("[WARN] Unmapped labels (showing up to 10):")
        print(df.loc[bad, ["slide_id", label_col]].head(10).to_string(index=False))
    slide_ok = slide.loc[~bad].values
    label_ok = mapped.loc[~bad].astype(int).values

    # build a fresh, index-aligned frame
    out = pd.DataFrame({"slide_id": slide_ok, "label": label_ok})
    out = out[(out["slide_id"] != "") & out["slide_id"].notna()]
    out = out.drop_duplicates(subset=["slide_id"], keep="first").reset_index(drop=True)

    if out.empty:
        raise ValueError("After normalization, split is empty.")
    # sanity
    assert not out["label"].isna().any(), "Labels still NaN after normalization."
    return out

SPLITS["train"] = normalize_split(SPLITS["train"])
SPLITS["val"]   = normalize_split(SPLITS["val"])
SPLITS["test"]  = normalize_split(SPLITS["test"])

print("TRAIN head:\n", SPLITS["train"].head())
print("VAL   head:\n", SPLITS["val"].head())
print("TEST  head:\n", SPLITS["test"].head())
print("NaN counts:", {k:int(SPLITS[k]['label'].isna().sum()) for k in SPLITS})


LABEL_MAP = {"FA": 0, "PT": 1}

# If coordinate systems differ by a constant scale vs 2.5×, set it here.
# Example below assumes 5× has 2x finer grid than 2.5×, 10× 4x finer; adjust if needed.
coord_scale = {"2p5x": 1.0, "5x": 0.5, "10x": 0.25}

# Robust nearest-neighbor pooling for misalignment
MATCH_RADIUS = 48.0   # in base (2.5×) coord units; increase if grids are offset/jittery
KNN = 3               # average up to k neighbors within radius (1 = pure NN)
GAUSS_SIGMA = 24.0    # distance weighting (pixels in base units). If None, use equal weights

# Add a per-mag present/missing mask to the instance features?
USE_MAG_MASK = True

# Optimization
EPOCHS = 30
LR = 2e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
HIDDEN = 128
ATTN_DIM = 64
BATCH_BAGS = 1  # bags are variable length; iterate one slide at a time

# ---------- Helpers ----------
def canon(s: str) -> str:
    import re
    return re.sub(r"[\s_-]+", "", str(s).strip().upper())

def index_npz_dir(dirpath: Path):
    idx = {}
    for p in dirpath.glob("*.npz"):
        idx[canon(p.stem)] = p
    return idx

def load_npz_for_slide(dirpath: Path, slide_id: str):
    want = canon(slide_id)
    for p in dirpath.glob("*.npz"):
        if canon(p.stem) == want:
            d = np.load(p, allow_pickle=False)
            qq = d["qq"]; coords = d["coords"]
            mask = d["mask"] if "mask" in d.files else np.ones((qq.shape[0],), dtype=bool)
            qq = qq[mask].astype(np.float32)
            coords = coords[mask].astype(np.float32)
            return {"qq": qq, "coords": coords}
    return None

def build_mag_store(slide_id: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load from train/val/test folders (whichever contains the slide).
    """
    stores = {}
    for m in MAGS:
        for sub in ["train", "val", "test"]:
            d = load_npz_for_slide(ROOTS[m] / sub, slide_id)
            if d is not None:
                stores[m] = d
                break
        if m not in stores:
            stores[m] = None
    return stores

def knn_pull(tree: KDTree, coords_ref: np.ndarray, query_xy: np.ndarray,
             radius: float, k: int, sigma: float = None):
    """
    Return weighted average qq vector over neighbors within radius.
    """
    # radius query
    idxs = tree.query_radius(query_xy[None, :], r=radius)[0]
    if len(idxs) == 0:
        return None
    # if many, keep the closest k
    if len(idxs) > k:
        dists, nn = tree.query(query_xy[None, :], k=k)
        idxs = nn[0]
        dists = dists[0]
    else:
        # compute distances for weighting
        d = coords_ref[idxs] - query_xy[None, :]
        dists = np.sqrt((d**2).sum(axis=1))
    # weights
    if sigma is not None and sigma > 0:
        w = np.exp(- (dists**2) / (2*sigma*sigma))
    else:
        w = np.ones_like(dists)
    w = w / (w.sum() + 1e-9)
    return idxs, w

def build_instances_for_slide_qq_only(slide_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X_instances: [n_base_patches, K*|MAGS| (+|MAGS| if USE_MAG_MASK)]
      Mag_mask:    [n_base_patches, |MAGS|] with 1=present, 0=missing (post-pull)
    """
    stores = build_mag_store(slide_id)
    base = stores["2p5x"]
    if base is None:
        raise FileNotFoundError(f"Missing 2.5× NPZ for slide {slide_id}")
    K = base["qq"].shape[1]

    # Prepare KD-trees for each mag (coords scaled to base system)
    mag_info = {}
    for m in MAGS:
        d = stores[m]
        if d is None or d["qq"].shape[0] == 0:
            mag_info[m] = {"coords": None, "qq": None, "tree": None}
            continue
        s = coord_scale[m]
        coords_scaled = d["coords"] * s
        tree = KDTree(coords_scaled) if coords_scaled.shape[0] else None
        mag_info[m] = {"coords": coords_scaled, "qq": d["qq"], "tree": tree}

    base_coords = mag_info["2p5x"]["coords"]
    base_qq     = mag_info["2p5x"]["qq"]

    X_list = []
    M_mask = []

    for i in range(base_coords.shape[0]):
        feats = []
        present = []
        q_base = base_qq[i]  # always present
        feats.append(q_base); present.append(1)

        q_center = base_coords[i]

        for m in ["5x", "10x"]:
            info = mag_info[m]
            if info["tree"] is None:
                feats.append(np.zeros((K,), dtype=np.float32)); present.append(0); continue
            res = knn_pull(info["tree"], info["coords"], q_center, MATCH_RADIUS, KNN, GAUSS_SIGMA)
            if res is None:
                feats.append(np.zeros((K,), dtype=np.float32)); present.append(0)
            else:
                idxs, w = res
                qavg = (info["qq"][idxs] * w[:, None]).sum(axis=0)
                # (optional) renormalize row to sum to 1
                rs = qavg.sum() + 1e-12
                qavg = qavg / rs
                feats.append(qavg.astype(np.float32)); present.append(1)

        x_inst = np.concatenate(feats, axis=0)  # [K*|MAGS|]
        if USE_MAG_MASK:
            x_inst = np.concatenate([x_inst, np.array(present, dtype=np.float32)], axis=0)
        X_list.append(x_inst)
        M_mask.append(present)

    X_instances = np.stack(X_list, axis=0).astype(np.float32)
    Mag_mask = np.array(M_mask, dtype=np.float32)
    return X_instances, Mag_mask

# --- helper: canonical filename existence check (ignores spaces/underscores/case) ---
def npz_exists_canon(dirpath: Path, slide_id: str) -> bool:
    want = canon(slide_id)
    for p in dirpath.glob("*.npz"):
        if canon(p.stem) == want:
            return True
    return False

def _coerce_label(lab):
    # Accept ints, numpy ints
    if isinstance(lab, (int, np.integer)):
        return int(lab)
    # Accept pandas nullable integer
    try:
        import pandas as pd
        if pd.isna(lab):
            return None
    except Exception:
        pass
    # Accept strings "0"/"1"/"FA"/"PT" (any case/whitespace)
    s = str(lab).strip().upper()
    if s in ("0", "1"):
        return int(s)
    if s in LABEL_MAP:
        return LABEL_MAP[s]
    return None  # unknown

class SlideBagDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame):
        df = df.copy()

        rows, dropped = [], []
        for sid, lab in df[["slide_id","label"]].itertuples(index=False):
            y = _coerce_label(lab)
            if y is None:
                dropped.append((sid, f"bad label: {repr(lab)}"))
                continue

            # require 2.5× npz somewhere (canonical check)
            found = any(npz_exists_canon(ROOTS["2p5x"]/sub, sid) for sub in ["train","val","test"])
            if not found:
                dropped.append((sid, "missing 2p5x npz"))
                continue

            rows.append((sid, y))

        if dropped:
            print(f"[WARN] Skipped {len(dropped)} slides. Examples:")
            for s, why in dropped[:8]:
                print(f"  - {s}: {why}")

        if not rows:
            # Add a quick debug hint to see what labels look like:
            print("[DEBUG] First 10 labels from df:", df["label"].head(10).tolist())
            print("[DEBUG] First 10 label types:", [type(v) for v in df["label"].head(10)])
            raise ValueError("No valid slides found after filtering. Check paths/filenames/labels.")

        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        sid, y = self.rows[i]
        X_inst, M_mask = build_instances_for_slide_qq_only(sid)
        return sid, torch.from_numpy(X_inst), torch.from_numpy(M_mask), torch.tensor(float(y))
    
def collate_bags(batch):
    sids, Xs, Ms, ys = zip(*batch)
    return list(sids), list(Xs), list(Ms), torch.stack(ys, dim=0)

# ---------- Attention MIL ----------
class AttentionMIL(nn.Module):
    def __init__(self, in_dim: int, h: int = 128, attn_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.inst = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, h),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h, h),
            nn.ReLU(inplace=True)
        )
        self.attn_V = nn.Linear(h, attn_dim)
        self.attn_U = nn.Linear(h, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)
        # logistic regression
        self.cls = nn.Sequential(
            nn.Linear(h, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        ) 
        # 3 layer mlp instead of logistic regression
        # self.cls = nn.Sequential(
        #     nn.Linear(h, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, 64),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(64, 1)
        # )

    def forward(self, X):
        # X: [N_inst, in_dim]
        H = self.inst(X)                       # [N, h]
        A_V = torch.tanh(self.attn_V(H))       # [N, a]
        A_U = torch.sigmoid(self.attn_U(H))    # [N, a]
        A = self.attn_w(A_V * A_U).squeeze(1)  # [N]
        A = torch.softmax(A, dim=0)
        M = torch.sum(A.unsqueeze(1) * H, dim=0, keepdim=True)  # [1, h]
        logit = self.cls(M).squeeze(1)         # [1]
        return logit, A

# ---------- Train / Eval ----------
def tune_tau(y, s):
    ts = np.linspace(0,1,201)
    best, best_t = -1, 0.5
    for t in ts:
        yhat = (s >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
        tpr = tp/(tp+fn+1e-9); tnr = tn/(tn+fp+1e-9)
        bal = 0.5*(tpr+tnr)
        if bal > best:
            best, best_t = bal, t
    return best_t

def train_epoch(model, loader, device, opt, bce):
    model.train()
    total, n = 0.0, 0
    for sids, Xs, Ms, ys in loader:
        opt.zero_grad()
        loss = 0.0
        for X, y in zip(Xs, ys):
            X = X.to(device).float()
            logit, _ = model(X)
            loss = loss + bce(logit, y.to(device).float().unsqueeze(0))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        total += loss.item(); n += len(ys)
    return total / max(1,n)

@torch.no_grad()
def infer_split(model, loader, device):
    model.eval()
    probs, labels, sids_all = [], [], []
    for sids, Xs, Ms, ys in loader:
        for sid, X, y in zip(sids, Xs, ys):
            X = X.to(device).float()
            prob = torch.sigmoid(model(X)[0]).item()
            probs.append(prob); labels.append(int(y.item())); sids_all.append(sid)
    return np.array(probs), np.array(labels), sids_all

# ---------- Build data ----------
for split in ["train","val","test"]:
    print(split, "n=", len(SPLITS[split]), 
          "unique slides=", SPLITS[split]["slide_id"].nunique(),
          "nan labels=", int(SPLITS[split]["label"].isna().sum()))


ds_tr = SlideBagDataset(SPLITS["train"])
ds_va = SlideBagDataset(SPLITS["val"])
ds_te = SlideBagDataset(SPLITS["test"])

if len(ds_tr) == 0 or len(ds_va) == 0 or len(ds_te) == 0:
    raise ValueError(f"Empty dataset(s): train={len(ds_tr)}, val={len(ds_va)}, test={len(ds_te)}")

print(f"Dataset sizes — train: {len(ds_tr)}, val: {len(ds_va)}, test: {len(ds_te)}")

dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=BATCH_BAGS, shuffle=True,  collate_fn=collate_bags)
dl_va = torch.utils.data.DataLoader(ds_va, batch_size=BATCH_BAGS, shuffle=False, collate_fn=collate_bags)
dl_te = torch.utils.data.DataLoader(ds_te, batch_size=BATCH_BAGS, shuffle=False, collate_fn=collate_bags)

# Infer input dim from one sample
_, X0, _, _ = next(iter(dl_tr))
in_dim = X0[0].shape[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AttentionMIL(in_dim=in_dim, h=HIDDEN, attn_dim=ATTN_DIM, dropout=DROPOUT).to(device)
bce = nn.BCEWithLogitsLoss()
opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ---------- Train ----------
for e in range(1, EPOCHS+1):
    tr_loss = train_epoch(model, dl_tr, device, opt, bce)
    with torch.no_grad():
        p_va, y_va, _ = infer_split(model, dl_va, device)
        try: auc_va = roc_auc_score(y_va, p_va)
        except ValueError: auc_va = float("nan")
    print(f"Epoch {e:02d} | train loss {tr_loss:.4f} | VAL AUC {auc_va:.3f}")

# ---------- Final eval ----------
p_va, y_va, _ = infer_split(model, dl_va, device)
tau = tune_tau(y_va, p_va)
print(f"Chosen τ on VAL = {tau:.2f}")

p_te, y_te, _ = infer_split(model, dl_te, device)
yhat_va = (p_va >= tau).astype(int)
yhat_te = (p_te >= tau).astype(int)

print("\n[VAL]")
print("acc=", accuracy_score(y_va, yhat_va), "auc=", roc_auc_score(y_va, p_va))
print(classification_report(y_va, yhat_va, target_names=["FA","PT"], digits=3))

print("\n[TEST]")
print("acc=", accuracy_score(y_te, yhat_te), "auc=", roc_auc_score(y_te, p_te))
print(classification_report(y_te, yhat_te, target_names=["FA","PT"], digits=3))
