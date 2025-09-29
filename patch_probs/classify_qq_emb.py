# Gated Attention MIL with engineered patch features + training curves (loss/acc/AUC)

import os, json, random, argparse
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             roc_auc_score, confusion_matrix, classification_report)
from collections import defaultdict
from typing import Dict, Tuple, List, Any, Optional
from datetime import datetime
import platform
import pickle, json as _json


# ======================= Config / Args =======================

def get_args():
    p = argparse.ArgumentParser(description="Gated Attention MIL training with Q / PCA(Z)+Q features")
    # Required paths
    p.add_argument("--merged-root", type=str, required=True,
                   help="Root dir with split subfolders train/val/test; each has _manifest.json of NPZs")
    p.add_argument("--label-csv", type=str, required=True,
                   help="CSV with columns [Filename, Class, Magnification] (Filename must match slide_id)")

    # Optional outputs
    p.add_argument("--save-dir", type=str, default=None,
                   help="Directory to save plots and history (created if not exist)")
    p.add_argument("--run-name", type=str, default=None,
               help="Optional custom prefix for the run folder name.")
    p.add_argument("--auto-name", action="store_true", default=True,
                help="If set, auto-append a run tag (head/feature/batch/lr/wd/seed) to the save dir.")

    # Feature options
    p.add_argument("--feature-mode", type=str, choices=["q", "zq", "z"], default="q",
               help="Which patch features to use: q (Q-only), zq (PCA(Z)+Q), or z (PCA(Z) only)")
    p.add_argument("--q-hard", action="store_true", default=False,
               help="If set (and feature-mode uses Q), replace Q with one-hot argmax per patch")
    p.add_argument("--use-only-q", action="store_true", default=True,
                   help="Use Q-only (plus engineered bits). If unset, will use PCA(Z)+Q.")
    p.add_argument("--no-use-only-q", dest="use_only_q", action="store_false")
    p.add_argument("--pca-dim", type=int, default=32, help="PCA dim for Z when use_only_q=False")
    p.add_argument("--use-entropy-gap", action="store_true", default=True, help="Append H(q) and (top1-top2)")
    p.add_argument("--no-use-entropy-gap", dest="use_entropy_gap", action="store_false")
    p.add_argument("--use-clr", action="store_true", default=True, help="Append clr(q) (K dims)")
    p.add_argument("--no-use-clr", dest="use_clr", action="store_false")

    # Training options
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size-slides", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=10, help="Early-stop patience (by val AUC)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pool", type=str, choices=["attn","mean"], default="attn",
               help="Slide aggregator: attention or masked mean")


    # Classifier
    p.add_argument("--head", type=str, choices=["lr", "mlp1", "mlp3", "mlp5"], default="mlp3",
               help="Classifier head: logistic regression (lr) or MLP with 1/3/5 hidden layers")
    p.add_argument("--mlp-width", type=int, default=256,
                help="Base hidden width for MLP heads (e.g., 256 -> mlp3: [256,128,64])")
    p.add_argument("--dropout-fe", type=float, default=0.25,
                help="Dropout used in the per-patch feature encoder (fe)")
    p.add_argument("--dropout-head", type=float, default=0.30,
                help="Dropout used inside the MLP head (ignored for lr)")

    # Class weighting
    p.add_argument("--use-class-weights", action="store_true", default=True)
    p.add_argument("--no-use-class-weights", dest="use_class_weights", action="store_false")

    # Plotting
    p.add_argument("--no-plots", action="store_true", default=False, help="Disable matplotlib plots")

    return p.parse_args()

LABEL_MAP = {"FA": 0, "PT": 1}
EPS = 1e-6

# ======================= Utils =======================
# ---------- Q-derived feature helpers ----------
EPS = 1e-6  # or reuse your global

def entropy_gap(Q: np.ndarray, eps: float = EPS):
    """
    Returns:
      H:  n×1 vector of per-row entropies (-Σ q_i log q_i)
      gap: n×1 vector of (top1 - top2) margins per row
    Works for both soft and one-hot distributions.
    """
    Qs = np.clip(Q, eps, 1.0).astype(np.float32)
    H  = (-np.sum(Qs * np.log(Qs), axis=1, keepdims=True)).astype(np.float32)

    # compute top-1 minus top-2 per row
    top2 = np.partition(Qs, -2, axis=1)[:, -2:]         # two largest (unsorted)
    top1 = np.max(top2, axis=1, keepdims=True)
    top2nd = np.min(top2, axis=1, keepdims=True)
    gap = (top1 - top2nd).astype(np.float32)            # shape (n,1)

    return H, gap

def clr(Q: np.ndarray, eps: float = EPS):
    """
    Centered log-ratio transform per row: log(q) - mean(log(q)).
    """
    Qs = np.clip(Q, eps, 1.0).astype(np.float32)
    logQ = np.log(Qs)
    return (logQ - logQ.mean(axis=1, keepdims=True)).astype(np.float32)

def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def safe_mkdir(path: Optional[str]):
    if path and len(path):
        os.makedirs(path, exist_ok=True)

def load_split_ZQ_SID(split_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    man = json.load(open(os.path.join(split_dir, "_manifest.json"), "r"))
    Zs, Qs, SIDs = [], [], []
    for rec in man:
        d = dict(np.load(rec["path"], allow_pickle=True))
        Zs.append(np.asarray(d["Z"], dtype=np.float32))
        Qs.append(np.asarray(d["Q"], dtype=np.float32))
        SIDs.append(np.asarray(d["slide_id"]).astype(str))
    Z = np.concatenate(Zs, axis=0)
    Q = np.concatenate(Qs, axis=0)
    SID = np.concatenate(SIDs, axis=0)
    return Z, Q, SID

def labels_from_csv(csv_path: str, label_map: Dict[str, int]) -> Dict[str, int]:
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    fn = cols.get("filename", "Filename")
    cl = cols.get("class", "Class")
    return {str(r[fn]): label_map[str(r[cl])] for _, r in df.iterrows()}

def make_run_tag(args, in_dim: int, slides_tr: int, slides_va: int, slides_te: int) -> str:
    """
    Build a compact tag encoding head and feature mode.
    Examples: '20250915_1453_mlp3_qq_bs4_lr0.001_wd0.0005_seed0'
              '20250915_1453_lr_zq64_bs6_lr0.0005_wd0.0001_seed0'
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    # feat = "qq" if args.use_only_q else f"zq{args.pca_dim}"
    # tag  = f"{ts}_{args.head}-{args.pool}_{feat}_bs{args.batch_size_slides}_lr{args.lr:g}_wd{args.weight_decay:g}_seed{args.seed}"
    feat = {"q":"qq", "zq":f"zq{args.pca_dim}", "z":f"z{args.pca_dim}"}[args.feature_mode]
    tag  = f"{ts}_{args.head}-{args.pool}_{feat}_bs{args.batch_size_slides}_lr{args.lr:g}_wd{args.weight_decay:g}_seed{args.seed}"

    if args.run_name:
        tag = f"{args.run_name}_{tag}"
    return tag

def dump_hparams_json(path: str, payload: dict):
    with open(path, "w") as f:
        _json.dump(payload, f, indent=2, sort_keys=True)

def create_hparams_dict(
    args,
    *,
    save_dir: str,
    run_tag: str,
    in_dim: int,
    cls_counts,  # numpy array or list, length = num_classes
    n_patches_train: int,
    n_patches_val: int,
    n_patches_test: int,
    n_slides_train: int,
    n_slides_val: int,
    n_slides_test: int,
    device,
    label_map: dict,
    attn_dim: int = 128,  # keep in sync with your model instantiation
):
    return {
        "run_tag": run_tag,
        "paths": {
            "merged_root": args.merged_root,
            "label_csv": args.label_csv,
            "save_dir": save_dir,
        },
        "label_map": label_map,
        "data_summary": {
            "n_patches_train": int(n_patches_train),
            "n_patches_val":   int(n_patches_val),
            "n_patches_test":  int(n_patches_test),
            "n_slides_train":  int(n_slides_train),
            "n_slides_val":    int(n_slides_val),
            "n_slides_test":   int(n_slides_test),
            "class_counts_train": list(map(int, cls_counts)) if cls_counts is not None else None,
        },
        "feature_opts": {
            "mode": args.feature_mode,          
            "q_hard": args.q_hard,     
            "use_only_q": args.use_only_q,
            "pca_dim": args.pca_dim,
            "use_entropy_gap": args.use_entropy_gap,
            "use_clr": args.use_clr,
            "feature_dim_after_build": int(in_dim),
        },
        "model_opts": {
            "head": args.head,
            "pool": args.pool,
            "mlp_width": args.mlp_width,
            "dropout_fe": args.dropout_fe,
            "dropout_head": args.dropout_head,
            "attn_dim": attn_dim,
        },
        "train_opts": {
            "epochs": args.epochs,
            "batch_size_slides": args.batch_size_slides,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "seed": args.seed,
            "use_class_weights": args.use_class_weights,
        },
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }

# feature engineering, adding functionality to use Z only
def _maybe_q_process(Q, use_entropy_gap, use_clr, q_hard=False):
    # optionally convert Q to hard one-hot
    if q_hard:
        k = Q.shape[1]
        onehot = np.eye(k, dtype=np.float32)[np.argmax(Q, axis=1)]
        Q_use = onehot
    else:
        Q_use = Q.astype(np.float32)

    blocks = [Q_use]
    if use_entropy_gap:
        H, g = entropy_gap(Q_use)  # uses Q_use; H/g still meaningful for one-hot (H≈0, gap≈1-0)
        blocks += [H, g]
    if use_clr:
        blocks += [clr(Q_use)]
    return blocks

def build_features_fit(Z_tr, Q_tr, *, feature_mode, pca_dim, use_entropy_gap, use_clr_bits, seed, q_hard=False):
    blocks_tr = []
    state = {
        "feature_mode": feature_mode, "pca": None, "scaler": None,
        "use_entropy_gap": use_entropy_gap, "use_clr": use_clr_bits, "q_hard": q_hard
    }

    if feature_mode in ("zq", "z"):
        pca = PCA(n_components=pca_dim, random_state=seed)
        Zp_tr = pca.fit_transform(Z_tr).astype(np.float32)
        state["pca"] = pca

    if feature_mode == "q":
        blocks_tr = _maybe_q_process(Q_tr, use_entropy_gap, use_clr_bits, q_hard)
    elif feature_mode == "zq":
        blocks_tr = [Zp_tr]
        blocks_tr += _maybe_q_process(Q_tr, use_entropy_gap, use_clr_bits, q_hard)
    elif feature_mode == "z":
        blocks_tr = [Zp_tr]
        # no Q-derived extras in pure-Z mode
        state["use_entropy_gap"] = False
        state["use_clr"] = False
    else:
        raise ValueError("feature_mode must be one of {'q','zq','z'}")

    X_tr = np.concatenate(blocks_tr, axis=1)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    state["scaler"] = scaler
    return X_tr, state

def build_features_apply(Z, Q, state):
    fm = state["feature_mode"]
    blocks = []
    if fm in ("zq", "z"):
        Zp = state["pca"].transform(Z).astype(np.float32)
    if fm == "q":
        blocks = _maybe_q_process(Q, state["use_entropy_gap"], state["use_clr"], state["q_hard"])
    elif fm == "zq":
        blocks = [Zp]
        blocks += _maybe_q_process(Q, state["use_entropy_gap"], state["use_clr"], state["q_hard"])
    elif fm == "z":
        blocks = [Zp]
    X = np.concatenate(blocks, axis=1)
    X = state["scaler"].transform(X).astype(np.float32)
    return X

# ===== Feature engineering (Q-only or PCA(Z)+Q (+entropy/gap/CLR)) =====
# def entropy_gap(Q: np.ndarray, eps: float = EPS) -> Tuple[np.ndarray, np.ndarray]:
#     Qs = np.clip(Q, eps, 1.0)
#     H  = (-np.sum(Qs * np.log(Qs), axis=1, keepdims=True)).astype(np.float32)
#     top2 = np.partition(Q, -2, axis=1)[:, -2:]
#     gap = (top2[:,1] - top2[:,0]).reshape(-1,1).astype(np.float32)  # top1 - top2
#     return H, gap

# def clr(Q: np.ndarray, eps: float = EPS) -> np.ndarray:
#     Qs = np.clip(Q, eps, 1.0)
#     logQ = np.log(Qs)
#     return (logQ - logQ.mean(axis=1, keepdims=True)).astype(np.float32)

# def build_features_fit(Z_tr: np.ndarray, Q_tr: np.ndarray, use_only_q: bool, pca_dim: int,
#                        use_entropy_gap: bool, use_clr_bits: bool, seed: int) -> Tuple[np.ndarray, Dict[str, Any]]:
#     blocks_tr = []
#     state = {"use_only_q": use_only_q, "use_entropy_gap": use_entropy_gap,
#              "use_clr": use_clr_bits, "scaler": None, "pca": None}

#     if use_only_q:
#         blocks_tr = [Q_tr.astype(np.float32)]
#     else:
#         pca = PCA(n_components=pca_dim, random_state=seed)
#         Zp_tr = pca.fit_transform(Z_tr).astype(np.float32)
#         state["pca"] = pca
#         blocks_tr = [Zp_tr, Q_tr.astype(np.float32)]

#     if use_entropy_gap:
#         H, g = entropy_gap(Q_tr); blocks_tr += [H, g]
#     if use_clr_bits:
#         blocks_tr += [clr(Q_tr)]

#     X_tr = np.concatenate(blocks_tr, axis=1)
#     scaler = StandardScaler(with_mean=True, with_std=True)
#     X_tr = scaler.fit_transform(X_tr).astype(np.float32)
#     state["scaler"] = scaler
#     return X_tr, state

# def build_features_apply(Z: np.ndarray, Q: np.ndarray, state: Dict[str, Any]) -> np.ndarray:
#     blocks = []
#     if state["use_only_q"]:
#         blocks = [Q.astype(np.float32)]
#     else:
#         Zp = state["pca"].transform(Z).astype(np.float32)
#         blocks = [Zp, Q.astype(np.float32)]
#     if state["use_entropy_gap"]:
#         H, g = entropy_gap(Q); blocks += [H, g]
#     if state["use_clr"]:
#         blocks += [clr(Q)]
#     X = np.concatenate(blocks, axis=1)
#     X = state["scaler"].transform(X).astype(np.float32)
#     return X

# =================== Bag dataset (one item = one slide) ===================

class SlideBagDataset(Dataset):
    def __init__(self, X: np.ndarray, y_patch: np.ndarray, sids: np.ndarray):
        self.slide_to_idx = defaultdict(list)
        for i, s in enumerate(sids):
            self.slide_to_idx[s].append(i)
        self.slides = list(self.slide_to_idx.keys())
        self.slide_labels = []
        for s in self.slides:
            idxs = self.slide_to_idx[s]
            self.slide_labels.append(int(y_patch[idxs[0]]))
        self.slide_labels = np.array(self.slide_labels, dtype=np.int64)
        self.X = X
        self.sids = sids

    def __len__(self): return len(self.slides)

    def __getitem__(self, i):
        s = self.slides[i]
        idxs = self.slide_to_idx[s]
        Xi = self.X[idxs]                 # (ni, d)
        yi = int(self.slide_labels[i])    # scalar
        return torch.from_numpy(Xi).float(), torch.tensor(yi).long(), s

def collate_bags(batch):
    lengths = [x[0].shape[0] for x in batch]
    d = batch[0][0].shape[1]
    B = len(batch); T = max(lengths)
    Xp = torch.zeros(B, T, d, dtype=torch.float32)
    mask = torch.zeros(B, T, dtype=torch.bool)
    ys = torch.zeros(B, dtype=torch.long)
    sids = []
    for b, (Xi, yi, sid) in enumerate(batch):
        n = Xi.shape[0]
        Xp[b, :n] = Xi
        mask[b, :n] = 1
        ys[b] = yi
        sids.append(sid)
    return Xp, mask, ys, sids

# =================== Classifier heads ===================
def make_classifier_head(in_dim: int, num_classes: int, head: str,
                         base_width: int = 256, pdrop: float = 0.30) -> nn.Module:
    head = head.lower()
    if head == "lr":
        # Logistic regression (no hidden layers, no dropout)
        return nn.Linear(in_dim, num_classes)

    elif head == "mlp1":
        # 1 hidden layer: [in -> W -> ReLU -> Dropout -> C]
        W = max(base_width, 16)
        return nn.Sequential(
            nn.Linear(in_dim, W), nn.ReLU(), nn.Dropout(pdrop),
            nn.Linear(W, num_classes)
        )

    elif head == "mlp3":
        # 3 hidden layers: [W, W/2, W/4]
        W1 = max(base_width, 32)
        W2 = max(base_width // 2, 32)
        W3 = max(base_width // 4, 16)
        return nn.Sequential(
            nn.Linear(in_dim, W1), nn.ReLU(), nn.Dropout(pdrop),
            nn.Linear(W1, W2),    nn.ReLU(), nn.Dropout(pdrop),
            nn.Linear(W2, W3),    nn.ReLU(), nn.Dropout(pdrop),
            nn.Linear(W3, num_classes)
        )

    elif head == "mlp5":
        # 5 hidden layers: [2W, W, W/2, W/4, W/8]
        W1 = max(2 * base_width, 64)
        W2 = max(base_width, 64)
        W3 = max(base_width // 2, 32)
        W4 = max(base_width // 4, 16)
        W5 = max(base_width // 8, 8)
        return nn.Sequential(
            nn.Linear(in_dim, W1), nn.ReLU(), nn.Dropout(pdrop),
            nn.Linear(W1, W2),     nn.ReLU(), nn.Dropout(pdrop),
            nn.Linear(W2, W3),     nn.ReLU(), nn.Dropout(pdrop),
            nn.Linear(W3, W4),     nn.ReLU(), nn.Dropout(pdrop),
            nn.Linear(W4, W5),     nn.ReLU(), nn.Dropout(pdrop),
            nn.Linear(W5, num_classes)
        )

    else:
        raise ValueError(f"Unknown head type: {head}")


# =================== Gated Attention MIL (Ilse et al., 2018) ===================

class GatedAttentionMIL(nn.Module):
    def __init__(self, in_dim, attn_dim=128, num_classes=2,
                 dropout_fe=0.25, head="mlp3", mlp_width=256, dropout_head=0.30):
        super().__init__()
        self.fe = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Dropout(dropout_fe),
        )
        self.V = nn.Linear(in_dim, attn_dim)
        self.U = nn.Linear(in_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1, bias=False)

        # Classifier head (switched by flag)
        self.fc = make_classifier_head(
            in_dim=in_dim, num_classes=num_classes,
            head=head, base_width=mlp_width, pdrop=dropout_head
        )

    def forward(self, X, mask):
        B, T, d = X.shape
        H = self.fe(X)                        # [B,T,d]
        Vh = torch.tanh(self.V(H))            # [B,T,h]
        Uh = torch.sigmoid(self.U(H))         # [B,T,h]
        A = self.w(Vh * Uh).squeeze(-1)       # [B,T]
        A[~mask] = -1e9
        A = torch.softmax(A, dim=1)           # [B,T]
        M = torch.sum(A.unsqueeze(-1) * H, dim=1)  # [B,d]
        logits = self.fc(M)                   # [B,C]
        return logits, A

# =================== Mean Pooling MIL (for comparison) ===================
class MeanPoolMIL(nn.Module):
    def __init__(self, in_dim, num_classes=2, dropout_fe=0.25,
                 head="mlp3", mlp_width=256, dropout_head=0.30):
        super().__init__()
        # same per-patch fe as before
        self.fe = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Dropout(dropout_fe),
        )
        self.fc = make_classifier_head(
            in_dim=in_dim, num_classes=num_classes,
            head=head, base_width=mlp_width, pdrop=dropout_head
        )

    def forward(self, X, mask):
        """
        X: [B,T,d], mask: [B,T] (True for valid positions)
        """
        H = self.fe(X)                            # [B,T,d]
        m = mask.float().unsqueeze(-1)            # [B,T,1]
        H_sum = (H * m).sum(dim=1)                # [B,d]
        cnt   = m.sum(dim=1).clamp_min(1e-9)      # [B,1]
        M = H_sum / cnt                           # masked mean [B,d]
        logits = self.fc(M)                       # [B,C]
        return logits, mask.float()               # keep same return shape as attention model

# =================== Metrics & Eval helpers ===================

def eval_slide_metrics(logits_list: List[torch.Tensor], y_list: List[torch.Tensor]) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    probs = torch.softmax(torch.cat(logits_list, dim=0), dim=1)[:,1].cpu().numpy()
    gts   = torch.cat(y_list, dim=0).cpu().numpy()
    pred  = (probs >= 0.5).astype(int)
    try:
        auc = roc_auc_score(gts, probs)
    except ValueError:
        auc = 0.0
    acc = accuracy_score(gts, pred)
    bAcc = balanced_accuracy_score(gts, pred)
    cm = confusion_matrix(gts, pred)
    rep = classification_report(gts, pred, target_names=["FA","PT"], digits=3)
    return {"auc":auc, "acc":acc, "bAcc":bAcc, "cm":cm, "report":rep}, probs, gts

def run_eval(model, loader, device, split_name="SPLIT"):
    model.eval()
    with torch.no_grad():
        logits_all, ys_all = [], []
        sids_all = []
        attn_list = []

        for Xb, maskb, yb, sids in loader:
            Xb, maskb, yb = Xb.to(device), maskb.to(device), yb.to(device)
            logits, A = model(Xb, maskb)
            logits_all.append(logits)
            ys_all.append(yb)

            A_cpu = A.cpu()
            mask_cpu = maskb.cpu()
            for b in range(A_cpu.size(0)):
                attn_vec = A_cpu[b][mask_cpu[b]].numpy()
                attn_list.append(attn_vec)
            sids_all += sids

    metrics, probs, gts = eval_slide_metrics(logits_all, ys_all)
    print(f"\n=== {split_name} SLIDE-LEVEL ===")
    print(f"acc={metrics['acc']:.3f} | bAcc={metrics['bAcc']:.3f} | AUC={metrics['auc']:.3f}")
    print(metrics["cm"], "\n", metrics["report"])

    # Optional: padded attention
    max_len = max(len(a) for a in attn_list) if attn_list else 0
    if max_len > 0:
        attn_padded = np.zeros((len(attn_list), max_len), dtype=np.float32)
        attn_mask   = np.zeros((len(attn_list), max_len), dtype=bool)
        for i, a in enumerate(attn_list):
            L = len(a)
            attn_padded[i, :L] = a
            attn_mask[i, :L] = True
    else:
        attn_padded, attn_mask = None, None

    return metrics, probs, gts, attn_list, sids_all, attn_padded, attn_mask

# =================== Training (this is where the HISTORY BLOCK lives) ===================

def train(model, loaders, device, epochs, lr, weight_decay, patience, class_weights=None):
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_auc = -1.0
    best_state = None
    no_improve = 0

    history = {
        "train_loss": [],
        "train_acc":  [],
        "val_loss":   [],
        "val_acc":    [],
        "val_auc":    []
    }

    def batch_acc(logits, y):
        return (torch.argmax(logits, dim=1) == y).float().mean().item()

    loader_tr, loader_va = loaders["train"], loaders["val"]

    for epoch in range(1, epochs + 1):
        # ---------- Train ----------
        model.train()
        total_loss = 0.0
        total_n = 0
        total_correct = 0
        total_seen = 0

        for Xb, maskb, yb, _ in loader_tr:
            Xb, maskb, yb = Xb.to(device), maskb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, _ = model(Xb, maskb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = yb.size(0)
            total_loss += loss.item() * bs
            total_n += bs
            total_correct += (torch.argmax(logits, dim=1) == yb).sum().item()
            total_seen += bs

        tr_loss = total_loss / max(1, total_n)
        tr_acc = total_correct / max(1, total_seen)

        # ---------- Validation ----------
        model.eval()
        val_total_loss = 0.0
        val_total_n = 0
        logits_all, ys_all = [], []

        with torch.no_grad():
            for Xb, maskb, yb, _ in loader_va:
                Xb, maskb, yb = Xb.to(device), maskb.to(device), yb.to(device)
                logits, _ = model(Xb, maskb)
                loss = criterion(logits, yb)

                bs = yb.size(0)
                val_total_loss += loss.item() * bs
                val_total_n += bs

                logits_all.append(logits)
                ys_all.append(yb)

        val_metrics, _, _ = eval_slide_metrics(logits_all, ys_all)
        val_loss = val_total_loss / max(1, val_total_n)
        val_acc = val_metrics["acc"]
        val_auc = val_metrics["auc"]

        # ----- History block you asked about -----
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        improved = val_auc > best_auc + 1e-4
        if improved:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        print(f"Epoch {epoch:03d} | "
              f"train loss {tr_loss:.4f} | train acc {tr_acc:.3f} || "
              f"VAL loss {val_loss:.4f} | VAL acc {val_acc:.3f} | VAL AUC {val_auc:.3f} "
              f"{'[*]' if improved else ''}")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}. Best VAL AUC = {best_auc:.3f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_auc

# =================== Plotting ===================

def plot_history(history: Dict[str, List[float]], save_path_prefix: Optional[str] = None, show=True):
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"],   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Train/Val Loss")
    plt.legend(); plt.grid(True, alpha=0.3)
    if save_path_prefix:
        plt.savefig(f"{save_path_prefix}_loss.png", bbox_inches="tight", dpi=150)
    if show: plt.show(); plt.close()

    # Accuracy
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"],   label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Train/Val Accuracy")
    plt.legend(); plt.grid(True, alpha=0.3)
    if save_path_prefix:
        plt.savefig(f"{save_path_prefix}_acc.png", bbox_inches="tight", dpi=150)
    if show: plt.show(); plt.close()

    # AUC
    plt.figure()
    plt.plot(epochs, history["val_auc"], label="Val AUC")
    plt.xlabel("Epoch"); plt.ylabel("AUC"); plt.title("Validation AUC")
    plt.legend(); plt.grid(True, alpha=0.3)
    if save_path_prefix:
        plt.savefig(f"{save_path_prefix}_auc.png", bbox_inches="tight", dpi=150)
    if show: plt.show(); plt.close()

# =================== Main ===================

def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TR_DIR = os.path.join(args.merged_root, "train")
    VA_DIR = os.path.join(args.merged_root, "val")
    TE_DIR = os.path.join(args.merged_root, "test")

    print("Loading merged Z/Q...")
    Z_tr, Q_tr, sid_tr = load_split_ZQ_SID(TR_DIR)
    Z_va, Q_va, sid_va = load_split_ZQ_SID(VA_DIR)
    Z_te, Q_te, sid_te = load_split_ZQ_SID(TE_DIR)

    slide2y = labels_from_csv(args.label_csv, LABEL_MAP)
    y_tr = np.array([slide2y[s] for s in sid_tr], dtype=np.int64)
    y_va = np.array([slide2y[s] for s in sid_va], dtype=np.int64)
    y_te = np.array([slide2y[s] for s in sid_te], dtype=np.int64)

    print("Building features...")
    # X_tr, state = build_features_fit(
    #     Z_tr, Q_tr,
    #     use_only_q=args.use_only_q,
    #     pca_dim=args.pca_dim,
    #     use_entropy_gap=args.use_entropy_gap,
    #     use_clr_bits=args.use_clr,
    #     seed=args.seed
    # )
    X_tr, state = build_features_fit(
    Z_tr, Q_tr,
    feature_mode=args.feature_mode,
    pca_dim=args.pca_dim,
    use_entropy_gap=args.use_entropy_gap,
    use_clr_bits=args.use_clr,
    seed=args.seed,
    q_hard=args.q_hard
)
    X_va = build_features_apply(Z_va, Q_va, state)
    X_te = build_features_apply(Z_te, Q_te, state)

    print("Feature dims:", X_tr.shape[1], "| slides(train/val/test):",
          len(set(sid_tr)), len(set(sid_va)), len(set(sid_te)))

    # Build datasets
    ds_tr = SlideBagDataset(X_tr, y_tr, sid_tr)
    ds_va = SlideBagDataset(X_va, y_va, sid_va)
    ds_te = SlideBagDataset(X_te, y_te, sid_te)

    # Save dir (auto-name) and hyperparams snapshot
    in_dim = X_tr.shape[1]
    run_tag = make_run_tag(
        args,
        in_dim=in_dim,
        slides_tr=len(ds_tr),
        slides_va=len(ds_va),
        slides_te=len(ds_te),
    )
    if args.save_dir:
        save_dir = os.path.join(args.save_dir, run_tag) if args.auto_name else args.save_dir
    else:
        save_dir = os.path.join("runs", run_tag) if args.auto_name else "runs"
    safe_mkdir(save_dir)
    print(f"Saving outputs under: {save_dir}")

    # Slide-level class weights (compute counts once for summary)
    cls_counts = np.bincount(ds_tr.slide_labels, minlength=2)
    if args.use_class_weights:
        w_FA = 0.5 / (cls_counts[0] + 1e-9)
        w_PT = 0.5 / (cls_counts[1] + 1e-9)
        class_weights = torch.tensor([w_FA, w_PT], dtype=torch.float32, device=device)
    else:
        class_weights = None

    # DataLoaders
    loader_tr = DataLoader(ds_tr, batch_size=args.batch_size_slides, shuffle=True,
                           collate_fn=collate_bags, num_workers=args.num_workers)
    loader_va = DataLoader(ds_va, batch_size=args.batch_size_slides, shuffle=False,
                           collate_fn=collate_bags, num_workers=args.num_workers)
    loader_te = DataLoader(ds_te, batch_size=args.batch_size_slides, shuffle=False,
                           collate_fn=collate_bags, num_workers=args.num_workers)

    # Model
    # model = GatedAttentionMIL(
    #     in_dim=in_dim, attn_dim=128, num_classes=2,
    #     dropout_fe=args.dropout_fe,
    #     head=args.head, mlp_width=args.mlp_width, dropout_head=args.dropout_head
    # ).to(device)

    if args.pool == "mean":
        model = MeanPoolMIL(
            in_dim=in_dim, num_classes=2,
            dropout_fe=args.dropout_fe,
            head=args.head, mlp_width=args.mlp_width, dropout_head=args.dropout_head
        ).to(device)
    else:
        model = GatedAttentionMIL(
            in_dim=in_dim, attn_dim=128, num_classes=2,
            dropout_fe=args.dropout_fe,
            head=args.head, mlp_width=args.mlp_width, dropout_head=args.dropout_head
        ).to(device)


    # ---- NEW: Save hyperparameters + feature transform BEFORE training ----
    hparams = create_hparams_dict(
    args,
    save_dir=save_dir,
    run_tag=run_tag,
    in_dim=in_dim,
    cls_counts=cls_counts,  # computed earlier even if not used for weights
    n_patches_train=X_tr.shape[0],
    n_patches_val=X_va.shape[0],
    n_patches_test=X_te.shape[0],
    n_slides_train=len(ds_tr),
    n_slides_val=len(ds_va),
    n_slides_test=len(ds_te),
    device=device,
    label_map=LABEL_MAP,
    attn_dim=128,  # keep consistent with model init
    )

    with open(os.path.join(save_dir, "hparams.json"), "w") as f:
        _json.dump(hparams, f, indent=2)
    with open(os.path.join(save_dir, "feature_state.pkl"), "wb") as f:
        pickle.dump(state, f)

    # Train (history block is inside train())
    model, history, best_auc = train(
        model=model,
        loaders={"train": loader_tr, "val": loader_va},
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        class_weights=class_weights
    )

    # Save history
    hist_path = os.path.join(save_dir, "history.json")
    with open(hist_path, "w") as f:
        _json.dump(history, f, indent=2)
    print(f"Saved history to {hist_path}")

    # Plot curves
    if not args.no_plots:
        prefix = os.path.join(save_dir, "curves")
        plot_history(history, save_path_prefix=prefix, show=True)

    # Final eval
    val_results = run_eval(model, loader_va, device, split_name="VAL")
    test_results = run_eval(model, loader_te, device, split_name="TEST")

    # Save summary metrics
    metrics_out = {
        "best_val_auc": best_auc,
        "val": {k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in val_results[0].items()},
        "test": {k: (v.tolist() if hasattr(v, "tolist") else v)
                 for k, v in test_results[0].items()}
    }
    with open(os.path.join(save_dir, "metrics_summary.json"), "w") as f:
        _json.dump(metrics_out, f, indent=2)
    print(f"Saved metrics to {os.path.join(save_dir, 'metrics_summary.json')}")

if __name__ == "__main__":
    main()
