import os, h5py, torch
import numpy as np
import pandas as pd

def build_panther_h5(
    fused_pt_path,           # e.g., ".../FA_57B_fused.pt" -> tensor shape [N, D_concat]
    coords_source,           # e.g., path to 2.5× H5 with 'coords' OR CSV with x,y
    out_h5_path,
    slide_id=None,
    size_um=None,            # scalar or per-row iterable; optional
    coords_from='h5'         # 'h5' or 'csv'
):
    # --- load features ---
    fused = torch.load(fused_pt_path, map_location='cpu')
    if isinstance(fused, dict) and 'features' in fused:
        fused = fused['features']
    assert fused.ndim == 2, f"Expected [N,D], got {fused.shape}"
    N, D = fused.shape

    # --- load coords (2.5× anchors) ---
    if coords_from == 'h5':
        with h5py.File(coords_source, 'r') as h5:
            coords = h5['coords'][:]
            # try to keep slide_id if present
            if slide_id is None and 'slide_id' in h5:
                slide_id = h5['slide_id'][()]  # could be bytes
                if isinstance(slide_id, bytes): slide_id = slide_id.decode('utf-8')
            if size_um is None and 'size_um' in h5:
                size_um = h5['size_um'][:]
    else:
        # CSV must have columns: x,y (pixel coords at 2.5× level)
        df = pd.read_csv(coords_source)
        coords = df[['x','y']].to_numpy(dtype=np.int32)
        if size_um is None and 'size_um' in df.columns:
            size_um = df['size_um'].to_numpy(dtype=np.float32)

    assert coords.shape[0] == N, f"Coord count {coords.shape[0]} != feature count {N}"
    assert coords.shape[1] == 2, "Coords must be (N,2)"

    # --- normalize blocks (optional but recommended) ---
    # Example: assume first D2 are 2.5× block, last D5 are pooled 5× block
    # Comment out if you already normalized upstream.
    # D2 = <put your 2.5× dim here>
    # fused_np = fused.numpy().astype(np.float32)
    # a = fused_np[:, :D2]; b = fused_np[:, D2:]
    # a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    # b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    # fused_np = np.concatenate([a, b], axis=1)
    # fused_np = fused_np / np.maximum(np.linalg.norm(fused_np, axis=1, keepdims=True), 1e-12)

    fused_np = fused.numpy().astype(np.float32)
    coords_np = coords.astype(np.int32)

    # --- write H5 ---
    os.makedirs(os.path.dirname(out_h5_path), exist_ok=True)
    with h5py.File(out_h5_path, 'w') as h5o:
        h5o.create_dataset('features', data=fused_np, dtype='float32', compression='gzip', compression_opts=4)
        h5o.create_dataset('coords',   data=coords_np, dtype='int32',  compression='gzip', compression_opts=4)
        if size_um is not None:
            size_um_arr = np.array(size_um, dtype=np.float32)
            if size_um_arr.ndim == 0:
                size_um_arr = np.full((N,), float(size_um_arr), dtype=np.float32)
            assert size_um_arr.shape[0] == N
            h5o.create_dataset('size_um', data=size_um_arr, dtype='float32')
        if slide_id is not None:
            # save as attribute; or create a fixed-length string dataset
            h5o.attrs['slide_id'] = str(slide_id)
        h5o.attrs['feature_dim'] = int(D)
        h5o.attrs['schema'] = 'panther_multiscale_v1'
    print(f"Wrote {out_h5_path} with features {fused_np.shape} and coords {coords_np.shape}")
