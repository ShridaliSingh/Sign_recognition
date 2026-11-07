import numpy as np
import pandas as pd

from features.landmarks import SELECTED, FLIP_ORDER, DOM_HAND_BLOCK, NONDOM_HAND_BLOCK,NOSE,L_SHOULDER,R_SHOULDER,REL_REFS

N_LANDMARKS = 543


def load_sequence(path):
    df = pd.read_parquet(path, columns=["frame", "x", "y", "z"])
    df = df.sort_values("frame", kind="stable")
    n_frames = df["frame"].nunique()
    data = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    return data.reshape(n_frames, N_LANDMARKS, 3)

def select_landmarks(arr):
    return arr[:, SELECTED, :]

def flip(sel):
    out = sel[:, FLIP_ORDER, :]
    out[..., 0] = -out[..., 0]
    return out

def make_right_dominant(sel):
    right_frames = (~np.isnan(sel[:, DOM_HAND_BLOCK.start, 0])).sum()
    left_frames = (~np.isnan(sel[:, NONDOM_HAND_BLOCK.start, 0])).sum()
    if left_frames > right_frames:
        return flip(sel), True
    return sel, False

def normalize(sel):
    mid = (sel[:, L_SHOULDER] + sel[:, R_SHOULDER]) / 2
    centre = np.nanmean(mid, axis=0)
    dist = np.linalg.norm(sel[:, NOSE, :2] - mid[:, :2], axis=-1)
    scale = np.nanmean(dist)
    if not np.isfinite(scale) or scale < 1e-6:
        return sel
    return (sel - centre) / scale

def build_features(norm):
    T = norm.shape[0]
    hand_idx = [DOM_HAND_BLOCK.start, NONDOM_HAND_BLOCK.start]
    wrists = norm[:, hand_idx, :]
    refs = norm[:, REL_REFS, :]
    rel = wrists[:, :, None, :] - refs[:, None, :, :]
    flags = (~np.isnan(norm[:, hand_idx, 0])).astype(np.float32)
    feats = np.concatenate([norm.reshape(T, -1), rel.reshape(T, -1), flags], axis=1)
    return np.nan_to_num(feats, nan=0.0)

def process_array(arr):
    sel = select_landmarks(arr)
    sel, _ = make_right_dominant(sel)
    return build_features(normalize(sel))


def process_sequence(path):
    return process_array(load_sequence(path))
