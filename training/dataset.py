import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

MAX_LEN = 128
N_FLAGS = 2


def load_cache(cache_dir, label_map_path):
    feats = np.load(f"{cache_dir}/features.npy")
    offsets = np.load(f"{cache_dir}/offsets.npy")
    lengths = np.load(f"{cache_dir}/lengths.npy")
    meta = pd.read_csv(f"{cache_dir}/meta.csv")
    with open(label_map_path) as f:
        sign_to_idx = json.load(f)
    labels = meta["sign"].map(sign_to_idx).to_numpy()
    return feats, offsets, lengths, labels, meta


def resample(x, max_len=MAX_LEN):
    T = len(x)
    if T <= max_len:
        return x
    idx = np.linspace(0, T - 1, max_len).round().astype(int)
    return x[idx]


def augment(x, rng):
    T = len(x)
    new_T = max(1, int(round(T / rng.uniform(0.8, 1.2))))
    x = x[np.linspace(0, T - 1, new_T).round().astype(int)]

    n_vec = (x.shape[1] - N_FLAGS) // 3
    vec = x[:, :n_vec * 3].reshape(len(x), n_vec, 3) * rng.uniform(0.9, 1.1)
    angle = np.deg2rad(rng.uniform(-10, 10))
    c, s = np.cos(angle), np.sin(angle)
    px, py = vec[..., 0].copy(), vec[..., 1].copy()
    vec[..., 0] = c * px - s * py
    vec[..., 1] = s * px + c * py
    return np.concatenate([vec.reshape(len(x), -1), x[:, n_vec * 3:]], axis=1).astype(np.float32)

class SignDataset(Dataset):
    def __init__(self, feats, offsets, lengths, labels, indices, augment=False):
        self.feats = feats
        self.offsets = offsets
        self.lengths = lengths
        self.labels = labels
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        start = self.offsets[j]
        x = self.feats[start:start + self.lengths[j]].astype(np.float32)
        if self.augment:
            x = augment(x, np.random.default_rng())
        x = resample(x)
        return torch.from_numpy(x), int(self.labels[j])


def collate(batch):
    xs, ys = zip(*batch)
    T = max(len(x) for x in xs)
    out = torch.zeros(len(xs), T, xs[0].shape[1])
    mask = torch.zeros(len(xs), T, dtype=torch.bool)
    for k, x in enumerate(xs):
        out[k, :len(x)] = x
        mask[k, :len(x)] = True
    return out, mask, torch.tensor(ys)