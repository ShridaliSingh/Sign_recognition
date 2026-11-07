import numpy as np
from training.dataset import augment
import torch
from training.models import SignTransformer


def test_augment_keeps_flags_and_missing_hands():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 320)).astype(np.float32)
    x[:, 318:] = 1.0
    x[:, 30:60] = 0.0
    out = augment(x, rng)
    assert out.shape[1] == 320
    assert (out[:, 318:] == 1.0).all()
    assert (out[:, 30:60] == 0.0).all()

def test_transformer_shape_and_padding_ignored():
    torch.manual_seed(0)
    model = SignTransformer().eval()
    x = torch.randn(2, 20, 320)
    mask = torch.ones(2, 20, dtype=torch.bool)
    mask[1, 12:] = False
    out = model(x, mask)
    assert out.shape == (2, 250)

    x2 = x.clone()
    x2[1, 12:] = 99.0
    assert torch.allclose(model(x2, mask)[1], out[1], atol=1e-5)

def test_hand_drop_looks_like_missing_hand():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 320)).astype(np.float32)
    x[:, 318:] = 1.0
    out = augment(x, rng, hand_drop=0.5)
    dropped = out[:, 318] == 0
    assert dropped.any()
    assert (out[dropped, 27:90] == 0).all()
    assert (out[dropped, 282:300] == 0).all()