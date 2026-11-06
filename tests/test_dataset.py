import numpy as np
from training.dataset import augment


def test_augment_keeps_flags_and_missing_hands():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 320)).astype(np.float32)
    x[:, 318:] = 1.0
    x[:, 30:60] = 0.0
    out = augment(x, rng)
    assert out.shape[1] == 320
    assert (out[:, 318:] == 1.0).all()
    assert (out[:, 30:60] == 0.0).all()