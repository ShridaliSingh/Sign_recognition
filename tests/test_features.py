import numpy as np
from features.landmarks import (N_POINTS, FLIP_ORDER, FACE, FACE_BLOCK, POSE_BLOCK,
                                DOM_HAND_BLOCK, NONDOM_HAND_BLOCK)
from features.pipeline import flip, make_right_dominant, normalize, build_features


def random_sequence(T=10, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(T, N_POINTS, 3)).astype(np.float32)


def test_flip_twice_is_identity():
    sel = random_sequence()
    sel[3:6, NONDOM_HAND_BLOCK] = np.nan
    assert np.allclose(flip(flip(sel)), sel, equal_nan=True)


def test_flip_order_keeps_midline_and_is_permutation():
    for i in [0, 17, 13, 14, 1, 10, 152]:
        pos = FACE_BLOCK.start + FACE.index(i)
        assert FLIP_ORDER[pos] == pos
    assert FLIP_ORDER[POSE_BLOCK.start] == POSE_BLOCK.start
    assert sorted(FLIP_ORDER) == list(range(N_POINTS))


def test_left_dominant_gets_flipped():
    sel = random_sequence()
    sel[:, DOM_HAND_BLOCK] = np.nan
    out, flipped = make_right_dominant(sel)
    assert flipped
    assert not np.isnan(out[:, DOM_HAND_BLOCK]).any()
    assert np.isnan(out[:, NONDOM_HAND_BLOCK]).all()


def test_feature_shape_and_no_nan():
    sel = random_sequence(T=7)
    sel[:, NONDOM_HAND_BLOCK] = np.nan
    feats = build_features(normalize(sel))
    assert feats.shape == (7, 320)
    assert not np.isnan(feats).any()
    assert feats[:, -1].sum() == 0