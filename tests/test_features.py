import numpy as np

from cryotriage import features


def test_feature_shape_and_names():
    img = np.zeros((128, 128), dtype=float)
    coords = [[20, 20], [64, 64], [100, 100]]
    feats = features.extract_features(img, coords, box=32)
    assert feats.shape == (3, len(features.FEATURE_NAMES))


def test_bright_blob_has_higher_central_ratio():
    img = np.full((128, 128), 0.1, dtype=float)
    ys, xs = np.mgrid[0:128, 0:128]
    img += np.exp(-(((ys - 64) ** 2 + (xs - 64) ** 2) / (2 * 5.0 ** 2)))  # blob at center
    f_blob = features.crop_features(img[64 - 16:64 + 16, 64 - 16:64 + 16])
    f_flat = features.crop_features(np.full((32, 32), 0.1))
    ratio_idx = features.FEATURE_NAMES.index("central_ratio")
    assert f_blob[ratio_idx] > f_flat[ratio_idx]


def test_empty_crop_is_zeros():
    f = features.crop_features(np.zeros((0, 0)))
    assert f.shape == (len(features.FEATURE_NAMES),)
    assert np.all(f == 0)
