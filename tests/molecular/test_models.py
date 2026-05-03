from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification

from idh_glioma.molecular.models import LightGBMIDH, LogisticIDH, MLPIDH, _MLPConfig


def _toy_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=200,
        n_features=50,
        n_informative=10,
        n_redundant=5,
        random_state=42,
    )
    return X.astype(np.float32), y.astype(np.int64)


def test_logistic_predict_proba_and_roundtrip(tmp_path: Path) -> None:
    X, y = _toy_data()
    model = LogisticIDH(random_state=42)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (200, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    save_path = tmp_path / "logistic.joblib"
    model.save(save_path)
    loaded = LogisticIDH.load(save_path)
    probs_loaded = loaded.predict_proba(X)
    assert np.allclose(probs, probs_loaded, atol=1e-6)


def test_lightgbm_predict_proba_and_roundtrip(tmp_path: Path) -> None:
    X, y = _toy_data()
    model = LightGBMIDH(random_state=42)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (200, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    save_path = tmp_path / "lightgbm.txt"
    model.save(save_path)
    loaded = LightGBMIDH.load(save_path)
    probs_loaded = loaded.predict_proba(X)
    assert np.allclose(probs, probs_loaded, atol=1e-6)


def test_mlp_predict_proba_and_roundtrip(tmp_path: Path) -> None:
    X, y = _toy_data()
    model = MLPIDH(
        input_dim=X.shape[1],
        config=_MLPConfig(max_epochs=4, patience=2, random_state=42),
    )
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (200, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    save_path = tmp_path / "mlp.pt"
    model.save(save_path)
    loaded = MLPIDH.load(save_path)
    probs_loaded = loaded.predict_proba(X)
    assert np.allclose(probs, probs_loaded, atol=1e-6)
