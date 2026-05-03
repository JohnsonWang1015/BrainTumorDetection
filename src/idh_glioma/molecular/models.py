from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import joblib
import lightgbm as lgb
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MolecularIDHModel(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> "MolecularIDHModel": ...


class LogisticIDH:
    def __init__(self, random_state: int = 42) -> None:
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state,
        )
        self._is_fit = False

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y, sample_weight=sample_weight)
        self._is_fit = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("Model is not fit")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self.scaler, "model": self.model, "is_fit": self._is_fit}, path)

    @classmethod
    def load(cls, path: Path) -> "LogisticIDH":
        payload = joblib.load(path)
        inst = cls()
        inst.scaler = payload["scaler"]
        inst.model = payload["model"]
        inst._is_fit = bool(payload.get("is_fit", True))
        return inst


class LightGBMIDH:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.model: lgb.LGBMClassifier | lgb.Booster | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
        model = lgb.LGBMClassifier(
            is_unbalance=True,
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            min_data_in_leaf=20,
            random_state=self.random_state,
            n_jobs=2,
            verbosity=-1,
            force_col_wise=True,
        )
        model.fit(X, y, sample_weight=sample_weight)
        self.model = model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fit")
        if isinstance(self.model, lgb.Booster):
            pos = self.model.predict(X)
            pos = np.asarray(pos, dtype=float)
            return np.column_stack([1.0 - pos, pos])
        return self.model.predict_proba(X)

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Model is not fit")
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(self.model, lgb.Booster):
            self.model.save_model(str(path))
            return
        booster = self.model.booster_
        booster.save_model(str(path))

    @classmethod
    def load(cls, path: Path) -> "LightGBMIDH":
        inst = cls()
        inst.model = lgb.Booster(model_file=str(path))
        return inst


class _MLPNet(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class _MLPConfig:
    max_epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    patience: int = 8
    random_state: int = 42


class MLPIDH:
    def __init__(self, input_dim: int | None = None, config: _MLPConfig | None = None) -> None:
        self.input_dim = input_dim
        self.config = config or _MLPConfig()
        self.scaler = StandardScaler()
        self.model: _MLPNet | None = _MLPNet(input_dim) if input_dim is not None else None
        self._is_fit = False

    def _ensure_model(self, input_dim: int) -> None:
        if self.model is None:
            self.input_dim = input_dim
            self.model = _MLPNet(input_dim)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float32)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self._ensure_model(X.shape[1])
        if self.model is None:
            raise RuntimeError("MLP model initialization failed")

        stratify = y if len(np.unique(y)) > 1 and len(y) >= 20 else None
        train_idx, val_idx = np.arange(len(y)), np.arange(len(y))
        if len(y) >= 20:
            train_idx, val_idx = train_test_split(
                np.arange(len(y)),
                test_size=0.2,
                random_state=self.config.random_state,
                stratify=stratify,
            )

        X_train = self.scaler.fit_transform(X[train_idx])
        y_train = y[train_idx]
        X_val = self.scaler.transform(X[val_idx])
        y_val = y[val_idx]

        dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        pos = max(float(np.sum(y_train == 1.0)), 1.0)
        neg = max(float(np.sum(y_train == 0.0)), 1.0)
        pos_weight = torch.tensor([float(np.sqrt(neg / pos))], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)

        best_state: dict[str, torch.Tensor] | None = None
        best_score = -np.inf
        patience_left = self.config.patience

        for _ in range(self.config.max_epochs):
            self.model.train()
            for xb, yb in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(torch.from_numpy(X_val))
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
            if len(np.unique(y_val)) > 1:
                score = float(roc_auc_score(y_val, val_probs))
            else:
                score = float(-criterion(val_logits, torch.from_numpy(y_val)).item())

            if score > best_score:
                best_score = score
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                patience_left = self.config.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self._is_fit = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or not self._is_fit:
            raise RuntimeError("Model is not fit")
        X_scaled = self.scaler.transform(np.asarray(X, dtype=np.float32))
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X_scaled))
            pos = torch.sigmoid(logits).cpu().numpy().astype(float)
        return np.column_stack([1.0 - pos, pos])

    def save(self, path: Path) -> None:
        if self.model is None or not self._is_fit:
            raise RuntimeError("Model is not fit")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "scaler_mean": self.scaler.mean_,
            "scaler_scale": self.scaler.scale_,
            "config": self.config.__dict__,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path) -> "MLPIDH":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        config = _MLPConfig(**payload["config"])
        inst = cls(input_dim=int(payload["input_dim"]), config=config)
        if inst.model is None:
            raise RuntimeError("MLP model initialization failed")
        inst.model.load_state_dict(payload["state_dict"])
        inst.scaler.mean_ = np.asarray(payload["scaler_mean"])
        inst.scaler.scale_ = np.asarray(payload["scaler_scale"])
        inst.scaler.var_ = inst.scaler.scale_ ** 2
        inst.scaler.n_features_in_ = int(inst.scaler.mean_.shape[0])
        inst.scaler.n_samples_seen_ = np.array([1], dtype=np.int64)
        inst._is_fit = True
        return inst
