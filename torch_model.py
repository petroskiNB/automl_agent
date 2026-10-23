"""CPU tabular MLPs trained from scratch; no pretrained assets or downloads."""
import copy
import time

from torch_resources import configure
configure()

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from threadpoolctl import threadpool_limits

from torch_resources import configure_torch


class TabularPreprocessor:
    """Fitted on the training fold only. 0=unknown, 1=missing categories."""
    def __init__(self, max_categories=20000):
        self.max_categories = max_categories

    def fit(self, frame):
        self.columns = list(frame.columns)
        self.numeric = [c for c in frame if pd.api.types.is_numeric_dtype(frame[c])]
        self.categorical = [c for c in frame if c not in self.numeric]
        numeric = frame[self.numeric].astype(float).replace([np.inf, -np.inf], np.nan)
        self.medians = numeric.median().fillna(0)
        self.scales = (numeric.quantile(.75) - numeric.quantile(.25)).fillna(1)
        self.scales = self.scales.mask(self.scales < 1e-8, 1)
        self.vocabularies = {}
        for c in self.categorical:
            values = frame[c].dropna().astype(str).value_counts().head(self.max_categories).index
            self.vocabularies[c] = {v: i + 2 for i, v in enumerate(values)}
        return self

    def transform(self, frame):
        if list(frame.columns) != self.columns:
            raise ValueError("Feature names/order differ from fitted preprocessing")
        numeric = frame[self.numeric].astype(float).replace([np.inf, -np.inf], np.nan)
        missing = numeric.isna().to_numpy(dtype=np.float32)
        values = ((numeric.fillna(self.medians) - self.medians) / self.scales).clip(-20, 20)
        values = np.concatenate([values.to_numpy(dtype=np.float32), missing], axis=1)
        categories = np.empty((len(frame), len(self.categorical)), dtype=np.int64)
        for i, c in enumerate(self.categorical):
            encoded = frame[c].astype(str).map(self.vocabularies[c]).fillna(0).to_numpy(dtype=np.int64)
            encoded[frame[c].isna().to_numpy()] = 1
            categories[:, i] = encoded
        return torch.from_numpy(values), torch.from_numpy(categories)


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.ReLU(),
                                    nn.Dropout(dropout), nn.Linear(width, width))

    def forward(self, x):
        return x + self.layers(x)


class TabularNetwork(nn.Module):
    def __init__(self, numeric_size, cardinalities, output_size, width, layers, dropout, residual):
        super().__init__()
        dimensions = [min(32, max(2, int(n ** .5))) for n in cardinalities]
        self.embeddings = nn.ModuleList([nn.Embedding(n, d, padding_idx=0)
                                        for n, d in zip(cardinalities, dimensions)])
        self.input = nn.Linear(numeric_size + sum(dimensions), width)
        blocks = []
        for _ in range(layers):
            blocks.append(ResidualBlock(width, dropout) if residual else nn.Sequential(
                nn.Linear(width, width), nn.LayerNorm(width), nn.ReLU(), nn.Dropout(dropout)))
        self.blocks = nn.Sequential(*blocks)
        self.output = nn.Linear(width, output_size)

    def forward(self, numeric, categorical):
        features = [numeric] + [embedding(categorical[:, i]) for i, embedding in enumerate(self.embeddings)]
        return self.output(self.blocks(torch.relu(self.input(torch.cat(features, dim=1)))))


class TorchModel:
    def __init__(self, task_type, metric, params, residual=False):
        self.task_type, self.metric, self.residual = task_type, metric, residual
        defaults = dict(iterations=100, width=256, layers=3, dropout=.1, learning_rate=.001,
                        weight_decay=.0001, batch_size=512, seed=42, max_categories=20000,
                        auto_class_weights=None, loss="auto", target_transform="standard")
        if set(params) - defaults.keys():
            raise ValueError(f"Unsupported torch parameters: {set(params) - defaults.keys()}")
        self.params = dict(defaults, **params)
        p = self.params
        for key, low, high in (("iterations", 1, 2000), ("width", 8, 1024), ("layers", 1, 8),
                               ("batch_size", 16, 8192), ("max_categories", 2, 50000), ("seed", 0, 2**31-1)):
            if type(p[key]) is not int or not low <= p[key] <= high:
                raise ValueError(f"Invalid {key}: expected integer in {low}..{high}")
        for key in ("dropout", "learning_rate", "weight_decay"):
            if not np.isfinite(p[key]):
                raise ValueError(f"Non-finite {key}")
        if not 0 <= p["dropout"] < 1 or not 0 < p["learning_rate"] <= .1 or not 0 <= p["weight_decay"] <= 1:
            raise ValueError("Invalid dropout/learning_rate/weight_decay")
        if p["loss"] not in ("auto", "mse", "huber", "mae") or p["target_transform"] not in ("standard", "log1p"):
            raise ValueError("Unsupported regression loss or target transform")
        if p["auto_class_weights"] not in (None, "Balanced", "SqrtBalanced"):
            raise ValueError("Unknown class weighting")
        if task_type != "classification" and p["auto_class_weights"]:
            raise ValueError("Class weights require classification")
        if task_type == "classification" and (p["loss"] != "auto" or p["target_transform"] != "standard"):
            raise ValueError("Regression loss/target transform do not apply to classification")

    def _build(self):
        prep, p = self.preprocessor, self.params
        return TabularNetwork(len(prep.numeric) * 2,
            [len(prep.vocabularies[c]) + 2 for c in prep.categorical], self.output_size,
            p["width"], p["layers"], p["dropout"], self.residual)

    def __getstate__(self):
        state = self.__dict__.copy()
        network = state.pop("network", None)
        if network is not None:
            state["network_weights"] = {k: v.detach().cpu().numpy().copy() for k, v in network.state_dict().items()}
        return state

    def __setstate__(self, state):
        weights = state.pop("network_weights", None)
        self.__dict__.update(state)
        if weights is not None:
            configure_torch()
            self.network = self._build()
            self.network.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in weights.items()})
            self.network.eval()

    def _predict_tensors(self, tensors):
        self.network.eval()
        rows = []
        with torch.inference_mode():
            for start in range(0, len(tensors[0]), self.params["batch_size"]):
                rows.append(self.network(*(x[start:start+self.params["batch_size"]] for x in tensors)))
        raw = torch.cat(rows) if rows else torch.empty((0, self.output_size))
        if self.task_type == "classification":
            return torch.softmax(raw, dim=1).numpy()
        pred = raw[:, 0].numpy().astype(np.float64) * self.target_scale + self.target_mean
        if self.params["target_transform"] == "log1p":
            pred = np.expm1(np.clip(pred, -30, 30))
        return pred

    def fit(self, x, y, eval_set=None, early_stopping_rounds=10, deadline=None):
        started = time.monotonic()
        self.threads = configure_torch()
        torch.manual_seed(self.params["seed"])
        p = self.params
        self.preprocessor = TabularPreprocessor(p["max_categories"]).fit(x)
        tensors = self.preprocessor.transform(x)
        y = np.asarray(y)
        classification = self.task_type == "classification"
        if classification:
            self.classes_, encoded = np.unique(y, return_inverse=True)
            self.output_size = len(self.classes_)
            if self.output_size < 2:
                raise ValueError("Need at least two classes")
            if eval_set is not None and set(eval_set[1]) - set(self.classes_):
                raise ValueError("Validation contains unseen target class")
            target = torch.tensor(encoded, dtype=torch.long)
            weights = None
            if p["auto_class_weights"]:
                counts = np.bincount(encoded)
                weights = counts.max() / counts.astype(float)
                if p["auto_class_weights"] == "SqrtBalanced":
                    weights = np.sqrt(weights)
                weights = torch.tensor(weights / weights.mean(), dtype=torch.float32)
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.output_size = 1
            y = y.astype(np.float64)
            if not np.isfinite(y).all():
                raise ValueError("Non-finite target")
            if p["target_transform"] == "log1p":
                if (y < 0).any():
                    raise ValueError("log1p target requires nonnegative training targets")
                y = np.log1p(y)
            self.target_mean, self.target_scale = float(y.mean()), max(float(y.std()), 1e-6)
            target = torch.tensor((y-self.target_mean)/self.target_scale, dtype=torch.float32).reshape(-1, 1)
            loss = p["loss"] if p["loss"] != "auto" else ("mae" if self.metric == "mae" else "mse")
            criterion = {"mse": nn.MSELoss, "mae": nn.L1Loss, "huber": nn.SmoothL1Loss}[loss]()
        self.network = self._build()
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"])
        validation = self.preprocessor.transform(eval_set[0]) if eval_set is not None else None
        best, best_weights, stale = None, None, 0
        self.evaluation_history, self.selected_iterations, self.timed_out = [], 0, False
        with threadpool_limits(limits=self.threads):
            for epoch in range(1, p["iterations"] + 1):
                self.network.train()
                order = torch.randperm(len(y))
                for indices in order.split(p["batch_size"]):
                    if deadline is not None and time.monotonic() >= deadline:
                        self.timed_out = True
                        break
                    optimizer.zero_grad(set_to_none=True)
                    output = self.network(*(t[indices] for t in tensors))
                    loss = criterion(output, target[indices])
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite torch training loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), 5)
                    optimizer.step()
                if self.timed_out:
                    break
                if validation is not None:
                    raw = self._predict_tensors(validation)
                    if not np.isfinite(raw).all():
                        raise ValueError("Non-finite predictions")
                    pred = self.classes_[raw.argmax(1)] if classification else raw
                    truth = np.asarray(eval_set[1])
                    if self.metric == "f1_macro":
                        score = f1_score(truth, pred, labels=self.classes_, average="macro", zero_division=0)
                    elif self.metric == "accuracy":
                        score = accuracy_score(truth, pred)
                    elif self.metric == "mae":
                        score = mean_absolute_error(truth, pred)
                    else:
                        score = np.sqrt(mean_squared_error(truth, pred))
                    self.evaluation_history.append(float(score))
                    improved = best is None or (score > best if classification else score < best)
                else:
                    improved = True
                if improved:
                    best = score if validation is not None else None
                    best_weights = copy.deepcopy(self.network.state_dict())
                    self.selected_iterations, stale = epoch, 0
                else:
                    stale += 1
                if validation is not None and early_stopping_rounds and stale >= early_stopping_rounds:
                    break
        if best_weights is None or (self.timed_out and validation is None):
            raise TimeoutError("Torch deadline reached before a usable completed fit")
        self.network.load_state_dict(best_weights)
        self.network.eval()
        self.fit_seconds = time.monotonic() - started
        return self

    def predict_proba(self, x):
        if self.task_type != "classification":
            raise ValueError("Probabilities require classification")
        return self._predict_tensors(self.preprocessor.transform(x))

    def predict(self, x):
        raw = self._predict_tensors(self.preprocessor.transform(x))
        return self.classes_[raw.argmax(1)] if self.task_type == "classification" else raw
