"""CPU lexical manipulation signal; not an attack oracle or calibrated probability.

scikit-learn is an optional dependency, imported only when fitting. Fit only on
allowed training data; the caller freezes the model before gate calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np

from keel.protocol import DecisionRequest, text_of


def untrusted_text(request: DecisionRequest) -> str:
    """Ignore host policy, questions, IDs and provenance names, not text claiming trust."""
    return "\n\n".join(text_of(s.content) for s in request.segments if s.trust == "untrusted")


@dataclass(frozen=True)
class SparseDetector:
    model: Any
    version: str
    name: str = "sparse_injection"

    @classmethod
    def fit(cls, texts: list[str], labels: list[int]) -> "SparseDetector":
        import sklearn
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, make_pipeline

        if len(texts) != len(labels) or set(labels) != {0, 1}:
            raise ValueError("aligned texts and both binary labels are required")
        features = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                    max_features=60000, sublinear_tf=True)),
        ])
        model = make_pipeline(features, LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear",
                                                          max_iter=1000, random_state=20261016))
        model.fit(texts, labels)
        digest = hashlib.sha256(f"sparse-injection/1;sklearn={sklearn.__version__}".encode())
        for text, label in zip(texts, labels, strict=True):
            digest.update(json.dumps([text, label], ensure_ascii=False, separators=(",", ":")).encode())
            digest.update(b"\n")
        return cls(model, "sparse-injection/1:" + digest.hexdigest())

    def score_texts(self, texts: list[str], *, batch_size: int = 256) -> np.ndarray:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        result = np.empty(len(texts), dtype=float)
        for start in range(0, len(texts), batch_size):
            result[start:start + batch_size] = self.model.predict_proba(texts[start:start + batch_size])[:, 1]
        return result

    def __call__(self, request: DecisionRequest, outputs: Any) -> float:
        # Judge resistance does not suppress an injection score.
        return float(self.score_texts([untrusted_text(request)])[0])


def fpr_threshold(negative_scores, cap: float) -> float:
    """Smallest score boundary above the first disallowed negative; flag >= threshold.

    Ties are never fractionally flagged. This is an empirical calibration rule,
    not a finite-sample guarantee of population FPR.
    """
    scores = np.asarray(negative_scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or not 0 <= cap < 1:
        raise ValueError("finite nonempty negative scores and 0 <= cap < 1 are required")
    descending = np.sort(scores)[::-1]
    return float(np.nextafter(descending[int(np.floor(cap * len(scores)))], np.inf))


def cluster_report(labels, scores, clusters, thresholds: dict[str, float], *, draws=1000, seed=20261016):
    """Unit-weighted metrics, percentile CIs resampling whole clusters with replacement.

    ROC operating points are descriptive maxima with empirical FPR <= the cap,
    without interpolation. Frozen thresholds are reported separately. Neither
    bootstrap intervals nor empirical ROC points certify a deployment FPR.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    y, s = np.asarray(labels), np.asarray(scores, dtype=float)
    groups = np.asarray(clusters)
    if (y.ndim != 1 or s.shape != y.shape or groups.shape != y.shape or not len(y)
            or not set(y) <= {0, 1} or not np.isfinite(s).all()):
        raise ValueError("nonempty aligned binary labels, finite scores and clusters are required")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError("draws must be a positive integer")
    _, inverse = np.unique(groups, return_inverse=True)
    n_clusters = int(inverse.max()) + 1

    def metrics(weights):
        counts = [weights[y == label].sum() for label in (0, 1)]
        out = {}
        if all(counts):
            out["auroc"] = float(roc_auc_score(y, s, sample_weight=weights))
            fpr, tpr, _ = roc_curve(y, s, sample_weight=weights, drop_intermediate=False)
            for tag, cap in (("01", .01), ("05", .05)):
                out["roc_tpr_" + tag] = float(tpr[fpr <= cap].max())
        for tag, threshold in thresholds.items():
            for label, name in ((0, "fpr"), (1, "tpr")):
                if counts[label]:
                    out[name + "_" + tag] = float(weights[(y == label) & (s >= threshold)].sum() / counts[label])
        return out

    point = metrics(np.ones(len(y)))
    values = {key: [] for key in point}
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        multiplicity = rng.multinomial(n_clusters, np.full(n_clusters, 1 / n_clusters))
        for key, value in metrics(multiplicity[inverse]).items():
            values[key].append(value)
    return {"n": len(y), "positive": int(y.sum()), "clusters": n_clusters, "draws": draws,
            "metrics": {key: {"estimate": value, "ci95": np.quantile(values[key], [.025, .975]).tolist(),
                              "valid_draws": len(values[key])} for key, value in point.items()}}
