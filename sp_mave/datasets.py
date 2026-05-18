"""Dataset loaders returning DatasetBundle objects.

To add a new dataset, write a loader function with signature
    def load_<name>(**kwargs) -> DatasetBundle
and register it in the REGISTRY dict at the bottom of this module.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.datasets import make_friedman1

from sp_mave.types import DatasetBundle

DatasetLoader = Callable[..., DatasetBundle]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
logger = logging.getLogger(__name__)


def _ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def _download(url: str, dest: Path) -> Path:
    """Download a file from *url* to *dest*."""
    import urllib.request

    logger.info("Downloading %s -> %s", url, dest)
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download {url}. "
            f"Please download manually and place at {dest}.\n"
            f"Original error: {exc}"
        ) from exc
    return dest


# ---------------------------------------------------------------------------
# Friedman-1 (synthetic, ground truth available)
# ---------------------------------------------------------------------------

def load_friedman1(
    n_samples: int = 100,
    n_noise: int = 20,
    noise: float = 1.0,
    random_state: int = 42,
    **_extra: Any,
) -> DatasetBundle:
    """Generate the Friedman-1 benchmark with appended noise features.

    The response is
        y = 10 sin(pi x1 x2) + 20 (x3 - 0.5)^2 + 10 x4 + 5 x5 + noise

    so y depends on x1..x5.  Additional *n_noise* columns of U(0,1)
    noise are appended, giving p = 5 + n_noise total features.

    The true sufficient dimension reduction subspace is span{e1,...,e5}
    because the five informative features enter through nonlinear and
    interaction terms that cannot be compressed to fewer than five
    independent linear combinations.
    """
    n_informative = 5
    p = n_informative + n_noise
    X, y = make_friedman1(
        n_samples=n_samples,
        n_features=p,
        noise=noise,
        random_state=random_state,
    )
    feature_names = [f"x{i+1}" for i in range(p)]

    true_subspace = np.eye(p, n_informative)

    metadata = {
        "generator": "sklearn.datasets.make_friedman1",
        "formula": "y = 10*sin(pi*x1*x2) + 20*(x3-0.5)^2 + 10*x4 + 5*x5 + N(0,noise)",
        "n_informative": n_informative,
        "n_noise": n_noise,
        "noise_level": noise,
        "true_d": n_informative,
    }
    return DatasetBundle(
        X=X,
        y=y,
        feature_names=feature_names,
        true_subspace=true_subspace,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Diabetes (sklearn built-in, always available)
# ---------------------------------------------------------------------------

def load_diabetes(**_extra: Any) -> DatasetBundle:
    """sklearn diabetes dataset (442 samples, 10 features).

    Response: quantitative measure of disease progression one year
    after baseline.  Features are already mean-centered and scaled
    to unit variance in sklearn's version.
    No ground-truth subspace is available.
    """
    from sklearn.datasets import load_diabetes as _load

    data = _load()
    return DatasetBundle(
        X=data.data,
        y=data.target,
        feature_names=list(data.feature_names),
        true_subspace=None,
        metadata={
            "source": "sklearn.datasets.load_diabetes",
            "n_samples": int(data.data.shape[0]),
            "n_features": int(data.data.shape[1]),
            "is_real": True,
        },
    )


# ---------------------------------------------------------------------------
# Communities and Crime (UCI, requires download on first use)
# ---------------------------------------------------------------------------

_COMMUNITIES_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "communities/communities.data"
)


def load_communities_crime(
    missing_threshold: float = 0.1,
    **_extra: Any,
) -> DatasetBundle:
    """UCI Communities and Crime dataset.

    Response: ViolentCrimesPerPop (total violent crimes per 100K
    population).  The first five non-predictive columns (state, county,
    community, communityname, fold) are dropped.  Features with more
    than *missing_threshold* fraction of missing values are removed,
    then rows with any remaining NaN are dropped.
    No ground-truth subspace is available.
    """
    cache_path = DATA_DIR / "communities_crime.npz"
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return DatasetBundle(
            X=data["X"],
            y=data["y"],
            feature_names=list(data["feature_names"]),
            true_subspace=None,
            metadata=data["metadata"].item(),
        )

    _ensure_data_dir()
    raw_path = DATA_DIR / "communities.data"
    if not raw_path.exists():
        _download(_COMMUNITIES_URL, raw_path)

    raw_rows: list[list[str]] = []
    with open(raw_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                raw_rows.append(row)

    n_total = len(raw_rows)
    n_cols = len(raw_rows[0])

    # Convert to float matrix; '?' and non-numeric values become NaN
    data_matrix = np.full((n_total, n_cols), np.nan)
    for i, row in enumerate(raw_rows):
        for j, val in enumerate(row):
            val = val.strip()
            if val != "?":
                try:
                    data_matrix[i, j] = float(val)
                except ValueError:
                    pass

    # Columns 0-4 are non-predictive; last column is the response
    X_full = data_matrix[:, 5 : n_cols - 1]
    y_full = data_matrix[:, n_cols - 1]

    # Drop columns with excessive missing
    col_missing = np.isnan(X_full).sum(axis=0) / n_total
    keep_cols = np.where(col_missing <= missing_threshold)[0]
    X_filt = X_full[:, keep_cols]

    # Drop rows with any remaining NaN (in X or y)
    valid = ~np.isnan(X_filt).any(axis=1) & ~np.isnan(y_full)
    X = X_filt[valid].astype(np.float64)
    y = y_full[valid].astype(np.float64)

    feature_names = [f"community_feat_{i}" for i in range(X.shape[1])]

    metadata: dict[str, Any] = {
        "source": "UCI Communities and Crime",
        "url": _COMMUNITIES_URL,
        "n_samples_original": n_total,
        "n_samples_clean": int(X.shape[0]),
        "n_features_original": int(n_cols - 6),
        "n_features_clean": int(X.shape[1]),
        "missing_threshold": missing_threshold,
        "response": "ViolentCrimesPerPop",
        "is_real": True,
    }

    np.savez(
        cache_path,
        X=X,
        y=y,
        feature_names=np.array(feature_names),
        metadata=np.array(metadata),
    )
    logger.info(
        "Communities & Crime: %d samples, %d features "
        "(from %d original features)",
        X.shape[0],
        X.shape[1],
        n_cols - 6,
    )

    return DatasetBundle(
        X=X,
        y=y,
        feature_names=feature_names,
        true_subspace=None,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Tecator (spectroscopy benchmark, requires download on first use)
# ---------------------------------------------------------------------------

_TECATOR_URL = "http://lib.stat.cmu.edu/datasets/tecator"


def _parse_tecator_text(
    text: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse the CMU StatLib Tecator file format.

    Each sample occupies 25 lines of 5 values each (125 values total):
      values 0–99   : 100 absorbance channels
      values 100–121: 22 principal components (discarded here)
      values 122–124: moisture, fat, protein

    Returns (X_absorbances, y_moisture, y_fat, y_protein).
    """
    lines = text.strip().split("\n")

    # Find data start: after the LAST "extrapolation_examples=..." line
    # (the marker appears twice — once in the description, once before data)
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("extrapolation_examples="):
            data_start = i + 1

    # Collect all numeric values after the header
    all_values: list[float] = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            all_values.extend(float(v) for v in stripped.split())
        except ValueError:
            continue

    # 125 values per sample: 100 absorbances + 22 PCs + 3 responses
    vals_per_sample = 125
    n_samples = len(all_values) // vals_per_sample

    X_list: list[list[float]] = []
    moisture_list: list[float] = []
    fat_list: list[float] = []
    protein_list: list[float] = []

    for s in range(n_samples):
        offset = s * vals_per_sample
        sv = all_values[offset : offset + vals_per_sample]
        X_list.append(sv[:100])
        moisture_list.append(sv[122])
        fat_list.append(sv[123])
        protein_list.append(sv[124])

    return (
        np.array(X_list),
        np.array(moisture_list),
        np.array(fat_list),
        np.array(protein_list),
    )


def load_tecator(response: str = "fat", **_extra: Any) -> DatasetBundle:
    """Tecator spectroscopy dataset (215 samples, 100 absorbance channels).

    Each sample is a 100-channel near-infrared absorbance spectrum.
    Three response variables are available: moisture, fat, protein.
    The default response is fat content (standard benchmark choice).
    No ground-truth subspace is available.
    """
    cache_path = DATA_DIR / "tecator.npz"
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        resp_key = f"y_{response}"
        if resp_key not in data:
            available = [k[2:] for k in data.files if k.startswith("y_")]
            raise ValueError(
                f"Unknown Tecator response '{response}'. "
                f"Available: {available}"
            )
        return DatasetBundle(
            X=data["X"],
            y=data[resp_key],
            feature_names=list(data["feature_names"]),
            true_subspace=None,
            metadata={**data["metadata"].item(), "response_used": response},
        )

    _ensure_data_dir()
    raw_path = DATA_DIR / "tecator.txt"
    if not raw_path.exists():
        _download(_TECATOR_URL, raw_path)

    text = raw_path.read_text()
    X, y_moisture, y_fat, y_protein = _parse_tecator_text(text)

    if X.shape[0] == 0:
        raise RuntimeError(
            "Failed to parse Tecator data. The file may have an unexpected "
            f"format. Check {raw_path}"
        )

    feature_names = [f"absorbance_{i + 1}" for i in range(X.shape[1])]

    metadata: dict[str, Any] = {
        "source": "CMU StatLib (http://lib.stat.cmu.edu/datasets/tecator)",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "responses_available": ["moisture", "fat", "protein"],
        "is_real": True,
    }

    np.savez(
        cache_path,
        X=X,
        y_moisture=y_moisture,
        y_fat=y_fat,
        y_protein=y_protein,
        feature_names=np.array(feature_names),
        metadata=np.array(metadata),
    )

    response_map = {"moisture": y_moisture, "fat": y_fat, "protein": y_protein}
    if response not in response_map:
        raise ValueError(
            f"Unknown Tecator response '{response}'. "
            f"Use: moisture, fat, or protein."
        )

    logger.info(
        "Tecator: %d samples, %d features, response=%s",
        X.shape[0],
        X.shape[1],
        response,
    )

    return DatasetBundle(
        X=X,
        y=response_map[response],
        feature_names=feature_names,
        true_subspace=None,
        metadata={**metadata, "response_used": response},
    )


# ---------------------------------------------------------------------------
# California housing, CCPP (energy), semi-synthetic on real X
# ---------------------------------------------------------------------------

_CCPP_UCI_CSV = "https://archive.ics.uci.edu/static/public/294/data.csv"


def _subsample_xy(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    if max_samples is None or max_samples >= n:
        return X, y
    rng = np.random.default_rng(int(random_state))
    idx = rng.choice(n, size=int(max_samples), replace=False)
    return X[idx], y[idx]


def load_california_housing_regression(
    max_samples: int | None = 4000,
    random_state: int = 42,
    **_extra: Any,
) -> DatasetBundle:
    """California housing (MedHouseVal); numeric features only.

    ``max_samples`` caps *n* for dense ``(n,n)`` MAVE weight matrices
    (methodological compromise; document in papers).
    """
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing()
    X = np.asarray(data.data, dtype=np.float64)
    y = np.asarray(data.target, dtype=np.float64)
    names = list(data.feature_names)
    X, y = _subsample_xy(X, y, max_samples, random_state)
    meta: dict[str, Any] = {
        "source": "sklearn.datasets.fetch_california_housing",
        "is_real": True,
        "max_samples_applied": max_samples,
        "uci_page": "https://www.kaggle.com/datasets/camnugent/california-housing-prices",
    }
    return DatasetBundle(
        X=X,
        y=y,
        feature_names=names,
        true_subspace=None,
        metadata=meta,
    )


def load_combined_cycle_power_plant(
    max_samples: int | None = 4000,
    random_state: int = 42,
    **_extra: Any,
) -> DatasetBundle:
    """UCI Combined Cycle Power Plant (dataset 294) — tabular regression.

    Features: AT, V, AP, RH; target: PE (net hourly electrical energy output).
    CSV: https://archive.ics.uci.edu/static/public/294/data.csv
    """
    cache_path = DATA_DIR / "ccpp.npz"
    feature_names = ["AT", "V", "AP", "RH"]

    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        X_all = np.asarray(data["X"], dtype=np.float64)
        y_all = np.asarray(data["y"], dtype=np.float64)
        meta = data["metadata"].item()
        X_use, y_use = _subsample_xy(X_all, y_all, max_samples, random_state)
        meta = {
            **meta,
            "n_samples_returned": int(X_use.shape[0]),
            "max_samples_applied": max_samples,
        }
        return DatasetBundle(
            X=X_use,
            y=y_use,
            feature_names=list(feature_names),
            true_subspace=None,
            metadata=meta,
        )

    _ensure_data_dir()
    dest = DATA_DIR / "ccpp_data.csv"
    if not dest.exists():
        _download(_CCPP_UCI_CSV, dest)

    rows: list[list[float]] = []
    with open(dest, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                vals = [float(row[k]) for k in ("AT", "V", "AP", "RH", "PE")]
            except (KeyError, ValueError):
                continue
            rows.append(vals)
    if not rows:
        raise RuntimeError(f"No valid rows parsed from {dest}")

    mat = np.array(rows, dtype=np.float64)
    X_all, y_all = mat[:, :4], mat[:, 4]

    metadata: dict[str, Any] = {
        "source": "UCI Combined Cycle Power Plant (id 294)",
        "url": _CCPP_UCI_CSV,
        "n_samples_full": int(X_all.shape[0]),
        "n_features": 4,
        "target": "PE",
        "is_real": True,
    }
    np.savez(
        cache_path,
        X=X_all,
        y=y_all,
        feature_names=np.array(feature_names),
        metadata=np.array(metadata),
    )
    X_use, y_use = _subsample_xy(X_all, y_all, max_samples, random_state)
    metadata_out = {
        **metadata,
        "n_samples_returned": int(X_use.shape[0]),
        "max_samples_applied": max_samples,
    }
    logger.info("CCPP: returning %d samples (full %d)", X_use.shape[0], X_all.shape[0])
    return DatasetBundle(
        X=X_use,
        y=y_use,
        feature_names=feature_names,
        true_subspace=None,
        metadata=metadata_out,
    )


def load_semi_synthetic_real_x(
    base: str = "california_housing",
    max_samples: int | None = 3000,
    random_state: int = 42,
    d_true: int = 3,
    b_true_seed: int = 0,
    noise_std: float = 1.0,
    **_extra: Any,
) -> DatasetBundle:
    """Real covariates only; ``y`` is a placeholder.

    After :func:`sp_mave.preprocessing.split_bundle`, responses are
    ``y = g(X_std @ B_true) + ε`` with train-only scaling (see
    preprocessing module).  ``true_subspace`` is set during splitting.

    Parameters
    ----------
    base
        ``"california_housing"`` or ``"ccpp"`` (same raw ``X`` sources as
        the housing / energy tracks).
    """
    if base == "california_housing":
        inner = load_california_housing_regression(
            max_samples=max_samples, random_state=random_state
        )
    elif base in ("ccpp", "combined_cycle_power_plant"):
        inner = load_combined_cycle_power_plant(
            max_samples=max_samples, random_state=random_state
        )
    else:
        raise ValueError(
            f"semi_synthetic base must be 'california_housing' or 'ccpp', got {base!r}"
        )
    X, fn = inner.X, inner.feature_names
    y = np.zeros(X.shape[0], dtype=np.float64)
    meta: dict[str, Any] = {
        "semi_synthetic": True,
        "base": base,
        "d_true": int(d_true),
        "b_true_seed": int(b_true_seed),
        "noise_std": float(noise_std),
        "g": "g_friedman_adapted",
        "description": (
            "Train/test split on raw X; StandardScaler fit on train only; "
            "B_true orthonormal QR of Gaussian with b_true_seed; "
            "y = g(Z)+noise on each split."
        ),
    }
    return DatasetBundle(
        X=X,
        y=y,
        feature_names=list(fn),
        true_subspace=None,
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Dispatcher and registry
# ---------------------------------------------------------------------------

def load_from_config(cfg: dict[str, Any]) -> DatasetBundle:
    """Dispatch to the appropriate loader based on config['dataset']."""
    ds_cfg = cfg["dataset"]
    name = ds_cfg["name"]
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(REGISTRY.keys())}"
        )
    loader = REGISTRY[name]
    kwargs = {k: v for k, v in ds_cfg.items() if k != "name"}
    return loader(**kwargs)


REGISTRY: dict[str, DatasetLoader] = {
    "friedman1": load_friedman1,
    "diabetes": load_diabetes,
    "communities_crime": load_communities_crime,
    "tecator": load_tecator,
    "california_housing": load_california_housing_regression,
    "combined_cycle_power_plant": load_combined_cycle_power_plant,
    "semi_synthetic_real_x": load_semi_synthetic_real_x,
}
