"""Machine learning and multiscale analysis for geophysical grids, logs and sections (scikit-learn based)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from .grid_tools import GridData
from .profile import ProfileData
from .seismic import Section, ricker


def _stack(grids: list[GridData], texture: bool = False) -> tuple[np.ndarray, np.ndarray]:
    shapes = {g.values.shape for g in grids}
    if len(shapes) != 1:
        raise ValueError("All input grids must share one shape.")
    feats = [g.values for g in grids]
    if texture:
        feats += [ndimage.generic_filter(g.values, np.std, size=5) for g in grids]
    x = np.column_stack([f.ravel() for f in feats])
    ok = np.isfinite(x).all(1)
    return x, ok


def _standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(0)) / np.where(x.std(0) > 0, x.std(0), 1)


def pca_grids(grids: list[GridData], n_components: int = 3) -> tuple[list[GridData], dict]:
    from sklearn.decomposition import PCA

    x, ok = _stack(grids)
    n = min(n_components, x.shape[1])
    pca = PCA(n).fit(_standardize(x[ok]))
    scores = np.full((x.shape[0], n), np.nan)
    scores[ok] = pca.transform(_standardize(x[ok]))
    ref = grids[0]
    outs = [ref.with_values(scores[:, i].reshape(ref.values.shape), f"PC{i + 1}", "score", {"method": "pca"}) for i in range(n)]
    return outs, {"explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                  "loadings": {f"PC{i + 1}": dict(zip([g.name for g in grids], pca.components_[i].round(4).tolist())) for i in range(n)}}


def pca_table(table: pd.DataFrame, columns: str, n_components: int = 3) -> tuple[pd.DataFrame, dict]:
    from sklearn.decomposition import PCA

    cols = [c.strip() for c in columns.split(",") if c.strip()]
    x = table[cols].apply(pd.to_numeric, errors="coerce")
    ok = x.notna().all(1)
    pca = PCA(min(n_components, len(cols))).fit(_standardize(x[ok].to_numpy()))
    out = table.copy()
    sc = pca.transform(_standardize(x[ok].to_numpy()))
    for i in range(sc.shape[1]):
        out.loc[ok, f"PC{i + 1}"] = sc[:, i]
    return out, {"explained_variance_ratio": pca.explained_variance_ratio_.tolist()}


def kmeans_zones(grids: list[GridData], n_clusters: int = 5, texture: bool = True, seed: int = 0) -> tuple[GridData, dict]:
    """K-means domains from amplitude (+ local-std texture) features."""

    from sklearn.cluster import KMeans

    x, ok = _stack(grids, texture)
    km = KMeans(n_clusters, n_init=10, random_state=seed).fit(_standardize(x[ok]))
    lab = np.full(x.shape[0], np.nan)
    lab[ok] = km.labels_
    ref = grids[0]
    return (ref.with_values(lab.reshape(ref.values.shape), "KMeans_zones", "class", {"method": "kmeans", "k": n_clusters}),
            {"inertia": float(km.inertia_), "cluster_sizes": np.bincount(km.labels_).tolist()})


def self_organizing_map(data: np.ndarray, rows: int = 6, cols: int = 6, iterations: int = 3000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Kohonen SOM with Gaussian neighbourhood and exponentially decaying learning rate/radius; returns (weights, bmu)."""

    rng = np.random.default_rng(seed)
    x = _standardize(data)
    w = rng.normal(0, 1, (rows * cols, x.shape[1]))
    gr, gc = np.divmod(np.arange(rows * cols), cols)
    sigma0, lr0 = max(rows, cols) / 2, 0.5
    for t in range(iterations):
        v = x[rng.integers(len(x))]
        bmu = int(np.argmin(((w - v) ** 2).sum(1)))
        frac = t / iterations
        sigma, lr = sigma0 * np.exp(-3 * frac), lr0 * np.exp(-3 * frac)
        h = np.exp(-((gr - gr[bmu]) ** 2 + (gc - gc[bmu]) ** 2) / (2 * sigma**2))
        w += lr * h[:, None] * (v - w)
    bmus = np.argmin(((x[:, None, :] - w[None]) ** 2).sum(2), 1)
    return w, bmus


def som_facies(grids: list[GridData], rows: int = 5, cols: int = 5, n_facies: int = 6) -> tuple[GridData, GridData, dict]:
    """SOM on grid attributes, then K-means on the SOM prototypes to group neurons into facies (Vesanto & Alhoniemi, 2000)."""

    from sklearn.cluster import KMeans

    x, ok = _stack(grids)
    w, bmu = self_organizing_map(x[ok], rows, cols)
    groups = KMeans(min(n_facies, rows * cols), n_init=10, random_state=0).fit_predict(w)
    node = np.full(x.shape[0], np.nan)
    fac = np.full(x.shape[0], np.nan)
    node[ok], fac[ok] = bmu, groups[bmu]
    ref = grids[0]
    return (ref.with_values(fac.reshape(ref.values.shape), "SOM_facies", "class", {"method": "som"}),
            ref.with_values(node.reshape(ref.values.shape), "SOM_bmu_node", "node", {"method": "som"}),
            {"map_size": [rows, cols], "facies": int(n_facies)})


def som_table(table: pd.DataFrame, columns: str, rows: int = 5, cols: int = 5, n_facies: int = 5) -> pd.DataFrame:
    from sklearn.cluster import KMeans

    names = [c.strip() for c in columns.split(",") if c.strip()]
    x = table[names].apply(pd.to_numeric, errors="coerce")
    ok = x.notna().all(1).to_numpy()
    w, bmu = self_organizing_map(x[ok].to_numpy(), rows, cols)
    groups = KMeans(min(n_facies, rows * cols), n_init=10, random_state=0).fit_predict(w)
    out = table.copy()
    out.loc[ok, "SOM_node"] = bmu
    out.loc[ok, "SOM_facies"] = groups[bmu]
    return out


def svm_classification(table: pd.DataFrame, features: str, label_col: str = "lithology", kernel: str = "rbf", c: float = 10.0,
                       test_fraction: float = 0.3) -> tuple[pd.DataFrame, dict]:
    """Supervised lithology classification with an RBF SVM; rows with empty labels are predicted."""

    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    cols = [f.strip() for f in features.split(",") if f.strip()]
    x = table[cols].apply(pd.to_numeric, errors="coerce")
    ok = x.notna().all(1)
    labeled = ok & table[label_col].notna() & (table[label_col].astype(str) != "")
    xtr, xte, ytr, yte = train_test_split(x[labeled], table.loc[labeled, label_col].astype(str), test_size=test_fraction,
                                          random_state=0, stratify=table.loc[labeled, label_col].astype(str))
    model = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=c)).fit(xtr, ytr)
    pred = model.predict(xte)
    out = table.copy()
    out.loc[ok, "predicted_" + label_col] = model.predict(x[ok])
    labels = sorted(set(ytr) | set(yte))
    return out, {"test_accuracy": float(accuracy_score(yte, pred)), "labels": labels,
                 "confusion_matrix": confusion_matrix(yte, pred, labels=labels).tolist()}


def random_forest_regression(table: pd.DataFrame, features: str, target_col: str = "porosity", n_trees: int = 300,
                             test_fraction: float = 0.3) -> tuple[pd.DataFrame, dict]:
    """Random forest regression (Breiman, 2001) with hold-out R^2 and feature importances."""

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    cols = [f.strip() for f in features.split(",") if f.strip()]
    x = table[cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(table[target_col], errors="coerce")
    ok = x.notna().all(1)
    train = ok & y.notna()
    xtr, xte, ytr, yte = train_test_split(x[train], y[train], test_size=test_fraction, random_state=0)
    rf = RandomForestRegressor(n_trees, random_state=0, oob_score=True).fit(xtr, ytr)
    out = table.copy()
    out.loc[ok, "predicted_" + target_col] = rf.predict(x[ok])
    return out, {"test_r2": float(r2_score(yte, rf.predict(xte))), "oob_r2": float(rf.oob_score_),
                 "feature_importance": dict(zip(cols, rf.feature_importances_.round(4).tolist()))}


# ---------------------------------------------------------------- neural pickers / fault detection
def _windows(trace: np.ndarray, half: int) -> np.ndarray:
    pad = np.pad(trace, half, mode="edge")
    return np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1)


def neural_first_break_picking(gather: Section, window: float = 0.02, training_picks: pd.DataFrame | None = None,
                               hidden: str = "64,32", seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Neural first-break picker: an MLP classifies samples as onset / not onset from normalized windows
    (a dense network standing in for a CNN; no deep-learning framework needed).

    Training labels come from `training_picks` (trace, pick_time) or, if absent, from STA/LTA+AIC picks on the
    highest-SNR half of traces (self-supervised); the network then picks every trace.
    """

    from sklearn.neural_network import MLPClassifier

    from .refraction import pick_first_arrivals

    half = max(int(window / gather.dt), 4)
    if training_picks is None or training_picks.empty:
        auto = pick_first_arrivals(gather)
        snr = [np.abs(gather.data[:, j]).max() / (np.std(gather.data[: max(int(len(gather.data) * 0.05), 5), j]) + 1e-12)
               for j in auto["trace"]]
        training_picks = auto[np.asarray(snr) >= np.median(snr)]
    x, y = [], []
    rng = np.random.default_rng(seed)
    for _, r in training_picks.iterrows():
        tr = gather.data[:, int(r.trace)]
        k = int(round((r.pick_time - gather.t0) / gather.dt))
        w = _windows(tr, half) / max(np.abs(tr).max(), 1e-12)  # trace-normalized windows keep relative amplitude
        pos = [k + d for d in (-1, 0, 1) if 0 <= k + d < len(tr)]
        neg = rng.choice(np.setdiff1d(np.arange(len(tr)), np.arange(k - 3 * half // 2, k + 3 * half // 2)), size=10, replace=False)
        x += [w[i] for i in pos] + [w[i] for i in neg]
        y += [1] * len(pos) + [0] * len(neg)
    clf = MLPClassifier(tuple(int(v) for v in hidden.split(",")), max_iter=600, random_state=seed).fit(np.array(x), np.array(y))
    rows = []
    for j in range(gather.ntr):
        tr = gather.data[:, j]
        prob = clf.predict_proba(_windows(tr, half) / max(np.abs(tr).max(), 1e-12))[:, 1]
        k = int(np.argmax(prob))
        rows.append({"trace": j, "offset": float(gather.x[j]), "pick_time": gather.t0 + k * gather.dt, "confidence": float(prob[k])})
    return pd.DataFrame(rows), {"training_examples": len(y), "training_accuracy": float(clf.score(np.array(x), np.array(y)))}


def _synthetic_fault_patches(n: int, size: int, rng) -> tuple[np.ndarray, np.ndarray]:
    patches, labels = [], []
    w = ricker(25, 0.004)
    for i in range(n):
        refl = rng.normal(0, 1, size * 3) * (rng.random(size * 3) < 0.2)
        trace = np.convolve(refl, w, "same")
        img = np.array([np.roll(trace, int(rng.integers(-1, 2))) for _ in range(size)]).T[size: 2 * size]
        dip = rng.uniform(-0.3, 0.3)
        img = np.array([np.roll(img[:, j], int(dip * j)) for j in range(size)]).T
        fault = i % 2 == 1
        if fault:
            throw = int(rng.integers(3, 8)) * (1 if rng.random() < 0.5 else -1)
            col = int(rng.integers(size // 3, 2 * size // 3))
            img[:, col:] = np.roll(img[:, col:], throw, axis=0)
        patches.append((img / max(np.abs(img).max(), 1e-12)).ravel())
        labels.append(int(fault))
    return np.array(patches), np.array(labels)


def fault_detection(section: Section, patch: int = 16, stride: int = 4, n_training: int = 1200, seed: int = 0) -> tuple[Section, dict]:
    """Fault probability section from a patch classifier trained on synthetic faulted/unfaulted reflectivity
    (Wu et al., 2019 idea; MLP instead of a CNN, trained on-the-fly on synthetic structural data)."""

    from sklearn.neural_network import MLPClassifier

    rng = np.random.default_rng(seed)
    x, y = _synthetic_fault_patches(n_training, patch, rng)
    clf = MLPClassifier((128, 32), max_iter=300, random_state=seed).fit(x, y)
    d = section.data
    prob = np.zeros_like(d)
    count = np.zeros_like(d)
    for i in range(0, d.shape[0] - patch + 1, stride):
        for j in range(0, d.shape[1] - patch + 1, stride):
            p = d[i:i + patch, j:j + patch]
            pr = clf.predict_proba((p / max(np.abs(p).max(), 1e-12)).ravel()[None])[0, 1]
            prob[i:i + patch, j:j + patch] += pr
            count[i:i + patch, j:j + patch] += 1
    return (section.derive(prob / np.maximum(count, 1), f"{section.name}_fault_probability", units="probability",
                           metadata={"method": "patch_classifier_fault_detection"}),
            {"synthetic_training_accuracy": float(clf.score(x, y)), "patch": patch})


# ---------------------------------------------------------------- multiscale / fractal
def starlet_denoise(values: np.ndarray, levels: int = 4, k_sigma: float = 3.0) -> np.ndarray:
    """Isotropic undecimated (a trous) B3-spline wavelet (starlet; Starck et al., 2007) hard-threshold denoising, 1D or 2D."""

    kernel = np.array([1, 4, 6, 4, 1], float) / 16
    c = np.asarray(values, float)
    details = []
    for j in range(levels):
        k = np.zeros(4 * 2**j + 1)
        k[:: 2**j] = kernel
        smooth = c
        for ax in range(c.ndim):
            smooth = ndimage.convolve1d(smooth, k, axis=ax, mode="mirror")
        w = c - smooth
        sigma = np.median(np.abs(w - np.median(w))) / 0.6745
        details.append(np.where(np.abs(w) > k_sigma * sigma, w, 0.0))
        c = smooth
    return c + sum(details)


def wavelet_filter_profile(profile: ProfileData, levels: int = 4, k_sigma: float = 3.0) -> tuple[ProfileData, ProfileData]:
    clean = starlet_denoise(profile.values, levels, k_sigma)
    return (profile.with_values(profile.distance, clean, f"{profile.name}_wavelet", {"method": "starlet_denoise"}),
            profile.with_values(profile.distance, profile.values - clean, f"{profile.name}_wavelet_noise", {"method": "starlet_noise"}))


def wavelet_filter_grid(grid: GridData, levels: int = 4, k_sigma: float = 3.0) -> GridData:
    return grid.with_values(starlet_denoise(grid.values, levels, k_sigma), f"{grid.name}_wavelet", metadata={"method": "starlet_denoise"})


def fractal_dimension(grid: GridData, max_lag: int = 10) -> tuple[ProfileData, dict]:
    """Variogram method for surfaces: gamma(h) ~ h^(2H), D = 3 - H (Mandelbrot; Turcotte, 1997)."""

    v = grid.values
    lags, gam = [], []
    for h in range(1, max_lag + 1):
        dx = (v[:, h:] - v[:, :-h]) ** 2
        dy = (v[h:, :] - v[:-h, :]) ** 2
        gam.append(0.5 * np.nanmean(np.concatenate([dx.ravel(), dy.ravel()])))
        lags.append(h * np.sqrt(grid.dx * grid.dy))
    lags, gam = np.array(lags), np.array(gam)
    slope = np.polyfit(np.log(lags), np.log(gam), 1)[0]
    hurst = np.clip(slope / 2, 0, 1)
    return (ProfileData(lags, gam, name="fractal_variogram", units="semivariance", metadata={"axis": "lag_m", "log_axes": True}),
            {"hurst_exponent": float(hurst), "fractal_dimension": float(3 - hurst)})


def fractal_dimension_map(grid: GridData, window: int = 16, step: int = 4, max_lag: int = 5) -> GridData:
    """Moving-window fractal dimension map to highlight changes in texture/complexity (tectonic domains)."""

    ny, nx = grid.values.shape
    ys, xs = list(range(0, ny - window + 1, step)), list(range(0, nx - window + 1, step))
    out = np.full((len(ys), len(xs)), np.nan)
    for a, i in enumerate(ys):
        for b, j in enumerate(xs):
            sub = GridData(grid.values[i:i + window, j:j + window], grid.x_vector[j:j + window], grid.y_vector[i:i + window], grid.dx, grid.dy)
            out[a, b] = fractal_dimension(sub, max_lag)[1]["fractal_dimension"]
    cx = np.array([grid.x_vector[j:j + window].mean() for j in xs])
    cy = np.array([grid.y_vector[i:i + window].mean() for i in ys])
    return GridData(out, cx, cy, grid.dx * step, grid.dy * step, name=f"{grid.name}_fractal_D", units="D",
                    metadata={"method": "fractal_dimension_map", "window_cells": window})
