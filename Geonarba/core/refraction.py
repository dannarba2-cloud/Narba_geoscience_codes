"""Seismic refraction: first-break picking, plus-minus, GRM, shortest-path ray tracing and tomography."""

from __future__ import annotations

from math import gcd

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import lsqr

from .grid_tools import GridData
from .seismic import Section


# ---------------------------------------------------------------- picking
def sta_lta(trace: np.ndarray, n_sta: int, n_lta: int) -> np.ndarray:
    """Classic STA/LTA energy ratio (trailing windows)."""

    e = np.concatenate([[0.0], np.cumsum(trace.astype(float) ** 2)])
    idx = np.arange(len(trace))
    ok = idx >= n_lta
    ratio = np.zeros(len(trace))
    sta = (e[idx[ok] + 1] - e[idx[ok] + 1 - n_sta]) / n_sta
    lta = (e[idx[ok] + 1] - e[idx[ok] + 1 - n_lta]) / n_lta
    ratio[ok] = np.where(lta > 0, sta / np.maximum(lta, 1e-30), 0.0)
    return ratio


def aic_pick(segment: np.ndarray) -> int:
    """Maeda (1985) AIC: k log var(x[:k]) + (N-k-1) log var(x[k:]); pick = argmin."""

    n = len(segment)
    aic = np.full(n, np.inf)
    for k in range(2, n - 2):
        v1, v2 = np.var(segment[:k]), np.var(segment[k:])
        if v1 > 0 and v2 > 0:
            aic[k] = k * np.log(v1) + (n - k - 1) * np.log(v2)
    return int(np.argmin(aic))


def pick_first_arrivals(
    gather: Section, method: str = "sta_lta+aic", sta: float = 0.005, lta: float = 0.05, threshold: float = 3.0,
) -> pd.DataFrame:
    """First-break picks per trace: STA/LTA trigger, optionally refined by AIC in a window around the trigger."""

    ns, nl = max(int(sta / gather.dt), 1), max(int(lta / gather.dt), 2)
    rows = []
    for j in range(gather.ntr):
        tr = gather.data[:, j]
        if method == "aic":
            k = aic_pick(tr)
        else:
            ratio = sta_lta(tr, ns, nl)
            hits = np.where(ratio > threshold)[0]
            if hits.size == 0:
                continue
            k = int(hits[0])
            if "aic" in method:
                lo, hi = max(k - 2 * nl, 0), min(k + nl, len(tr))
                k = lo + aic_pick(tr[lo:hi])
        rows.append({"trace": j, "offset": float(gather.x[j]), "pick_time": gather.t0 + k * gather.dt})
    return pd.DataFrame(rows, columns=["trace", "offset", "pick_time"])


# ---------------------------------------------------------------- plus-minus / GRM
def plus_minus(
    table: pd.DataFrame, x_col: str = "x", tf_col: str = "t_forward", tr_col: str = "t_reverse",
    v1: float = 500.0, reciprocal_time: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Hagedoorn plus-minus: t- = tf - tr + T gives v2 (slope 2/v2); t+ = tf + tr - T gives z = t+ v1 v2 / (2 sqrt(v2^2 - v1^2))."""

    x = table[x_col].to_numpy(float)
    tf = table[tf_col].to_numpy(float)
    tr = table[tr_col].to_numpy(float)
    t_ab = reciprocal_time if reciprocal_time is not None else float(tf[np.argmax(x)])
    minus = tf - tr + t_ab
    slope = np.polyfit(x, minus, 1)[0]
    v2 = 2.0 / slope
    if v2 <= v1:
        raise ValueError(f"Refractor velocity {v2:.0f} m/s must exceed v1 {v1:.0f} m/s; check picks or v1.")
    plus = tf + tr - t_ab
    depth = plus * v1 * v2 / (2 * np.sqrt(v2**2 - v1**2))
    out = pd.DataFrame({"x": x, "plus_time": plus, "minus_time": minus, "depth": depth})
    return out, {"v2_m_per_s": float(v2), "reciprocal_time_s": t_ab, "mean_depth_m": float(np.mean(depth))}


def generalized_reciprocal_method(
    table: pd.DataFrame, x_col: str = "x", tf_col: str = "t_forward", tr_col: str = "t_reverse",
    v1: float = 500.0, xy_values: str = "0,2,4,6,8,10,15,20", reciprocal_time: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Palmer (1980) GRM. tV(G) = [tf(Y) - tr(X) + T]/2, tG = [tf(Y) + tr(X) - (T + XY/V')]/2,
    z = tG V V' / sqrt(V'^2 - V^2). Optimum XY maximizes linearity of the velocity-analysis function."""

    x = table[x_col].to_numpy(float)
    tf = table[tf_col].to_numpy(float)
    tr = table[tr_col].to_numpy(float)
    t_ab = reciprocal_time if reciprocal_time is not None else float(tf[np.argmax(x)])
    best = None
    scores = {}
    for xy in [float(v) for v in xy_values.split(",") if v.strip()]:
        g = x[(x - xy / 2 >= x.min()) & (x + xy / 2 <= x.max())]
        if g.size < 4:
            continue
        tfy = np.interp(g + xy / 2, x, tf)
        trx = np.interp(g - xy / 2, x, tr)
        tv = (tfy - trx + t_ab) / 2
        coef = np.polyfit(g, tv, 1)
        rough = float(np.std(tv - np.polyval(coef, g)))
        scores[xy] = rough
        if best is None or rough < best[0]:
            best = (rough, xy, g, tfy, trx, coef)
    if best is None:
        raise ValueError("No XY value leaves enough stations; reduce XY values.")
    _, xy, g, tfy, trx, coef = best
    vp = 1.0 / coef[0]
    if vp <= v1:
        raise ValueError("Refractor velocity from GRM must exceed v1.")
    tg = (tfy + trx - (t_ab + xy / vp)) / 2
    depth = tg * v1 * vp / np.sqrt(vp**2 - v1**2)
    return (pd.DataFrame({"x": g, "time_depth": tg, "depth": depth}),
            {"optimum_XY_m": xy, "refractor_velocity_m_per_s": float(vp), "velocity_function_roughness": scores})


# ---------------------------------------------------------------- shortest path ray tracing / tomography
def _stencil(radius: int) -> list[tuple[int, int]]:
    out = []
    for di in range(0, radius + 1):
        for dj in range(-radius, radius + 1):
            if (di == 0 and dj <= 0) or gcd(di, abs(dj)) != 1:
                continue
            out.append((di, dj))
    return out


def _graph(model: GridData, slowness: np.ndarray, radius: int = 3) -> sparse.csr_matrix:
    ny, nx = slowness.shape
    idx = np.arange(nx * ny).reshape(ny, nx)
    rows, cols, w = [], [], []
    for di, dj in _stencil(radius):
        jlo, jhi = max(0, -dj), nx - max(0, dj)
        a = idx[: ny - di, jlo:jhi]
        b = idx[di:, jlo + dj: jhi + dj]
        length = np.hypot(di * model.dy, dj * model.dx)
        sa, sb = slowness.ravel()[a.ravel()], slowness.ravel()[b.ravel()]
        rows.append(a.ravel())
        cols.append(b.ravel())
        w.append(length * 0.5 * (sa + sb))
    return sparse.csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(nx * ny, nx * ny))


def _node(model: GridData, x: float, z: float) -> int:
    j = int(np.argmin(np.abs(model.x_vector - x)))
    i = int(np.argmin(np.abs(model.y_vector - z)))
    return i * model.values.shape[1] + j


def traveltime_field(velocity: GridData, source_x: float, source_z: float, radius: int = 3) -> np.ndarray:
    """First-arrival traveltimes (s) from a source by Dijkstra shortest paths (Moser, 1991). y axis = depth."""

    graph = _graph(velocity, 1.0 / velocity.values, radius)
    t = dijkstra(graph, directed=False, indices=_node(velocity, source_x, source_z))
    return t.reshape(velocity.values.shape)


def refraction_tomography(
    picks: pd.DataFrame,
    source_col: str = "source_x",
    receiver_col: str = "receiver_x",
    time_col: str = "pick_time",
    max_depth: float = 50.0,
    cell_size: float = 2.0,
    v_top: float = 400.0,
    v_bottom: float = 3000.0,
    iterations: int = 8,
    smoothing: float = 5.0,
    v_min: float = 150.0,
    v_max: float = 6000.0,
) -> tuple[GridData, GridData, dict]:
    """Shortest-path first-arrival tomography: linearized damped/smoothed LSQR slowness updates (SIRT-like)."""

    s_all = picks[source_col].to_numpy(float)
    r_all = picks[receiver_col].to_numpy(float)
    t_obs = picks[time_col].to_numpy(float)
    x = np.arange(min(s_all.min(), r_all.min()), max(s_all.max(), r_all.max()) + cell_size / 2, cell_size)
    z = np.arange(0, max_depth + cell_size / 2, cell_size)
    vel = np.tile(np.linspace(v_top, v_bottom, len(z))[:, None], (1, len(x)))
    model = GridData(vel, x, z, cell_size, cell_size, name="tomography_velocity", units="m/s",
                     metadata={"vertical_axis": "depth"})
    ny, nx = vel.shape
    n = nx * ny
    lap = sparse.vstack([
        sparse.kron(sparse.identity(ny), sparse.diags([1, -2, 1], [-1, 0, 1], shape=(nx, nx))),
        sparse.kron(sparse.diags([1, -2, 1], [-1, 0, 1], shape=(ny, ny)), sparse.identity(nx)),
    ]).tocsr()
    rms_hist = []
    coverage = np.zeros(n)
    for _ in range(iterations):
        slow = 1.0 / model.values
        graph = _graph(model, slow)
        rows, cols, vals, t_calc = [], [], [], np.zeros(len(t_obs))
        coverage[:] = 0
        for src in np.unique(s_all):
            sel = np.where(s_all == src)[0]
            dist, pred = dijkstra(graph, directed=False, indices=_node(model, src, 0.0), return_predecessors=True)
            for p in sel:
                node = _node(model, r_all[p], 0.0)
                t_calc[p] = dist[node]
                while pred[node] >= 0:
                    prev = pred[node]
                    i1, j1 = divmod(node, nx)
                    i0, j0 = divmod(prev, nx)
                    seg = np.hypot((i1 - i0) * cell_size, (j1 - j0) * cell_size)
                    rows += [p, p]
                    cols += [node, prev]
                    vals += [seg / 2, seg / 2]
                    node = prev
        gmat = sparse.csr_matrix((vals, (rows, cols)), shape=(len(t_obs), n))
        coverage = np.asarray(gmat.sum(axis=0)).ravel()
        resid = t_obs - t_calc
        rms_hist.append(float(np.sqrt(np.mean(resid**2))))
        a = sparse.vstack([gmat, smoothing * cell_size * lap])
        ds = lsqr(a, np.concatenate([resid, np.zeros(lap.shape[0])]), damp=1e-4, iter_lim=500)[0]
        new_slow = np.clip(slow.ravel() + ds, 1 / v_max, 1 / v_min)
        model = model.with_values((1.0 / new_slow).reshape(ny, nx), name="tomography_velocity")
    cov = model.with_values(coverage.reshape(ny, nx), name="tomography_ray_coverage", units="m")
    return model, cov, {"rms_history_s": rms_hist, "n_picks": int(len(t_obs)), "cells": n}
