"""Declarative catalogue of every GEONARBA method.

Each Method declares its inputs (data objects from the session), typed parameters (with choices = sub-options)
and a `run` callable. The desktop toolbox, menus and parameter forms are generated from this list, and the
test-suite runs every entry on its demo dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.optimize import least_squares

from . import depth, electrical, em, gis, gravity, interpolation, inversion, ml, modeling, optimize, passive, potential_field
from . import refraction, seismic, seismic_wave, wells
from .analytic_signal import compute_analytic_signal
from .derivatives import compute_thg, compute_vertical_derivative_fft
from .edge_detection import compute_fsed, threshold_edges
from .fft_filters import compute_butterworth_filter, compute_residual_from_upward
from .grid_operations import crop_grid, extended_grid_statistics, flip_grid, rotate_grid
from .grid_tools import GridData
from .profile import (ProfileData, extract_cross_section, interpolate_profile, polynomial_fit_profile,
                      remove_linear_trend_profile, running_average_profile)
from .spectral import compute_raps, fit_raps_segment
from .tilt import compute_tilt_derivative

INPUT_KINDS = ("grid", "grids", "profile", "section", "sections", "table")
PADDING = ("reflect", "zero", "none")


@dataclass
class Param:
    name: str
    label: str
    default: Any
    choices: tuple = ()
    help: str = ""
    kind: str = ""  # float, int, bool, choice, text, column
    source: str = ""  # for kind == 'column': name of the table input

    def __post_init__(self) -> None:
        if not self.kind:
            if self.choices:
                self.kind = "choice"
            elif isinstance(self.default, bool):
                self.kind = "bool"
            elif isinstance(self.default, int):
                self.kind = "int"
            elif isinstance(self.default, float):
                self.kind = "float"
            else:
                self.kind = "text"


@dataclass
class Input:
    name: str
    kind: str
    label: str
    optional: bool = False
    demo: str | tuple[str, ...] = ""


@dataclass
class Method:
    number: str
    key: str
    name: str
    domain: str
    category: str
    description: str
    inputs: list[Input]
    params: list[Param]
    run: Callable[..., Any]
    reference: str = ""
    slow: bool = False

    @property
    def label(self) -> str:
        return f"{self.number}. {self.name}" if self.number else self.name


REGISTRY: list[Method] = []


def add(number, key, name, domain, category, description, inputs, params, run, reference="", slow=False) -> None:
    REGISTRY.append(Method(str(number), key, name, domain, category, description, inputs, params, run, reference, slow))


def col(name: str, label: str, default: str, source: str = "table") -> Param:
    return Param(name, label, default, kind="column", source=source)


def pad() -> Param:
    return Param("padding", "FFT padding", "reflect", PADDING, "Reflect padding limits edge ringing.")


def inversion_params(error_percent: float, error_floor: float | None = None, alpha_s: float = 1e-4, alpha_y: bool = False,
                     solver: bool = False) -> list[Param]:
    """Shared Tikhonov/Occam controls: trade-off rule, smoothing operator, anisotropy, data errors (and solver)."""

    from .inversion import LAMBDA_MODES, OPERATORS, SOLVERS

    out = [Param("lambda_mode", "Trade-off (lambda) selection", "discrepancy", LAMBDA_MODES,
                 "discrepancy = Occam target RMS 1 (recommended); l-curve = max-curvature corner; gcv = generalized cross-validation"),
           Param("operator", "Smoothing operator", "first-derivative", OPERATORS,
                 "1st derivative: blocky with transitions; 2nd derivative: smoothest; smallness: closest to reference"),
           Param("alpha_s", "Smallness weight alpha_s", alpha_s), Param("alpha_x", "Horizontal smoothing alpha_x", 1.0)]
    if alpha_y:
        out.append(Param("alpha_y", "Horizontal smoothing alpha_y", 1.0))
    out += [Param("alpha_z", "Vertical smoothing alpha_z", 1.0, help="alpha_z < alpha_x favours layered (sub-horizontal) models"),
            Param("error_percent", "Data error (% of |d|)", error_percent, help="W_d = 1/sigma")]
    if error_floor is not None:
        out.append(Param("error_floor", "Error floor (data units, 0 = auto)", error_floor))
    if solver:
        out.append(Param("solver", "Solver", "cg", SOLVERS, "direct = normal equations; cg = LSQR/CG; subspace = few-vector updates"))
    return out


G = lambda demo="Demo_TMI", name="grid", label="Input grid": Input(name, "grid", label, demo=demo)  # noqa: E731
T = lambda demo, name="table", label="Input table": Input(name, "table", label, demo=demo)  # noqa: E731
S = lambda demo, name="section", label="Seismic / radar section": Input(name, "section", label, demo=demo)  # noqa: E731
PR = lambda demo, name="profile", label="Input profile": Input(name, "profile", label, demo=demo)  # noqa: E731

INC = Param("inclination", "Field inclination (deg)", 30.0)
DEC = Param("declination", "Field declination (deg)", 5.0)
INDUCED = Param("use_induced", "Induced magnetization only", True, help="Untick to give a remanent magnetization direction.")
MINC = Param("mag_inclination", "Magnetization inclination (deg)", 30.0)
MDEC = Param("mag_declination", "Magnetization declination (deg)", 5.0)


def _mag(use_induced, mag_inclination, mag_declination):
    return (None, None) if use_induced else (mag_inclination, mag_declination)


# ================================================================ POTENTIAL FIELDS
PF = "Potential Fields (Gravity & Magnetic)"

add(1, "rtp", "Reduction to the Pole (RTP)", PF, "Magnetic Reduction",
    "Centres magnetic anomalies over sources by simulating a vertical field and magnetization (wavenumber-domain, "
    "Wiener-stabilized for low latitudes).", [G()],
    [INC, DEC, INDUCED, MINC, MDEC, Param("stabilization", "Stabilization epsilon", 0.0, help="0 = exact RTP; 0.1-0.5 for |I| < 20 deg"), pad()],
    lambda grid, inclination, declination, use_induced, mag_inclination, mag_declination, stabilization, padding:
    potential_field.reduce_to_pole(grid, inclination, declination, *_mag(use_induced, mag_inclination, mag_declination), stabilization, padding),
    "Baranov (1957); Blakely (1995)")

add(2, "rte", "Reduction to the Equator (RTE)", PF, "Magnetic Reduction",
    "Transforms data to an equatorial (horizontal) field: stable alternative to RTP at low magnetic latitudes.", [G()],
    [INC, DEC, INDUCED, MINC, MDEC, Param("stabilization", "Stabilization epsilon", 0.0), pad()],
    lambda grid, inclination, declination, use_induced, mag_inclination, mag_declination, stabilization, padding:
    potential_field.reduce_to_equator(grid, inclination, declination, *_mag(use_induced, mag_inclination, mag_declination), stabilization, padding),
    "Leu (1982); Li (2008)")

add(30, "igrf", "IGRF Removal", PF, "Magnetic Reduction",
    "Subtracts the IGRF-14 total field at each station (lon/lat/date) to isolate crustal anomalies.",
    [T("Demo_magnetic_stations")],
    [col("lon_col", "Longitude column", "lon"), col("lat_col", "Latitude column", "lat"), col("field_col", "Total field column (nT)", "total_field"),
     Param("date", "Survey date (YYYY-MM-DD)", "2025-01-01"), Param("height_m", "Sensor height (m)", 0.0)],
    lambda table, lon_col, lat_col, field_col, date, height_m: potential_field.igrf_removal_table(table, lon_col, lat_col, field_col, "", date, height_m),
    "Alken et al. (2021) IGRF; ppigrf")

add("30.1", "igrf_direction", "IGRF Field Direction at a Point", PF, "Magnetic Reduction",
    "Inclination, declination and intensity of the IGRF-14 field (inputs for RTP/RTE/pseudogravity).", [],
    [Param("lon", "Longitude", 10.0), Param("lat", "Latitude", 35.0), Param("height_m", "Height (m)", 0.0), Param("date", "Date", "2025-01-01")],
    lambda lon, lat, height_m, date: potential_field.igrf_field_direction(lon, lat, height_m, __import__("datetime").datetime.fromisoformat(date)))

add(22, "pseudogravity", "Pseudogravity Transformation", PF, "Magnetic Reduction",
    "Poisson relation: converts TMI into the gravity anomaly of the same bodies for a given density/magnetization ratio.", [G()],
    [INC, DEC, Param("density_contrast", "Density contrast (kg/m3)", 100.0), Param("magnetization", "Magnetization (A/m)", 1.0),
     INDUCED, MINC, MDEC, Param("stabilization", "Stabilization", 0.01), pad()],
    lambda grid, inclination, declination, density_contrast, magnetization, use_induced, mag_inclination, mag_declination, stabilization, padding:
    potential_field.pseudogravity(grid, inclination, declination, density_contrast, magnetization,
                                  *_mag(use_induced, mag_inclination, mag_declination), stabilization, padding),
    "Baranov (1957); Blakely (1995)")

_GRAV_COLS = [col("lat_col", "Latitude column", "lat"), col("height_col", "Elevation column (m)", "elevation"),
              col("gravity_col", "Observed gravity column (mGal)", "gravity")]

add(18, "free_air", "Free-Air Anomaly", PF, "Gravity Reduction",
    "FA = g_obs - gamma(lat) + FAC(h) with Somigliana normal gravity and 1st/2nd-order free-air gradient.",
    [T("Demo_gravity_stations")],
    _GRAV_COLS + [Param("ellipsoid", "Reference ellipsoid", "GRS80", ("GRS80", "WGS84")), Param("second_order_free_air", "2nd-order free-air (Hinze 2005)", True)],
    lambda table, lat_col, height_col, gravity_col, ellipsoid, second_order_free_air:
    gravity.gravity_anomalies(table, lat_col, height_col, gravity_col, 2670.0, ellipsoid, second_order_free_air)
    .drop(columns=["bouguer_correction_mGal", "simple_bouguer_anomaly_mGal"]),
    "Hinze et al. (2005)")

add(17, "bouguer", "Bouguer Anomaly", PF, "Gravity Reduction",
    "Simple (slab) and complete Bouguer anomaly with optional Bullard-B curvature and terrain-correction column.",
    [T("Demo_gravity_stations")],
    _GRAV_COLS + [Param("density", "Reduction density (kg/m3)", 2670.0), Param("ellipsoid", "Reference ellipsoid", "GRS80", ("GRS80", "WGS84")),
                  Param("curvature", "Bullard B curvature correction", False),
                  Param("terrain_col", "Terrain-correction column (optional)", "", help="Leave empty for simple Bouguer")],
    lambda table, lat_col, height_col, gravity_col, density, ellipsoid, curvature, terrain_col:
    gravity.gravity_anomalies(table, lat_col, height_col, gravity_col, density, ellipsoid, True, curvature, terrain_col),
    "LaFehr (1991); Hinze et al. (2005)")

add(19, "terrain", "Terrain Correction", PF, "Gravity Reduction",
    "Prism-based terrain correction from a DEM, at stations (if a table is given) or at every DEM node.",
    [G("Demo_DEM", "dem", "DEM grid (m)"), Input("table", "table", "Stations (optional)", optional=True, demo="Demo_gravity_stations")],
    [Param("density", "Density (kg/m3)", 2670.0), Param("radius", "Outer radius (m)", 8000.0), col("x_col", "X column", "x"),
     col("y_col", "Y column", "y"), col("height_col", "Elevation column", "elevation"), Param("stride", "Grid-node stride (DEM mode)", 4)],
    lambda dem, table, density, radius, x_col, y_col, height_col, stride:
    gravity.terrain_correction_table(table, dem, x_col, y_col, height_col, density, radius) if table is not None
    else gravity.terrain_correction_grid(dem, density, radius, stride),
    "Hammer (1939); Nagy (1966)", slow=True)

add(20, "isostatic", "Isostatic Residual Anomaly (Airy)", PF, "Gravity Reduction",
    "Removes the gravity effect of Airy compensating roots computed from topography.",
    [G("Demo_Bouguer", "bouguer", "Bouguer anomaly grid"), G("Demo_DEM", "topography", "Topography grid (m)")],
    [Param("crust_density", "Crust density", 2670.0), Param("mantle_density", "Mantle density", 3270.0),
     Param("compensation_depth", "Normal crust thickness Tc (m)", 30000.0), pad()],
    lambda bouguer, topography, crust_density, mantle_density, compensation_depth, padding:
    gravity.isostatic_residual(bouguer, topography, crust_density, mantle_density, 1030.0, compensation_depth, padding),
    "Heiskanen & Vening Meinesz (1958); Simpson et al. (1986)")

add(29, "drift", "Microgravity Drift & Earth-Tide Correction", PF, "Gravity Reduction",
    "Longman (1959) tide plus piecewise-linear instrument drift from base-station re-occupations.",
    [T("Demo_gravity_drift_survey")],
    [col("time_col", "Time column (UTC)", "time"), col("station_col", "Station column", "station"), col("reading_col", "Reading column (mGal)", "reading"),
     Param("base_station", "Base station name", "BASE"), Param("latitude", "Latitude", 35.0), Param("longitude", "Longitude", 10.0),
     Param("apply_tide", "Apply Longman tide", True)],
    gravity.drift_tide_correction, "Longman (1959)")

add(3, "upward", "Upward Continuation", PF, "Continuation & Filtering",
    "Regional field at height h: IFFT(FFT(T) e^{-|k|h}); the residual is Original - Regional.", [G()],
    [Param("height", "Continuation height (m)", 2000.0), pad(), Param("remove_mean", "Remove mean first", False)],
    lambda grid, height, padding, remove_mean: compute_residual_from_upward(grid, height, padding, remove_mean), "Blakely (1995)")

add(4, "downward", "Downward Continuation", PF, "Continuation & Filtering",
    "Regularized (Wiener/Tikhonov) downward continuation: enhances shallow features; unstable with noise.", [G()],
    [Param("depth", "Continuation depth (m)", 300.0), Param("regularization", "Regularization alpha", 1e-3), pad()],
    potential_field.downward_continuation, "Pasteka et al. (2012)")


def _butterworth(grid, filter_type, cutoff, cutoff_short, order, padding, output_residual):
    c = (cutoff, cutoff_short) if filter_type == "bandpass" else cutoff
    out = compute_butterworth_filter(grid, c, order, filter_type, padding=padding)
    if output_residual and filter_type == "lowpass":
        return out, grid.with_values(grid.values - out.values, f"{grid.name}_BW_residual_{cutoff:g}m", metadata={"method": "butterworth_residual"})
    return out


add(21, "butterworth", "Wavelength Filtering (Butterworth)", PF, "Continuation & Filtering",
    "Radial Butterworth low/high/band-pass filters for regional-residual separation.", [G()],
    [Param("filter_type", "Filter type", "lowpass", ("lowpass", "highpass", "bandpass")), Param("cutoff", "Cutoff / long wavelength (m)", 8000.0),
     Param("cutoff_short", "Short wavelength for band-pass (m)", 2000.0), Param("order", "Order", 4), pad(),
     Param("output_residual", "Also output residual (low-pass)", True)],
    _butterworth, "Butterworth (1930)")

add(26, "matched", "Matched Filtering (Equivalent Layers)", PF, "Continuation & Filtering",
    "Fits the radial spectrum with equivalent source layers and separates the field into depth slices.", [G()],
    [Param("n_layers", "Number of equivalent layers", 2), pad()],
    potential_field.matched_filter, "Syberg (1972); Phillips (2001)")

add(5, "fvd", "First Vertical Derivative (FVD)", PF, "Derivatives & Edge Detection",
    "dT/dz = IFFT(|k| FFT(T)); sharpens anomalies over shallow sources.", [G()], [pad()],
    lambda grid, padding: compute_vertical_derivative_fft(grid, 1, padding))

add(6, "svd", "Second Vertical Derivative (SVD)", PF, "Derivatives & Edge Detection",
    "d2T/dz2 = IFFT(|k|^2 FFT(T)); resolves closely spaced shallow sources (noise sensitive).", [G()], [pad()],
    lambda grid, padding: compute_vertical_derivative_fft(grid, 2, padding))

add(7, "thdr", "Total Horizontal Derivative (THDR)", PF, "Derivatives & Edge Detection",
    "sqrt((dT/dx)^2 + (dT/dy)^2): maxima over body edges (best on RTP or gravity data).", [G()], [],
    compute_thg, "Cordell & Grauch (1985)")

add(8, "as", "Analytic Signal (AS)", PF, "Derivatives & Edge Detection",
    "|A| = sqrt(Tx^2 + Ty^2 + Tz^2); maxima over edges largely independent of magnetization direction.", [G()], [pad()],
    compute_analytic_signal, "Nabighian (1972); Roest et al. (1992)")


def _tilt(grid, padding, units):
    t = compute_tilt_derivative(grid, padding=padding)
    return t if units == "radians" else t.with_values(np.degrees(t.values), t.name + "_deg", "deg")


add(9, "tdr", "Tilt Angle Derivative (TDR)", PF, "Derivatives & Edge Detection",
    "arctan(VDR / THDR): equalizes strong and weak anomalies; zero contour tracks contacts.", [G()],
    [pad(), Param("units", "Output units", "degrees", ("degrees", "radians"))], _tilt, "Miller & Singh (1994)")

add(10, "theta", "Theta Map (cos TDR)", PF, "Derivatives & Edge Detection",
    "cos(theta) = THDR / |A|: normalized edge map, amplitude-balanced.", [G()], [pad()],
    potential_field.theta_map, "Wijns et al. (2005)")


def _fsed(grid, base, method, gain, threshold):
    out = compute_fsed(grid, base=base, gain=gain, method=method)
    return (out, threshold_edges(out, threshold)) if threshold > 0 else out


add(11, "fsed", "Fast Sigmoid Edge Detection (FSED)", PF, "Derivatives & Edge Detection",
    "Bounded sigmoid of the vertical-to-horizontal derivative ratio of THDR: sharp, balanced contact maps.", [G()],
    [Param("base", "Base function", "THG ratio", ("THG ratio", "THG", "ASA", "absolute TILT", "direct input")),
     Param("method", "Sigmoid", "fast_sigmoid", ("fast_sigmoid", "logistic")), Param("gain", "Logistic gain", 5.0),
     Param("threshold", "Binary edge threshold (0 = none)", 0.0)],
    _fsed, "Pham et al. (2020, 2021)")


def _euler(grid, structural_index, window, max_depth_error):
    return depth.euler_deconvolution(grid, float(structural_index.split()[0]), window, None, max_depth_error)


SI = ("0 (contact)", "0.5 (thick step)", "1 (dike/sill, mag) / contact (grav)", "2 (pipe, mag) / sphere-like (grav)", "3 (sphere, mag)")
add(12, "euler", "3D Euler Deconvolution", PF, "Depth Estimation",
    "Solves Euler's homogeneity equation in moving windows for source position and depth.", [G("Demo_Bouguer")],
    [Param("structural_index", "Structural index", SI[3], SI), Param("window", "Window (cells)", 10),
     Param("max_depth_error", "Max relative depth uncertainty", 0.15)],
    _euler, "Reid et al. (1990); Thompson (1982)")

add(13, "werner", "Werner Deconvolution", PF, "Depth Estimation",
    "Thin-sheet (dike) or contact depth solutions from sliding windows along a profile.",
    [PR("Demo_TMI_profile")], [Param("window", "Window (samples)", 7), Param("model", "Source model", "dike", ("dike", "contact")),
                               Param("min_depth", "Minimum depth (m)", 0.0)],
    depth.werner_deconvolution, "Werner (1953); Hartman et al. (1971)")


def _raps(grid, window, detrend, auto_fit, k_min, k_max):
    res = compute_raps(grid, detrend=detrend, window=window)
    fit = fit_raps_segment(res, k_min or None, k_max or None, automatic=auto_fit)
    prof = ProfileData(res.k, res.log_power, name=f"{grid.name}_RAPS", units="ln(P)", metadata={"axis": "k_rad_per_m", "method": "raps"})
    return prof, {"slope": fit.slope, "r_squared": fit.r_squared, "depth_m": fit.depth_m, "k_range": (fit.k_min, fit.k_max),
                  "formula": fit.formula}


add(14, "raps", "Radially Averaged Power Spectrum", PF, "Depth Estimation",
    "Ensemble source depths from the slope of ln(P) vs k: depth = -slope/2.", [G()],
    [Param("window", "Taper", "hanning", ("hanning", "cosine", "none")), Param("detrend", "Remove plane", False),
     Param("auto_fit", "Automatic segment", True), Param("k_min", "Manual k min (rad/m)", 0.0), Param("k_max", "Manual k max (rad/m)", 0.0)],
    _raps, "Spector & Grant (1970)")

add(25, "spi", "Source Parameter Imaging (SPI)", PF, "Depth Estimation",
    "Depth = 1/local wavenumber of the tilt angle (contacts).", [G()],
    [Param("max_depth", "Max depth to display (m)", 10000.0), pad()], depth.source_parameter_imaging, "Thurston & Smith (1997)")

add(27, "tilt_depth", "Tilt-Depth Method", PF, "Depth Estimation",
    "Depth = 1/|grad(tilt)| on the zero tilt contour (half-distance between +-45 deg contours).", [G()],
    [Param("tilt_tolerance_deg", "Zero-contour tolerance (deg)", 2.0), pad()], depth.tilt_depth, "Salem et al. (2007)")


def _cpd(grid, mode, window_size, step, thermal_conductivity, curie_temperature):
    if mode == "single window":
        return depth.curie_point_depth(grid)
    return depth.curie_depth_map(grid, window_size, step, thermal_conductivity, curie_temperature)


add(15, "cpd", "Curie Point Depth (Centroid Method)", PF, "Thermal Analysis",
    "Top (Zt), centroid (Z0) and bottom Zb = 2Z0 - Zt of magnetic sources from spectral slopes; single window or map.",
    [G()], [Param("mode", "Mode", "single window", ("single window", "moving-window map")),
            Param("window_size", "Map window (m)", 20000.0), Param("step", "Map step (m)", 5000.0),
            Param("thermal_conductivity", "Conductivity (W/m/K)", 2.5), Param("curie_temperature", "Curie temperature (C)", 580.0)],
    _cpd, "Okubo et al. (1985); Tanaka et al. (1999)")


def _heat_flow(grid, curie_depth_m, thermal_conductivity, curie_temperature, surface_temperature, heat_production):
    if grid is not None:
        return grid.with_values(depth.heat_flow_from_curie(grid.values, thermal_conductivity, curie_temperature, surface_temperature, heat_production),
                                f"{grid.name}_HeatFlow", "mW/m^2", {"method": "heat_flow"})
    q = float(depth.heat_flow_from_curie(curie_depth_m, thermal_conductivity, curie_temperature, surface_temperature, heat_production))
    return {"heat_flow_mW_m2": q, "geothermal_gradient_C_per_km": (curie_temperature - surface_temperature) / curie_depth_m * 1000}


add(16, "heat_flow", "Heat Flow from Curie Depth", PF, "Thermal Analysis",
    "q = k (Tc - T0)/Zb + A Zb/2 from a Curie-depth grid or a single depth value.",
    [Input("grid", "grid", "Curie depth grid (m, optional)", optional=True)],
    [Param("curie_depth_m", "Curie depth if no grid (m)", 20000.0), Param("thermal_conductivity", "k (W/m/K)", 2.5),
     Param("curie_temperature", "Tc (C)", 580.0), Param("surface_temperature", "T0 (C)", 15.0),
     Param("heat_production", "Heat production A (uW/m3)", 0.0)], _heat_flow, "Tanaka et al. (1999)")


def _forward25(profile, x_min, x_max, spacing, polygon, property_value, strike_half_length, field, inclination, declination):
    return modeling.forward_2_5d_profile(x_min, x_max, spacing, polygon, property_value, strike_half_length, field, inclination, declination, profile)


add(23, "forward25d", "2.5D Forward Modeling", PF, "Modeling & Inversion",
    "Gravity/magnetic response of a polygonal body with finite strike (+-L) using exact prism kernels.",
    [Input("profile", "profile", "Observed profile (optional)", optional=True)],
    [Param("x_min", "Profile start (m)", 0.0), Param("x_max", "Profile end (m)", 20000.0), Param("spacing", "Station spacing (m)", 250.0),
     Param("polygon", "Polygon vertices 'x,z; ...' (z down, m)", "8000,1000; 12000,1000; 13000,3000; 7000,3000"),
     Param("property_value", "Density contrast (kg/m3) or magnetization (A/m)", 300.0),
     Param("strike_half_length", "Strike half-length L (m)", 10000.0), Param("field", "Field", "gravity", ("gravity", "magnetic")), INC, DEC],
    _forward25, "Talwani et al. (1959); Nagy et al. (2000)")

add(24, "voxel", "3D Voxel Inversion", PF, "Modeling & Inversion",
    "Depth-weighted Tikhonov inversion for density or magnetization with automatic lambda, predicted/residual maps, L-curve and QC.",
    [G("Demo_Bouguer")],
    [Param("field", "Data type", "gravity", ("gravity", "magnetic")), Param("n_z", "Depth layers", 6), Param("max_depth", "Mesh depth (m)", 8000.0),
     Param("stride", "Horizontal decimation", 4), Param("depth_weight_beta", "Depth weighting beta", 2.0),
     Param("reference_value", "Reference model value", 0.0), *inversion_params(0.0, 0.1, alpha_y=True, solver=True),
     Param("regularization", "Lambda (fixed mode only)", 1.0), INC, DEC], modeling.voxel_inversion, "Li & Oldenburg (1996, 1998)", slow=True)

add(28, "eqsource", "Equivalent Source Technique", PF, "Modeling & Inversion",
    "Point sources below uneven stations reproduce the data, then predict it on a regular grid at constant height.",
    [T("Demo_gravity_stations")],
    [col("x_col", "X column", "x"), col("y_col", "Y column", "y"), col("z_col", "Station elevation column", "elevation"),
     col("value_col", "Value column", "gravity"), Param("spacing", "Output spacing (m)", 1000.0),
     Param("output_elevation", "Output elevation (m)", 1000.0), Param("source_depth_factor", "Source depth / station spacing", 3.0),
     Param("damping", "Damping", 1e-3)], modeling.equivalent_sources, "Dampney (1969); Cordell (1992)")

# ================================================================ SEISMIC
SE = "Seismic (Reflection, Refraction, Passive)"
VEL = Param("velocity", "RMS velocity (m/s or 't0:v, ...')", "0.35:1800, 0.6:2200, 0.85:2600, 1.1:3000")


def _vel(v):
    return float(v) if ":" not in str(v) else str(v)


add(31, "nmo", "Normal Moveout (NMO) Correction", SE, "Reflection Processing",
    "Flattens reflection hyperbolas with t(x) = sqrt(t0^2 + x^2/v^2); stretch mute.", [S("Demo_CMP_gathers")],
    [VEL, Param("stretch_mute", "Stretch mute (fraction)", 0.5)],
    lambda section, velocity, stretch_mute: seismic.nmo_correction(section, _vel(velocity), stretch_mute), "Yilmaz (2001)")

add(32, "dmo", "Dip Moveout (DMO) Correction", SE, "Reflection Processing",
    "Kirchhoff DMO on NMO-corrected common-offset panels removes reflection-point smear of dipping events.",
    [S("Demo_CMP_gathers")], [Param("apply_nmo", "Apply NMO first", True), VEL, Param("n_b", "Operator samples", 21)],
    lambda section, apply_nmo, velocity, n_b: seismic.dmo_correction(seismic.nmo_correction(section, _vel(velocity)) if apply_nmo else section, n_b=n_b),
    "Deregowski & Rocca (1981); Hale (1984)")

add(33, "stack", "Common Midpoint (CMP) Stacking", SE, "Reflection Processing",
    "NMO + fold-normalized stack per CMP header.", [S("Demo_CMP_gathers")],
    [Param("apply_nmo", "Apply NMO", True), VEL, Param("stretch_mute", "Stretch mute", 0.5)],
    lambda section, apply_nmo, velocity, stretch_mute: seismic.cmp_stack(section, _vel(velocity) if apply_nmo else None, stretch_mute))

add(36, "decon", "Seismic Deconvolution", SE, "Reflection Processing",
    "Wiener spiking or predictive (gap) deconvolution to compress the wavelet.", [S("Demo_stack_section")],
    [Param("mode", "Type", "spiking", ("spiking", "predictive")), Param("operator_length", "Operator length (s)", 0.08),
     Param("prediction_lag", "Prediction lag (s)", 0.016), Param("prewhitening", "Pre-whitening", 0.01)],
    seismic.deconvolution, "Robinson & Treitel (1980)")

add(37, "fk", "f-k Filtering", SE, "Reflection Processing",
    "Velocity-fan filter in the frequency-wavenumber domain (e.g. ground-roll rejection).", [S("Demo_CMP_gathers")],
    [Param("min_velocity", "Cut velocity (m/s)", 1200.0), Param("taper", "Taper fraction", 0.2),
     Param("mode", "Mode", "reject_slow", ("reject_slow", "pass_slow"))], seismic.fk_filter, "Yilmaz (2001)")

add(34, "prestm", "Pre-stack Time Migration (Kirchhoff)", SE, "Imaging",
    "Double-square-root Kirchhoff summation using source/receiver headers and v_rms(t).", [S("Demo_CMP_gathers")],
    [VEL, Param("aperture", "Aperture (m)", 400.0)],
    lambda section, velocity, aperture: seismic.kirchhoff_prestack_time_migration(section, _vel(velocity), aperture), slow=True)

add(35, "presdm", "Pre-stack Depth Migration (Kirchhoff)", SE, "Imaging",
    "Kirchhoff depth imaging with shortest-path traveltime tables through an interval-velocity model.",
    [S("Demo_CMP_gathers", "gathers"), G("Demo_velocity_model", "velocity_model", "Interval velocity model (x, depth)")],
    [Param("aperture", "Aperture (m)", 400.0)], seismic.kirchhoff_prestack_depth_migration, slow=True)

add("34.1", "stolt", "Post-stack Stolt f-k Migration", SE, "Imaging",
    "Constant-velocity exploding-reflector migration of a stacked (zero-offset) section.", [S("Demo_stack_section")],
    [Param("velocity", "Velocity (m/s)", 2000.0)], lambda section, velocity: seismic.stolt_migration(section, velocity), "Stolt (1978)")

add(38, "avo", "AVO Intercept/Gradient Analysis", SE, "Reservoir Characterization",
    "Shuey two-term fit R(theta) = A + B sin^2(theta) per sample of an angle gather (A, B, A*B).", [S("Demo_angle_gather", "angle_gather")], [],
    seismic.avo_intercept_gradient, "Shuey (1985)")

add("38.1", "avo_model", "AVO Modeling (Aki-Richards)", SE, "Reservoir Characterization",
    "Reflection coefficient vs angle for a two-layer interface; Rutherford-Williams class.", [],
    [Param("vp1", "Vp1", 2400.0), Param("vs1", "Vs1", 1200.0), Param("rho1", "Rho1", 2300.0), Param("vp2", "Vp2", 2200.0),
     Param("vs2", "Vs2", 1400.0), Param("rho2", "Rho2", 2100.0), Param("max_angle", "Max angle (deg)", 40.0)], seismic.avo_modeling,
    "Aki & Richards (1980); Rutherford & Williams (1989)")

add(39, "ai", "Acoustic Impedance Inversion", SE, "Reservoir Characterization",
    "Model-based inversion of traces for impedance with a Ricker wavelet and low-frequency prior.", [S("Demo_stack_section")],
    [Param("wavelet_freq", "Wavelet peak frequency (Hz)", 30.0), Param("initial_impedance", "Background impedance", 5e6),
     Param("regularization", "Prior weight", 0.1)],
    lambda section, wavelet_freq, initial_impedance, regularization: seismic.acoustic_impedance_inversion(section, wavelet_freq, initial_impedance, 0.0, regularization),
    "Russell & Hampson (1991)")

add(53, "complex_trace", "Complex Trace Attributes", SE, "Reservoir Characterization",
    "Envelope, instantaneous phase/frequency, cosine of phase and sweetness via the Hilbert transform.",
    [S("Demo_stack_section")], [], seismic.complex_trace_attributes, "Taner et al. (1979)")

add(54, "coherence", "Seismic Coherence / Variance", SE, "Reservoir Characterization",
    "Semblance coherence highlights faults and channel edges (1 = continuous).", [S("Demo_stack_section")],
    [Param("window_traces", "Traces in window", 3), Param("window_time", "Half window (s)", 0.012)], seismic.coherence, "Marfurt et al. (1998)")

add(55, "specdecomp", "Spectral Decomposition", SE, "Reservoir Characterization",
    "STFT amplitude sections at selected frequencies (thin-bed tuning).", [S("Demo_stack_section")],
    [Param("frequencies", "Frequencies (Hz, comma list)", "15,30,45"), Param("window", "Window (s)", 0.064)],
    seismic.spectral_decomposition, "Partyka et al. (1999)")

add(52, "q", "Q-factor Estimation (Spectral Ratio)", SE, "Reservoir Characterization",
    "ln(A2/A1) = -pi f dt / Q + c between two arrivals (e.g. VSP direct waves).", [S("Demo_VSP")],
    [Param("trace_1", "Trace 1", 0), Param("trace_2", "Trace 2", -1), Param("window", "Window (s)", 0.2),
     Param("f_min", "f min (Hz)", 10.0), Param("f_max", "f max (Hz)", 60.0)],
    lambda section, trace_1, trace_2, window, f_min, f_max: seismic.q_spectral_ratio(
        section, trace_1, trace_2, *( (section.headers["first_break"][trace_1], section.headers["first_break"][trace_2])
                                     if "first_break" in section.headers else (None, None)), window, f_min, f_max),
    "Bath (1974); Tonn (1991)")

add("40.1", "acoustic_model", "2D Acoustic FD Modeling", SE, "Waveform Modeling & Inversion",
    "4th-order finite-difference shot gathers from a velocity model (surface receivers).",
    [G("Demo_velocity_model", "velocity", "Velocity model (x, depth)")],
    [Param("source_x", "Source x (m)", 300.0), Param("frequency", "Ricker frequency (Hz)", 10.0), Param("duration", "Record length (s)", 0.8)],
    seismic_wave.acoustic_modeling)


def _fwi(sections, initial_velocity, smooth_initial, frequency, iterations):
    init = initial_velocity
    if smooth_initial:
        from scipy.ndimage import gaussian_filter

        init = init.with_values(gaussian_filter(init.values, 6), init.name + "_smoothed")
    return seismic_wave.full_waveform_inversion(sections, init, frequency, iterations)


add(40, "fwi", "Full Waveform Inversion (Acoustic)", SE, "Waveform Modeling & Inversion",
    "Adjoint-state time-domain FWI (L-BFGS) of shot gathers produced on the modeling grid.",
    [Input("sections", "sections", "Observed shot gathers (from 40.1 geometry)", demo=("Demo_FWI_shot_1", "Demo_FWI_shot_2")),
     G("Demo_velocity_model", "initial_velocity", "Starting model (x, depth)")],
    [Param("smooth_initial", "Smooth the starting model", True), Param("frequency", "Source frequency (Hz)", 10.0), Param("iterations", "L-BFGS iterations", 8)],
    _fwi, "Tarantola (1984); Virieux & Operto (2009)", slow=True)

add(41, "first_break", "First Arrival Picking", SE, "Refraction",
    "STA/LTA trigger refined by the AIC picker on each trace.", [S("Demo_refraction_shot", "gather")],
    [Param("method", "Picker", "sta_lta+aic", ("sta_lta+aic", "sta_lta", "aic")), Param("sta", "STA (s)", 0.002),
     Param("lta", "LTA (s)", 0.02), Param("threshold", "Trigger ratio", 3.0)], refraction.pick_first_arrivals, "Allen (1978); Maeda (1985)")

_FR = [col("x_col", "Position column", "x"), col("tf_col", "Forward time column", "t_forward"), col("tr_col", "Reverse time column", "t_reverse"),
       Param("v1", "Upper-layer velocity V1 (m/s)", 500.0)]
add(42, "plus_minus", "Plus-Minus Method", SE, "Refraction",
    "Hagedoorn plus/minus times give refractor velocity and depth under each station.",
    [T("Demo_refraction_forward_reverse")], _FR + [Param("reciprocal_time", "Reciprocal time (s, 0 = auto)", 0.0)],
    lambda table, x_col, tf_col, tr_col, v1, reciprocal_time: refraction.plus_minus(table, x_col, tf_col, tr_col, v1, reciprocal_time or None),
    "Hagedoorn (1959)")

add(43, "grm", "Generalized Reciprocal Method (GRM)", SE, "Refraction",
    "Velocity-analysis and time-depth functions over a range of XY; optimum XY by linearity.",
    [T("Demo_refraction_forward_reverse")], _FR + [Param("xy_values", "XY values (m)", "0,3,6,9,12,15"),
                                                   Param("reciprocal_time", "Reciprocal time (s, 0 = auto)", 0.0)],
    lambda table, x_col, tf_col, tr_col, v1, xy_values, reciprocal_time: refraction.generalized_reciprocal_method(
        table, x_col, tf_col, tr_col, v1, xy_values, reciprocal_time or None), "Palmer (1980)")

add(44, "tomography", "Refraction Tomography", SE, "Refraction",
    "Shortest-path ray tracing and smoothed LSQR slowness updates from first-break picks.", [T("Demo_refraction_picks", "picks")],
    [col("source_col", "Source x column", "source_x"), col("receiver_col", "Receiver x column", "receiver_x"), col("time_col", "Pick time column", "pick_time"),
     Param("max_depth", "Model depth (m)", 40.0), Param("cell_size", "Cell size (m)", 2.0), Param("v_top", "Start v top", 500.0),
     Param("v_bottom", "Start v bottom", 2500.0), Param("iterations", "Iterations", 6), Param("smoothing", "Smoothing", 5.0)],
    refraction.refraction_tomography, "Moser (1991); Zhang & Toksoz (1998)", slow=True)

add(45, "hvsr", "HVSR (Nakamura)", SE, "Passive & Surface Waves",
    "Site resonance f0 and bedrock depth from single-station 3C ambient noise.", [T("Demo_3C_ambient_noise")],
    [col("north_col", "North", "north"), col("east_col", "East", "east"), col("vertical_col", "Vertical", "vertical"),
     Param("sampling_rate", "Sampling rate (Hz)", 100.0), Param("window_length", "Window (s)", 30.0), Param("f_min", "f min", 0.3),
     Param("f_max", "f max", 20.0), Param("smoothing_b", "Konno-Ohmachi b", 40.0), Param("shear_velocity", "Sediment Vs (m/s)", 300.0)],
    passive.hvsr, "Nakamura (1989); SESAME (2004)")

add(46, "spac", "Spatial Autocorrelation (SPAC)", SE, "Passive & Surface Waves",
    "Rayleigh phase velocities from ring-array coherency: rho(f) = J0(2 pi f r / c).", [T("Demo_SPAC_array")],
    [col("center_col", "Centre station", "center"), Param("ring_prefix", "Ring column prefix", "ring"), Param("radius", "Radius (m)", 20.0),
     Param("sampling_rate", "Sampling rate (Hz)", 100.0), Param("window_length", "Window (s)", 20.0), Param("f_min", "f min", 1.0),
     Param("f_max", "f max", 25.0)], passive.spac, "Aki (1957); Okada (2003)")

add(47, "masw", "MASW", SE, "Passive & Surface Waves",
    "Phase-shift dispersion image, fundamental-mode pick and wavelength-depth Vs profile.", [S("Demo_MASW_shot")],
    [Param("c_min", "c min (m/s)", 100.0), Param("c_max", "c max (m/s)", 600.0), Param("f_min", "f min (Hz)", 5.0), Param("f_max", "f max (Hz)", 60.0)],
    lambda section, c_min, c_max, f_min, f_max: passive.masw_dispersion(section, c_min, c_max, 181, f_min, f_max), "Park et al. (1998, 1999)")

add(48, "rf", "Receiver Function Analysis", SE, "Passive & Surface Waves",
    "Water-level deconvolution of radial by vertical teleseismic motion.", [T("Demo_receiver_function_ZR")],
    [col("vertical_col", "Vertical", "vertical"), col("radial_col", "Radial", "radial"), Param("sampling_rate", "Sampling rate (Hz)", 20.0),
     Param("water_level", "Water level", 0.01), Param("gaussian", "Gaussian a", 2.5)],
    lambda table, vertical_col, radial_col, sampling_rate, water_level, gaussian: passive.receiver_function(
        table, vertical_col, radial_col, sampling_rate, water_level, gaussian, 10.0), "Langston (1979)")

add("48.1", "hk", "H-kappa Stacking", SE, "Passive & Surface Waves",
    "Crustal thickness and Vp/Vs from Ps and multiples of a receiver function.", [PR("Demo_receiver_function")],
    [Param("vp", "Crustal Vp (km/s)", 6.3), Param("ray_parameter", "Ray parameter (s/km)", 0.06)],
    lambda profile, vp, ray_parameter: passive.h_kappa_stack(profile, vp, ray_parameter), "Zhu & Kanamori (2000)")

add(49, "focal", "Focal Mechanism (P First Motions)", SE, "Passive & Surface Waves",
    "Grid search of strike/dip/rake fitting P-wave polarities.", [T("Demo_first_motion_polarities")],
    [col("azimuth_col", "Azimuth", "azimuth"), col("takeoff_col", "Take-off angle", "takeoff"), col("polarity_col", "Polarity (+1/-1)", "polarity"),
     Param("strike_step", "Strike step", 10.0), Param("dip_step", "Dip step", 10.0), Param("rake_step", "Rake step", 15.0)],
    passive.focal_mechanism, "Reasenberg & Oppenheimer (1985)")

add(50, "interferometry", "Seismic Interferometry", SE, "Passive & Surface Waves",
    "Noise cross-correlation stacking to retrieve the inter-station Green's function.", [T("Demo_noise_station_pair")],
    [col("station_a", "Station A", "a"), col("station_b", "Station B", "b"), Param("sampling_rate", "Sampling rate (Hz)", 50.0),
     Param("window_length", "Window (s)", 60.0), Param("max_lag", "Max lag (s)", 5.0), Param("one_bit", "One-bit normalization", True),
     Param("whiten", "Spectral whitening", True)], passive.ambient_noise_interferometry, "Bensen et al. (2007)")

add(51, "vsp", "VSP Corridor Stack", SE, "Borehole Seismic",
    "Up/down-going separation (median filter), TWT alignment and corridor stack for well ties.", [S("Demo_VSP", "vsp")],
    [Param("first_break_velocity", "Velocity for first breaks", 2500.0), Param("corridor_length", "Corridor (s)", 0.1),
     Param("median_traces", "Median filter (traces)", 7)], seismic.vsp_corridor_stack, "Hardage (1985)")

# ================================================================ ELECTRICAL & EM
EE = "Electrical & Electromagnetic"
add(56, "ves", "1D VES Inversion", EE, "DC Resistivity & IP",
    "Layered-earth inversion of Schlumberger/Wenner soundings (FFTLog Hankel forward).", [T("Demo_VES")],
    [col("spacing_col", "AB/2 or a column", "ab2"), col("rhoa_col", "Apparent resistivity", "rhoa"), Param("n_layers", "Layers", 3),
     Param("array", "Array", "schlumberger", ("schlumberger", "wenner"))], electrical.ves_inversion, "Koefoed (1979)")

add("57.1", "ert_design", "ERT Array Design", EE, "DC Resistivity & IP",
    "Generates electrode configurations (Wenner, Schlumberger, dipole-dipole) with homogeneous apparent resistivity.", [],
    [Param("n_electrodes", "Electrodes", 24), Param("spacing", "Spacing (m)", 5.0),
     Param("array", "Array", "dipole-dipole", ("dipole-dipole", "wenner", "schlumberger")), Param("max_n", "Max n-level", 6)],
    lambda n_electrodes, spacing, array, max_n: electrical.make_array(n_electrodes, spacing, array, max_n).assign(rhoa=100.0))

add(57, "ert", "2D ERT Inversion", EE, "DC Resistivity & IP",
    "2.5D finite-volume forward, adjoint sensitivities, Occam Gauss-Newton with 1/sigma weights; residual table + QC.",
    [T("Demo_ERT_dipole_dipole")], [col("rhoa_col", "Apparent resistivity", "rhoa"), Param("iterations", "Max iterations", 6),
                                   *inversion_params(2.0, alpha_s=1e-3), Param("reference_resistivity", "Reference half-space (ohm.m, 0 = median)", 0.0),
                                   Param("smoothness", "Relative smoothness (fixed mode)", 0.3)],
    lambda table, **kw: electrical.ert_inversion(table, **kw),
    "Dey & Morrison (1979); deGroot-Hedlin & Constable (1990)", slow=True)

add(58, "ip", "IP Chargeability Inversion", EE, "DC Resistivity & IP",
    "Linearized chargeability inversion (Seigel formulation) on the ERT resistivity model.", [T("Demo_ERT_dipole_dipole")],
    [col("rhoa_col", "Apparent resistivity", "rhoa"), col("chargeability_col", "Apparent chargeability (mV/V)", "chargeability_mV_V"),
     Param("iterations", "Resistivity iterations", 5), *inversion_params(2.0, alpha_s=1e-3)],
    lambda table, **kw: electrical.ert_inversion(table, **kw),
    "Seigel (1959); Oldenburg & Li (1994)", slow=True)

add(59, "sip", "Spectral IP (Cole-Cole)", EE, "DC Resistivity & IP",
    "Fits Cole-Cole parameters (rho0, m, tau, c) to complex resistivity spectra.", [T("Demo_SIP_spectrum")],
    [col("freq_col", "Frequency", "frequency"), col("amp_col", "Amplitude", "amplitude"), col("phase_col", "Phase (mrad)", "phase_mrad")],
    electrical.sip_cole_cole_fit, "Pelton et al. (1978)")


def _sp_ls(profile, shape, depth_guess):
    q = electrical.SP_SHAPES[shape]
    x, v = profile.distance, profile.values
    best = None
    for x0 in np.linspace(x.min(), x.max(), 7):
        for a in (-60.0, 0.0, 60.0):
            p0 = [x0, depth_guess, a]
            f = lambda p: (lambda b: (b @ v / max(b @ b, 1e-300)) * b - v)(electrical.sp_forward(x, 1.0, *p, q))  # noqa: E731
            r = least_squares(f, p0, bounds=([x.min(), 0.1, -90], [x.max(), 10 * np.ptp(x), 90]))
            if best is None or r.cost < best.cost:
                best = r
    b = electrical.sp_forward(x, 1.0, *best.x, q)
    k = float(b @ v / (b @ b))
    return (profile.with_values(x, k * b, f"{profile.name}_SPfit", {"method": "sp_curve_matching"}),
            {"K": k, "x0_m": best.x[0], "depth_m": best.x[1], "polarization_deg": best.x[2], "rms_mV": float(np.sqrt(np.mean(best.fun**2)))})


add(70, "sp_model", "Self-Potential Quantitative Modeling", EE, "DC Resistivity & IP",
    "Multi-start least-squares curve matching of polarized sphere/cylinder SP anomalies.", [PR("Demo_SP")],
    [Param("shape", "Source shape", "sphere", tuple(electrical.SP_SHAPES)), Param("depth_guess", "Starting depth (m)", 20.0)],
    _sp_ls, "Yungul (1950)")

_MT = [col("ex", "Ex (mV/km)", "ex"), col("ey", "Ey (mV/km)", "ey"), col("hx", "Hx/Bx (nT)", "hx"), col("hy", "Hy/By (nT)", "hy")]
add(60, "mt_z", "MT Impedance Tensor", EE, "Magnetotellurics",
    "Band-averaged least-squares impedance tensor, apparent resistivity & phase (xy, yx, determinant).",
    [T("Demo_MT_time_series")], _MT + [Param("sampling_rate", "Sampling rate (Hz)", 100.0), Param("window_length", "Window (s)", 20.0)],
    lambda table, ex, ey, hx, hy, sampling_rate, window_length: (lambda res: (res[0], em.mt_apparent_profile(res[0]), res[1]))(
        em.mt_impedance_tensor(table, ex, ey, hx, hy, "", sampling_rate, window_length)), "Cagniard (1953); Vozoff (1972)")

add(61, "tipper", "MT Tipper / Induction Arrows", EE, "Magnetotellurics",
    "Hz = Tx Hx + Ty Hy; real induction arrows (Parkinson) point to conductors.", [T("Demo_MT_time_series")],
    _MT + [col("hz", "Hz/Bz (nT)", "hz"), Param("sampling_rate", "Sampling rate (Hz)", 100.0), Param("window_length", "Window (s)", 20.0)],
    lambda table, ex, ey, hx, hy, hz, sampling_rate, window_length: em.mt_tipper_arrows(
        em.mt_impedance_tensor(table, ex, ey, hx, hy, hz, sampling_rate, window_length)[0]), "Parkinson (1959)")

add(62, "occam", "MT Occam 1D Inversion", EE, "Magnetotellurics",
    "Smoothest 1D resistivity model fitting apparent resistivity and phase to a target misfit.", [T("Demo_MT_response")],
    [col("freq_col", "Frequency", "frequency"), col("rho_col", "Apparent resistivity", "rho_det"), col("phase_col", "Phase (deg)", "phase_det"),
     Param("n_layers", "Layers", 40), Param("error_percent", "Data error (%)", 5.0), Param("target_rms", "Target normalized RMS", 1.0),
     Param("operator", "Roughness operator", "first-derivative", ("first-derivative", "second-derivative"))],
    em.mt_occam_1d, "Constable et al. (1987)")

add(63, "csamt", "CSAMT Processing", EE, "Magnetotellurics",
    "Cagniard resistivity, skin depth, far-field check and Bostick depth transform.", [T("Demo_CSAMT")],
    [col("freq_col", "Frequency", "frequency"), col("ex_col", "Ex (mV/km)", "ex"), col("hy_col", "Hy (nT)", "hy"),
     Param("phase_col", "Phase column (optional)", "phase"), Param("tx_rx_distance", "Tx-Rx distance (m)", 5000.0)],
    em.csamt_cagniard, "Zonge & Hughes (1991)")

add(64, "tem", "TEM Decay Analysis", EE, "Time & Frequency-Domain EM",
    "Late-time apparent resistivity and exponential decay constant (conductor indicator).", [T("Demo_TEM_decay")],
    [col("time_col", "Time (s)", "time"), col("response_col", "dB/dt (V/m^2 per rx area)", "dbdt"), Param("tx_area", "Tx area (m2)", 10000.0),
     Param("tx_turns", "Tx turns", 1), Param("current", "Current (A)", 1.0), Param("rx_area", "Rx area (m2)", 1.0)],
    em.tem_late_time_rho, "Spies & Frischknecht (1991)")

add(65, "aem_cdt", "Airborne EM Conductivity-Depth Transform", EE, "Time & Frequency-Domain EM",
    "Half-space apparent conductivity per gate at diffusion depth, assembled into a section.", [T("Demo_AEM_line")],
    [col("x_col", "Station column", "x"), col("time_col", "Time", "time"), col("response_col", "Response", "dbdt"),
     Param("moment", "Tx moment (A m2)", 5e5), Param("depth_step", "Depth step (m)", 5.0), Param("max_depth", "Max depth (m)", 300.0)],
    em.aem_cdt, "Macnae & Lamontagne (1987)")

add(66, "fdem", "FDEM Quadrature Conductivity", EE, "Time & Frequency-Domain EM",
    "McNeill low-induction-number apparent conductivity from the quadrature component.", [T("Demo_FDEM_survey")],
    [col("quadrature_col", "Quadrature (ppt)", "quadrature_ppt"), Param("frequency", "Frequency (Hz)", 9800.0),
     Param("coil_separation", "Coil separation (m)", 3.66)], em.fdem_lin_conductivity, "McNeill (1980)")

add(71, "vlf", "VLF-EM Fraser Filtering", EE, "Time & Frequency-Domain EM",
    "Converts dip-angle crossovers into positive peaks over conductors.", [PR("Demo_VLF_tilt")], [], electrical.vlf_fraser_filter, "Fraser (1969)")

add(67, "dewow", "GPR Dewow", EE, "Ground Penetrating Radar",
    "Running-mean subtraction removes low-frequency instrument 'wow'.", [S("Demo_GPR")], [Param("window_ns", "Window (ns)", 5.0)],
    lambda section, window_ns: em.gpr_dewow(section, window_ns * 1e-9))

add(68, "bgr", "GPR Background Removal", EE, "Ground Penetrating Radar",
    "Subtracts the mean (or moving-average) trace to remove horizontal banding.", [S("Demo_GPR")],
    [Param("window_traces", "Moving window (traces, 0 = global)", 0)], em.gpr_background_removal)

add(69, "gpr_mig", "GPR Migration", EE, "Ground Penetrating Radar",
    "Stolt f-k migration collapses diffraction hyperbolas.", [S("Demo_GPR")], [Param("velocity_m_per_ns", "Velocity (m/ns)", 0.1)],
    lambda section, velocity_m_per_ns: em.gpr_migration(section, velocity_m_per_ns * 1e9), "Stolt (1978)")

# ================================================================ WELLS & RADIOMETRICS
WR = "Well Logging & Radiometrics"
add(72, "gamma_spec", "Gamma-Ray Spectrometry Analysis", WR, "Radiometrics",
    "Th/K, U/Th, U/K ratios, F-parameter and air dose rate from K, eU, eTh.", [T("Demo_well_logs_radiometric")],
    [col("k_col", "K (%)", "K"), col("u_col", "eU (ppm)", "eU"), col("th_col", "eTh (ppm)", "eTh")], wells.gamma_spectrometry, "IAEA (2003)")

add(73, "heat_prod", "Radiogenic Heat Production (grids)", WR, "Radiometrics",
    "A = 1e-5 rho (9.52 U + 2.56 Th + 3.48 K) uW/m3 from K, eU, eTh grids.",
    [G("Demo_K", "k", "K grid (%)"), G("Demo_eU", "u", "eU grid (ppm)"), G("Demo_eTh", "th", "eTh grid (ppm)")],
    [Param("density", "Rock density (kg/m3)", 2670.0)], wells.heat_production_grid, "Rybach (1988)")

add("73.1", "heat_prod_table", "Radiogenic Heat Production (table)", WR, "Radiometrics",
    "Heat production per sample from K, eU, eTh columns.", [T("Demo_well_logs_radiometric")],
    [col("k_col", "K (%)", "K"), col("u_col", "eU", "eU"), col("th_col", "eTh", "eTh"), Param("density", "Density", 2670.0)],
    wells.heat_production_table, "Rybach (1988)")

_LOG = T("Demo_well_logs")
add("74.1", "vsh_gr", "Shale Volume from Gamma Ray", WR, "Petrophysics",
    "IGR with linear or Larionov corrections.", [_LOG],
    [col("gr_col", "GR", "GR"), Param("method", "Method", "linear", ("linear", "larionov_tertiary", "larionov_older"))],
    lambda table, gr_col, method: wells.shale_volume_gr(table, gr_col, None, None, method), "Larionov (1969)")

add(74, "sp_baseline", "SP Baseline Shift", WR, "Petrophysics",
    "Tracks and removes the shale baseline drift; Vsh from SP and permeable-bed flag.", [_LOG],
    [col("depth_col", "Depth", "DEPTH"), col("sp_col", "SP", "SP"), Param("window", "Baseline window (samples)", 101),
     Param("permeable_threshold_mv", "Permeable deflection (mV)", 15.0)],
    lambda table, depth_col, sp_col, window, permeable_threshold_mv: wells.sp_baseline_shift(table, depth_col, sp_col, window, permeable_threshold_mv),
    "Doveton (1994)")

add(75, "archie", "Archie Saturation Modeling", WR, "Petrophysics",
    "Sw = (a Rw / (phi^m Rt))^(1/n); hydrocarbon saturation and bulk-volume water.", [_LOG],
    [col("rt_col", "Rt", "RT"), col("phi_col", "Porosity", "PHI"), Param("rw", "Rw (ohm.m)", 0.05), Param("a", "a", 1.0), Param("m", "m", 2.0),
     Param("n", "n", 2.0)], wells.archie_saturation, "Archie (1942)")

add(76, "dn", "Density-Neutron Crossplot", WR, "Petrophysics",
    "Density, crossplot porosity, apparent matrix density, lithology and gas flag.", [_LOG],
    [col("rhob_col", "RHOB", "RHOB"), col("nphi_col", "NPHI", "NPHI"), Param("rho_matrix", "Matrix density", 2.65), Param("rho_fluid", "Fluid density", 1.0)],
    wells.density_neutron, "Schlumberger (1989)")

add(77, "sonic", "Sonic Log Integration", WR, "Petrophysics",
    "Time-depth relationship (OWT/TWT), interval and RMS velocity for seismic-well ties.", [_LOG],
    [col("depth_col", "Depth", "DEPTH"), col("dt_col", "DT", "DT"), Param("dt_unit", "DT units", "us/ft", ("us/ft", "us/m")),
     Param("start_twt", "TWT at first sample (s)", 1.2)],
    lambda table, depth_col, dt_col, dt_unit, start_twt: wells.sonic_integration(table, depth_col, dt_col, dt_unit, start_twt))

add(78, "dipmeter", "Dipmeter / Image Log Analysis", WR, "Petrophysics",
    "Sinusoid fitting of image-log picks: bed/fracture dip and dip azimuth.", [T("Demo_dipmeter_picks")],
    [col("feature_col", "Feature id", "feature"), col("depth_col", "Depth", "depth"), col("azimuth_col", "Borehole azimuth", "azimuth"),
     Param("borehole_diameter", "Borehole diameter (m)", 0.216)], wells.dipmeter_sinusoid_fit)

add(80, "nmr", "NMR T2 Processing", WR, "Petrophysics",
    "Regularized NNLS T2 distribution, BVI/FFI and Coates permeability.", [T("Demo_NMR_echo_train")],
    [col("time_col", "Echo time (ms)", "time_ms"), col("amp_col", "Amplitude", "amplitude"), Param("t2_cutoff", "T2 cutoff (ms)", 33.0),
     Param("regularization", "Regularization", 0.5)],
    lambda table, time_col, amp_col, t2_cutoff, regularization: wells.nmr_t2_inversion(table, time_col, amp_col, 64, 0.3, 3000.0, regularization, t2_cutoff),
    "Coates et al. (1999)")

add(79, "crosshole", "Cross-hole Electrical Tomography", WR, "Borehole Geophysics",
    "2.5D resistivity inversion with electrodes in two boreholes.", [T("Demo_crosshole_ERT")],
    [col("rhoa_col", "Apparent resistivity", "rhoa"), Param("iterations", "Max iterations", 5), *inversion_params(2.0, alpha_s=1e-3)],
    lambda table, **kw: electrical.ert_inversion(table, **kw),
    "Daily & Owen (1991)", slow=True)

# ================================================================ ADVANCED & ML
AD = "Advanced Processing & Machine Learning"
add(81, "joint", "Joint Inversion (Cross-gradient)", AD, "Inversion & Optimization",
    "Simultaneous gravity-magnetic voxel inversion coupled by cross-gradients.",
    [G("Demo_Bouguer", "gravity", "Gravity grid"), G("Demo_TMI", "magnetic", "Magnetic grid")],
    [Param("n_z", "Depth layers", 5), Param("max_depth", "Mesh depth (m)", 8000.0), Param("stride", "Decimation", 5),
     Param("cross_gradient_weight", "Cross-gradient weight", 1.0), Param("iterations", "Gauss-Newton iterations", 4), INC, DEC],
    lambda gravity, magnetic, n_z, max_depth, stride, cross_gradient_weight, iterations, inclination, declination:
    modeling.joint_inversion_cross_gradient(gravity, magnetic, n_z, max_depth, stride, 1e-2, cross_gradient_weight, iterations, inclination, declination),
    "Gallardo & Meju (2003)", slow=True)

add(82, "pinn", "Physics-Informed Neural Network (Geotherm)", AD, "Inversion & Optimization",
    "PINN solving k T'' = -A; with temperature data it also learns heat production (inverse).",
    [Input("observations", "table", "Temperature observations (optional)", optional=True, demo="Demo_temperature_observations")],
    [Param("depth_max", "Depth (m)", 30000.0), Param("surface_temperature", "T0 (C)", 15.0), Param("surface_heat_flow", "q0 (W/m2)", 0.07),
     Param("conductivity", "k (W/m/K)", 2.5), Param("heat_production", "A (W/m3, forward or start)", 1e-6)],
    lambda observations, depth_max, surface_temperature, surface_heat_flow, conductivity, heat_production:
    optimize.pinn_geotherm(depth_max, surface_temperature, surface_heat_flow, conductivity, heat_production, observations),
    "Raissi et al. (2019)")

add(93, "mcmc", "Bayesian Inversion (MCMC)", AD, "Inversion & Optimization",
    "Metropolis-Hastings posterior sampling of a layered VES model (uncertainty percentiles).", [T("Demo_VES")],
    [col("spacing_col", "AB/2", "ab2"), col("rhoa_col", "rho_a", "rhoa"), Param("n_layers", "Layers", 3), Param("noise_percent", "Noise (%)", 3.0),
     Param("n_samples", "Samples", 6000), Param("burn_in", "Burn-in", 2000)], optimize.mcmc_ves_inversion, "Malinverno (2002)", slow=True)

add(94, "pso", "Particle Swarm Optimization (SP)", AD, "Inversion & Optimization",
    "Global PSO inversion of SP anomalies for position, depth and polarization.", [PR("Demo_SP")],
    [Param("shape", "Source shape", "sphere", tuple(electrical.SP_SHAPES)), Param("depth_max", "Max depth (m)", 200.0),
     Param("particles", "Particles", 30), Param("iterations", "Iterations", 120)], optimize.pso_sp_inversion, "Kennedy & Eberhart (1995)")

add(95, "sa", "Simulated Annealing (Residual Gravity)", AD, "Inversion & Optimization",
    "Global SA inversion of a residual gravity profile for depth, position and shape factor.", [PR("Demo_Bouguer_profile")],
    [Param("depth_max", "Max depth (m)", 8000.0), Param("iterations", "Iterations", 3000)], optimize.sa_residual_gravity_inversion,
    "Kirkpatrick et al. (1983)")

add(96, "ga", "Genetic Algorithm (MT 1D)", AD, "Inversion & Optimization",
    "Real-coded GA inversion of layered MT resistivity and thickness.", [T("Demo_MT_response")],
    [col("freq_col", "Frequency", "frequency"), col("rho_col", "rho_a", "rho_det"), col("phase_col", "Phase", "phase_det"),
     Param("n_layers", "Layers", 3), Param("generations", "Generations", 80)], optimize.ga_mt_inversion, "Holland (1975)")

add(97, "tsvd", "Truncated SVD Inversion (Gravity)", AD, "Inversion & Optimization",
    "Linear 2D density inversion of a gravity profile stabilized by truncating small singular values.", [PR("Demo_Bouguer_profile")],
    [Param("depth_top", "Top depth (m)", 500.0), Param("depth_bottom", "Bottom depth (m)", 6000.0), Param("n_z", "Layers", 8),
     Param("truncation", "Kept singular values (0 = auto)", 0)], optimize.tsvd_gravity_inversion, "Hansen (1987)")

add("97.1", "tikhonov", "Tikhonov / Occam Inversion Studio (profile)", AD, "Inversion & Optimization",
    "2D density inversion exposing every trade-off control: lambda by discrepancy / L-curve / GCV, operators, anisotropy, "
    "reference model, 1/sigma weights, direct / CG / subspace solvers; outputs L-curve, residuals and QC.", [PR("Demo_Bouguer_profile")],
    [Param("depth_top", "Top depth (m)", 500.0), Param("depth_bottom", "Bottom depth (m)", 6000.0), Param("n_z", "Layers", 10),
     Param("reference_density", "Reference density contrast", 0.0), *inversion_params(0.0, 0.1, solver=True),
     Param("lam", "Lambda (fixed mode only)", 1.0)],
    optimize.tikhonov_gravity_profile, "Tikhonov (1963); Constable et al. (1987); Hansen (1992)")

add("97.2", "residual_qc", "Inversion Residual Analysis", AD, "Inversion & Optimization",
    "Observed - calculated maps, normalized residuals, chi^2 / normalized RMS, Durbin-Watson, normality and Moran I "
    "(coherent residuals = missing structure or over-smoothing).",
    [G("Demo_Bouguer", "observed", "Observed grid"), G("Demo_Bouguer_trend", "predicted", "Calculated / predicted grid")],
    [Param("error_percent", "Data error (%)", 0.0), Param("error_floor", "Error floor (data units)", 0.1)],
    inversion.residual_analysis, "Aster et al. (2018)")

_XYV = [col("x_col", "X", "x"), col("y_col", "Y", "y"), col("value_col", "Value", "gravity")]
add(83, "mincurv", "Minimum Curvature Gridding", AD, "Gridding & Geostatistics",
    "Smoothest (biharmonic) surface honoring data, optional tension.", [T("Demo_gravity_stations")],
    _XYV + [Param("spacing", "Cell size (m)", 1000.0), Param("tension", "Tension (0-1)", 0.25)], interpolation.minimum_curvature,
    "Briggs (1974); Smith & Wessel (1990)")

add(84, "kriging", "Ordinary Kriging", AD, "Gridding & Geostatistics",
    "Variogram-based BLUE estimate with kriging variance.", [T("Demo_gravity_stations")],
    _XYV + [Param("spacing", "Cell size (m)", 1000.0), Param("model", "Variogram model", "spherical", tuple(interpolation.VARIOGRAMS)),
            Param("neighbors", "Neighbours", 16)], interpolation.ordinary_kriging, "Matheron (1963)")

add(85, "cokriging", "Collocated Cokriging", AD, "Gridding & Geostatistics",
    "Sparse primary data + dense correlated secondary grid (Markov model 1).",
    [T("Demo_gravity_stations"), G("Demo_Bouguer", "secondary", "Secondary grid")],
    _XYV + [Param("model", "Variogram model", "spherical", tuple(interpolation.VARIOGRAMS)), Param("neighbors", "Neighbours", 16)],
    lambda table, secondary, x_col, y_col, value_col, model, neighbors: interpolation.collocated_cokriging(table, secondary, x_col, y_col, value_col, model, neighbors),
    "Xu et al. (1992)")

GRIDS = lambda demo=("Demo_TMI", "Demo_Bouguer"), label="Input grids (multi-select)": Input("grids", "grids", label, demo=demo)  # noqa: E731
add(86, "pca", "Principal Component Analysis (grids)", AD, "Machine Learning",
    "Standardized PCA of co-registered grids (e.g. K, eU, eTh).", [GRIDS(("Demo_K", "Demo_eU", "Demo_eTh"))],
    [Param("n_components", "Components", 3)], ml.pca_grids, "Pearson (1901)")

add("86.1", "pca_table", "Principal Component Analysis (table)", AD, "Machine Learning",
    "PCA scores for selected numeric columns.", [_LOG], [Param("columns", "Columns (comma list)", "GR,RT,RHOB,NPHI,DT"), Param("n_components", "Components", 3)],
    ml.pca_table)

add(87, "dl_fb", "Neural First-Break Picking", AD, "Machine Learning",
    "Neural classifier trained on high-SNR automatic picks, then applied to every trace.", [S("Demo_refraction_shot")],
    [Param("window", "Window half-length (s)", 0.006), Param("hidden", "Hidden layers", "64,32")],
    lambda section, window, hidden: ml.neural_first_break_picking(section, window, None, hidden), "Yuan et al. (2018)")

add(88, "som", "Self-Organizing Maps (grids)", AD, "Machine Learning",
    "Kohonen SOM + clustering of prototypes into facies/domains.", [GRIDS(("Demo_K", "Demo_eU", "Demo_eTh"))],
    [Param("rows", "SOM rows", 5), Param("cols", "SOM cols", 5), Param("n_facies", "Facies", 5)], ml.som_facies, "Kohonen (1982)")

add("88.1", "som_table", "Self-Organizing Maps (logs)", AD, "Machine Learning",
    "Electrofacies from well-log columns.", [_LOG], [Param("columns", "Columns", "GR,RT,RHOB,NPHI,DT"), Param("rows", "Rows", 5),
                                                  Param("cols", "Cols", 5), Param("n_facies", "Facies", 4)], ml.som_table)

add(89, "svm", "Support Vector Machine Lithology", AD, "Machine Learning",
    "RBF-SVM classification of lithology from logs; hold-out accuracy and confusion matrix.", [_LOG],
    [Param("features", "Feature columns", "GR,RT,RHOB,NPHI,DT"), col("label_col", "Label column", "lithology"),
     Param("kernel", "Kernel", "rbf", ("rbf", "linear", "poly")), Param("c", "C", 10.0)],
    lambda table, features, label_col, kernel, c: ml.svm_classification(table, features, label_col, kernel, c), "Cortes & Vapnik (1995)")

add(90, "rforest", "Random Forest Regression", AD, "Machine Learning",
    "Predicts porosity/permeability from logs with feature importance.", [_LOG],
    [Param("features", "Feature columns", "GR,RT,RHOB,NPHI,DT"), col("target_col", "Target column", "porosity"), Param("n_trees", "Trees", 200)],
    lambda table, features, target_col, n_trees: ml.random_forest_regression(table, features, target_col, n_trees), "Breiman (2001)")

add(91, "kmeans", "K-Means Clustering", AD, "Machine Learning",
    "Partitions grids into domains using amplitude and texture.", [GRIDS()],
    [Param("n_clusters", "Clusters", 5), Param("texture", "Add local-std texture", True)], ml.kmeans_zones, "MacQueen (1967)")

add(92, "fault_ml", "Neural Fault Detection", AD, "Machine Learning",
    "Patch classifier trained on synthetic faulted reflectivity; outputs fault probability.", [S("Demo_stack_section")],
    [Param("patch", "Patch size", 16), Param("stride", "Stride", 4)], lambda section, patch, stride: ml.fault_detection(section, patch, stride),
    "Wu et al. (2019)", slow=True)

add(98, "wavelet", "Wavelet Transform Filtering (profile)", AD, "Signal & Texture Analysis",
    "Starlet (a trous) wavelet denoising of non-stationary noise.", [PR("Demo_TMI_profile")],
    [Param("levels", "Scales", 4), Param("k_sigma", "Threshold (x sigma)", 3.0)], ml.wavelet_filter_profile, "Starck et al. (2007)")

add("98.1", "wavelet_grid", "Wavelet Transform Filtering (grid)", AD, "Signal & Texture Analysis",
    "2D starlet denoising.", [G()], [Param("levels", "Scales", 4), Param("k_sigma", "Threshold", 3.0)], ml.wavelet_filter_grid)

add(99, "fractal", "Fractal Dimension Analysis", AD, "Signal & Texture Analysis",
    "Variogram-based Hurst exponent and fractal dimension D = 3 - H.", [G()], [Param("max_lag", "Max lag (cells)", 10)], ml.fractal_dimension, "Turcotte (1997)")

add("99.1", "fractal_map", "Fractal Dimension Map", AD, "Signal & Texture Analysis",
    "Moving-window fractal dimension to delineate textural/tectonic domains.", [G()],
    [Param("window", "Window (cells)", 16), Param("step", "Step (cells)", 4)], ml.fractal_dimension_map)

add(100, "dircos", "Directional Cosine Filtering", AD, "Signal & Texture Analysis",
    "Passes or rejects features of a chosen strike in the wavenumber domain.", [G()],
    [Param("strike_azimuth", "Strike azimuth (deg from N)", 45.0), Param("power", "Cosine power", 2.0),
     Param("mode", "Mode", "pass", ("pass", "reject")), pad()], potential_field.directional_cosine_filter)

# ================================================================ GIS
GI = "GIS & Spatial Analysis"


def _map_algebra(grids, expression):
    return gis.map_algebra(expression, {chr(65 + i): g for i, g in enumerate(grids)})


add(101, "map_algebra", "Map Algebra / Raster Math", GI, "Raster Analysis",
    "Expression over selected grids named A, B, C... in selection order, e.g. where((A>0)&(B>1), A*B, 0).",
    [GRIDS(("Demo_K", "Demo_eTh"))], [Param("expression", "Expression", "B / A")], _map_algebra)

add(107, "hillshade", "Hillshade & Multidirectional Shading", GI, "Raster Analysis",
    "Simulated illumination of DEMs or geophysical grids.", [G("Demo_DEM")],
    [Param("azimuth", "Sun azimuth", 315.0), Param("altitude", "Sun altitude", 45.0), Param("z_factor", "Z factor", 1.0),
     Param("multidirectional", "Multidirectional", False)], gis.hillshade, "Mark (1992)")

add(115, "topo", "Topographic Surface Analysis", GI, "Raster Analysis",
    "Slope, aspect, profile and plan curvature.", [G("Demo_DEM", "dem", "DEM")], [], gis.terrain_derivatives, "Zevenbergen & Thorne (1987)")

add(117, "trend", "Trend Surface Analysis", GI, "Raster Analysis",
    "Polynomial regional surface and residual.", [G("Demo_Bouguer")], [Param("order", "Polynomial order", 2)], gis.trend_surface)

add(111, "zonal", "Zonal Statistics", GI, "Raster Analysis",
    "Statistics of a value grid within class zones.", [G("Demo_Bouguer", "values", "Value grid"), G("Demo_geology_zones", "zones", "Zone grid")], [],
    gis.zonal_statistics)

add(123, "cutfill", "Volumetric (Cut/Fill) Calculation", GI, "Raster Analysis",
    "Volume between two surfaces.", [G("Demo_DEM", "surface_top", "Top surface"), G("Demo_DEM_smoothed", "surface_base", "Base surface")], [],
    gis.cut_fill)

add(102, "lineaments", "Lineament Extraction", GI, "Structural Interpretation",
    "Non-maximum-suppressed gradient ridges grouped into lineaments with azimuth and length.", [G()],
    [Param("threshold_percentile", "Threshold percentile", 85.0), Param("min_pixels", "Min pixels", 6), Param("use_thg", "Use THDR of input", True)],
    gis.extract_lineaments)

add("102.1", "lin_density", "Lineament Density Mapping", GI, "Structural Interpretation",
    "Length of lineaments per area within a search radius.", [T("Demo_lineaments", "lineaments", "Lineament table"), G("Demo_TMI", "reference", "Reference grid")],
    [Param("radius", "Search radius (m)", 4000.0)], gis.lineament_density)

add(103, "rose", "Rose Diagram", GI, "Structural Interpretation",
    "Azimuth-frequency (length-weighted) histogram and dominant trend.", [T("Demo_lineaments", "lineaments", "Lineament table")],
    [Param("bin_deg", "Bin width (deg)", 10.0), Param("weight_by_length", "Weight by length", True)], gis.rose_diagram)

add(104, "woe", "Weights of Evidence", GI, "Prospectivity & MCDA",
    "Bayesian W+/W-, contrast and posterior probability for a binary evidence layer.",
    [G("Demo_K", "evidence", "Evidence grid"), T("Demo_deposits", "deposits", "Known deposits (x, y)")],
    [col("x_col", "X", "x", "deposits"), col("y_col", "Y", "y", "deposits"), Param("threshold", "Evidence threshold", 2.5)],
    lambda evidence, deposits, x_col, y_col, threshold: gis.weights_of_evidence(evidence, deposits, x_col, y_col, threshold), "Bonham-Carter et al. (1989)")

add(105, "fuzzy", "Fuzzy Logic Overlay", GI, "Prospectivity & MCDA",
    "Fuzzy memberships combined with AND/OR/product/sum/gamma operators.", [GRIDS(("Demo_K", "Demo_eU", "Demo_eTh"))],
    [Param("operator", "Operator", "gamma", ("gamma", "and", "or", "product", "sum")), Param("gamma", "Gamma", 0.9),
     Param("membership", "Membership", "linear", ("linear", "sigmoid"))], gis.fuzzy_overlay, "Bonham-Carter (1994)")

add(106, "ahp", "MCDA / AHP Weighted Overlay", GI, "Prospectivity & MCDA",
    "Saaty pairwise comparisons -> weights (consistency ratio) -> weighted linear combination.", [GRIDS(("Demo_K", "Demo_eU", "Demo_eTh"))],
    [Param("pairwise", "Pairwise matrix rows 'a,b,c; ...'", "1,3,5; 1/3,1,3; 1/5,1/3,1"), Param("invert", "Invert layers (indices)", "")],
    gis.ahp_overlay, "Saaty (1980)")

add(108, "georef", "Georeferencing & Rectification", GI, "Cartography & Conversion",
    "Affine/2nd-order polynomial fit from GCPs and resampling to world coordinates.",
    [G("Demo_scanned_map", "image", "Raster in pixel coordinates"), T("Demo_GCPs", "gcps", "GCP table (col,row,x,y)")],
    [Param("spacing", "Output cell size", 25.0), Param("order", "Polynomial order", 1, (1, 2), kind="choice")],
    lambda image, gcps, spacing, order: gis.rectify_image(image, gcps, spacing, int(order)))

add(116, "rgb", "RGB Ternary Blending", GI, "Cartography & Conversion",
    "Three grids stretched into one RGB composite (e.g. K-Th-U).",
    [G("Demo_K", "red", "Red"), G("Demo_eTh", "green", "Green"), G("Demo_eU", "blue", "Blue")], [Param("clip_percent", "Clip (%)", 2.0)],
    gis.ternary_rgb)

add(119, "contour", "Contouring / Isolines", GI, "Cartography & Conversion",
    "Vector isolines (vertex table) from a grid.", [G()], [Param("levels", "Levels (count or comma list)", "12")],
    lambda grid, levels: gis.contour_lines(grid, levels if "," in levels else int(levels)))

add(125, "vec2ras", "Vector-to-Raster Conversion", GI, "Cartography & Conversion",
    "Rasterizes polygons (id, x, y vertex rows).", [T("Demo_polygons", "polygons", "Polygon vertices"), G("Demo_Bouguer", "reference", "Reference grid")],
    [col("id_col", "Polygon id", "id", "polygons"), Param("value_col", "Value column (optional)", "value")],
    lambda polygons, reference, id_col, value_col: gis.vector_to_raster(polygons, reference, id_col, "x", "y", value_col))

add("125.1", "ras2vec", "Raster-to-Vector Conversion", GI, "Cartography & Conversion",
    "Class boundaries as polygon vertex rows.", [G("Demo_geology_zones", "classes", "Class grid")], [], gis.raster_to_vector)

add(109, "idw", "IDW Interpolation", GI, "Interpolation",
    "Inverse-distance weighted gridding.", [T("Demo_gravity_stations")],
    _XYV + [Param("spacing", "Cell size", 1000.0), Param("power", "Power", 2.0), Param("neighbors", "Neighbours", 12)], interpolation.idw)

add(110, "spline", "Spline Interpolation", GI, "Interpolation",
    "Thin-plate (minimum bending) radial-basis spline.", [T("Demo_gravity_stations")],
    _XYV + [Param("spacing", "Cell size", 1000.0), Param("smoothing", "Smoothing", 0.0),
            Param("kernel", "Kernel", "thin_plate_spline", ("thin_plate_spline", "cubic", "quintic", "linear"))], interpolation.spline_interpolation)

add(120, "tin", "TIN Generation", GI, "Interpolation",
    "Delaunay triangulation honoring every sample (rasterized + triangle table).", [T("Demo_gravity_stations")],
    _XYV + [Param("spacing", "Cell size", 1000.0)], interpolation.tin_interpolation, "Delaunay (1934)")

add(112, "distance", "Euclidean Distance / Proximity", GI, "Proximity & Statistics",
    "Distance to nearest feature cell (value >= threshold).", [G("Demo_lineament_pixels", "features", "Feature grid")],
    [Param("threshold", "Feature threshold", 0.5)], gis.euclidean_distance)

add(118, "buffer", "Buffer Analysis", GI, "Proximity & Statistics",
    "Zone within a distance of feature cells.", [G("Demo_lineament_pixels", "features", "Feature grid")],
    [Param("distance", "Buffer distance (m)", 1500.0), Param("threshold", "Feature threshold", 0.5)], gis.buffer_zone)

add("118.1", "buffer_pts", "Buffer Analysis (points)", GI, "Proximity & Statistics",
    "Circular buffers around point locations.", [T("Demo_deposits", "points", "Points"), G("Demo_Bouguer", "reference", "Reference grid")],
    [Param("distance", "Buffer distance (m)", 2000.0), col("x_col", "X", "x", "points"), col("y_col", "Y", "y", "points")], gis.buffer_points)

add(113, "moran", "Spatial Autocorrelation (Moran's I)", GI, "Proximity & Statistics",
    "Global Moran's I with z-score.", [G()], [Param("distance", "Distance band (m)", 1500.0)], gis.morans_i, "Moran (1950)")

add(114, "gistar", "Hot Spot Analysis (Getis-Ord Gi*)", GI, "Proximity & Statistics",
    "Local Gi* z-scores for hot/cold spots.", [G()], [Param("distance", "Distance band (m)", 2000.0)], gis.getis_ord_gi_star, "Getis & Ord (1992)")

add(121, "viewshed", "Viewshed / Line-of-Sight", GI, "Terrain & Hydrology",
    "Visible cells from an observer.", [G("Demo_DEM", "dem", "DEM")],
    [Param("observer_x", "Observer x", 20000.0), Param("observer_y", "Observer y", 18000.0), Param("observer_height", "Observer height (m)", 10.0)],
    lambda dem, observer_x, observer_y, observer_height: gis.viewshed(dem, observer_x, observer_y, observer_height))

add(122, "lcp", "Least Cost Path", GI, "Terrain & Hydrology",
    "Dijkstra optimal route across a cost surface (e.g. slope).", [G("Demo_DEM_slope", "cost", "Cost grid")],
    [Param("start_x", "Start x", 1000.0), Param("start_y", "Start y", 1000.0), Param("end_x", "End x", 30000.0), Param("end_y", "End y", 30000.0)],
    gis.least_cost_path)

add(124, "watershed", "Watershed & Flow Accumulation", GI, "Terrain & Hydrology",
    "Priority-flood fill, D8 directions, flow accumulation and optional catchment.", [G("Demo_DEM", "dem", "DEM")],
    [Param("delineate", "Delineate watershed", True), Param("pour_x", "Pour point x", 20000.0), Param("pour_y", "Pour point y", 5000.0)],
    lambda dem, delineate, pour_x, pour_y: gis.flow_accumulation(dem, pour_x if delineate else None, pour_y if delineate else None),
    "O'Callaghan & Mark (1984)")

# ================================================================ DATA & GRID TOOLS
DT = "Data & Grid Tools"
add("", "stats", "Grid Statistics", DT, "Grid Tools", "Min/max/mean/std/median/RMS and NaN count.", [G()], [], extended_grid_statistics)
add("", "crop", "Crop Grid", DT, "Grid Tools", "Crop by coordinate bounds.", [G()],
    [Param("xmin", "X min", 0.0), Param("xmax", "X max", 20000.0), Param("ymin", "Y min", 0.0), Param("ymax", "Y max", 20000.0)], crop_grid)
add("", "flip", "Flip Grid", DT, "Grid Tools", "Flip values left-right or up-down.", [G()], [Param("axis", "Axis", "x", ("x", "y", "both"))], flip_grid)
add("", "rotate", "Rotate Grid", DT, "Grid Tools", "Rotate by 90-degree steps.", [G()],
    [Param("degrees_clockwise", "Degrees clockwise", 90, (90, 180, 270), kind="choice")], lambda grid, degrees_clockwise: rotate_grid(grid, int(degrees_clockwise)))
add("", "section", "Extract Cross-Section", DT, "Profile Tools", "Sample a grid along a straight line.", [G()],
    [Param("x1", "Start x", 0.0), Param("y1", "Start y", 16000.0), Param("x2", "End x", 31500.0), Param("y2", "End y", 16000.0), Param("spacing", "Spacing", 250.0)],
    lambda grid, x1, y1, x2, y2, spacing: extract_cross_section(grid, (x1, y1), (x2, y2), spacing))
add("", "prof_avg", "Profile Running Average", DT, "Profile Tools", "Centered moving average.", [PR("Demo_TMI_profile")],
    [Param("window_size", "Window (samples)", 5)], running_average_profile)
add("", "prof_detrend", "Profile Linear Detrend", DT, "Profile Tools", "Remove best-fit line.", [PR("Demo_TMI_profile")], [], remove_linear_trend_profile)
add("", "prof_poly", "Profile Polynomial Fit", DT, "Profile Tools", "Polynomial regional and residual.", [PR("Demo_TMI_profile")],
    [Param("degree", "Degree", 2)], polynomial_fit_profile)
add("", "prof_interp", "Profile Resampling", DT, "Profile Tools", "Resample to regular spacing.", [PR("Demo_TMI_profile")],
    [Param("spacing", "Spacing", 100.0)], interpolate_profile)
add("", "las_curve", "Well Log Curve to Profile", DT, "Profile Tools", "Plot a log curve against depth.", [_LOG],
    [col("depth_col", "Depth", "DEPTH"), col("value_col", "Curve", "GR")], wells.log_profile)


# ================================================================ lookup helpers
BY_KEY = {m.key: m for m in REGISTRY}


def domains() -> dict[str, dict[str, list[Method]]]:
    """Ordered tree: domain -> category -> methods (sorted by number)."""

    def order(m: Method) -> tuple:
        try:
            return (0, float(m.number))
        except ValueError:
            return (1, m.name)

    tree: dict[str, dict[str, list[Method]]] = {}
    for m in REGISTRY:
        tree.setdefault(m.domain, {}).setdefault(m.category, []).append(m)
    for cats in tree.values():
        for methods in cats.values():
            methods.sort(key=order)
    return tree


def numbered_count() -> int:
    return len({m.number.split(".")[0] for m in REGISTRY if m.number})
