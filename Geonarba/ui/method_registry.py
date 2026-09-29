"""Method catalogue labels for the Streamlit prototype (the full, runnable catalogue lives in core.registry)."""

from core.registry import domains


def method_catalogue() -> dict[str, list[str]]:
    """Domain -> list of 'N. Method name' labels for every implemented method."""

    return {domain: [m.label for methods in cats.values() for m in methods] for domain, cats in domains().items()}


# Kept for backward compatibility: these modules are now implemented (desktop Processing Toolbox).
FUTURE_MODULES = method_catalogue()


SCIENTIFIC_WARNINGS = [
    "FFT derivatives, RAPS, continuation and wavelength filters require regular projected coordinates in meters.",
    "Coordinates in degrees should be reprojected to UTM before interpretation.",
    "First vertical derivative enhances shallow sources and noise; the second derivative strongly amplifies noise.",
    "Upward continuation smooths anomalies and is not a unique geological separation.",
    "Downward continuation is unstable: never continue below the shallowest source and increase regularization if noisy.",
    "RTP is unstable at low magnetic latitudes (|I| < 20 deg): use stabilization, RTE or the analytic signal.",
    "Butterworth regional/residual separation depends on cutoff wavelength and filter order.",
    "RAPS, matched-filter and Curie-depth estimates depend on windowing, detrending, window size and the fitted wavenumber band.",
    "Curie-depth windows should be several times larger than the expected Curie depth.",
    "FFT methods may suffer from boundary artifacts. Use tapering and padding.",
    "Filters highlight mathematical gradients and edges, not automatically faults. Geological validation is required.",
    "Inversions (voxel, joint, ERT, MT, FWI, tomography) are non-unique: report regularization, misfit and data coverage.",
    "Machine-learning outputs depend on training data; validate on independent wells/areas.",
]
