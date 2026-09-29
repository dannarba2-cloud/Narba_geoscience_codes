# GEONARBA V1

GEONARBA is a gravity and magnetic potential-field anomaly processing workspace. The project now contains:

- `desktop/`: the GeoFieldLab Pro Desktop V1 app for professional interpreter workflows.
- `app.py`: the original Streamlit prototype, kept as a reference and quick web demo.
- `core/`: UI-independent scientific routines shared by tests, Streamlit, and desktop controllers.

## Install

```powershell
cd Geonarba
pip install -r requirements.txt
python -m desktop
```

The Streamlit prototype can still be launched with:

```powershell
streamlit run app.py
```

## Test

```powershell
python -m pytest tests
```

## Supported Input

Upload a CSV, TXT, XYZ, or DAT file with columns for:

- `x`: projected x coordinate in meters
- `y`: projected y coordinate in meters
- anomaly value, such as Complete Bouguer anomaly in mGal or magnetic anomaly in nT

The UI lets you choose the x, y, and anomaly columns. If points are scattered, GEONARBA interpolates them to a regular grid using `scipy.interpolate.griddata`.

The example file is `examples/synthetic_anomaly.csv`. It covers a 50 km x 50 km projected area with 1000 m spacing and contains a shallow Gaussian source, a broad deeper source, and small noise.

## Desktop V1

The desktop app is started with `python -m desktop` or `python desktop_launcher.py`. It provides:

- Project/session state with raw table data, layers, profiles, history, settings, RAPS summary, and `.gflp` save/load.
- Navigation pages: Project, Data, Map, Profile, Regional/Residual, Derivatives, Filters, Edges, Spectrum, and Export.
- Layer and profile inventories, Matplotlib map/profile previews, processing history, and export actions.
- Controller-backed workflows that call pure `core/` functions instead of duplicating numerical logic in the UI.

## V1 Methods

- Regular grid detection and scattered interpolation.
- Matrix-to-XYZ, XYZ-to-grid, crop, flip, rotate, grid statistics, and profile statistics.
- Cross-section extraction and profile operations: interpolation, running average, linear detrending, and polynomial fit/residual.
- Total horizontal gradient: `THG = sqrt((dT/dx)^2 + (dT/dy)^2)`.
- First vertical derivative by FFT: `FVD = IFFT(|k| * FFT(T))`.
- Second vertical derivative by FFT: `SVD = IFFT(|k|^2 * FFT(T))`.
- Tilt derivative: `TILT = arctan(FVD / (THG + epsilon))`.
- Analytic Signal Amplitude: `ASA = sqrt((dT/dx)^2 + (dT/dy)^2 + (dT/dz)^2)`.
- Upward continuation: `UC = IFFT(FFT(T) * exp(-|k| * height))`.
- Residual anomaly from upward continuation: `Residual = Original - UC`.
- Butterworth low-pass, high-pass, band-pass, regional, and residual filters.
- FSED edge detection using THG, ASA, absolute TILT, or direct input.
- RAPS spectral analysis with radial angular wavenumber bins and depth estimate.

## RAPS Depth Convention

GEONARBA uses angular radial wavenumber `k` in rad/m and natural log power:

```text
y = ln(radially averaged power)
x = k in rad/m
depth = -slope / 2
```

If a spectrum is built from cycles/m or cycles/km elsewhere, convert to angular wavenumber first with `k = 2*pi*f`.

## Scientific Warnings

- FFT derivatives, upward continuation, RAPS, and wavelength filters require regular projected coordinates in meters.
- Longitude/latitude coordinates should be reprojected to UTM or another projected CRS before interpretation.
- FVD enhances shallow sources and noise.
- SVD strongly amplifies noise and should be interpreted cautiously.
- Upward continuation is a smoothing transform, not a unique geological separation.
- Butterworth regional/residual separation depends on cutoff wavelength and filter order.
- RAPS is sensitive to windowing, padding, detrending, grid size, and slope segment choice.
- A 50 km x 50 km study area cannot reliably resolve very deep sources.
- Edge filters show mathematical gradients; geological validation is required before interpreting faults.

## Export

Each computed grid can be exported as:

- CSV grid with `x,y,value,layer_name,units`
- PNG map figure
- Processing history JSON
- Markdown processing report

Profiles can be exported as CSV with `distance,value,profile_name,units`. Desktop project state can be saved as `.gflp` JSON.

## Windows Packaging

After installing requirements, build a first Windows desktop distribution with:

```powershell
pyinstaller GeoFieldLabPro.spec
```

The PyInstaller spec bundles the desktop launcher and the `examples/` folder. A future installer step can wrap `dist\GeoFieldLabPro` with Inno Setup or WiX for desktop shortcuts and Start Menu entries.

## Project Structure

```text
Geonarba/
|-- app.py
|-- requirements.txt
|-- README.md
|-- desktop/
|   |-- main.py
|   |-- controller.py
|   |-- project.py
|   `-- __main__.py
|-- core/
|   |-- data_io.py
|   |-- grid_tools.py
|   |-- grid_operations.py
|   |-- profile.py
|   |-- derivatives.py
|   |-- fft_filters.py
|   |-- tilt.py
|   |-- analytic_signal.py
|   |-- edge_detection.py
|   |-- spectral.py
|   |-- visualization.py
|   |-- exports.py
|   `-- history.py
|-- ui/
|   |-- panels.py
|   `-- method_registry.py
|-- examples/
|   `-- synthetic_anomaly.csv
|-- tests/
|-- docs/
`-- outputs/
```

## First Workflow

1. Load the example dataset or upload CSV/XYZ data.
2. Select x, y, and anomaly columns.
3. Create the regular grid or interpolate scattered points.
4. Display the original anomaly map.
5. Compute upward or Butterworth regional anomaly.
6. Compute residual anomaly.
7. Compute THG.
8. Compute FVD and SVD.
9. Compute TILT.
10. Compute ASA.
11. Extract a cross-section profile if needed.
12. Compute FSED.
13. Compute RAPS and depth estimate.
14. Export maps, profiles, grids, history, report, and `.gflp` project state.

## Roadmap

Future versions are planned to add gravity corrections, magnetic corrections, RTP/RTE, Euler deconvolution, SPI, tilt-depth, Werner, Peters, Curie depth, Parker-Oldenburg inversion, GIS lineament extraction, rose diagrams, and geothermal favorability mapping.
