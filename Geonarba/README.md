# GEONARBA

GEONARBA is a geophysical processing and interpretation workspace. It started as a gravity and magnetic
potential-field tool and now covers potential fields, seismic, electrical and EM methods, well logs and
radiometrics, machine learning and GIS: 125 numbered methods plus their sub-options (150 tools in total).

- `desktop/`: the GeoFieldLab Pro desktop app (PySide6), the main professional interface.
- `core/`: UI-independent scientific routines shared by the desktop app, Streamlit and the tests.
- `app.py`: the original Streamlit prototype, kept as a reference and quick web demo.

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

The suite runs every registered method on its demo dataset (`tests/test_method_library.py`) and checks the physics
against analytic solutions (`tests/test_physics.py`): prism vs point mass and dipole, GRS80 normal gravity, RTP,
Euler/Werner depths, 2-layer VES image series, MT half-space, NMO and plus-minus.

## Desktop App Structure

The desktop window uses a standard processing-suite layout:

| Area | Contents |
|------|----------|
| Menu bar | File, Data, one menu per domain (category sub-menus, then methods), View, Help |
| Project Explorer (left) | Grids, Profiles, Sections, Tables, Images, with right-click View / Export / Delete |
| Processing Toolbox (right) | Searchable Domain > Category > Method tree, plus a parameter panel generated for each method |
| Centre tabs | Viewer (maps, profiles, sections, rose diagrams, RGB), Table, Report, Data Import, History |
| Log (bottom) | Run messages, timings and scientific warnings |

Every method panel shows the description, reference, inputs (a single object or a multi-selection) and typed
parameters. Choice lists are the sub-options, for example padding mode, filter type, structural index, array type,
fuzzy operator or kriging variogram. **Use demo data** loads a matching synthetic dataset, so every method can be
tried immediately, and **File > Load Demo Library** loads all of them. Heavy methods run on a background thread.

All methods are declared once in `core/registry.py`. The toolbox, menus, forms and tests are generated from that list,
so adding a method means adding one `add(...)` entry.

## Method Library

| Domain | Categories | Methods |
|--------|-----------|---------|
| Potential Fields | Magnetic Reduction, Gravity Reduction, Continuation & Filtering, Derivatives & Edge Detection, Depth Estimation, Thermal Analysis, Modeling & Inversion | 1-30 (RTP, RTE, UC/DC, FVD/SVD, THDR, AS, TDR, Theta, FSED, Euler, Werner, RAPS, CPD, heat flow, FA/Bouguer/terrain/isostatic, Butterworth, pseudogravity, 2.5D, voxel inversion, SPI, matched filter, tilt-depth, equivalent sources, drift & tide, IGRF) |
| Seismic | Reflection Processing, Imaging, Reservoir Characterization, Waveform Modeling & Inversion, Refraction, Passive & Surface Waves, Borehole Seismic | 31-55 (NMO, DMO, CMP stack, PreSTM, PreSDM, Stolt, deconvolution, f-k, AVO, AI inversion, FWI, first breaks, plus-minus, GRM, tomography, HVSR, SPAC, MASW, receiver functions + H-kappa, focal mechanisms, interferometry, VSP, Q, attributes, coherence, spectral decomposition) |
| Electrical & EM | DC Resistivity & IP, Magnetotellurics, Time & Frequency-Domain EM, Ground Penetrating Radar | 56-71 (VES, 2.5D ERT, IP, SIP Cole-Cole, MT tensor, tipper, Occam 1D, CSAMT, TEM, AEM CDT, FDEM, GPR dewow/background/migration, SP modeling, VLF Fraser) |
| Well Logging & Radiometrics | Radiometrics, Petrophysics, Borehole Geophysics | 72-80 (gamma spectrometry, heat production, SP baseline, Vsh, Archie, density-neutron, sonic integration, dipmeter, cross-hole ERT, NMR T2) |
| Advanced Processing & ML | Inversion & Optimization, Gridding & Geostatistics, Machine Learning, Signal & Texture Analysis | 81-100 (joint cross-gradient inversion, PINN, minimum curvature, kriging, cokriging, PCA, neural first-break picking, SOM, SVM, random forest, K-means, neural fault detection, MCMC, PSO, SA, GA, TSVD, wavelets, fractal dimension, directional cosine) |
| GIS & Spatial Analysis | Raster Analysis, Structural Interpretation, Prospectivity & MCDA, Cartography & Conversion, Interpolation, Proximity & Statistics, Terrain & Hydrology | 101-125 (map algebra, lineaments + density, rose diagram, WofE, fuzzy, AHP, hillshade, georeferencing, IDW, spline, zonal stats, distance, Moran's I, Gi*, terrain analysis, RGB ternary, trend surface, buffer, contours, TIN, viewshed, least-cost path, cut/fill, watershed, vector/raster) |
| Data & Grid Tools | Grid Tools, Profile Tools | statistics, crop, flip, rotate, cross-section, profile filters, log curves |

**Help > Method Catalogue** in the app lists all 150 tools.

### Inversion framework (`core/inversion.py`)

The voxel, ERT, IP, cross-hole and profile inversions share one Tikhonov/Occam engine that minimizes
`Phi = ||W_d (G m - d)||^2 + lambda ||W_m (m - m_ref)||^2`:

- **Data weighting:** `W_d = 1/sigma`, with sigma = error % x |d| + floor.
- **Smoothing operator:** 1st derivative (blocky with transitions), 2nd derivative (smoothest) or smallness.
  Weights `alpha_s`, `alpha_x`, `alpha_y`, `alpha_z` set the balance. For example, `alpha_z < alpha_x` favors layered models.
- **Reference model** `m_ref`: a background half-space or a well-derived value.
- **Lambda selection:**
  - **discrepancy** (default, Occam): the smoothest model with normalized RMS = 1, refined by bisection.
  - **l-curve**: the point of maximum curvature.
  - **gcv**: generalized cross-validation, using a randomized trace estimate for large models.
  - **fixed**
- **Solvers:** direct normal equations, CG (LSQR), or subspace.
- **QC report:**
  - chi^2 and normalized RMS, with a verdict (overfitting < 0.8 < fits noise < 1.2 < underfitting)
  - Durbin-Watson and lag-1 autocorrelation
  - a normality test
  - Moran's I of residual maps
  - a flag when residuals are coherent rather than random
- **Outputs:** predicted and residual maps or tables, and an L-curve plot with the chosen lambda starred.

For underdetermined problems, L-curve and GCV often choose lambdas that fit below the noise level. The report flags
this, which is why the discrepancy principle is the default. Method 97.1 (*Tikhonov / Occam Inversion Studio*) exposes
every control on a gravity profile. Method 97.2 (*Inversion Residual Analysis*) runs the residual QC on any
observed/calculated grid pair.

### Implementation notes and known limits

- Magnetic and gravity forward modeling, voxel/joint inversion and terrain corrections use exact right-rectangular prism
  kernels (Nagy et al., 2000). Magnetics are built through the Poisson relation from the gradient tensor.
- 2.5D modeling discretizes polygons into vertical prisms of finite strike +-L.
- PreSDM, tomography and ray traveltimes use shortest-path (Dijkstra) ray tracing on the model grid.
- ERT, IP and cross-hole tomography share one 2.5D finite-volume engine with adjoint sensitivities.
- The neural first-break picker and fault detector use scikit-learn multilayer perceptrons, not CNNs (no deep-learning
  dependency). The PINN is a one-hidden-layer network with analytic derivatives, applied to the 1D steady geotherm.
- FWI is 2D acoustic, time-domain, and meant for small models. MT inversion (Occam, GA) is 1D. MASW converts dispersion
  to Vs with the wavelength-depth approximation rather than a full inversion.
- IGRF uses the IGRF-14 coefficients shipped with `ppigrf`.

## Supported Input

- **Tables** (CSV, TXT, XYZ, DAT): station data, logs, time series, picks. Grids are created from x/y/value columns
  in the *Data Import* tab (regular reshape or interpolation), or with the gridding methods (83-85, 109, 110, 120).
- **LAS 2.0** well logs are imported as tables.
- **Sections** (seismic/GPR): a samples x traces matrix from CSV/TXT/NPY, via *File > Import Section*.
- Coordinates for FFT methods must be projected (meters); longitude/latitude grids trigger a warning.

## Conventions

- x east, y north, z down. Angular wavenumber k = 2*pi/wavelength (rad/m).
- FFT derivative d/dz = |k|; upward continuation exp(-|k| h). RAPS depth = -slope/2 of ln(P) vs k.
- Gravity in mGal (GRS80 Somigliana normal gravity), magnetics in nT, magnetization in A/m (SI susceptibility).
- MT impedance in (mV/km)/nT, rho_a = 0.2 |Z|^2 / f, with the half-space phase at 45 deg.

## Scientific Warnings

- FFT derivatives, continuation, RAPS and wavelength filters require regular projected coordinates in meters.
- FVD/SVD and downward continuation amplify noise; do not continue below the shallowest source.
- RTP is unstable at low magnetic latitudes: use stabilization, RTE or the analytic signal.
- Regional/residual separation, spectral depths and Curie depths depend on the chosen parameters and window size.
- Edge filters show mathematical gradients; geological validation is required before interpreting faults.
- Inversions are non-unique: report regularization, misfit and coverage. ML results depend on the training data.

## Export

Grids, profiles, tables and sections export as CSV. The current figure exports as PNG, PDF or SVG. The processing
history exports as JSON and the processing report as Markdown. The full project (every data type and every report)
saves as `.gflp` JSON.

## Windows Packaging

```powershell
pyinstaller GeoFieldLabPro.spec
```

The spec bundles the desktop launcher, `examples/` and the IGRF-14 coefficients.

## Project Structure

```text
Geonarba/
|-- app.py                     Streamlit prototype
|-- desktop/
|   |-- qt_app.py              main window: menus, explorer, toolbox, parameter panel, viewers
|   |-- controller.py          run_method(), imports/exports, demo library
|   `-- project.py             session with grids, profiles, sections, tables, images, reports
|-- core/
|   |-- registry.py            catalogue of every method (drives the UI and tests)
|   |-- fourier.py             shared wavenumber-domain helpers
|   |-- prisms.py              gravity / tensor / magnetic prism kernels
|   |-- potential_field.py     RTP, RTE, DC, pseudogravity, theta, matched & directional filters, IGRF
|   |-- gravity.py             normal gravity, FA, Bouguer, terrain, isostasy, drift & Longman tide
|   |-- depth.py               Euler, Werner, SPI, tilt-depth, Curie depth, heat flow
|   |-- modeling.py            2.5D forward, voxel, equivalent sources, joint inversion
|   |-- seismic.py             Section type, NMO/DMO/stack, migration, decon, f-k, AVO, AI, attributes, VSP, Q
|   |-- seismic_wave.py        2D acoustic FD modeling and FWI
|   |-- refraction.py          picking, plus-minus, GRM, shortest-path tomography
|   |-- passive.py             HVSR, SPAC, MASW, receiver functions, focal mechanisms, interferometry
|   |-- electrical.py          VES, 2.5D ERT/IP/cross-hole, SIP, SP, VLF
|   |-- em.py                  MT, CSAMT, TEM, AEM, FDEM, GPR
|   |-- wells.py               LAS, petrophysics, NMR, radiometrics
|   |-- optimize.py            PSO, SA, GA, MCMC, TSVD, PINN
|   |-- interpolation.py       minimum curvature, kriging, cokriging, IDW, spline, TIN
|   |-- ml.py                  PCA, K-means, SOM, SVM, RF, neural picking / faults, wavelets, fractals
|   |-- gis.py                 map algebra, terrain, lineaments, prospectivity, georeferencing, hydrology
|   |-- synthetic.py           physically consistent demo datasets
|   `-- (derivatives, fft_filters, tilt, analytic_signal, edge_detection, spectral, grid_tools, profile, ...)
|-- ui/                        Streamlit panels and catalogue labels
|-- examples/
`-- tests/
```
