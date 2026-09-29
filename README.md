# Geonarba

GEONARBA is a gravity and magnetic potential-field anomaly processing workspace. The project now contains:

- `desktop/`: the GeoFieldLab Pro Desktop V1 app for professional interpreter workflows.
- `app.py`: the original Streamlit prototype, kept as a reference and quick web demo.
- `core/`: UI-independent scientific routines shared by tests, Streamlit, and desktop controllers.

All of these live in [`Geonarba/`](./Geonarba/).

## Install

```powershell
cd Geonarba
pip install -r requirements.txt
python -m desktop
```

See [Geonarba/README.md](./Geonarba/README.md) for the Streamlit demo, tests, supported input, and methods.

## License

This repository is released under the MIT License.
