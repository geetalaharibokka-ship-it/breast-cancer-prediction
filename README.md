# ML Workspace — Python Structure

This is a Python-only machine-learning website structure using **Streamlit**.
Open this folder in VS Code and build each feature inside the matching module.

## Folders

```text
ml-workspace-python/
├── app.py                     # Landing page and shared navigation
├── pages/                     # Streamlit pages shown in the sidebar
├── core/                      # Dataset, preprocessing, algorithms, metrics
├── components/                # Reusable visual interface sections
├── assets/                    # Images, icons, and custom CSS
├── data/                      # Temporary/sample datasets (not user uploads)
├── exports/                   # Generated reports and result CSV files
├── tests/                     # Automated checks
└── .streamlit/config.toml     # Theme configuration
```

## Start in VS Code

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Build order

1. Complete `core/dataset_loader.py` for CSV, Excel, JSON, TSV, and ARFF upload.
2. Complete preprocessing and feature selection.
3. Add algorithms in `core/classification.py` and `core/clustering.py`.
4. Use the pages and components folders to present metrics, trees, charts, and exports.
