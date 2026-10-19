"""OGE cropland binary semantic segmentation."""
import importlib.util
import os
from pathlib import Path


# Windows GIS installations often export a different PROJ/GDAL database globally.
# Prefer resources bundled with the installed Rasterio wheel, for this process only.
_rasterio_spec = importlib.util.find_spec("rasterio")
if _rasterio_spec and _rasterio_spec.origin:
    _package = Path(_rasterio_spec.origin).parent
    if (_package / "proj_data/proj.db").is_file():
        os.environ["PROJ_DATA"] = str(_package / "proj_data")
        os.environ["PROJ_LIB"] = str(_package / "proj_data")
    if (_package / "gdal_data").is_dir():
        os.environ["GDAL_DATA"] = str(_package / "gdal_data")
