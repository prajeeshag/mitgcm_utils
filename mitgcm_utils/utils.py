# type: ignore
import logging
import math
import re
import tempfile
from pathlib import Path
from typing import Any

import f90nml  # type: ignore
import numpy as np
import xarray as xr
from sphericalpolygon import Sphericalpolygon

nmlparser = f90nml.Parser()
nmlparser.comment_tokens += "#"
nmlparser.comment_tokens += "$"

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
logger.addHandler(handler)

RSPHERE = 6370000.0

MITGCM_GRID_VARS = [
    "xC",
    "yC",
    "dxF",
    "dyF",
    "rA",
    "xG",
    "yG",
    "dxV",
    "dyU",
    "rAz",
    "dxC",
    "dyC",
    "rAw",
    "rAs",
    "dxG",
    "dyG",
    "angleCosC",
    "angleSinC",
]


class CaseInsensitiveDict(dict[str, Any]):
    def __setitem__(self, key: str, value: Any):
        super().__setitem__(key.lower(), value)

    def __getitem__(self, key: str):
        return super().__getitem__(key.lower())

    def __contains__(self, key: str):
        return super().__contains__(key.lower())

    def get(self, key: str, default: Any = None):
        return super().get(key.lower(), default)

    def setdefault(self, key: str, default: Any = None):
        return super().setdefault(key.lower(), default)

    def update(self, other: Any = None, **kwargs):
        if other:
            if isinstance(other, dict):
                for key, value in other.items():
                    self[key.lower()] = value
            else:
                for key, value in other:
                    self[key.lower()] = value
        for key, value in kwargs.items():
            self[key.lower()] = value


def load_bathy(bathy_file: Path, nx: int, ny: int):
    z = np.fromfile(bathy_file, ">f4")
    if len(z) != nx * ny:
        if len(z) / 2 == nx * ny:
            z = np.fromfile(bathy_file, ">f8")
        else:
            logger.error(f"Expected size of bathymetry {nx*ny} but got {len(z)}")
            raise ValueError(
                f"Dimension mismatch for bathymetry field from file {bathy_file}"
            )
    return z.reshape(ny, nx)


def get_bathy(run_dir_path: str, nx: int, ny: int):
    nml_file = Path(run_dir_path) / "data"
    nml = nmlparser.read(nml_file)
    nml = CaseInsensitiveDict(nml)
    try:
        parm05 = nml["parm05"]
        parm05 = CaseInsensitiveDict(parm05)
    except KeyError as e:
        logger.error("&parm05 namelist does not exist")
        raise e
    bathy_file_name = parm05["bathyfile"].strip()
    if bathy_file_name[0] == "/":
        bathy_file = bathy_file_name
    else:
        bathy_file = Path(run_dir_path) / parm05["bathyfile"]
    return load_bathy(bathy_file, nx, ny)


def get_hgrid(run_dir_path: str, nx: int, ny: int, as_ds=False):
    nml_file = Path(run_dir_path) / "data"
    grid_file = Path(run_dir_path) / "tile001.mitgrid"
    nml = nmlparser.read(nml_file)
    nml = CaseInsensitiveDict(nml)
    try:
        parm04 = nml["parm04"]
        parm04 = CaseInsensitiveDict(parm04)
    except KeyError as e:
        logger.error("&parm04 namelist does not exist")
        raise e

    is_sp = False
    is_curv = False
    try:
        is_sp = parm04["usingsphericalpolargrid"]
    except KeyError:
        pass
    try:
        is_curv = parm04["usingcurvilineargrid"]
    except KeyError:
        pass

    if not is_curv and not is_sp:
        logger.error("Both `usingCurvilinearGrid` & `usingsphericalpolargrid` is False")
        raise ValueError(
            "Both `usingCurvilinearGrid` & `usingsphericalpolargrid` is False"
        )
    if is_curv and is_sp:
        logger.error("Both `usingCurvilinearGrid` & `usingsphericalpolargrid` is True")
        raise ValueError(
            "Both `usingCurvilinearGrid` & `usingsphericalpolargrid` is True"
        )

    if is_sp:
        xC, yC = get_hcoords_from_nml(parm04, nx, ny)

    if is_curv:
        gA = read_mitgcm_grid(grid_file, nx, ny)
        yC = gA["yC"][:-1, :-1]
        xC = gA["xC"][:-1, :-1]

    if as_ds:
        return _get_grid_ds(xC, yC)
    return _create_grid_file(xC, yC)


def get_hcoords_from_nml(nml, nx, ny):
    xgOrigin = nml["xgorigin"]
    ygOrigin = nml["ygorigin"]
    delX = nml["delX"]
    delY = nml["delY"]
    if nx != len(delX):
        raise ValueError("len(delX) != nx")
    if ny != len(delY):
        raise ValueError("len(delY) != ny")
    xG = np.zeros(nx + 1)
    yG = np.zeros(ny + 1)
    xC = np.zeros([ny, nx])
    yC = np.zeros([ny, nx])
    xG[0], yG[0] = xgOrigin, ygOrigin

    for i in range(nx):
        xG[i + 1] = xG[i] + delX[i]
        xC[:, i] = xG[i] + 0.5 * delX[i]

    for i in range(ny):
        yG[i + 1] = yG[i] + delY[i]
        yC[i, :] = yG[i] + 0.5 * delY[i]

    return xC, yC


def _create_grid_file(xC, yC, var=None):
    ds_out = _get_grid_ds(xC, yC, var)
    encoding = {var: {"_FillValue": None} for var in ds_out.variables}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmpfile:
        logger.info(f"Writing grid file to {tmpfile.name}")
        ds_out.to_netcdf(tmpfile.name, encoding=encoding)
    return tmpfile.name


def _get_grid_ds(xC, yC, var=None):
    if var is None:
        var = xC
    ds_out = xr.Dataset(
        {
            "lat": (
                ["y", "x"],
                yC,
                {"units": "degrees_north"},
            ),
            "lon": (
                ["y", "x"],
                xC,
                {"units": "degrees_east"},
            ),
            "var": (
                ["y", "x"],
                var,
                {"units": "", "coordinates": "lat lon"},
            ),
        }
    )
    return ds_out


def read_mitgcm_grid(
    grid_file: Path, nx: int, ny: int
) -> dict[str, np.ndarray[Any, np.dtype[Any]]]:
    nx1, ny1 = nx + 1, ny + 1
    nxy1 = nx1 * ny1
    fdata = np.fromfile(grid_file, ">f8")
    nvars = int(fdata.shape[0] / nxy1)
    if nvars not in [len(MITGCM_GRID_VARS), len(MITGCM_GRID_VARS) - 2]:
        raise ValueError(
            f"{grid_file} does not contain enough variables needed for a mitgcm grid"
        )
    nele1 = nvars * nxy1
    nele = fdata.shape[0]
    if nvars * nxy1 != fdata.shape[0]:
        raise ValueError(
            f"nvars*(nx+1)*(ny+1) != shape of the data read: {nele1} != {nele}"
        )
    fdata = fdata.reshape([nvars, ny1, nx1])
    gridA: dict[str, np.ndarray[Any, np.dtype[Any]]] = {}
    for i in range(nvars):
        gridA[MITGCM_GRID_VARS[i]] = fdata[i, :, :]
    return gridA


def vgrid_from_parm04(nml_file):
    nml = nmlparser.read(nml_file)
    nml = CaseInsensitiveDict(nml)
    try:
        nml = nml["parm04"]
    except KeyError as e:
        logger.error("&parm04 namelist does not exist")
        raise e

    nml = CaseInsensitiveDict(nml)
    try:
        delz = np.array(nml["delr"])
    except KeyError:
        logger.error("delr does not exist in &parm04. trying delz")
        try:
            delz = np.array(nml["delz"])
        except KeyError as e:
            logger.error("delr and delz does not exist in &parm04")
            raise e

    zi = [0.0]
    for dz in delz:
        zi.append(zi[-1] + dz)

    z = np.array(zi[1:]) - delz * 0.5
    return z


def fill_missing3D(arr):
    for i in range(arr.shape[0]):
        arr2D = arr[i, :, :]
        if np.all(np.isnan(arr2D)):
            if i == 0:
                raise ValueError(f"layer {i} contains all missing values, cannot fill")
            arr2D = arr[i - 1, :, :]
        else:
            fill_missing2D(arr2D)
        arr[i, :, :] = arr2D


def fill_missing2D(arr):
    """
    Fill in missing values in a 2D NumPy array with their
    nearest non-missing neighbor.
    """

    # Get the indices of all missing values in the array
    missing_indices = np.argwhere(np.isnan(arr))

    # Get the shape of the array
    nrows, ncols = arr.shape
    # Define the directions to search for nearest neighbors
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    # Perform a spiral grid search to fill in missing values with nearest neighbors
    arr_copy = arr.copy()
    for r, c in missing_indices:
        for i in range(1, max(nrows, ncols)):
            for dr, dc in directions:
                nr, nc = r + i * dr, c + i * dc
                if (
                    nr >= 0
                    and nr < nrows
                    and nc >= 0
                    and nc < ncols
                    and not np.isnan(arr[nr, nc])
                ):
                    arr_copy[r, c] = arr[nr, nc]
                    break
            if not np.isnan(arr[r, c]):
                break
    arr[:, :] = arr_copy[:, :]
    return arr_copy


def get_dimlist_from_meta_file(fname: Path) -> list[list[int]]:
    """Get the dimList out of the MITgcm mds .meta file."""
    flds: dict[str, Any] = {}
    with open(fname) as f:
        text = f.read()
    # split into items
    for item in re.split(";", text):
        # remove whitespace at beginning
        item = re.sub(r"^\s+", "", item)
        match = re.match(r"(\w+) = (\[|\{)(.*)(\]|\})", item, re.DOTALL)
        if match:
            key, _, value, _ = match.groups()
            # remove more whitespace
            value = re.sub(r"^\s+", "", value)
            value = re.sub(r"\s+$", "", value)
            # print key,':', value
            flds[key] = value
    # now check the needed things are there
    needed_keys = ["dimList"]
    for k in needed_keys:
        assert k in flds
    dimList: list[list[int]] = [
        [int(h) for h in re.split(",", g)] for g in re.split(",\n", flds["dimList"])
    ]
    return dimList


def great_circle(lon1, lat1, lon2, lat2, input_in_radians=False, rearth=RSPHERE):
    """
    Calculates the great circle distance between two points on the Earth's surface,
    given their longitude and latitude coordinates.

    Arguments:

    lon1 (float): the longitude of the first point
    lat1 (float): the latitude of the first point
    lon2 (float): the longitude of the second point
    lat2 (float): the latitude of the second point
    input_in_radians (bool): a flag indicating whether the
            input coordinates are in radians (True) or degrees (False).
            Default is False.
    rearth (float): the radius of the Earth in kilometers. Default is 6370000 m.
    Returns:

    The great circle distance between the two points, in kilometers."""
    xlon1, xlat1, xlon2, xlat2 = lon1, lat1, lon2, lat2
    if not input_in_radians:
        xlon1, xlat1, xlon2, xlat2 = map(math.radians, [xlon1, xlat1, xlon2, xlat2])
    dlon = xlon2 - xlon1
    dlat = xlat2 - xlat1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(xlat1) * math.cos(xlat2) * math.sin(dlon / 2) ** 2
    )
    return 2.0 * rearth * math.asin(math.sqrt(a))


def quadrilateral_area_on_earth(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
    R: float = RSPHERE,
) -> float:
    # return polygon_area([a, b, c, d, a]) * R * R
    arr = [a, b, c, d]
    polygon = Sphericalpolygon.from_array([a, b, c, d])
    return polygon.area(R)


if __name__ == "__main__":
    print(get_dimlist_from_meta_file(Path("test_data/grid_data/XC.meta")))
    print(get_dimlist_from_meta_file(Path("test_data/grid_data/RC.meta")))
    print(get_dimlist_from_meta_file(Path("test_data/grid_data/hFacC.meta")))
