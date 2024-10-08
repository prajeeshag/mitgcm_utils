# type: ignore
import logging
import re
from pathlib import Path
from typing import Any

import f90nml  # type: ignore
import numpy as np

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
logger.addHandler(handler)

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
        raise ValueError(
            f"Dimension mismatch for bathymetry field from file {bathy_file}"
        )
    return z.reshape(ny, nx)


def load_grid(
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
    nmlparser = f90nml.Parser()
    nmlparser.comment_tokens += "#"
    nmlparser.comment_tokens += "$"

    nml = nmlparser.read(nml_file)
    nml = CaseInsensitiveDict(nml)
    try:
        nml = nml["parm04"]
    except KeyError as e:
        print(nml)
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
    print(arr.shape)
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
                    arr[r, c] = arr[nr, nc]
                    break
            if not np.isnan(arr[r, c]):
                break


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


if __name__ == "__main__":
    print(get_dimlist_from_meta_file(Path("test_data/grid_data/XC.meta")))
    print(get_dimlist_from_meta_file(Path("test_data/grid_data/RC.meta")))
    print(get_dimlist_from_meta_file(Path("test_data/grid_data/hFacC.meta")))
