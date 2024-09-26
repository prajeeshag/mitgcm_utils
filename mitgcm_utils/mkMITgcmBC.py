import logging
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import typer
import xarray as xr
from cdo import Cdo  # type: ignore

from .utils import (
    fill_missing2D,  # type: ignore
    fill_missing3D,  # type: ignore
    get_dimlist_from_meta_file,
    load_bathy,
    load_grid,
    vgrid_from_parm04,  # type: ignore
)

app = typer.Typer(add_completion=False)


class Boundary(str, Enum):
    south = "S"
    north = "N"
    east = "E"
    west = "W"


BNDDEF = {
    "W": (slice(None), slice(0, 1)),
    "S": (slice(0, 1), slice(None)),
    "E": (slice(None), slice(-1, None)),
    "N": (slice(-1, None), slice(None)),
}

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
logger.addHandler(handler)


def gen_bnd_grid(mitgrid: Path, nx: int, ny: int, bathy_file: Path):
    """Generate MITgcm boundary grid files later to be used to interpolate the boundary condition"""

    logger.info("Reading bathymetry and grid info")

    gA = load_grid(mitgrid, nx, ny)
    z = load_bathy(bathy_file, nx, ny)
    lat = gA["yC"][:-1, :-1]
    lon = gA["xC"][:-1, :-1]

    omask = np.array(z.shape, dtype=int)
    omask = np.where(z < 0, 1, 0)
    return lat, lon, omask


def mk_bnd_grid(
    lat: np.ndarray[Any, np.dtype[Any]],
    lon: np.ndarray[Any, np.dtype[Any]],
    omask: np.ndarray[Any, np.dtype[Any]],
    boundaries: list[Boundary] = [],
) -> list[tuple[str, str]]:
    bndAct: list[tuple[str, str]] = []

    for bnd in BNDDEF:
        if boundaries and bnd not in boundaries:
            continue

        bndMask = omask[BNDDEF[bnd]]
        isboundary = np.any(bndMask != 0)
        bndPoints = np.count_nonzero(bndMask)
        logger.info(f"{bnd}: {isboundary}, {bndPoints}")
        if not isboundary:
            continue

        latitude = lat[BNDDEF[bnd]]
        longitude = lon[BNDDEF[bnd]]
        valid_index_forword = -1
        valid_index_reverse = -1
        for i in range(bndMask.shape[0]):
            i_r = bndMask.shape[0] - 1 - i
            if bndMask[i, 0] != 0:
                valid_index_forword = i
            if bndMask[i_r, 0] != 0:
                valid_index_reverse = i_r
            if valid_index_forword != -1 and bndMask[i, 0] == 0:
                longitude[i, 0] = longitude[valid_index_forword, 0]
                latitude[i, 0] = latitude[valid_index_forword, 0]
            if valid_index_reverse != -1 and bndMask[i_r, 0] == 0:
                longitude[i_r, 0] = longitude[valid_index_reverse, 0]
                latitude[i_r, 0] = latitude[valid_index_reverse, 0]

        longitude = longitude.squeeze()
        latitude = latitude.squeeze()
        bndMask = bndMask.squeeze()
        ds_out = xr.Dataset(
            {
                "lat": (
                    ["y"],
                    latitude,
                    {"units": "degrees_north"},
                ),
                "lon": (
                    ["y"],
                    longitude,
                    {"units": "degrees_east"},
                ),
                "da": (
                    ["y"],
                    bndMask,
                    {"units": "1", "coordinates": "lat lon"},
                ),
            }
        )
        encoding = {var: {"_FillValue": None} for var in ds_out.variables}
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmpfile:
            logger.info(f"Writing {bnd} boundary grid to file {tmpfile.name}")
            ds_out.to_netcdf(tmpfile.name, encoding=encoding)  # type: ignore
        bndAct.append((bnd, tmpfile.name))
    return bndAct


@app.command(hidden=True)
def igrid(
    input: str = typer.Option(
        help="""
        Input can be: \n
         1. A NetCDF file.
         2. A valid cdo option which will generate a NetCDF file.
         e.g. "-mergetime input1.nc input2.nc input3.nc"
         """,
    ),
    nx: int = typer.Option(
        help="Number of points in x-direction",
    ),
    ny: int = typer.Option(
        help="Number of points in y-direction",
    ),
    field: str = typer.Option(
        help="""Boundary field name, i.e. T, S, U, V,... \n
            e.g.; This will be used to generate files <field>_E.bin, <field>_W.bin,..
            """,
    ),
    bathymetry: Path = typer.Option(
        default=Path("./bathymetry.bin"),
        exists=True,
        dir_okay=False,
        help="Path to the MITgcm bathymetry file",
    ),
    nml: Path = typer.Option(
        default=Path("./data"),
        exists=True,
        dir_okay=False,
        help="Path to the MITgcm `data` namelist file",
    ),
    mitgrid: Path = typer.Option(
        default=Path("./tile001.mitgrid"),
        exists=True,
        dir_okay=False,
        help="Path to the MITgcm data namelist file",
    ),
    addc: float = typer.Option(
        default=0.0,
        help="Add a constant to the input field",
    ),
    mulc: float = typer.Option(
        default=1.0,
        help="Multiply a constant to the input field",
    ),
):
    """
    Use input grid informations from namelist and bathymetry file to generate MITgcm boundary conditions
    """
    z = vgrid_from_parm04(nml)

    # generate boundary grids
    lat, lon, omask = gen_bnd_grid(mitgrid, nx, ny, bathymetry)

    mk_obcs(input, field, addc, mulc, z, lat, lon, omask)  # type: ignore


@app.command()
def mds(
    input: str = typer.Option(
        help="""
            Input can be: \n
             1. A NetCDF file. \n 
             2. A valid cdo option which will generate a NetCDF file. \n 
             e.g. "-mergetime input1.nc input2.nc input3.nc"
             """,
    ),
    field: str = typer.Option(
        help="field to be added to the output files, i.e. <field>_E.bin, <field>_W.bin,..",
    ),
    grid_path: Path = typer.Option(
        default=Path("./"),
        exists=True,
        dir_okay=True,
        help="Directory path where grid info mds files are",
    ),
    boundaries: list[Boundary] = typer.Option(
        default=[],
        help="A list of Boundaries",
    ),
    addc: float = typer.Option(
        default=0.0,
        help="Add a constant to the input field",
    ),
    mulc: float = typer.Option(
        default=1.0,
        help="Multiply a constant to the input field",
    ),
):
    """
    Use input grid informations from grid info mds (XC,YC,RC,hFacC,...) files to generate MITgcm boundary conditions
    """

    mask_file = grid_path / "hFacC.data"
    lat_file = grid_path / "YC.data"
    lon_file = grid_path / "XC.data"
    depth_file = grid_path / "RC.data"
    if field == "U":
        mask_file = grid_path / "hFacW.data"
        lon_file = grid_path / "XG.data"
    elif field == "V":
        mask_file = grid_path / "hFacC.data"
        lat_file = grid_path / "YG.data"

    dimList = get_dimlist_from_meta_file(mask_file.with_suffix(".meta"))
    nx = dimList[0][0]
    ny = dimList[1][0]
    nz = dimList[2][0]

    logger.info(f"Grid size: nx, ny, nz = {nx} {ny} {nz}")
    logger.info(f"Reading grid longitudes from {lon_file}")
    lon = np.fromfile(lon_file, ">f4").reshape(ny, nx)
    logger.info(f"Reading grid latitudes from {lat_file}")
    lat = np.fromfile(lat_file, ">f4").reshape(ny, nx)
    logger.info(f"Reading grid depths from {depth_file}")
    z = np.fromfile(depth_file, ">f4").reshape(nz) * -1
    logger.info(f"Reading ocean mask from {mask_file}")
    omask3d = np.fromfile(mask_file, ">f4").reshape(nz, ny, nx)
    omask3d = np.where(omask3d != 0, 1, 0)
    omask = omask3d[0, :, :]

    bndDict = mk_obcs(input, addc, mulc, z, lat, lon, omask, boundaries)
    for bnd, arr in bndDict.items():
        out_file = f"{field}_{bnd}.bin"
        omask = omask3d[:, BNDDEF[bnd][0], BNDDEF[bnd][1]].squeeze()
        arr.values = arr.values * omask  # type: ignore
        logger.info(f"Writing {out_file}")
        arr.values.astype(">f4").tofile(out_file)  # type: ignore


def mk_obcs(
    input: str,
    addc: float,
    mulc: float,
    z: np.ndarray[Any, np.dtype[Any]],
    lat: np.ndarray[Any, np.dtype[Any]],
    lon: np.ndarray[Any, np.dtype[Any]],
    omask: np.ndarray[Any, np.dtype[Any]],
    boundaries: list[Boundary] = [],
) -> dict[str, xr.DataArray]:
    """Generate MITgcm boundary conditions"""
    res: dict[str, xr.DataArray] = {}
    bndAct = mk_bnd_grid(lat, lon, omask, boundaries)
    levels = ",".join(["{:.3f}".format(i) for i in z])
    cdo = Cdo(tempdir="tmp/", options=["-f", "nc"])  # type: ignore
    for bnd, gridfile in bndAct:
        logger.info(f"Processing {bnd} boundary")

        cdoOpr1 = input
        cdoOpr2 = f" -setlevel,0 -sellevidx,1 {cdoOpr1}"
        cdoOpr1 = f" -merge {cdoOpr2} {cdoOpr1}"
        cdoOpr1 = f" -remapnn,{gridfile} {cdoOpr1}"
        cdoOpr1 = f" -intlevel,{levels} " + cdoOpr1
        cdoOpr1 = f" -vertfillmiss {cdoOpr1}"
        cdoOpr1 = f" -setmisstonn {cdoOpr1}"
        cdoOpr1 = f" -addc,{addc} {cdoOpr1}"
        logger.info(f"CDO operation: {cdoOpr1}")

        out_file = cdo.mulc(mulc, input=cdoOpr1, output="out.nc")  # type: ignore
        ds = xr.open_dataset(out_file, decode_times=False, engine="netcdf4")  # type: ignore
        arr = get_data_array(ds)
        arr = arr.squeeze()
        shape = arr.shape
        is2D = len(shape) == 2
        field = arr.name  # type: ignore
        out_file = f"{field} at {bnd} boundary"
        if np.any(np.isnan(arr.values)):  # type: ignore
            logger.info(f"NaN Values present in {out_file}")
            logger.info("Trying to fill NaN Values with Nearest Neighbhour")
            if is2D:
                fill_missing2D(arr.values)  # type: ignore
            else:
                fill_missing3D(arr.values)  # type: ignore

        if np.any(np.isnan(arr.values)):  # type: ignore
            raise RuntimeError(f"Nan Values present in {out_file}")

        logger.info(f"Shape of {out_file} is {arr.shape}")
        logger.info(f"Maximum value of {out_file} is {arr.values.max()}")  # type: ignore
        logger.info(f"Minimum value of {out_file} is {arr.values.min()}")  # type: ignore

        res[bnd] = arr
    return res


def get_data_array(dset: xr.Dataset) -> xr.DataArray:
    data_vars: list[str] = list(dset.data_vars)
    if len(data_vars) == 1:
        return dset[data_vars[0]]  # type: ignore
    else:
        raise ValueError(
            "The dataset contains multiple data variables. Use cdo -selvar to select a single variable"
        )


app_click = typer.main.get_command(app)

if __name__ == "__main__":
    app()
