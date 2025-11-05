import logging
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import typer
import xarray as xr
from cdo import Cdo  # type: ignore
from scipy.ndimage import label  # type: ignore

from .utils import (
    _create_grid_file,
    fill_missing2D,  # type: ignore
    fill_missing3D,  # type: ignore
    get_bathy,
    get_dimlist_from_meta_file,
    get_hgrid,
    load_bathy,
    read_mitgcm_grid,
    vgrid_from_parm04,  # type: ignore
)

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

cdo = Cdo(tempdir="tmp/", options=["-f", "nc"])  # type: ignore


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


def get_bnd_grid(run_dir: str, nx: int, ny: int):  # type: ignore
    """Generate MITgcm boundary grid files later to be used to interpolate the boundary condition"""

    logger.info("Reading bathymetry and grid info")

    grid_ds = get_hgrid(run_dir, nx, ny, as_ds=True)

    z = get_bathy(run_dir, nx, ny)
    lat = grid_ds["lat"].values  # type: ignore
    lon = grid_ds["lon"].values  # type: ignore

    omask = np.array(z.shape, dtype=int)
    omask = np.where(z < 0, 1, 0)
    return lat, lon, omask  # type: ignore


def mk_bnd_basins(
    lat: np.ndarray[Any, np.dtype[Any]],
    lon: np.ndarray[Any, np.dtype[Any]],
    omask: np.ndarray[Any, np.dtype[Any]],
    boundary: list[Boundary] = [],
) -> tuple[dict[int, str], list[str]]:
    larray, num_features = label(omask)  # type: ignore

    basin_mask_files: dict[int, str] = {}

    bnd_feature_set: set[int] = set()
    bndAct: list[str] = []

    for bnd in BNDDEF:
        if boundary and bnd not in boundary:
            continue
        bndMask = omask[BNDDEF[bnd]]
        bndlarray = larray[BNDDEF[bnd]]  # type: ignore
        features = np.sort(np.unique(bndlarray))  # type: ignore
        isboundary = np.any(bndMask != 0)
        bndPoints = np.count_nonzero(bndMask)
        logger.info(f"{bnd}: {isboundary}, {bndPoints}")
        if not isboundary:
            continue
        bndAct.append(bnd)
        if len(features) == 1 and features[0] == 0:
            raise RuntimeError(
                "An open boundary contains no detected basin features (possible bug)!!"
            )
        bnd_feature_set.update(features[1:])

    for feature in bnd_feature_set:
        farray = np.where(larray == feature, 1, 0)  # type: ignore
        basin_mask_files[int(feature)] = _create_grid_file(lon, lat, farray)

    return basin_mask_files, bndAct


def mk_bnd_grid(
    lat: np.ndarray[Any, np.dtype[Any]],
    lon: np.ndarray[Any, np.dtype[Any]],
    omask: np.ndarray[Any, np.dtype[Any]],
    boundary: list[Boundary] = [],
) -> list[tuple[str, str]]:
    bndAct: list[tuple[str, str]] = []

    for bnd in BNDDEF:
        if boundary and bnd not in boundary:
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


@app.command()
def igrid(
    input: str = typer.Option(
        help="""
        Input can be: \n
         1. A NetCDF file. \n
         2. A valid cdo option which will generate a NetCDF file. \n
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
        help="""Boundary field name, i.e. T, S, U, V, Eta \n
            e.g.; This will be used to generate files <field>_E.bin, <field>_W.bin,..
            For field "U" and "V", West and South grid coordinates of Arakawa-C will be used respectively.
            For all other fields Center grid coordinates of Arakawa-C will be used.
            """,
    ),
    run_dir: Path = typer.Option(
        default=Path("./"),
        exists=True,
        dir_okay=True,
        help="Path to the MITgcm run directory where all neccesary files are present",
    ),
    boundary: list[Boundary] = typer.Option(
        default=[],
        help="""
            boundary; can be defined multiple times \n
            e.g. --boundary S --boundary N --boundary E 
            """,
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
    nml = Path(run_dir) / "data"

    # generate boundary grids
    lat, lon, omask = get_bnd_grid(run_dir, nx, ny)  # type: ignore

    if field == "Eta":
        bndDict = mk_obcs_eta(input, addc, mulc, lat, lon, omask, boundary)  # type: ignore
        for bnd, arr in bndDict.items():
            out_file = f"{field}_{bnd}.bin"
            logger.info(f"Writing {out_file}")
            arr.astype(">f4").tofile(out_file)  # type: ignore
            print(np.count_nonzero(arr))
    else:
        z = vgrid_from_parm04(nml)
        bndDict = mk_obcs(input, addc, mulc, z, lat, lon, omask, boundary)  # type: ignore
        for bnd, arr in bndDict.items():
            out_file = f"{field}_{bnd}.bin"
            logger.info(f"Writing {out_file}")
            arr.values.astype(">f4").tofile(out_file)  # type: ignore


@app.command()
def mds(
    input: str = typer.Option(
        help="""
            Input can be: \n
             1. A CF-compliant NetCDF file. \n 
             2. A valid cdo option which will generate a NetCDF file. \n 
             e.g. "-mergetime input1.nc input2.nc input3.nc"
             """,
    ),
    field: str = typer.Option(
        help="""Boundary field name, i.e. T, S, U, V \n
            e.g.; This will be used to generate files <field>_E.bin, <field>_W.bin,.. \n
            For field "U" and "V", West and South grid coordinates of Arakawa-C will be used respectively. \n
            For all other fields Center grid coordinates of Arakawa-C will be used. \n
            """,
    ),
    grid_path: Path = typer.Option(
        default=Path("./"),
        exists=True,
        dir_okay=True,
        help="Directory path where grid info mds files are",
    ),
    boundary: list[Boundary] = typer.Option(
        default=[],
        help="""
            boundary; can be defined multiple times \n
            e.g. --boundary S --boundary N --boundary E 
            """,
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
    Use input grid informations from grid info mds (XC,YC,RC,hFacC,...) files to generate MITgcm boundary conditions.
    The input file must be a CF-compliant NetCDF file and should contain a single data variable.
    If the input file contains multiple data variables, use cdo operator "-selvar" to select a single data variable.
    Example: \n
    - `mkMITgcmBC mds --grid-path mds_grid_info_files_directory_path/ --field T --boundary E --input "-mergetime [ input_data_*.nc ]"` \n
    - `mkMITgcmBC mds --grid-path mds_grid_info_files_directory_path/ --field U --boundary E --boundary N --input "-selvar,uvel input_data.nc"`
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

    bndDict = mk_obcs(input, addc, mulc, z, lat, lon, omask, boundary)
    for bnd, arr in bndDict.items():
        out_file = f"{field}_{bnd}.bin"
        omask = omask3d[:, BNDDEF[bnd][0], BNDDEF[bnd][1]].squeeze()
        arr.values = arr.values * omask  # type: ignore
        logger.info(f"Writing {out_file}")
        arr.values.astype(">f4").tofile(out_file)  # type: ignore


def mk_obcs_eta(
    input: str,
    addc: float,
    mulc: float,
    lat: np.ndarray[Any, np.dtype[Any]],
    lon: np.ndarray[Any, np.dtype[Any]],
    omask: np.ndarray[Any, np.dtype[Any]],
    boundary: list[Boundary] = [],
) -> dict[str, np.ndarray]:  # type: ignore
    """
    Generate MITgcm ETA boundary condition

    Steps:
        1. Create mask files for basins which contains boundaries
        2. remap input file
        3. for each basin multiply the basin mask and take fldmean and create timeseries
    """

    basins, bndAct = mk_bnd_basins(lat, lon, omask, boundary)
    basin_mean_vals: dict[str, np.ndarray] = {}  # type: ignore
    for _, basin_file in basins.items():
        cdoOpr1 = input
        cdoOpr1 = f" -fldmean -ifthen [ {basin_file} -remapnn,{basin_file} {cdoOpr1} ]"
        cdoOpr1 = f" -addc,{addc} {cdoOpr1}"
        logger.info(f"CDO operation: {cdoOpr1}")

        out_file = cdo.mulc(mulc, input=cdoOpr1)  # type: ignore
        ds = xr.open_dataset(out_file, decode_times=False, engine="netcdf4")  # type: ignore
        da = get_data_array(ds).squeeze()
        ds = xr.open_dataset(basin_file, decode_times=False, engine="netcdf4")  # type: ignore
        da_mask = get_data_array(ds).squeeze()
        nt = len(da.values)  # type: ignore
        for bnd in bndAct:
            mask = da_mask[BNDDEF[bnd]].squeeze()
            if bnd not in basin_mean_vals:
                shp = mask.shape
                basin_mean_vals[bnd] = np.zeros([nt, *shp])
                logger.info(f"Shape of Eta at {bnd} boundary is {(nt, *shp)}")
            for i in range(nt):
                basin_mean_vals[bnd][i, :] += da.values[i] * mask.values  # type: ignore

    # fill missing with nearest neighbour
    logger.info(f"Filling missing values with nearest neighbour")
    for bnd in bndAct:
        mask = da_mask[BNDDEF[bnd]].squeeze()
        for i in range(nt):
            arr = basin_mean_vals[bnd][i : i + 1, :]
            arr[0, :] = np.where(mask.values == 0, np.nan, arr[0, :])  # type: ignore
            basin_mean_vals[bnd][i : i + 1, :] = fill_missing2D(arr)  # type: ignore

    return basin_mean_vals  # type: ignore


def mk_obcs(
    input: str,
    addc: float,
    mulc: float,
    z: np.ndarray[Any, np.dtype[Any]],
    lat: np.ndarray[Any, np.dtype[Any]],
    lon: np.ndarray[Any, np.dtype[Any]],
    omask: np.ndarray[Any, np.dtype[Any]],
    boundary: list[Boundary] = [],
) -> dict[str, xr.DataArray]:
    """Generate MITgcm boundary conditions"""
    res: dict[str, xr.DataArray] = {}
    bndAct = mk_bnd_grid(lat, lon, omask, boundary)
    levels = ",".join(["{:.3f}".format(i) for i in z])
    for bnd, gridfile in bndAct:
        logger.info(f"Processing {bnd} boundary")

        cdoOpr1 = input
        cdoOpr2 = f" -setlevel,0 -sellevidx,1 {cdoOpr1}"
        cdoOpr1 = f" -merge {cdoOpr2} {cdoOpr1}"
        cdoOpr1 = f" -remapbil,{gridfile} {cdoOpr1}"
        cdoOpr1 = f" -setmisstonn {cdoOpr1}"
        cdoOpr1 = f" -intlevel,{levels} " + cdoOpr1
        cdoOpr1 = f" -vertfillmiss {cdoOpr1}"
        cdoOpr1 = f" -addc,{addc} {cdoOpr1}"
        logger.info(f"CDO operation: {cdoOpr1}")

        out_file = cdo.mulc(mulc, input=cdoOpr1)  # type: ignore
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
