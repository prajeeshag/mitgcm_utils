import logging
from pathlib import Path

import f90nml  # type: ignore
import typer
import xarray as xr
import xesmf as xe  # type: ignore

from .utils import MITGCM_GRID_VARS, gridinfo_from_parm04, load_grid  # type: ignore

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
logger.addHandler(handler)


app = typer.Typer(pretty_exceptions_show_locals=False, add_completion=False)

GRID_VARS = MITGCM_GRID_VARS


@app.command()
def mk_bathy(
    in_file: Path = typer.Option(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path of input bathymetry netcdf file",
    ),
    grid_file: Path = typer.Option(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="grid file",
    ),
    nx: int = typer.Option(
        None, help="no of grid points in x-dir (must be given if grid_file is given)"
    ),
    ny: int = typer.Option(
        None, help="no of grid points in y-dir (must be given if grid_file is given)"
    ),
    out_file: Path = typer.Option(
        None,
        file_okay=True,
        dir_okay=False,
        writable=True,
        help="output bathymetry file (must be given if grid_file is given)",
    ),
):
    """
    Create bathymetry file for MITgcm:
    1. From the grid information taken from `data` namelist (Default)
    2. From the grid information from the `grid-file`, (--nx, --ny, --out-file) should be provided
    """
    BDATAVAR = {"z": "SRTM+", "elevation": "Gebco"}
    ds_input_bathy = xr.open_dataset(in_file)  # type: ignore

    input_bathy = None
    for key in BDATAVAR:
        try:
            input_bathy = ds_input_bathy[key]
            dset = BDATAVAR[key]
            logger.info(f"Using variable `{key}`: assuming `{dset}` Bathymetry")
            break
        except KeyError:
            continue

    if input_bathy is None:
        keys = BDATAVAR.keys()
        raise KeyError(
            f"Bathymetry file doest not contain any variable with names {keys}"
        )

    if grid_file is not None:  # type: ignore
        if (nx is None) or (ny is None):  # type: ignore
            raise typer.BadParameter(
                "nx or ny not provided when grid_file was provided"
            )
        if out_file is None:  # type: ignore
            raise typer.BadParameter("out_file was not when grid_file was provided")

        gD = load_grid(grid_file, nx, ny)  # type: ignore
        grid_out = xr.Dataset(
            {
                "lat": (["y", "x"], gD["yC"][:-1, :-1], {"units": "degrees_north"}),
                "lon": (["y", "x"], gD["xC"][:-1, :-1], {"units": "degrees_east"}),
                "lat_b": (["y_b", "x_b"], gD["yG"][:, :], {"units": "degrees_north"}),
                "lon_b": (["y_b", "x_b"], gD["xG"][:, :], {"units": "degrees_east"}),
            }
        )
    else:
        logger.info("Reading `data`")
        nml = f90nml.read("data")
        usingsphericalpolargrid = nml["parm04"]["usingsphericalpolargrid"]
        if not usingsphericalpolargrid:
            raise NotImplementedError(
                "Not implemented for any other grid apart from spherical-polar grid"
            )

        logger.info("Generating grid from `data`")
        nx, ny, lon, lat = gridinfo_from_parm04(nml["parm04"])
        if out_file is None:
            out_file = nml["parm05"]["bathyfile"]

        grid_out = xr.Dataset(
            {
                "lat": (["lat"], lat, {"units": "degrees_north"}),
                "lon": (["lon"], lon, {"units": "degrees_east"}),
            }
        )

    logger.info("Creating regridder")
    regridder = xe.Regridder(ds_input_bathy, grid_out, "conservative")

    logger.info("Remapping Bathymery")
    dr_out = regridder(input_bathy, keep_attrs=True)  # type: ignore

    logger.info(f"Writing to bathymetry to `{out_file}`")
    _da2bin(dr_out, out_file)  # type: ignore


def _da2bin(da: xr.DataArray, binfile: Path, typ: str = ">f4"):
    """
    write xarray data array to with big-endian byte ordering
    as single-precision real numbers (which is NumPy float32 or
    equivalently, Fortran real*4 format)
    """
    da.values.astype(typ).tofile(binfile)  # type: ignore


app_click = typer.main.get_command(app)
