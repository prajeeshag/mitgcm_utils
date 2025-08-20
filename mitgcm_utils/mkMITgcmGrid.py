import itertools
import logging
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import typer
import xarray as xr

from .utils import (  # type: ignore
    MITGCM_GRID_VARS,
    great_circle,  # type: ignore
    quadrilateral_area_on_earth,
)

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
def wrfgrid(
    in_file: Path = typer.Option(
        Path("geo_em.d01.nc"), readable=True, exists=True, dir_okay=False
    )
):
    """
    Create MITgcm grid from geo_em.d??.nc file of WRF
    """
    """
    xG  -> (j=0, i=0), (j=0, i=nx  ), (j=ny,  i=nx  ), (j=ny,  i=0)
    yG  -> (j=0, i=0), (j=0, i=nx  ), (j=ny,  i=nx  ), (j=ny,  i=0)
    xC  -> (j=0, i=0), (j=0, i=nx-1), (j=ny-1,i=nx-1), (j=ny-1,i=0)
    yC  -> (j=0, i=0), (j=0, i=nx-1), (j=ny-1,i=nx-1), (j=ny-1,i=0)
    """

    def print_info(var: str):
        min_val = np.amin(gA[var][:, :])  # type: ignore
        max_val = np.amax(gA[var][:, :])  # type: ignore
        logger.info(f"{var} min = {min_val}")
        logger.info(f"{var} max = {max_val}")

    sl = slice(None, None, None)
    sl1 = slice(None, -1, None)
    vMap = {
        "xG": ["XLONG_C", sl, sl],
        "yG": ["XLAT_C", sl, sl],
        "xC": ["XLONG_M", sl1, sl1],
        "yC": ["XLAT_M", sl1, sl1],
        "xU": ["XLONG_U", sl1, sl],
        "yU": ["XLAT_U", sl1, sl],
        "xV": ["XLONG_V", sl, sl1],
        "yV": ["XLAT_V", sl, sl1],
    }

    geo_ds = xr.open_dataset(in_file)
    XLAT_C = geo_ds["XLAT_C"].squeeze().values.astype(np.float64)
    nyp1, nxp1 = XLAT_C.shape

    gA = {}
    for var in GRID_VARS + ["xU", "yU", "xV", "yV"]:
        gA[var] = np.zeros_like(XLAT_C)

    for var in vMap:
        vin = vMap[var][0]
        dy = vMap[var][1]
        dx = vMap[var][2]
        fld = geo_ds[vin].squeeze().values.astype(np.float64)
        gA[var][dy, dx] = fld

    # dxG
    # dxG -> (j=0, i=0), (j=0, i=nx-1), (j=ny,  i=nx-1), (j=ny,  i=0)
    logger.info("computing dxG")
    X1 = gA["xG"][:, :-1]
    Y1 = gA["yG"][:, :-1]
    X2 = gA["xG"][:, 1:]
    Y2 = gA["yG"][:, 1:]
    gA["dxG"][:, :-1] = great_circle_a(X1, X2, Y1, Y2)
    gA["dxG"][:, -1] = gA["dxG"][:, -2]
    print_info("dxG")

    # dyG -> (j=0, i=0), (j=0, i=nx  ), (j=ny-1,i=nx  ), (j=ny-1,i=0)
    logger.info("computing dyG")
    X1 = gA["xG"][:-1, :]
    Y1 = gA["yG"][:-1, :]
    X2 = gA["xG"][1:, :]
    Y2 = gA["yG"][1:, :]
    gA["dyG"][:-1, :] = great_circle_a(X1, X2, Y1, Y2)
    gA["dyG"][-1, :] = gA["dyG"][-2, :]
    print_info("dyG")

    # dxC -> (j=0, i=1), (j=0, i=nx-1), (j=ny-1,i=nx-1), (j=ny-1,i=1)
    logger.info("computing dxC")
    X1 = gA["xC"][:-1, :-2]
    Y1 = gA["yC"][:-1, :-2]
    X2 = gA["xC"][:-1, 1:-1]
    Y2 = gA["yC"][:-1, 1:-1]
    gA["dxC"][:-1, 1:-1] = great_circle_a(X1, X2, Y1, Y2)
    gA["dxC"][:, 0] = gA["dxC"][:, 1]
    gA["dxC"][:, -1] = gA["dxC"][:, -2]
    gA["dxC"][-1, :] = gA["dxC"][-2, :]
    print_info("dxC")

    # dyC -> (j=1, i=0), (j=1, i=nx-1), (j=ny-1,i=nx-1), (j=ny-1,i=0)
    logger.info("computing dyC")
    X1 = gA["xC"][:-2, :-1]
    Y1 = gA["yC"][:-2, :-1]
    X2 = gA["xC"][1:-1, :-1]
    Y2 = gA["yC"][1:-1, :-1]
    gA["dyC"][1:-1, :-1] = great_circle_a(X1, X2, Y1, Y2)
    gA["dyC"][0, :] = gA["dyC"][1, :]
    gA["dyC"][-1, :] = gA["dyC"][-2, :]
    gA["dyC"][:, -1] = gA["dyC"][:, -2]
    print_info("dyC")

    # dxF -> (j=0, i=0), (j=0, i=nx-1), (j=ny-1,i=nx-1), (j=ny-1,i=0)
    logger.info("computing dxF")
    X1 = gA["xU"][:-1, :-1]
    Y1 = gA["yU"][:-1, :-1]
    X2 = gA["xU"][:-1, 1:]
    Y2 = gA["yU"][:-1, 1:]
    gA["dxF"][:-1, :-1] = great_circle_a(X1, X2, Y1, Y2)
    gA["dxF"][:, -1] = gA["dxF"][:, -2]
    gA["dxF"][-1, :] = gA["dxF"][-2, :]
    print_info("dxF")

    # dyF -> (j=0, i=0), (j=0, i=nx-1), (j=ny-1,i=nx-1), (j=ny-1,i=0)
    logger.info("computing dyF")
    X1 = gA["xV"][:-1, :-1]
    Y1 = gA["yV"][:-1, :-1]
    X2 = gA["xV"][1:, :-1]
    Y2 = gA["yV"][1:, :-1]
    gA["dyF"][:-1, :-1] = great_circle_a(X1, X2, Y1, Y2)
    gA["dyF"][:, -1] = gA["dyF"][:, -2]
    gA["dyF"][-1, :] = gA["dyF"][-2, :]
    print_info("dxF")

    # dxV -> (j=0, i=1), (j=0, i=nx-1), (j=ny,  i=nx-1), (j=ny,  i=1)
    logger.info("computing dxV")
    X1 = gA["xV"][:-1, :-2]
    Y1 = gA["yV"][:-1, :-2]
    X2 = gA["xV"][:-1, 1:-1]
    Y2 = gA["yV"][:-1, 1:-1]
    gA["dxV"][:-1, 1:-1] = great_circle_a(X1, X2, Y1, Y2)
    gA["dxV"][:, 0] = gA["dxV"][:, 1]
    gA["dxV"][:, -1] = gA["dxV"][:, -2]
    gA["dxV"][-1, :] = gA["dxV"][-2, :]
    print_info("dxV")

    # dyU -> (j=1, i=0), (j=1, i=nx  ), (j=ny-1,i=nx  ), (j=ny-1,i=0)
    logger.info("computing dyU")
    X1 = gA["xU"][:-2, :-1]
    Y1 = gA["yU"][:-2, :-1]
    X2 = gA["xU"][1:-1, :-1]
    Y2 = gA["yU"][1:-1, :-1]
    gA["dyU"][1:-1, :-1] = great_circle_a(X1, X2, Y1, Y2)
    gA["dyU"][0, :] = gA["dyU"][1, :]
    gA["dyU"][-1, :] = gA["dyU"][-2, :]
    gA["dyU"][:, -1] = gA["dyU"][:, -2]
    print_info("dyU")

    # rA  -> (j=0, i=0), (j=0, i=nx-1), (j=ny-1,i=nx-1), (j=ny-1,i=0)
    logger.info("computing rA")
    X1, Y1 = gA["xG"][:-1, :-1], gA["yG"][:-1, :-1]
    X2, Y2 = gA["xG"][:-1, 1:], gA["yG"][:-1, 1:]
    X3, Y3 = gA["xG"][1:, 1:], gA["yG"][1:, 1:]
    X4, Y4 = gA["xG"][1:, :-1], gA["yG"][1:, :-1]
    gA["rA"][:-1, :-1] = quad_area_a(X1, X2, X3, X4, Y1, Y2, Y3, Y4)
    gA["rA"][-1, :] = gA["rA"][-2, :]
    gA["rA"][:, -1] = gA["rA"][:, -2]
    print_info("rA")

    # rAz -> (j=1, i=1), (j=1, i=nx-1), (j=ny-1,i=nx-1), (j=ny-1,i=1)
    logger.info("computing rAz")
    X1, Y1 = gA["xC"][:-2, :-2], gA["yC"][:-2, :-2]
    X2, Y2 = gA["xC"][:-2, 1:-1], gA["yC"][:-2, 1:-1]
    X3, Y3 = gA["xC"][1:-1, 1:-1], gA["yC"][1:-1, 1:-1]
    X4, Y4 = gA["xC"][1:-1, :-2], gA["yC"][1:-1, :-2]
    gA["rAz"][1:-1, 1:-1] = quad_area_a(X1, X2, X3, X4, Y1, Y2, Y3, Y4)
    gA["rAz"][0, :] = gA["rAz"][1, :]
    gA["rAz"][:, 0] = gA["rAz"][:, 1]
    gA["rAz"][-1, :] = gA["rAz"][-2, :]
    gA["rAz"][:, -1] = gA["rAz"][:, -2]
    print_info("rAz")

    # rAw -> (j=0, i=1), (j=0, i=nx-1), (j=ny-1,i=nx-1), (j=ny-1,i=1)
    logger.info("computing rAw")
    X1, Y1 = gA["xV"][:-1, :-2], gA["yV"][:-1, :-2]
    X2, Y2 = gA["xV"][:-1, 1:-1], gA["yV"][:-1, 1:-1]
    X3, Y3 = gA["xV"][1:, 1:-1], gA["yV"][1:, 1:-1]
    X4, Y4 = gA["xV"][1:, :-2], gA["yV"][1:, :-2]
    gA["rAw"][:-1, 1:-1] = quad_area_a(X1, X2, X3, X4, Y1, Y2, Y3, Y4)
    gA["rAw"][:, 0] = gA["rAw"][:, 1]
    gA["rAw"][:, -1] = gA["rAw"][:, -2]
    gA["rAw"][-1, :] = gA["rAw"][-2, :]
    print_info("rAw")

    # rAs -> (j=1, i=0), (j=1, i=nx-1), (j=ny-1,i=nx-1), (j=ny-1,i=0)
    logger.info("computing rAs")
    X1, Y1 = gA["xU"][:-2, :-1], gA["yU"][:-2, :-1]
    X2, Y2 = gA["xU"][:-2, 1:], gA["yU"][:-2, 1:]
    X3, Y3 = gA["xU"][1:-1, 1:], gA["yU"][1:-1, 1:]
    X4, Y4 = gA["xU"][1:-1, :-1], gA["yU"][1:-1, :-1]
    gA["rAs"][1:-1, :-1] = quad_area_a(X1, X2, X3, X4, Y1, Y2, Y3, Y4)
    gA["rAs"][0, :] = gA["rAs"][1, :]
    gA["rAs"][-1, :] = gA["rAs"][-2, :]
    gA["rAs"][:, -1] = gA["rAs"][:, -2]
    print_info("rAs")

    out_file_prefix = "tile001.mitgrid"
    out_file = f"{out_file_prefix}.nc"
    logger.info(f"writing {out_file}")
    dump_grid_nc(gA, out_file)

    out_file = f"{out_file_prefix}"
    logger.info(f"writing {out_file}")
    dump_grid(gA, out_file)


def dump_grid_nc(gA, out_file):
    datavars = {}
    for varnm in GRID_VARS[:-2]:
        datavars[varnm] = (("ny1", "nx1"), gA[varnm])
    ds = xr.Dataset(data_vars=datavars)
    ds.to_netcdf(out_file)


def great_circle_a(X1, X2, Y1, Y2):
    dc = list(itertools.starmap(great_circle, zip(*map(np.ravel, [X1, Y1, X2, Y2]))))
    return np.array(dc).reshape(X1.shape)


def quad_area_a(X1, X2, X3, X4, Y1, Y2, Y3, Y4, nprocs: int = None):
    if nprocs is None:
        pool = Pool()
    else:
        pool = Pool(nprocs)

    area = pool.starmap(
        quadrilateral_area_on_earth,
        (
            ((y1, x1), (y2, x2), (y3, x3), (y4, x4))
            for x1, x2, x3, x4, y1, y2, y3, y4 in zip(
                *map(np.ravel, [X1, X2, X3, X4, Y1, Y2, Y3, Y4])
            )
        ),
    )
    pool.close()
    return np.array(area).reshape(X1.shape)


def dump_grid(grid: dict, out_file: Path) -> None:
    nvars = len(GRID_VARS) - 2
    ny, nx = grid["xC"].shape
    fdata = np.zeros((nvars, ny, nx), dtype=np.float64)
    for i, var in enumerate(GRID_VARS[:-2]):
        fdata[i, :, :] = grid[var][:, :]
    fdata.astype(">f8").tofile(out_file)


app_click = typer.main.get_command(app)
