import logging
import tempfile
from typing import Annotated, Any

import numpy as np
import typer
import xarray as xr
from cdo import Cdo  # type: ignore

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
logger.addHandler(handler)


app = typer.Typer(pretty_exceptions_show_locals=False, add_completion=False)


@app.command(hidden=True)
def from_era5(wrf_output_file: str):
    pass


def ocean_mask_from_geo_em(geo_em: str) -> str:
    landusef = xr.open_dataset(geo_em)["LANDUSEF"][0, 16, :, :]  # type: ignore
    xlat = xr.open_dataset(geo_em)["XLAT_M"][0, :, :]  # type: ignore
    xlon = xr.open_dataset(geo_em)["XLONG_M"][0, :, :]  # type: ignore
    xlat.attrs["units"] = "degrees_north"
    xlon.attrs["units"] = "degrees_east"
    dset = xr.Dataset({"ocean_mask": landusef, "XLAT": xlat, "XLONG": xlon})
    dset["ocean_mask"].attrs["coordinates"] = "XLONG XLAT"
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as temp_file:
        temp_file_path = temp_file.name
        dset.to_netcdf(temp_file_path)  # type: ignore
        return temp_file_path


FIELD_DEF: dict[str, dict[str, Any]] = {
    "uwind": {
        "units": "m s-1",
    },
    "vwind": {
        "units": "m s-1",
    },
    "apressure": {
        "units": "N/m^2",
    },
    "atemp": {
        "units": "K",
    },
    "aqh": {
        "units": "kg/kg",
    },
    "lwdown": {
        "units": "W/m^2",
    },
    "swdown": {
        "units": "W/m^2",
    },
    "precip": {
        "units": "m/sec",
    },
}


def check_data_range(ds: xr.Dataset) -> None:
    for name in ds.data_vars:
        if name in FIELD_DEF:
            if name in ["precip", "aqh", "swdown"]:
                logger.info(f"{name} minimum values clipped to 0")
                ds[name].values = np.where(ds[name].values < 0, 0, ds[name].values)  # type: ignore
            min_val = ds[name].min().item()
            max_val = ds[name].max().item()
            logger.info(f"{name} min value is: {min_val} {FIELD_DEF[name]['units']}")
            logger.info(f"{name} max value is: {max_val} {FIELD_DEF[name]['units']}")


@app.command()
def from_wrf(
    wrf_output_files: Annotated[list[str], typer.Argument(help="WRF output files")],
    geo_em: Annotated[str, typer.Option(help="geo_em file")] = "",
    lonlatbox: Annotated[
        str, typer.Option(help="lonlatbox as lon1,lon2,lat1,lat2")
    ] = "",
    suffix: Annotated[str, typer.Option(help="suffix for output files")] = ".bin",
):
    """
    Create MITGCM external forcing files from WRF output.\n
        - Creates a land sea mask (LSM) file from the geo_em file (if specified) or the first WRF output file.\n
        - Merges the WRF output files and applies the sellonlatbox operation if specified.\n
        - Infer the time interval in seconds from the time difference between the first two time steps.\n
        - Mask the land points and extrapolate it with the nearest neighbor method for T2, Q2 and PSFC.\n
        - Compute the time difference quantities of accumulated fields.\n
        - Unit conversions for RAIN: mm/sec to m/sec.\n
        - Fields are renamed to match the names in https://mitgcm.readthedocs.io/en/latest/phys_pkgs/exf.html#exf-input-fields-and-units
        - Writes the resulting data to MITgcm compatible binary files with the specified suffix.\n
    Note:\n
        - The first time step is dropped since it is used only to compute the time difference quatities of accumulated fields.

    Example:\n
        `mkMITgcmEXF from-wrf --lonlatbox 29.8,50.2,9.8,30.2 --geo-em-file geo_em.d01.nc wrfout_d01_*`
    """

    cdo = Cdo(tempdir="tmp/", options=["-f", "nc"])

    sellonlatbox_operation = f" -sellonlatbox,{lonlatbox} :" if lonlatbox else ""
    if geo_em:
        logger.info(f"Land sea mask from {geo_em}")
        lsm_file = ocean_mask_from_geo_em(geo_em)
        if lonlatbox:
            lsm_file = cdo.sellonlatbox(lonlatbox, input=lsm_file)
    else:
        logger.info(f"Land sea mask from XLAND variable from {wrf_output_files[0]}")
        lsm_file = cdo.eqc(
            "2", input=f" -selvar,XLAND {sellonlatbox_operation} {wrf_output_files[0]}"
        )

    file_list_string = " ".join(wrf_output_files)

    masked_vars = ["PSFC", "T2", "Q2"]
    unmasked_vars = ["U10", "V10", "RAINC", "RAINNC", "ACSWDNB", "ACLWDNB"]

    masked_vars_string = ",".join(masked_vars)
    unmasked_vars_string = ",".join(unmasked_vars)

    logger.info(f"Fields to be masked and extrapolated: {masked_vars_string}")

    maskOpr = f"-ifthen {lsm_file}"
    extrapOpr = " -setmisstonn "
    cdoInput = f"{extrapOpr} {maskOpr} -mergetime -select,name={masked_vars_string} [ {sellonlatbox_operation} {file_list_string} ]"
    masked_field_file = cdo.addc(0, input=cdoInput)
    logger.info(f"Other fields: {unmasked_vars_string}")
    unmasked_field_file = cdo.mergetime(
        input=f" -select,name={unmasked_vars_string} [ {sellonlatbox_operation} {file_list_string} ]"
    )

    ds = xr.merge(  # type: ignore
        [xr.open_dataset(f) for f in [masked_field_file, unmasked_field_file]]  # type: ignore
    )

    time_delta = ds["XTIME"][1] - ds["XTIME"][0]
    interval_seconds = time_delta.astype("timedelta64[s]").item().total_seconds()  # type: ignore
    logger.info(f"Time interval (seconds): {interval_seconds}")
    ds1 = ds.isel(XTIME=slice(1, None))

    r_interval_seconds = 1.0 / interval_seconds
    for name in ds1.data_vars:
        if name in ["RAINC", "RAINNC", "ACSWDNB", "ACLWDNB"]:
            logger.info(f"Time difference quantities of accumulated field: {name}")
            ds1[name].values = (
                ds1[name].values - ds[name][0:-1, :, :].values  # type: ignore
            ) * r_interval_seconds

    logger.info("Creating RAIN=RAINC+RAINNC and converting units from mm/sec to m/")
    ds1["RAINC"].values = (ds1["RAINC"].values + ds1["RAINNC"].values) / 1000.0  # type: ignore
    ds1 = ds1.drop_vars(["RAINNC"])
    ds1 = ds1.rename_vars(
        {
            "RAINC": "precip",
            "ACSWDNB": "swdown",
            "ACLWDNB": "lwdown",
            "U10": "uwind",
            "V10": "vwind",
            "T2": "atemp",
            "Q2": "aqh",
            "PSFC": "apressure",
        }
    )

    check_data_range(ds1)

    for name in ds1.data_vars:
        binfile = f"{name}{suffix}"
        logger.info(f"Writing: {binfile}")
        ds1[name].values.astype(">f4").tofile(binfile)  # type: ignore
        logger.info(f"Shape of {name} : {ds1[name].values.shape}")  # type: ignore


app_click = typer.main.get_command(app)

if __name__ == "__main__":
    app()
