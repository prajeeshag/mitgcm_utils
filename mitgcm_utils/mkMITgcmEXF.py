import typer
import xarray as xr
from cdo import Cdo
import tempfile
import logging
from typing import Annotated

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


def ocean_mask_from_geo_em(geo_em_file: str) -> str:
    landusef = xr.open_dataset(geo_em_file)["LANDUSEF"][0, 16, :, :]
    xlat = xr.open_dataset(geo_em_file)["XLAT_M"][0, :, :]
    xlon = xr.open_dataset(geo_em_file)["XLONG_M"][0, :, :]
    xlat.attrs["units"] = "degrees_north"
    xlon.attrs["units"] = "degrees_east"
    dset = xr.Dataset({"ocean_mask": landusef, "XLAT": xlat, "XLONG": xlon})
    dset["ocean_mask"].attrs["coordinates"] = "XLONG XLAT"
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as temp_file:
        temp_file_path = temp_file.name
        dset.to_netcdf(temp_file_path)
        return temp_file_path


@app.command()
def from_wrf(
    wrf_output_files: Annotated[list[str], typer.Argument(help="WRF output files")],
    geo_em_file: Annotated[str, typer.Option(help="geo_em file")] = "",
    lonlatbox: Annotated[
        str, typer.Option(help="lonlatbox as lon1,lon2,lat1,lat2")
    ] = "",
    suffix: Annotated[str, typer.Option(help="suffix for output files")] = ".bin",
):
    """
    Create MITGCM external forcing files from WRF output.
    - Creates a land sea mask (LSM) file from the geo_em file (if specified) or the first WRF output file.
    - Merges the WRF output files and applies the sellonlatbox operation if specified.
    - Infer the time interval in seconds from the time difference between the first two time steps.
    - Compute the time difference quantities of accumulated fields.
    - Mask the land points and extrapolate it with the nearest neighbor method for T2, Q2 and PSFC
    - Writes the resulting data to MITgcm compatible binary files with the specified suffix.
    Note: The first time step is dropped since it is used only to compute the time difference quatities of accumulated fields.
    """

    cdo = Cdo()

    if geo_em_file:
        lsm_file = ocean_mask_from_geo_em(geo_em_file)
        if sellonlatbox:
            lsm_file = cdo.sellonlatbox(sellonlatbox, input=lsm_file)
    else:
        sellonlatbox_operation = (
            f" -sellonlatbox,{sellonlatbox}" if sellonlatbox else ""
        )
        lsm_file = cdo.eqc(
            "2", input=f" -selvar,XLAND {sellonlatbox_operation} {wrf_output_files[0]}"
        )

    file_list_string = " ".join(wrf_output_files)

    sellonlatbox_operation = f" -sellonlatbox,{sellonlatbox} :" if sellonlatbox else ""
    masked_field_file = cdo.fillmiss2(
        input=f" -ifthen {lsm_file} -mergetime -select,name=PSFC,T2,Q2 [ {sellonlatbox_operation} {file_list_string} ]"
    )
    unmasked_field_file = cdo.mergetime(
        input=f" -select,name=U10,V10,RAINC,RAINNC,ACSWDNB,ACLWDNB [ {sellonlatbox_operation} {file_list_string} ]"
    )

    ds = xr.merge(
        [xr.open_dataset(f) for f in [masked_field_file, unmasked_field_file]]
    )

    time_delta = ds["XTIME"][1] - ds["XTIME"][0]
    interval_seconds = time_delta.astype("timedelta64[s]").item().total_seconds()
    ds1 = ds.isel(XTIME=slice(1, None))

    r_interval_seconds = 1.0 / interval_seconds
    for name in ds1.data_vars:
        if name in ["RAINC", "RAINNC", "ACSWDNB", "ACLWDNB"]:
            ds1[name].values = (
                ds1[name].values - ds[name][0:-1, :, :].values
            ) * r_interval_seconds

    ds1["RAINC"].values = ds1["RAINC"].values + ds1["RAINNC"].values

    ds1 = ds1.drop_vars(["RAINNC"])

    for name in ds1.data_vars:
        logger.info(f"Writing: {name}")
        binfile = f"{name}{suffix}"
        ds1[name].values.astype(">f4").tofile(binfile)


app_click = typer.main.get_command(app)

if __name__ == "__main__":
    app()
