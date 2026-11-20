"""The variable catalogue (M1.1): which environmental variables the tool knows how to download,
declared as data.

Each entry says where a variable comes from (provider, dataset, variable name, depth range,
cadence), what unit it arrives in, what unit the analyses use, and the conversion between them,
named and written in one place (the CO2 in Pascal was an error of exactly this kind: the unit was
declared nowhere, 2026-09-10). A study refers to an entry by id (`environment: [{catalog_id: ...}]`
in format 2); it never carries dataset names of its own.

The catalogue holds names, units and numbers only: no field is executable and none is a path
(validated models, extra fields refused). It starts with the daily SST, the one variable the
vertical slice of milestone M1 needs; the monthly variables are added one at a time (M1.8).

Domain (decision D1 of the milestone): the Mediterranean product only. `covers()` tells whether a
point is inside the product's bounding box; whether the nearest grid cell is sea is checked when
the data are downloaded (M1.2), not here.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CatalogError(Exception):
    """An id that is not in the catalogue."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Conversion(_Strict):
    """A named, constant multiplicative conversion from the native unit to the analysis unit."""
    name: str
    factor: float


class Domain(_Strict):
    """Bounding box of the product, in degrees."""
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


class CatalogVariable(_Strict):
    id: str
    label: str
    column: str = Field(..., description="The name of the value's column in the series file, and of the variable in the dataset.")
    provider: Literal["copernicus_marine"]
    variable: str = Field(..., description="Variable name inside the dataset (e.g. thetao).")
    cadence: Literal["P1D", "P1M"]
    unit_native: str
    unit_analysis: str
    conversion: Conversion | None = Field(
        default=None, description="From unit_native to unit_analysis; absent when they are the same.")
    depth_min_m: float
    depth_max_m: float
    dataset_multiyear: str = Field(..., description="The reprocessed product, the series' backbone.")
    dataset_analysis_forecast: str | None = Field(
        default=None, description="The product that follows the multiyear one in time, if any.")
    domain: Domain


CATALOG: dict[str, CatalogVariable] = {
    entry.id: entry for entry in [
        CatalogVariable(
            id="sst_daily",
            label="Sea surface temperature, daily",
            column="Temperature",
            provider="copernicus_marine",
            variable="thetao",
            cadence="P1D",
            unit_native="degC",
            unit_analysis="degC",
            depth_min_m=0.0,
            depth_max_m=5.0,
            dataset_multiyear="cmems_mod_med_phy-temp_my_4.2km_P1D-m",
            # The id scripts/fetch_copernicus_daily.py names as its fallback
            # (cmems_mod_med_phy-temp_anfc_...) does not exist in the Copernicus catalogue
            # (DatasetNotFound, 2026-10-02); this one does.
            dataset_analysis_forecast="cmems_mod_med_phy-tem_anfc_4.2km_P1D-m",
            # Latitude and longitude range of the multiyear product, read from the Copernicus
            # catalogue on 2026-10-08 (the analysis-forecast product extends further west).
            domain=Domain(lat_min=30.1875, lat_max=45.97916793823242,
                          lon_min=-6.0, lon_max=36.29166793823242),
        ),
        # The monthly variables are the ones scripts/fetch_copernicus.py downloads for Livorno: the mean over the sea cells
        # of the box and over depth 0-10 m of the multiyear product (tests/test_monthly_variable.py keeps them equal).
        CatalogVariable(
            id="ph_monthly",
            label="pH (total scale), monthly",
            column="pH",
            provider="copernicus_marine",
            variable="ph",
            cadence="P1M",
            unit_native="1",
            unit_analysis="1",
            depth_min_m=0.0,
            depth_max_m=10.0,
            dataset_multiyear="cmems_mod_med_bgc-car_my_4.2km_P1M-m",
            dataset_analysis_forecast="cmems_mod_med_bgc-car_anfc_4.2km_P1M-m",
            # The biogeochemical product's own box, read from the Copernicus catalogue on 2026-11-20: it starts east of the
            # physics product's (lon -5.54 against -6.0).
            domain=Domain(lat_min=30.1875, lat_max=45.97916793823242,
                          lon_min=-5.541666507720947, lon_max=36.29166793823242),
        ),
        CatalogVariable(
            id="salinity_monthly",
            label="Salinity, monthly",
            column="Salinity",
            provider="copernicus_marine",
            variable="so",
            cadence="P1M",
            unit_native="0.001",
            unit_analysis="0.001",
            depth_min_m=0.0,
            depth_max_m=10.0,
            dataset_multiyear="cmems_mod_med_phy-sal_my_4.2km_P1M-m",
            dataset_analysis_forecast="cmems_mod_med_phy-sal_anfc_4.2km_P1M-m",
            # The physics product's box, the same as the temperature's (read from the Copernicus catalogue on 2026-11-20).
            domain=Domain(lat_min=30.1875, lat_max=45.97916793823242,
                          lon_min=-6.0, lon_max=36.29166793823242),
        ),
    ]
}


def known_ids() -> list[str]:
    return sorted(CATALOG)


def monthly_ids() -> list[str]:
    """The ids of the monthly variables, in the catalogue's order."""
    return [i for i, v in CATALOG.items() if v.cadence == "P1M"]


def get(catalog_id: str) -> CatalogVariable:
    try:
        return CATALOG[catalog_id]
    except KeyError:
        raise CatalogError(
            f"{catalog_id!r} is not in the variable catalogue; known ids: {', '.join(known_ids())}"
        ) from None


def covers(variable: CatalogVariable, lat: float, lon: float) -> bool:
    """Whether the point lies inside the product's bounding box."""
    d = variable.domain
    return d.lat_min <= lat <= d.lat_max and d.lon_min <= lon <= d.lon_max
