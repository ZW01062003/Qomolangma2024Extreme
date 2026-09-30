#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot OBS, GPM IMERG and ERA5-Land precipitation at P5200/P5800/P6500.

Panels (c)-(f) show hourly precipitation rather than accumulated precipitation.
Panel (c) is the station-mean hourly precipitation, while panels (d)-(f) show
the individual stations.
"""
from __future__ import annotations

import argparse
import math
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_POINTS = (
    ("P5200", 28.128911, 86.859978),
    ("P5800", 28.085467, 86.914133),
    ("P6500", 28.030633, 86.940464),
)

OBS_COLUMN_ALIASES = {
    "P5200": ("P5200", "5200", "Precip_5200m", "Precip_5220m", "prep/5200", "prep/5220"),
    "P5800": ("P5800", "5800", "Precip_5800m", "prep/5800"),
    # P6500 is the unified public name. Old P6400/6410 aliases are accepted.
    "P6500": (
        "P6500", "6500", "P6400", "6400", "Precip_6500m", "Precip_6400m",
        "Precip_6410m", "prep/6500", "prep/6400", "prep/6410",
    ),
}

GPM_FILENAME_TIME_RE = re.compile(r"\.(?P<date>\d{8})-S(?P<start>\d{6})-E(?P<end>\d{6})\.")
GPM_PRECIP_NAMES = ("precipitation", "precipitationCal", "precipitationUncal")


@dataclass(frozen=True)
class Point:
    name: str
    lat: float
    lon: float


@dataclass(frozen=True)
class GPMFile:
    path: Path
    start_utc: datetime
    nominal_end_utc: datetime

    @property
    def end_bjt(self) -> datetime:
        return self.nominal_end_utc + timedelta(hours=8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a reference-style event error-evaluation dot-plot figure for OBS/GPM/ERA5-Land precipitation."
    )
    parser.add_argument(
        "--obs-csv",
        default="/home/phadcloud027n16178/Python/observed_hourly_precip_mm.csv",
        help="Hourly OBS precipitation table with time_bjt and station columns. Supports .csv/.xlsx.",
    )
    parser.add_argument(
        "--gpm-dir",
        default="/home/phadcloud027n16178/GPM/data",
        help="Directory containing GPM IMERG half-hourly *.nc4 files. Ignored if --gpm-hourly-csv is used.",
    )
    parser.add_argument(
        "--gpm-hourly-csv",
        default=None,
        help="Optional precomputed GPM hourly station table (.csv/.xlsx). If set, raw GPM files are not read.",
    )
    parser.add_argument(
        "--era5-land-grib",
        default=None,
        help="ERA5-Land single-level GRIB file containing total precipitation tp. Required unless --era5-land-hourly-csv is used.",
    )
    parser.add_argument(
        "--era5-land-hourly-csv",
        default=None,
        help="Optional precomputed ERA5-Land hourly station table (.csv/.xlsx). If set, the GRIB file is not read.",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/phadcloud027n16178/GPMandERA5Land/event_error_eval_out_v4",
        help="Output directory.",
    )
    parser.add_argument("--start-bjt", default="2024-09-27 00:00", help="Start time in BJT.")
    parser.add_argument("--end-bjt", default="2024-09-29 00:00", help="End time in BJT.")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=(85.5, 88.0, 27.0, 29.5),
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        help="Small Everest bbox used when reading ERA5-Land/GPM grids.",
    )
    parser.add_argument(
        "--ci-mode",
        choices=("product_range", "normal95"),
        default="product_range",
        help=(
            "Grey band calculation. product_range = min/max of GPM and ERA5-Land; "
            "normal95 = mean ± 1.96*std from the two products."
        ),
    )
    parser.add_argument(
        "--ci-label",
        default="95% confidence interval",
        help="Legend label for the grey band. Kept as requested by default.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="Figure DPI.")
    parser.add_argument("--fig-width", type=float, default=10.5, help="Figure width in inches.")
    parser.add_argument("--fig-height", type=float, default=9.2, help="Figure height in inches.")
    return parser.parse_args()


def configure_plotting() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Arial", "DejaVu Sans", "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"


def parse_bjt(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"Invalid BJT timestamp: {value}")
    return timestamp.tz_localize(None)


def find_name(names: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    lookup = {str(name).lower(): str(name) for name in names}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def coordinate_edges(centers: np.ndarray, default_half_width: float) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    if centers.size == 1:
        return np.array([centers[0] - default_half_width, centers[0] + default_half_width])
    middle = (centers[:-1] + centers[1:]) / 2.0
    first = centers[0] - (middle[0] - centers[0])
    last = centers[-1] + (centers[-1] - middle[-1])
    return np.concatenate(([first], middle, [last]))


def build_station_mapping(points: Sequence[Point], latitudes: np.ndarray, longitudes: np.ndarray) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for point in points:
        lat_index = int(np.argmin(np.abs(latitudes - point.lat)))
        lon_index = int(np.argmin(np.abs(longitudes - point.lon)))
        rows.append(
            {
                "point": point.name,
                "station_lat": point.lat,
                "station_lon": point.lon,
                "grid_lat": float(latitudes[lat_index]),
                "grid_lon": float(longitudes[lon_index]),
                "grid_lat_index": lat_index,
                "grid_lon_index": lon_index,
            }
        )
    return pd.DataFrame(rows)


def parse_gpm_filename(path: Path) -> GPMFile:
    match = GPM_FILENAME_TIME_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot parse IMERG time from filename: {path.name}")
    start_utc = datetime.strptime(match.group("date") + match.group("start"), "%Y%m%d%H%M%S")
    return GPMFile(path=path, start_utc=start_utc, nominal_end_utc=start_utc + timedelta(minutes=30))


def discover_gpm_files(gpm_dir: Path, start_bjt: pd.Timestamp, end_bjt: pd.Timestamp) -> List[GPMFile]:
    paths = sorted(gpm_dir.glob("*.nc4"))
    if not paths:
        raise FileNotFoundError(f"No *.nc4 files found in {gpm_dir}")
    start_utc = start_bjt.to_pydatetime() - timedelta(hours=8)
    end_utc = end_bjt.to_pydatetime() - timedelta(hours=8)
    parsed: List[GPMFile] = []
    for path in paths:
        try:
            item = parse_gpm_filename(path)
        except ValueError:
            continue
        if start_utc <= item.start_utc < end_utc:
            parsed.append(item)
    if not parsed:
        raise ValueError(f"No GPM files in requested UTC interval [{start_utc}, {end_utc}).")
    # Keep one file per half-hour timestamp.
    grouped: Dict[datetime, List[GPMFile]] = defaultdict(list)
    for item in parsed:
        grouped[item.start_utc].append(item)
    selected = [sorted(items, key=lambda x: x.path.name)[0] for items in grouped.values()]
    selected.sort(key=lambda x: x.start_utc)
    return selected


def open_imerg_dataset(path: Path) -> Tuple[xr.Dataset, Optional[str]]:
    errors: List[str] = []
    for group in (None, "Grid"):
        kwargs = {"decode_times": False, "mask_and_scale": True}
        if group is not None:
            kwargs["group"] = group
        try:
            ds = xr.open_dataset(path, **kwargs)
        except Exception as exc:
            errors.append(f"group={group!r}: {exc}")
            continue
        if find_name(ds.data_vars, GPM_PRECIP_NAMES) is not None:
            return ds, group
        ds.close()
        errors.append(f"group={group!r}: precipitation variable not found")
    raise ValueError(f"Could not open precipitation in {path}. " + " | ".join(errors))


def standardize_gpm_precip(ds: xr.Dataset) -> Tuple[xr.DataArray, str, str]:
    precip_name = find_name(ds.data_vars, GPM_PRECIP_NAMES)
    lat_name = find_name(list(ds.coords) + list(ds.variables), ("lat", "latitude"))
    lon_name = find_name(list(ds.coords) + list(ds.variables), ("lon", "longitude"))
    if precip_name is None or lat_name is None or lon_name is None:
        raise KeyError(f"Cannot identify GPM precipitation/lat/lon in {list(ds.variables)}")
    da = ds[precip_name]
    for dim in list(da.dims):
        if dim not in (lat_name, lon_name):
            da = da.isel({dim: 0}, drop=True)
    da = da.transpose(lat_name, lon_name)
    if float(da[lat_name][0]) > float(da[lat_name][-1]):
        da = da.sortby(lat_name)
    if float(da[lon_name][0]) > float(da[lon_name][-1]):
        da = da.sortby(lon_name)
    return da, lat_name, lon_name


def subset_dataarray(da: xr.DataArray, lat_name: str, lon_name: str, bbox: Sequence[float]) -> xr.DataArray:
    lon_min, lon_max, lat_min, lat_max = bbox
    subset = da.sel({lat_name: slice(lat_min, lat_max), lon_name: slice(lon_min, lon_max)})
    if subset.sizes.get(lat_name, 0) == 0 or subset.sizes.get(lon_name, 0) == 0:
        raise ValueError(f"Requested bbox {tuple(bbox)} does not overlap the grid.")
    return subset


def clean_precip(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    arr[arr < 0.0] = np.nan
    return arr


def extract_gpm_hourly(gpm_dir: Path, start_bjt: pd.Timestamp, end_bjt: pd.Timestamp,
                       points: Sequence[Point], bbox: Sequence[float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    files = discover_gpm_files(gpm_dir, start_bjt, end_bjt)
    mapping: Optional[pd.DataFrame] = None
    rows: List[Dict[str, object]] = []
    for i, item in enumerate(files):
        ds, _ = open_imerg_dataset(item.path)
        try:
            da, lat_name, lon_name = standardize_gpm_precip(ds)
            subset = subset_dataarray(da, lat_name, lon_name, bbox)
            lats = np.asarray(subset[lat_name].values, dtype=float)
            lons = np.asarray(subset[lon_name].values, dtype=float)
            rate = clean_precip(subset.load().values)
        finally:
            ds.close()
        if i == 0:
            mapping = build_station_mapping(points, lats, lons)
        # IMERG precipitation is a rate in mm h-1; half-hour amount = rate * 0.5.
        amount = rate * 0.5
        row: Dict[str, object] = {"time_bjt": pd.Timestamp(item.end_bjt)}
        for station in mapping.itertuples(index=False):
            value = amount[int(station.grid_lat_index), int(station.grid_lon_index)]
            row[str(station.point)] = float(value) if np.isfinite(value) else np.nan
        rows.append(row)
    half_hourly = pd.DataFrame(rows).sort_values("time_bjt").reset_index(drop=True)
    hourly = half_hourly.set_index("time_bjt").resample("1h", closed="right", label="right").sum(min_count=1).reset_index()
    hourly = hourly[(hourly["time_bjt"] > start_bjt) & (hourly["time_bjt"] <= end_bjt)].copy()
    return hourly, mapping if mapping is not None else pd.DataFrame()


def normalize_longitudes(lons: np.ndarray) -> np.ndarray:
    return ((np.asarray(lons, dtype=float) + 180.0) % 360.0) - 180.0


def open_era5land_tp(path: Path) -> xr.Dataset:
    errors: List[str] = []
    for filter_by_keys in ({"shortName": "tp", "typeOfLevel": "surface"}, {"shortName": "tp"}, None):
        try:
            backend_kwargs = {"indexpath": ""}
            if filter_by_keys is not None:
                backend_kwargs["filter_by_keys"] = filter_by_keys
            ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs=backend_kwargs)
        except Exception as exc:
            errors.append(str(exc))
            continue
        if find_name(ds.data_vars, ("tp",)) is not None:
            return ds
        ds.close()
    raise ValueError("Could not open ERA5-Land tp from GRIB. " + " | ".join(errors[-3:]))


def collapse_aux_dims(da: xr.DataArray, lat_name: str, lon_name: str) -> xr.DataArray:
    protected = {lat_name, lon_name, "time", "valid_time", "step"}
    for dim in list(da.dims):
        if dim in protected:
            continue
        if da.sizes[dim] == 1:
            da = da.isel({dim: 0}, drop=True)
        elif dim.lower() in ("expver", "number"):
            da = da.max(dim=dim, skipna=True)
        else:
            raise ValueError(f"Unsupported dimension {dim!r} with size {da.sizes[dim]} in ERA5-Land tp.")
    return da


def subset_era5_da(da: xr.DataArray, lat_name: str, lon_name: str, bbox: Sequence[float]) -> xr.DataArray:
    lon_min, lon_max, lat_min, lat_max = map(float, bbox)
    lats = np.asarray(da[lat_name].values, dtype=float)
    lons = normalize_longitudes(np.asarray(da[lon_name].values, dtype=float))
    lat_idx = np.flatnonzero((lats >= lat_min) & (lats <= lat_max))
    lon_idx = np.flatnonzero((lons >= lon_min) & (lons <= lon_max))
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise ValueError(f"Requested bbox {tuple(bbox)} does not overlap ERA5-Land grid.")
    return da.isel({lat_name: lat_idx, lon_name: lon_idx})


def broadcast_coord(coord: Optional[xr.DataArray], template: xr.DataArray, sample_dims: Sequence[str]) -> Optional[np.ndarray]:
    if coord is None:
        return None
    if not sample_dims:
        return np.asarray([coord.values])
    out = xr.broadcast(coord, template)[0]
    out = out.transpose(*sample_dims)
    return np.asarray(out.values).reshape(-1)


def timedelta_hours(values: Optional[np.ndarray], size: int) -> np.ndarray:
    if values is None:
        return np.full(size, np.nan, dtype=float)
    flat = np.asarray(values).reshape(-1)
    out = np.full(flat.size, np.nan, dtype=float)
    for i, value in enumerate(flat):
        if np.issubdtype(np.asarray(value).dtype, np.timedelta64):
            out[i] = pd.to_timedelta(value).total_seconds() / 3600.0
        else:
            try:
                out[i] = pd.to_timedelta(value).total_seconds() / 3600.0
            except Exception:
                try:
                    out[i] = float(value)
                except Exception:
                    pass
    return out


def convert_to_mm(values: np.ndarray, units: str) -> Tuple[np.ndarray, str]:
    normalized = str(units).lower().replace(" ", "")
    if normalized in ("m", "metre", "meter", "mofwaterequivalent") or normalized.startswith("mofwater"):
        return values * 1000.0, "m to mm"
    if "kgm-2" in normalized or "kg/m2" in normalized or normalized.startswith("mm"):
        return values.copy(), "already mm equivalent"
    return values * 1000.0, f"unknown units {units!r}; assumed metres and multiplied by 1000"


def derive_era5_interval_amounts(raw_mm: np.ndarray, valid_times: pd.DatetimeIndex,
                                 init_times: pd.DatetimeIndex, step_hours: np.ndarray) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """Return hourly/interval precipitation amounts indexed by valid time.

    Handles both independent hourly accumulations and cumulative forecast accumulations.
    """
    finite_steps = step_hours[np.isfinite(step_hours)]
    repeated_init = len(np.unique(pd.DatetimeIndex(init_times).asi8)) < len(init_times)
    varying_step = finite_steps.size > 0 and (np.nanmax(finite_steps) - np.nanmin(finite_steps) > 0.01)
    cumulative = repeated_init and varying_step and np.nanmax(finite_steps) > 1.01

    candidates: List[Tuple[pd.Timestamp, np.ndarray, float, float]] = []
    if cumulative:
        groups: Dict[pd.Timestamp, List[int]] = defaultdict(list)
        for i, init_time in enumerate(init_times):
            groups[pd.Timestamp(init_time)].append(i)
        for _, indices in groups.items():
            indices.sort(key=lambda idx: (np.inf if not np.isfinite(step_hours[idx]) else step_hours[idx], valid_times[idx]))
            previous: Optional[int] = None
            for idx in indices:
                step = step_hours[idx]
                if previous is None:
                    if np.isfinite(step) and step > 1.01:
                        previous = idx
                        continue
                    amount = raw_mm[idx].copy()
                    interval = step if np.isfinite(step) and step > 0 else 1.0
                else:
                    previous_step = step_hours[previous]
                    interval = step - previous_step if np.isfinite(step) and np.isfinite(previous_step) else np.nan
                    if not np.isfinite(interval) or interval <= 0:
                        previous = idx
                        continue
                    amount = raw_mm[idx] - raw_mm[previous]
                amount[np.isfinite(amount) & (amount < 0.0)] = 0.0
                candidates.append((pd.Timestamp(valid_times[idx]), amount, float(interval), float(step)))
                previous = idx
    else:
        finite_positive = step_hours[np.isfinite(step_hours) & (step_hours > 0)]
        default_interval = float(np.nanmedian(finite_positive)) if finite_positive.size else 1.0
        for idx, valid_time in enumerate(valid_times):
            step = step_hours[idx] if np.isfinite(step_hours[idx]) else default_interval
            candidates.append((pd.Timestamp(valid_time), raw_mm[idx].copy(), float(step), float(step)))

    # Deduplicate valid times: prefer intervals closest to 1 h, then smallest forecast step.
    grouped: Dict[pd.Timestamp, List[Tuple[pd.Timestamp, np.ndarray, float, float]]] = defaultdict(list)
    for item in candidates:
        grouped[item[0]].append(item)
    selected: List[Tuple[pd.Timestamp, np.ndarray]] = []
    for timestamp, items in grouped.items():
        items.sort(key=lambda item: (abs(item[2] - 1.0), np.inf if not np.isfinite(item[3]) else item[3]))
        selected.append((timestamp, items[0][1]))
    selected.sort(key=lambda item: item[0])
    return pd.DatetimeIndex([item[0] for item in selected]), np.stack([item[1] for item in selected], axis=0)


def extract_era5land_hourly(grib_path: Path, start_bjt: pd.Timestamp, end_bjt: pd.Timestamp,
                             points: Sequence[Point], bbox: Sequence[float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ds = open_era5land_tp(grib_path)
    try:
        var_name = find_name(ds.data_vars, ("tp",))
        lat_name = find_name(list(ds.coords) + list(ds.variables), ("latitude", "lat"))
        lon_name = find_name(list(ds.coords) + list(ds.variables), ("longitude", "lon"))
        if var_name is None or lat_name is None or lon_name is None:
            raise KeyError("Cannot identify ERA5-Land tp/latitude/longitude coordinates.")
        da = collapse_aux_dims(ds[var_name], lat_name, lon_name)
        da = subset_era5_da(da, lat_name, lon_name, bbox)
        da.load()
        sample_dims = [dim for dim in da.dims if dim not in (lat_name, lon_name)]
        da = da.transpose(*(sample_dims + [lat_name, lon_name]))
        template = da.isel({lat_name: 0, lon_name: 0}, drop=True)
        valid_values = broadcast_coord(ds.coords.get("valid_time"), template, sample_dims)
        time_values = broadcast_coord(ds.coords.get("time"), template, sample_dims)
        step_values = broadcast_coord(ds.coords.get("step"), template, sample_dims)

        sample_count = int(np.prod([da.sizes[d] for d in sample_dims])) if sample_dims else 1
        raw = np.asarray(da.values, dtype=float).reshape(sample_count, da.sizes[lat_name], da.sizes[lon_name])
        raw[~np.isfinite(raw)] = np.nan
        raw[raw < 0] = np.nan
        units = str(da.attrs.get("units", "unknown"))
        raw_mm, conversion_msg = convert_to_mm(raw, units)
        if conversion_msg.startswith("unknown"):
            warnings.warn(conversion_msg)

        lats = np.asarray(da[lat_name].values, dtype=float)
        lons = normalize_longitudes(np.asarray(da[lon_name].values, dtype=float))
        # Sort lat/lon ascending after extraction.
        lat_order = np.argsort(lats)
        lon_order = np.argsort(lons)
        raw_mm = raw_mm[:, lat_order, :][:, :, lon_order]
        lats = lats[lat_order]
        lons = lons[lon_order]

        if valid_values is not None:
            valid_times = pd.DatetimeIndex(pd.to_datetime(valid_values))
        elif time_values is not None:
            valid_times = pd.DatetimeIndex(pd.to_datetime(time_values))
        else:
            raise ValueError("ERA5-Land tp has no time or valid_time coordinate.")
        if time_values is not None:
            init_times = pd.DatetimeIndex(pd.to_datetime(time_values))
        else:
            init_times = pd.DatetimeIndex(valid_times)
        step_hours = timedelta_hours(step_values, sample_count)
        times_utc, amounts = derive_era5_interval_amounts(raw_mm, valid_times, init_times, step_hours)
    finally:
        ds.close()

    start_utc = start_bjt - pd.Timedelta(hours=8)
    end_utc = end_bjt - pd.Timedelta(hours=8)
    mask = (times_utc > start_utc) & (times_utc <= end_utc)
    if not np.any(mask):
        raise ValueError(f"No ERA5-Land records in UTC interval ({start_utc}, {end_utc}].")
    times_utc = times_utc[mask]
    amounts = amounts[mask]
    mapping = build_station_mapping(points, lats, lons)
    rows: List[Dict[str, object]] = []
    for t_idx, utc_time in enumerate(times_utc):
        row: Dict[str, object] = {"time_bjt": pd.Timestamp(utc_time) + pd.Timedelta(hours=8)}
        for station in mapping.itertuples(index=False):
            value = amounts[t_idx, int(station.grid_lat_index), int(station.grid_lon_index)]
            row[str(station.point)] = float(value) if np.isfinite(value) else np.nan
        rows.append(row)
    hourly = pd.DataFrame(rows).sort_values("time_bjt").reset_index(drop=True)
    return hourly, mapping


def resolve_observation_columns(frame: pd.DataFrame, point_names: Sequence[str]) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for name in point_names:
        source = find_name(frame.columns, OBS_COLUMN_ALIASES.get(name, (name,)))
        if source is None:
            raise ValueError(f"Observation CSV lacks station column for {name}. Existing columns: {list(frame.columns)}")
        resolved[name] = source
    return resolved


def read_station_table(path: Path, sheet_name=0) -> pd.DataFrame:
    """Read a pre-extracted station table from CSV/TXT/XLSX/XLS.

    Expected columns are a BJT time column plus station columns P5200, P5800,
    and P6500. Old P6400 aliases are still accepted and shown as P6500.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    raise ValueError(f"Unsupported station table type: {path}. Use .csv, .txt, .xlsx, or .xls.")


def load_station_csv(path: Path, start_bjt: pd.Timestamp, end_bjt: pd.Timestamp,
                     point_names: Sequence[str], prefix: str = "") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = read_station_table(path)
    frame.columns = [str(c).strip() for c in frame.columns]
    time_col = find_name(frame.columns, ("time_bjt", "Beijing Time", "beijing_time", "time", "date", "datetime"))
    if time_col is None:
        raise ValueError(f"Station table lacks a time column: {path}")
    frame["time_bjt"] = pd.to_datetime(frame[time_col], errors="coerce").dt.tz_localize(None)
    frame = frame.dropna(subset=["time_bjt"]).copy()
    station_columns = resolve_observation_columns(frame, point_names)
    out = pd.DataFrame({"time_bjt": frame["time_bjt"]})
    for name in point_names:
        out[name] = pd.to_numeric(frame[station_columns[name]], errors="coerce")
        out.loc[out[name] < 0, name] = np.nan
    out = out[(out["time_bjt"] > start_bjt) & (out["time_bjt"] <= end_bjt)].copy()
    out = out.set_index("time_bjt").resample("1h", closed="right", label="right").sum(min_count=1).reset_index()
    if prefix:
        out = out.rename(columns={name: f"{prefix}_{name}" for name in point_names})
    return out


def load_obs(path: Path, start_bjt: pd.Timestamp, end_bjt: pd.Timestamp, point_names: Sequence[str]) -> pd.DataFrame:
    return load_station_csv(path, start_bjt, end_bjt, point_names, prefix="obs")


def rename_product(frame: pd.DataFrame, point_names: Sequence[str], prefix: str) -> pd.DataFrame:
    out = frame.copy()
    out["time_bjt"] = pd.to_datetime(out["time_bjt"], errors="coerce").dt.tz_localize(None)
    rename: Dict[str, str] = {}
    for name in point_names:
        source = find_name(out.columns, (name, f"{prefix}_{name}", f"tp_{name}"))
        if source is None:
            raise ValueError(f"Product frame lacks station {name}. Columns: {list(out.columns)}")
        rename[source] = f"{prefix}_{name}"
    return out[["time_bjt"] + list(rename.keys())].rename(columns=rename)


def merge_all(obs: pd.DataFrame, gpm: pd.DataFrame, era5land: pd.DataFrame, start_bjt: pd.Timestamp,
              end_bjt: pd.Timestamp, point_names: Sequence[str]) -> pd.DataFrame:
    merged = obs.copy()
    merged = pd.merge(merged, rename_product(gpm, point_names, "gpm"), on="time_bjt", how="outer")
    merged = pd.merge(merged, rename_product(era5land, point_names, "era5_land"), on="time_bjt", how="outer")
    expected = pd.DataFrame({"time_bjt": pd.date_range(start=start_bjt + pd.Timedelta(hours=1), end=end_bjt, freq="1h")})
    merged = pd.merge(expected, merged, on="time_bjt", how="left").sort_values("time_bjt").reset_index(drop=True)
    return merged


def compute_ci_band(gpm: object, era5: object, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return lower/upper grey-band bounds from two product series.

    Accepts either pandas Series or NumPy arrays. The default mode uses the
    product envelope [min(GPM, ERA5-Land), max(GPM, ERA5-Land)].
    """
    gpm_values = np.asarray(gpm, dtype=float)
    era5_values = np.asarray(era5, dtype=float)
    values = np.vstack([gpm_values, era5_values])
    if mode == "normal95":
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        lower = np.maximum(mean - 1.96 * std, 0.0)
        upper = mean + 1.96 * std
    else:
        with np.errstate(all="ignore"):
            lower = np.nanmin(values, axis=0)
            upper = np.nanmax(values, axis=0)
    invalid = ~np.isfinite(lower) | ~np.isfinite(upper)
    lower[invalid] = np.nan
    upper[invalid] = np.nan
    return lower, upper


def metrics_for_product(merged: pd.DataFrame, point_names: Sequence[str], product_prefix: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for name in point_names:
        obs_col = f"obs_{name}"
        prod_col = f"{product_prefix}_{name}"
        paired = merged[["time_bjt", obs_col, prod_col]].dropna()
        diff = paired[prod_col] - paired[obs_col] if not paired.empty else pd.Series(dtype=float)
        obs_total = merged[obs_col].sum(min_count=1)
        prod_total = merged[prod_col].sum(min_count=1)
        row: Dict[str, object] = {
            "station": name,
            "product": "GPM" if product_prefix == "gpm" else "ERA5-Land",
            "paired_hours": int(len(paired)),
            "obs_total_mm": float(obs_total) if pd.notna(obs_total) else np.nan,
            "product_total_mm": float(prod_total) if pd.notna(prod_total) else np.nan,
            "event_total_bias_mm": float(prod_total - obs_total) if pd.notna(obs_total) and pd.notna(prod_total) else np.nan,
            "event_total_relative_bias_percent": (
                float((prod_total - obs_total) / obs_total * 100.0)
                if pd.notna(obs_total) and pd.notna(prod_total) and obs_total != 0 else np.nan
            ),
            "mean_bias_mm_per_hour": float(diff.mean()) if len(diff) else np.nan,
            "mae_mm_per_hour": float(diff.abs().mean()) if len(diff) else np.nan,
            "rmse_mm_per_hour": float(np.sqrt(np.mean(np.square(diff)))) if len(diff) else np.nan,
            "correlation": (
                float(paired[prod_col].corr(paired[obs_col]))
                if len(paired) >= 2 and paired[prod_col].std() > 0 and paired[obs_col].std() > 0 else np.nan
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)



def setup_event_style(font_size: int = 9) -> None:
    """Reference-style plotting settings adapted from the JJAS error-evaluation script."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Microsoft YaHei", "SimHei"],
        "font.size": font_size,
        "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def style_event_axis(ax: plt.Axes, grid: bool = True, face: bool = False) -> None:
    if face:
        ax.set_facecolor("#EAEAF2")
    if grid:
        ax.grid(True, color="white" if face else "0.86", linewidth=0.9, alpha=1.0, linestyle="-")
        ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("0.45")
        spine.set_linewidth(0.8)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.020, y: float = 1.020) -> None:
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="normal",
        va="bottom",
        ha="left",
        clip_on=False,
    )


def pearson_r_event(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    xx = x[mask]
    yy = y[mask]
    if np.nanstd(xx) == 0 or np.nanstd(yy) == 0:
        return np.nan
    return float(np.corrcoef(xx, yy)[0, 1])


def compute_event_group_metrics(merged: pd.DataFrame, point_names: Sequence[str]) -> pd.DataFrame:
    """Compute event-scale RMSE and Pearson r for All Stations and each station."""
    groups: Dict[str, Sequence[str]] = {"All Stations": list(point_names)}
    for station in point_names:
        groups[station] = [station]

    product_specs = [("gpm", "GPM"), ("era5_land", "ERA5-Land")]
    rows: List[Dict[str, object]] = []
    for group_name, stations in groups.items():
        obs = np.concatenate([merged[f"obs_{s}"].to_numpy(dtype=float) for s in stations])
        for prefix, label in product_specs:
            prod = np.concatenate([merged[f"{prefix}_{s}"].to_numpy(dtype=float) for s in stations])
            mask = np.isfinite(obs) & np.isfinite(prod)
            if mask.sum() == 0:
                rmse = r = np.nan
                n = 0
            else:
                diff = prod[mask] - obs[mask]
                rmse = float(np.sqrt(np.mean(diff ** 2)))
                r = pearson_r_event(obs[mask], prod[mask])
                n = int(mask.sum())
            rows.append({
                "Group": group_name,
                "Product": label,
                "N": n,
                "RMSE_mm_h": rmse,
                "Pearson_r": r,
            })
    return pd.DataFrame(rows)


def hourly_event_series(
    merged: pd.DataFrame,
    prefix: str,
    stations: Sequence[str],
    station: Optional[str] = None,
) -> np.ndarray:
    """Return hourly precipitation; All stations is the station mean, not the sum."""
    if station is None:
        cols = [f"{prefix}_{s}" for s in stations]
        hourly = merged[cols].mean(axis=1, skipna=True)
    else:
        hourly = merged[f"{prefix}_{station}"]
    return hourly.to_numpy(dtype=float)


def plot_event_metric_dot_panel(
    ax: plt.Axes,
    metrics: pd.DataFrame,
    metric_col: str,
    xlabel: str,
    xlim: Optional[Tuple[float, float]] = None,
    show_legend: bool = False,
    zero_line: bool = False,
) -> None:
    """Draw a Cleveland-style paired dot plot for event metrics.

    This is better than bars for a short event, especially for Pearson r,
    because values near zero remain visually meaningful.
    """
    groups = ["P5200", "P5800", "P6500"]
    products = ["GPM", "ERA5-Land"]
    product_marker = {"GPM": "^", "ERA5-Land": "s"}
    product_color = {"GPM": "#d73027", "ERA5-Land": "#2166ac"}
    product_yoffset = {"GPM": -0.10, "ERA5-Land": 0.10}

    y_centers = np.arange(len(groups), dtype=float)

    # Light reference segment spanning the two products in each station group.
    for y, group in zip(y_centers, groups):
        vals = []
        for product in products:
            subset = metrics[(metrics["Group"] == group) & (metrics["Product"] == product)]
            if not subset.empty:
                val = float(subset[metric_col].iloc[0])
                if np.isfinite(val):
                    vals.append(val)
        if len(vals) >= 2:
            ax.plot(
                [min(vals), max(vals)],
                [y, y],
                color="0.65",
                linewidth=0.8,
                zorder=1,
            )

    for product in products:
        xs = []
        ys = []
        for y, group in zip(y_centers, groups):
            subset = metrics[(metrics["Group"] == group) & (metrics["Product"] == product)]
            if subset.empty:
                xs.append(np.nan)
            else:
                xs.append(float(subset[metric_col].iloc[0]))
            ys.append(y + product_yoffset[product])

        ax.scatter(
            xs,
            ys,
            marker=product_marker[product],
            s=30,
            color=product_color[product],
            edgecolor="black",
            linewidth=0.35,
            label=product,
            zorder=3,
        )

    if zero_line:
        ax.axvline(0.0, color="0.35", linewidth=0.9, linestyle="--", zorder=1)

    ax.set_yticks(y_centers)
    ax.set_yticklabels(groups)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Station group")
    if xlim is not None:
        ax.set_xlim(*xlim)
    style_event_axis(ax, grid=True, face=True)

    if show_legend:
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.98, 0.90),
            frameon=True,
            framealpha=0.95,
            edgecolor="0.5",
            ncol=1,
            borderaxespad=0.0,
        )

def plot_event_hourly_panel(
    ax: plt.Axes,
    merged: pd.DataFrame,
    stations: Sequence[str],
    title: str,
    station: Optional[str] = None,
    legend: bool = False,
) -> None:
    times = pd.to_datetime(merged["time_bjt"]).to_numpy(dtype="datetime64[ns]")
    series_specs = [
        ("obs", "OBS", "black", "*", 4.2, 1.15),
        ("gpm", "GPM", "#d73027", "^", 2.4, 1.05),
        ("era5_land", "ERA5-Land", "#2166ac", "s", 2.3, 1.05),
    ]

    for prefix, label, color, marker, ms, lw in series_specs:
        y = hourly_event_series(merged, prefix, stations, station=station)
        ax.plot(
            times,
            y,
            label=label,
            color=color,
            marker=marker,
            markersize=ms,
            linewidth=lw,
            markeredgewidth=0.4,
            markevery=2,
        )

    ax.set_title(title, pad=7)
    ax.set_ylabel("Precipitation (mm)")
    ax.set_xlabel("UTC +8")
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    ax.set_xlim(pd.to_datetime(merged["time_bjt"]).min(), pd.to_datetime(merged["time_bjt"]).max())
    for tick in ax.get_xticklabels():
        tick.set_rotation(0)
        tick.set_ha("center")
    style_event_axis(ax, grid=True, face=False)
    if legend:
        ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="0.5")


def make_event_error_evaluation_figure(
    merged: pd.DataFrame,
    metrics: pd.DataFrame,
    point_names: Sequence[str],
    output_base: Path,
    dpi: int,
) -> None:
    setup_event_style(font_size=9)

    fig, axes = plt.subplots(
        3, 2, figsize=(11.6, 9.3),
        gridspec_kw={"height_ratios": [0.75, 1.0, 1.0]},
    )
    axes = axes.ravel()

    max_rmse = metrics["RMSE_mm_h"].replace([np.inf, -np.inf], np.nan).max()
    rmse_ylim = (0, max(1.0, float(max_rmse) * 1.35)) if pd.notna(max_rmse) else None

    plot_event_metric_dot_panel(axes[0], metrics, "RMSE_mm_h", "RMSE (mm h$^{-1}$)", xlim=rmse_ylim, show_legend=True)
    plot_event_metric_dot_panel(axes[1], metrics, "Pearson_r", "Pearson correlation", xlim=(-1.0, 1.0), zero_line=True)

    plot_event_hourly_panel(axes[2], merged, point_names, "All stations", station=None, legend=True)
    plot_event_hourly_panel(axes[3], merged, point_names, "P5200", station="P5200", legend=True)
    plot_event_hourly_panel(axes[4], merged, point_names, "P5800", station="P5800", legend=True)
    plot_event_hourly_panel(axes[5], merged, point_names, "P6500", station="P6500", legend=True)

    for i, (lab, ax) in enumerate(zip(["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"], axes)):
        if i % 2 == 1:
            add_panel_label(ax, lab, x=0.000, y=1.020)
        else:
            add_panel_label(ax, lab, x=-0.020, y=1.020)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.975, bottom=0.075, hspace=0.45, wspace=0.28)

    png = output_base.with_suffix(".png")
    pdf = output_base.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"[OK] Saved figure: {png}")
    print(f"[OK] Saved figure: {pdf}")

def main() -> None:
    args = parse_args()
    configure_plotting()
    start_bjt = parse_bjt(args.start_bjt)
    end_bjt = parse_bjt(args.end_bjt)
    if end_bjt <= start_bjt:
        raise ValueError("--end-bjt must be later than --start-bjt")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    points = [Point(*item) for item in DEFAULT_POINTS]
    point_names = [point.name for point in points]

    print("Loading OBS...", flush=True)
    obs = load_obs(Path(args.obs_csv), start_bjt, end_bjt, point_names)

    if args.gpm_hourly_csv:
        print("Loading precomputed GPM hourly CSV...", flush=True)
        gpm_hourly = load_station_csv(Path(args.gpm_hourly_csv), start_bjt, end_bjt, point_names, prefix="")
        gpm_mapping = pd.DataFrame()
    else:
        print("Extracting GPM hourly station precipitation...", flush=True)
        gpm_hourly, gpm_mapping = extract_gpm_hourly(Path(args.gpm_dir), start_bjt, end_bjt, points, args.bbox)
    gpm_hourly.to_csv(output_dir / "gpm_hourly_points_for_combined_plot.csv", index=False, encoding="utf-8-sig")
    if not gpm_mapping.empty:
        gpm_mapping.to_csv(output_dir / "gpm_station_grid_mapping.csv", index=False, encoding="utf-8-sig")

    if args.era5_land_hourly_csv:
        print("Loading precomputed ERA5-Land hourly CSV...", flush=True)
        era5land_hourly = load_station_csv(Path(args.era5_land_hourly_csv), start_bjt, end_bjt, point_names, prefix="")
        era5_mapping = pd.DataFrame()
    else:
        if args.era5_land_grib is None:
            raise SystemExit("Please provide --era5-land-grib or --era5-land-hourly-csv.")
        print("Extracting ERA5-Land hourly station precipitation...", flush=True)
        era5land_hourly, era5_mapping = extract_era5land_hourly(
            Path(args.era5_land_grib), start_bjt, end_bjt, points, args.bbox
        )
    era5land_hourly.to_csv(output_dir / "era5_land_hourly_points_for_combined_plot.csv", index=False, encoding="utf-8-sig")
    if not era5_mapping.empty:
        era5_mapping.to_csv(output_dir / "era5_land_station_grid_mapping.csv", index=False, encoding="utf-8-sig")

    merged = merge_all(obs, gpm_hourly, era5land_hourly, start_bjt, end_bjt, point_names)
    merged.to_csv(output_dir / "obs_gpm_era5land_hourly_three_stations.csv", index=False, encoding="utf-8-sig")

    station_metrics = pd.concat(
        [metrics_for_product(merged, point_names, "gpm"), metrics_for_product(merged, point_names, "era5_land")],
        ignore_index=True,
    )
    station_metrics.to_csv(output_dir / "obs_gpm_era5land_station_error_metrics.csv", index=False, encoding="utf-8-sig")

    event_metrics = compute_event_group_metrics(merged, point_names)
    event_metrics.to_csv(output_dir / "event_group_metrics_OBS_GPM_ERA5Land.csv", index=False, encoding="utf-8-sig")

    tag = f"{start_bjt.strftime('%Y%m%d%H')}_{end_bjt.strftime('%Y%m%d%H')}_BJT"
    output_base = output_dir / f"fig_error_evaluation_OBS_GPM_ERA5Land_event_{tag}_v4"
    make_event_error_evaluation_figure(
        merged=merged,
        metrics=event_metrics,
        point_names=point_names,
        output_base=output_base,
        dpi=args.dpi,
    )
    print("Done.", flush=True)
    print(f"Figure: {output_base.with_suffix('.png')}", flush=True)
    print(f"CSV: {output_dir / 'obs_gpm_era5land_hourly_three_stations.csv'}", flush=True)
    print(f"Station metrics: {output_dir / 'obs_gpm_era5land_station_error_metrics.csv'}", flush=True)
    print(f"Event group metrics: {output_dir / 'event_group_metrics_OBS_GPM_ERA5Land.csv'}", flush=True)


if __name__ == "__main__":
    main()
