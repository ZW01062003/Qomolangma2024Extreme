#!/usr/bin/env python3
"""Extract daily WRF precipitation from d02 regional bands and d04 station cells.

The spatial box is 27–29°N, 86–89°E. A WRF mass-grid cell is included when
its centre lies inside the box, and is assigned to one of three half-open
native-terrain bands: [4900, 5500), [5500, 6100), [6200, 6800) m.

The default event is [2024-09-27 00:00, 2024-09-29 00:00) Beijing time and is
split into two complete BJT days: Sep 27 and Sep 28. The left panel compares
OBS and nearest-grid d04 at three stations; the right panel shows d02
area-weighted native-HGT elevation bands over 27–29°N, 86–89°E. Each day,
the left panel has six independent bars from zero, ordered OBS then WRF D04
at P5200, P5800 and P6500. The right panel has three independent WRF D02
elevation-band bars. Their blue, green and gray fills reproduce the supplied
2.pptx rectangles' color stops in both panels.
WRF timestamps are assumed to be UTC unless --wrf-time-zone BJT is supplied.
Precipitation is calculated from differences of accumulated WRF fields, never
by summing cumulative values. RAINC and RAINNC are required; nonzero RAINSH is
included automatically unless --rainsh exclude is used. Bucket counters are
honoured when present. Negative accumulation jumps are treated as restarts and
reconstructed interval by interval; all such cells are written to the audit.

This rectangle is not a geomorphological north-slope or Rongbuk-basin mask.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import xarray as xr


VERSION = "v14"
OUTPUT_STEM = "WRF_OBS_d04stations_d02regional_27_28Sep2024_BJT_v14"
DOMAIN = {"south": 27.0, "north": 29.0, "west": 86.0, "east": 89.0}
UTC_OFFSET = pd.Timedelta(hours=8)
GRID_COLOR = "#D8D8D8"
TEXT_COLOR = "#111111"
MUTED_TEXT = "#666666"


@dataclass(frozen=True)
class ElevationBand:
    key: str
    display_name: str
    lower_m: float
    upper_m: float
    color: str


@dataclass(frozen=True)
class Station:
    key: str
    latitude: float
    longitude: float
    observed_elevation_m: float


BANDS = (
    ElevationBand("P5200", "5200 ± 300 m", 4900.0, 5500.0, "#537EA6"),
    ElevationBand("P5800", "5800 ± 300 m", 5500.0, 6100.0, "#537EA6"),
    ElevationBand("P6500", "6500 ± 300 m", 6200.0, 6800.0, "#537EA6"),
)

STATIONS = (
    Station("P5200", 28.128911, 86.859978, 5221.0),
    Station("P5800", 28.085467, 86.914133, 5800.0),
    Station("P6500", 28.030633, 86.940464, 6409.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WRF Sep 27/28 daily precipitation for regional bands and station cells."
    )
    parser.add_argument(
        "--wrf-regional", type=Path, required=True,
        help="d02 WRF file used for the 27–29°N, 86–89°E regional elevation bands.",
    )
    parser.add_argument(
        "--wrf-station", type=Path, required=True,
        help="d04 WRF file used for P5200/P5800/P6500 nearest-grid precipitation.",
    )
    parser.add_argument(
        "--obs-hourly-file", type=Path, required=True,
        help="Station hourly precipitation CSV; its timestamps are already BJT.",
    )
    parser.add_argument(
        "--obs-hour-label", choices=("end", "start"), default="end",
        help="Whether an OBS timestamp labels the END (default) or START of its hour.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-bjt", default="2024-09-27 00:00:00")
    parser.add_argument("--end-bjt", default="2024-09-29 00:00:00")
    parser.add_argument(
        "--wrf-time-zone", choices=("UTC", "BJT"), default="UTC",
        help="Time basis of Times/XTIME stored in the WRF file (default: UTC).",
    )
    parser.add_argument(
        "--rainsh", choices=("auto", "include", "exclude"), default="auto",
        help="auto includes RAINSH only when present and nonzero.",
    )
    parser.add_argument(
        "--bucket-mm", type=float, default=None,
        help="WRF bucket_mm value; needed only if nonzero I_RAIN* exists and the file lacks BUCKET_MM.",
    )
    parser.add_argument(
        "--ymax", type=float, default=0.0,
        help="Y-axis maximum in mm; 0 selects an automatic maximum.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if args.ymax < 0:
        parser.error("--ymax must be 0 (automatic) or positive")
    if not 72 <= args.dpi <= 2400:
        parser.error("--dpi must be between 72 and 2400")
    if args.bucket_mm is not None and args.bucket_mm <= 0:
        parser.error("--bucket-mm must be positive")
    return args


def decode_wrf_times(ds: xr.Dataset) -> pd.DatetimeIndex:
    if "Times" in ds.variables:
        values = np.asarray(ds["Times"].values)
        decoded: list[str] = []
        if values.ndim == 2:
            for row in values:
                if row.dtype.kind == "S":
                    decoded.append(b"".join(row.tolist()).decode("utf-8").strip("\x00 "))
                else:
                    decoded.append("".join(str(item) for item in row).strip("\x00 "))
        elif values.ndim == 1:
            for item in values:
                decoded.append(
                    item.decode("utf-8").strip("\x00 ")
                    if isinstance(item, (bytes, np.bytes_))
                    else str(item).strip("\x00 ")
                )
        else:
            raise ValueError(f"Unsupported Times shape: {values.shape}")
        times = pd.DatetimeIndex(pd.to_datetime(decoded, format="%Y-%m-%d_%H:%M:%S"))
    elif "XTIME" in ds.variables:
        xtime = ds["XTIME"]
        units = str(xtime.attrs.get("units", ""))
        if "since" not in units.lower():
            start = ds.attrs.get("START_DATE") or ds.attrs.get("SIMULATION_START_DATE")
            if start is None:
                raise ValueError("XTIME lacks 'since' units and WRF START_DATE is unavailable")
            origin = pd.Timestamp(str(start).replace("_", " "))
        else:
            origin = pd.Timestamp(units.lower().split("since", 1)[1].strip().replace("_", " "))
        times = pd.DatetimeIndex(origin + pd.to_timedelta(np.asarray(xtime.values), unit="min"))
    else:
        raise KeyError("WRF file contains neither Times nor XTIME")
    if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("WRF timestamps must be valid, unique and strictly increasing")
    return times


def first_existing(ds: xr.Dataset, names: tuple[str, ...]) -> str:
    for name in names:
        if name in ds.variables:
            return name
    raise KeyError(f"None of these WRF variables is present: {', '.join(names)}")


def field_2d(
    da: xr.DataArray,
    y_dim: str,
    x_dim: str,
    time_dim: str | None = None,
    time_index: int = 0,
) -> np.ndarray:
    indexers = {
        dim: (time_index if dim == time_dim else 0)
        for dim in da.dims
        if dim not in (y_dim, x_dim)
    }
    result = da.isel(indexers).transpose(y_dim, x_dim)
    return np.asarray(result.values, dtype="float64")


def find_exact_time(times: pd.DatetimeIndex, target: pd.Timestamp, name: str) -> int:
    positions = np.flatnonzero(times == target)
    if len(positions) == 1:
        return int(positions[0])
    closest_i = int(np.argmin(np.abs(times - target)))
    raise ValueError(
        f"Exact {name} boundary {target} is absent from WRF Times. "
        f"Closest timestamp is {times[closest_i]}. Accumulations are not interpolated."
    )


def bucket_value(ds: xr.Dataset, override: float | None) -> float | None:
    if override is not None:
        return float(override)
    for key, value in ds.attrs.items():
        if str(key).lower() == "bucket_mm":
            try:
                number = float(np.asarray(value).squeeze())
            except (TypeError, ValueError):
                continue
            if number > 0:
                return number
    return None


def accumulated_component(
    ds: xr.Dataset,
    name: str,
    time_dim: str,
    time_indices: np.ndarray,
    y_dim: str,
    x_dim: str,
    y_slice: slice,
    x_slice: slice,
    bucket_mm: float | None,
) -> np.ndarray:
    da = ds[name]
    units = str(da.attrs.get("units", "mm")).lower().replace(" ", "")
    if units not in ("mm", "kgm-2", "kg/m2", "kgm**-2"):
        raise ValueError(f"{name} must be accumulated water depth in mm; found units={units!r}")
    selected = da.isel(
        {time_dim: time_indices, y_dim: y_slice, x_dim: x_slice}
    ).transpose(time_dim, y_dim, x_dim)
    accumulated = np.asarray(selected.values, dtype="float64")
    counter_name = f"I_{name}"
    if counter_name in ds.variables:
        counter_da = ds[counter_name].isel(
            {time_dim: time_indices, y_dim: y_slice, x_dim: x_slice}
        ).transpose(time_dim, y_dim, x_dim)
        counter = np.asarray(counter_da.values, dtype="float64")
        if np.any(counter != 0):
            if bucket_mm is None:
                raise ValueError(
                    f"{counter_name} is nonzero, but bucket_mm could not be read. "
                    "Supply --bucket-mm with the namelist value."
                )
            accumulated = accumulated + bucket_mm * counter
    if not np.isfinite(accumulated).all():
        raise ValueError(f"{name} contains missing/non-finite values in the selected window")
    return accumulated


def interval_increments(accumulated: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.diff(accumulated, axis=0)
    tiny_negative = (raw < 0.0) & (raw >= -0.01)
    reset = raw < -0.01
    increments = raw.copy()
    increments[tiny_negative] = 0.0
    # At a restart the new accumulated value is precipitation since the reset.
    increments[reset] = accumulated[1:][reset]
    if np.any(increments < -1e-8):
        raise ValueError("Negative interval precipitation remains after restart handling")
    return increments, reset.sum(axis=(1, 2)), tiny_negative.sum(axis=(1, 2))


def weighted_stats(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return {
            "area_weighted_mean_mm": np.nan,
            "area_weighted_sd_mm": np.nan,
            "minimum_mm": np.nan,
            "p25_mm": np.nan,
            "median_mm": np.nan,
            "p75_mm": np.nan,
            "maximum_mm": np.nan,
        }
    v, w = values[valid], weights[valid]
    mean = float(np.average(v, weights=w))
    variance = float(np.average((v - mean) ** 2, weights=w))
    return {
        "area_weighted_mean_mm": mean,
        "area_weighted_sd_mm": float(np.sqrt(max(0.0, variance))),
        "minimum_mm": float(np.min(v)),
        "p25_mm": float(np.percentile(v, 25)),
        "median_mm": float(np.median(v)),
        "p75_mm": float(np.percentile(v, 75)),
        "maximum_mm": float(np.max(v)),
    }


def nearest_station_cells(
    lat: np.ndarray,
    lon: np.ndarray,
    hgt: np.ndarray,
    box: np.ndarray,
    y_offset: int,
    x_offset: int,
) -> pd.DataFrame:
    """Return the nearest in-box WRF mass-grid centre for each station."""
    rows = []
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    iy, ix = np.indices(lat.shape)
    for station in STATIONS:
        station_lat = np.deg2rad(station.latitude)
        station_lon = np.deg2rad(station.longitude)
        dlat = lat_rad - station_lat
        dlon = lon_rad - station_lon
        haversine = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(station_lat) * np.cos(lat_rad) * np.sin(dlon / 2.0) ** 2
        )
        distance_km = 6371.0088 * 2.0 * np.arcsin(
            np.sqrt(np.clip(haversine, 0.0, 1.0))
        )
        distance_km = np.where(box & np.isfinite(distance_km), distance_km, np.inf)
        flat_index = int(np.argmin(distance_km))
        local_y, local_x = np.unravel_index(flat_index, distance_km.shape)
        if not np.isfinite(distance_km[local_y, local_x]):
            raise ValueError(f"No valid WRF grid cell found for {station.key}")
        rows.append(
            {
                "station": station.key,
                "station_latitude": station.latitude,
                "station_longitude": station.longitude,
                "observed_elevation_m": station.observed_elevation_m,
                "south_north_index": int(iy[local_y, local_x] + y_offset),
                "west_east_index": int(ix[local_y, local_x] + x_offset),
                "local_y_index": int(local_y),
                "local_x_index": int(local_x),
                "wrf_grid_latitude": float(lat[local_y, local_x]),
                "wrf_grid_longitude": float(lon[local_y, local_x]),
                "nearest_grid_distance_km": float(distance_km[local_y, local_x]),
                "wrf_HGT_m": float(hgt[local_y, local_x]),
            }
        )
    result = pd.DataFrame(rows)
    duplicated = result.duplicated(
        ["south_north_index", "west_east_index"], keep=False
    )
    result["shares_grid_cell_with_another_station"] = duplicated
    if duplicated.any():
        shared = ", ".join(result.loc[duplicated, "station"])
        warnings.warn(f"Some stations share the same nearest WRF grid cell: {shared}")
    return result


def extract_station_daily_from_d04(
    wrf_path: Path,
    args: argparse.Namespace,
    start_bjt: pd.Timestamp,
    end_bjt: pd.Timestamp,
    day_starts_bjt: list[pd.Timestamp],
    day_ends_bjt: list[pd.Timestamp],
    day_labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Extract two BJT days at the nearest d04 grid cell to each station."""
    start_wrf = start_bjt - UTC_OFFSET if args.wrf_time_zone == "UTC" else start_bjt
    end_wrf = end_bjt - UTC_OFFSET if args.wrf_time_zone == "UTC" else end_bjt

    with xr.open_dataset(wrf_path, decode_times=False, mask_and_scale=True) as ds:
        times = decode_wrf_times(ds)
        start_i = find_exact_time(times, start_wrf, "d04 start")
        end_i = find_exact_time(times, end_wrf, "d04 end")
        if end_i <= start_i:
            raise ValueError("Resolved d04 event endpoints are reversed")
        time_indices = np.arange(start_i, end_i + 1)
        selected_times = times[time_indices]
        gaps_hours = (
            np.diff(selected_times.values).astype("timedelta64[s]").astype(float)
            / 3600.0
        )
        if np.any(gaps_hours <= 0):
            raise ValueError("d04 event times are not strictly increasing")
        interval_starts_bjt = (
            selected_times[:-1] + UTC_OFFSET
            if args.wrf_time_zone == "UTC"
            else selected_times[:-1]
        )
        interval_ends_bjt = (
            selected_times[1:] + UTC_OFFSET
            if args.wrf_time_zone == "UTC"
            else selected_times[1:]
        )

        base = ds["RAINNC"] if "RAINNC" in ds.variables else None
        if base is None or "RAINC" not in ds.variables:
            raise KeyError("Both RAINC and RAINNC are required in d04")
        y_dim, x_dim = base.dims[-2:]
        time_candidates = [dim for dim in base.dims if dim not in (y_dim, x_dim)]
        if len(time_candidates) != 1:
            raise ValueError(f"Cannot identify one d04 time dimension in RAINNC dims={base.dims}")
        time_dim = time_candidates[0]

        lat_name = first_existing(ds, ("XLAT", "XLAT_M", "lat", "latitude"))
        lon_name = first_existing(ds, ("XLONG", "XLONG_M", "lon", "longitude"))
        hgt_name = first_existing(ds, ("HGT", "HGT_M", "ter", "terrain"))
        lat = field_2d(ds[lat_name], y_dim, x_dim, time_dim)
        lon = field_2d(ds[lon_name], y_dim, x_dim, time_dim)
        hgt = field_2d(ds[hgt_name], y_dim, x_dim, time_dim)
        valid_grid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(hgt)
        if not np.any(valid_grid):
            raise ValueError("d04 latitude/longitude/HGT contain no valid cells")
        coverage = {
            "south": float(np.nanmin(lat[valid_grid])),
            "north": float(np.nanmax(lat[valid_grid])),
            "west": float(np.nanmin(lon[valid_grid])),
            "east": float(np.nanmax(lon[valid_grid])),
        }
        station_cells = nearest_station_cells(lat, lon, hgt, valid_grid, 0, 0)
        if (station_cells.nearest_grid_distance_km > 1.0).any():
            far = ", ".join(
                f"{row.station}={row.nearest_grid_distance_km:.2f} km"
                for row in station_cells.itertuples()
                if row.nearest_grid_distance_km > 1.0
            )
            warnings.warn(
                "Some d04 nearest cells are unexpectedly far from their stations: " + far
            )

        y_min = int(station_cells.south_north_index.min())
        y_max = int(station_cells.south_north_index.max())
        x_min = int(station_cells.west_east_index.min())
        x_max = int(station_cells.west_east_index.max())
        y_slice = slice(y_min, y_max + 1)
        x_slice = slice(x_min, x_max + 1)

        component_names = ["RAINC", "RAINNC"]
        rainsh_status = "absent"
        if "RAINSH" in ds.variables and args.rainsh != "exclude":
            test = ds["RAINSH"].isel(
                {time_dim: time_indices, y_dim: y_slice, x_dim: x_slice}
            )
            nonzero = bool(
                np.nanmax(np.abs(np.asarray(test.values, dtype=float))) > 1e-10
            )
            if args.rainsh == "include" or nonzero:
                component_names.append("RAINSH")
                rainsh_status = "included"
            else:
                rainsh_status = "present_but_zero_excluded"
        elif "RAINSH" in ds.variables:
            rainsh_status = "excluded_by_user"

        bucket = bucket_value(ds, args.bucket_mm)
        component_intervals: dict[str, np.ndarray] = {}
        reset_counts: dict[str, np.ndarray] = {}
        tiny_negative_counts: dict[str, np.ndarray] = {}
        for name in component_names:
            accumulated = accumulated_component(
                ds, name, time_dim, time_indices, y_dim, x_dim,
                y_slice, x_slice, bucket,
            )
            increments, resets, tiny = interval_increments(accumulated)
            component_intervals[name] = increments
            reset_counts[name] = resets
            tiny_negative_counts[name] = tiny

        daily_masks: dict[str, np.ndarray] = {}
        for label, day_start, day_end in zip(
            day_labels, day_starts_bjt, day_ends_bjt
        ):
            mask = np.asarray(
                (interval_starts_bjt >= day_start) & (interval_ends_bjt <= day_end),
                dtype=bool,
            )
            integrated_hours = float(np.sum(gaps_hours[mask]))
            if not np.any(mask) or abs(integrated_hours - 24.0) > 1e-6:
                raise ValueError(
                    f"d04 {label} contains {integrated_hours:.6f} integrated hours, not 24 h"
                )
            if interval_starts_bjt[mask][0] != day_start or interval_ends_bjt[mask][-1] != day_end:
                raise ValueError(f"d04 {label} lacks exact 00:00 BJT day boundaries")
            daily_masks[label] = mask

        daily_component_totals: dict[str, dict[str, np.ndarray]] = {}
        daily_precip: dict[str, np.ndarray] = {}
        for label in day_labels:
            daily_component_totals[label] = {
                name: component_intervals[name][daily_masks[label]].sum(axis=0)
                for name in component_names
            }
            daily_precip[label] = np.sum(
                np.stack(list(daily_component_totals[label].values())), axis=0
            )

        station_rows = []
        for station in STATIONS:
            cell = station_cells.loc[station_cells.station == station.key].iloc[0]
            local_y = int(cell.south_north_index) - y_slice.start
            local_x = int(cell.west_east_index) - x_slice.start
            total_48h = float(
                sum(daily_precip[label][local_y, local_x] for label in day_labels)
            )
            for label, day_start, day_end in zip(
                day_labels, day_starts_bjt, day_ends_bjt
            ):
                row = {
                    "wrf_domain": "d04",
                    "day_label": label,
                    "day_start_bjt": str(day_start),
                    "day_end_bjt": str(day_end),
                    "station": station.key,
                    "station_latitude": station.latitude,
                    "station_longitude": station.longitude,
                    "observed_elevation_m": station.observed_elevation_m,
                    "south_north_index": int(cell.south_north_index),
                    "west_east_index": int(cell.west_east_index),
                    "wrf_grid_latitude": float(cell.wrf_grid_latitude),
                    "wrf_grid_longitude": float(cell.wrf_grid_longitude),
                    "nearest_grid_distance_km": float(cell.nearest_grid_distance_km),
                    "wrf_HGT_m": float(cell.wrf_HGT_m),
                    "precipitation_mm": float(daily_precip[label][local_y, local_x]),
                    "total_48h_precipitation_mm": total_48h,
                }
                for name in component_names:
                    row[f"{name}_precipitation_mm"] = float(
                        daily_component_totals[label][name][local_y, local_x]
                    )
                station_rows.append(row)
        station_summary = pd.DataFrame(station_rows)

        station_metadata = {
            "wrf_domain": "d04",
            "wrf_file": str(wrf_path),
            "wrf_title": str(ds.attrs.get("TITLE", "")),
            "wrf_start_date_attribute": str(ds.attrs.get("START_DATE", "")),
            "wrf_coordinate_extent": coverage,
            "terrain_variable": hgt_name,
            "components_included": component_names,
            "rainsh_status": rainsh_status,
            "bucket_mm_used": bucket,
            "event_output_intervals": int(len(selected_times) - 1),
            "minimum_interval_hours": float(np.min(gaps_hours)),
            "maximum_interval_hours": float(np.max(gaps_hours)),
            "total_reset_cells_by_component": {
                name: int(np.sum(reset_counts[name])) for name in component_names
            },
            "total_tiny_negative_cells_by_component": {
                name: int(np.sum(tiny_negative_counts[name])) for name in component_names
            },
        }
    return station_cells, station_summary, station_metadata


def extract_obs_daily(
    path: Path,
    day_starts: list[pd.Timestamp],
    day_labels: list[str],
    hour_label: str,
) -> tuple[pd.DataFrame, dict]:
    """Sum 24 observed hourly amounts per BJT day without shifting timestamps."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    time_candidates = [c for c in df if any(k in str(c).lower() for k in ("time", "date", "bjt"))]
    if not time_candidates:
        raise ValueError(f"OBS CSV lacks a time/date column: {list(df.columns)}")
    time_col = time_candidates[0]
    ts = pd.to_datetime(df[time_col], errors="coerce")
    if ts.isna().any():
        raise ValueError(f"OBS CSV has {int(ts.isna().sum())} invalid timestamps in {time_col}")
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    if ts.duplicated().any():
        raise ValueError("OBS CSV contains duplicate hourly timestamps")
    df = df.assign(_time_bjt=ts).set_index("_time_bjt").sort_index()
    columns = {}
    for station in STATIONS:
        aliases = (station.key.lower(), "p6400") if station.key == "P6500" else (station.key.lower(),)
        names = [c for c in df.columns if any(a in str(c).lower() for a in aliases)]
        qualified = [c for c in names if any(k in str(c).lower() for k in ("precip", "rain", "obs", "mm"))]
        candidates = qualified or [c for c in names if str(c).lower() in aliases]
        if len(candidates) != 1:
            raise ValueError(f"Expected one OBS precipitation column for {station.key}; found {candidates}")
        columns[station.key] = candidates[0]
    rows = []
    for start, label in zip(day_starts, day_labels):
        offset = pd.Timedelta(hours=1) if hour_label == "end" else pd.Timedelta(0)
        expected = pd.date_range(start + offset, periods=24, freq="h")
        absent = expected.difference(df.index)
        if len(absent):
            raise ValueError(f"OBS {label} lacks {len(absent)} hourly timestamps; first missing: {absent[0]}")
        for station in STATIONS:
            values = pd.to_numeric(df.loc[expected, columns[station.key]], errors="coerce")
            if values.isna().any() or (values < 0).any():
                raise ValueError(f"OBS {station.key} {label} has missing, nonnumeric or negative hourly precipitation")
            rows.append({"day_label": label, "station": station.key, "precipitation_mm": float(values.sum())})
    return pd.DataFrame(rows), {"file": str(path), "time_column": str(time_col),
                                "station_columns": {k: str(v) for k, v in columns.items()},
                                "time_zone": "BJT (unchanged)", "hour_label": hour_label,
                                "hourly_samples_per_station_per_day": 24}


def plot_bars(
    regional_summary: pd.DataFrame,
    station_summary: pd.DataFrame,
    obs_summary: pd.DataFrame,
    day_labels: list[str],
    output_dir: Path,
    dpi: int,
    ymax_arg: float,
) -> Path:
    """Each day has six separate station bars and three regional band bars."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 11, "font.weight": "normal",
        "axes.labelweight": "normal",
        "axes.unicode_minus": False, "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR, "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
    })
    if len(day_labels) != 2:
        raise ValueError("The comparison requires exactly two BJT days")

    def matrix(table, labels, key_col, value_col):
        values = np.empty((len(day_labels), len(labels)), dtype=float)
        for i, day in enumerate(day_labels):
            for j, key in enumerate(labels):
                row = table.loc[(table[key_col] == key) & (table.day_label == day)]
                if len(row) != 1:
                    raise ValueError(f"Expected one {key} / {day} value in {key_col}; got {len(row)}")
                values[i, j] = float(row[value_col].iloc[0])
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Nonfinite or negative precipitation in {key_col}")
        return values

    stations = [station.key for station in STATIONS]
    obs = matrix(obs_summary, stations, "station", "precipitation_mm")
    d04 = matrix(station_summary, stations, "station", "precipitation_mm")
    d02 = matrix(regional_summary, [band.key for band in BANDS],
                 "band_key", "area_weighted_mean_mm")
    largest = max(float(np.max(values)) for values in (obs, d04, d02))
    if ymax_arg > 0 and largest > ymax_arg:
        raise ValueError(f"Maximum daily value {largest:.2f} mm exceeds --ymax {ymax_arg:.2f} mm")
    ymax = ymax_arg if ymax_arg > 0 else max(10.0, largest * 1.35)

    # Literal DrawingML stops from the three rectangles in the user's 2.pptx.
    # The XML's 270-degree direction places pos=0 at the lower edge.
    # Blue: #00B0F0, tints 66%, 44.5%, 23.5%; satMod 160%.
    # Green: #82C241, shades 30%, 67.5%, 100%; satMod 115%.
    # Gray: theme bg2=#E7E6E6 with lumMod 50%, a solid fill.
    def ppt_color(hex_color, *, tint=None, shade=None, saturation=1.0):
        rgb = np.asarray(mpl.colors.to_rgb(hex_color))
        if tint is not None:
            rgb = rgb * (1 - tint) + tint
        if shade is not None:
            rgb = rgb * shade
        hue, lightness, sat = colorsys.rgb_to_hls(*rgb)
        return np.asarray(colorsys.hls_to_rgb(
            hue, lightness, min(1.0, sat * saturation)))

    gray = np.asarray(mpl.colors.to_rgb("#E7E6E6")) * 0.5
    ppt_stops = (
        ((0.0, ppt_color("#00B0F0", tint=0.66, saturation=1.60)),
         (0.5, ppt_color("#00B0F0", tint=0.445, saturation=1.60)),
         (1.0, ppt_color("#00B0F0", tint=0.235, saturation=1.60))),
        ((0.09, ppt_color("#82C241", shade=0.30, saturation=1.15)),
         (0.56, ppt_color("#82C241", shade=0.675, saturation=1.15)),
         (1.0, ppt_color("#82C241", shade=1.00, saturation=1.15))),
        ((0.0, gray), (1.0, gray)),
    )
    regional_gradients = []
    regional_legend_colors = []
    grid = np.linspace(0.0, 1.0, 256)
    for stops in ppt_stops:
        positions = [pos for pos, _ in stops]
        colors = np.asarray([rgb for _, rgb in stops])
        ascending = np.stack([np.interp(grid, positions, colors[:, channel])
                              for channel in range(3)], axis=-1)
        regional_gradients.append(ascending.reshape(256, 1, 3))
        regional_legend_colors.append(ascending[128])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 5.0), dpi=200,
                                  sharey=True)
    centers = np.arange(2, dtype=float)
    station_width = 0.124
    station_offsets = (np.arange(6) - 2.5) * 0.135
    regional_width = 0.164
    regional_offsets = (np.arange(3) - 1) * 0.178

    def gradient_bar(ax, xpos, height, band_index, width):
        if height <= 0:
            return
        ax.imshow(regional_gradients[band_index], origin="lower", aspect="auto",
                  interpolation="bilinear",
                  extent=(xpos - width / 2, xpos + width / 2, 0, height),
                  zorder=3, clip_on=True)

    for j, (station, band) in enumerate(zip(stations, BANDS)):
        for day_index, day_center in enumerate(centers):
            for source_index, (source, amount) in enumerate(
                    (("OBS", obs[day_index, j]), ("WRF", d04[day_index, j]))):
                xpos = day_center + station_offsets[2 * j + source_index]
                gradient_bar(ax1, xpos, amount, j, station_width)
                # Reference lettering is vertical, centred within a column.
                # The shortest OBS columns cannot hold even three letters.
                inside = amount >= max(15.0, ymax * 0.13)
                label_y = amount * 0.59 if inside else amount + 6.0
                ax1.text(xpos, label_y, source, rotation=90,
                         ha="center", va="center", fontsize=8.5,
                         color="#101010", clip_on=False, zorder=5)
            gradient_bar(ax2, day_center + regional_offsets[j],
                         d02[day_index, j], j, regional_width)

    day_titles = [f"{int(pd.Timestamp(label + ' 2024').day)} {pd.Timestamp(label + ' 2024').strftime('%B')}"
                  for label in day_labels]
    for ax in (ax1, ax2):
        ax.set_xlim(-0.49, 1.49)
        ax.set_ylim(0, ymax)
        ax.tick_params(axis="x", direction="out", length=4, width=0.75,
                       pad=5, labelsize=11)
        ax.tick_params(axis="y", direction="out", length=5, width=0.75,
                       labelsize=11)
        ax.grid(False)
        for side in ("left", "bottom", "top", "right"):
            ax.spines[side].set_color("#888888")
            ax.spines[side].set_linewidth(0.75)

    ax1.set_ylabel("Daily precipitation (mm)", fontsize=13, labelpad=8)
    ax2.tick_params(axis="y", labelleft=False)
    ax1.legend(handles=[
        Patch(facecolor=color, edgecolor="none", label=station)
        for station, color in zip(stations, regional_legend_colors)
    ], loc="upper right", frameon=False, fontsize=10,
       handlelength=1.05, handletextpad=0.5, labelspacing=0.3,
       borderaxespad=0.65)
    ax2.legend(handles=[
        Patch(facecolor=color, edgecolor="none", linewidth=0,
              label=f"WRF {band.display_name}")
        for band, color in zip(BANDS, regional_legend_colors)
    ], loc="upper right", frameon=False, fontsize=10,
       handlelength=1.05, handletextpad=0.5, labelspacing=0.3,
       borderaxespad=0.65)
    # Set fixed date ticks last; do not let numeric auto-ticks replace them.
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(centers))
        ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(day_titles))
        for label in ax.get_xticklabels():
            label.set_rotation(0)
            label.set_ha("center")
            label.set_rotation_mode("anchor")
    fig.subplots_adjust(left=0.105, right=0.98, bottom=0.30, top=0.91,
                        wspace=0.11)
    png = output_dir / f"{OUTPUT_STEM}.png"
    fig.savefig(png, dpi=dpi, facecolor="white")
    plt.close(fig)
    return png

def main() -> None:
    args = parse_args()
    regional_wrf_path = args.wrf_regional.expanduser().resolve()
    station_wrf_path = args.wrf_station.expanduser().resolve()
    obs_path = args.obs_hourly_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not regional_wrf_path.is_file():
        raise FileNotFoundError(f"d02 regional WRF file does not exist: {regional_wrf_path}")
    if not station_wrf_path.is_file():
        raise FileNotFoundError(f"d04 station WRF file does not exist: {station_wrf_path}")
    if not obs_path.is_file():
        raise FileNotFoundError(f"OBS hourly CSV does not exist: {obs_path}")
    if regional_wrf_path == station_wrf_path:
        raise ValueError("--wrf-regional and --wrf-station must be different d02/d04 files")
    output_dir.mkdir(parents=True, exist_ok=True)
    start_bjt = pd.Timestamp(args.start_bjt)
    end_bjt = pd.Timestamp(args.end_bjt)
    if end_bjt <= start_bjt:
        raise ValueError("--end-bjt must be later than --start-bjt")
    if end_bjt - start_bjt != pd.Timedelta(days=2):
        raise ValueError(
            "This daily comparison requires exactly 48 h: end-bjt = start-bjt + 2 days"
        )
    if start_bjt != start_bjt.normalize():
        raise ValueError("--start-bjt must be 00:00:00 for two complete BJT days")
    day_starts_bjt = [start_bjt, start_bjt + pd.Timedelta(days=1)]
    day_ends_bjt = [value + pd.Timedelta(days=1) for value in day_starts_bjt]
    day_labels = [value.strftime("%b %d") for value in day_starts_bjt]
    start_wrf = start_bjt - UTC_OFFSET if args.wrf_time_zone == "UTC" else start_bjt
    end_wrf = end_bjt - UTC_OFFSET if args.wrf_time_zone == "UTC" else end_bjt

    with xr.open_dataset(regional_wrf_path, decode_times=False, mask_and_scale=True) as ds:
        times = decode_wrf_times(ds)
        start_i = find_exact_time(times, start_wrf, "start")
        end_i = find_exact_time(times, end_wrf, "end")
        if end_i <= start_i:
            raise ValueError("Resolved event endpoints are reversed")
        time_indices = np.arange(start_i, end_i + 1)
        selected_times = times[time_indices]
        gaps_hours = np.diff(selected_times.values).astype("timedelta64[s]").astype(float) / 3600.0
        if np.any(gaps_hours <= 0):
            raise ValueError("WRF event times are not strictly increasing")
        interval_starts_bjt = (
            selected_times[:-1] + UTC_OFFSET
            if args.wrf_time_zone == "UTC"
            else selected_times[:-1]
        )
        interval_ends_bjt = (
            selected_times[1:] + UTC_OFFSET
            if args.wrf_time_zone == "UTC"
            else selected_times[1:]
        )

        base = ds["RAINNC"] if "RAINNC" in ds.variables else None
        if base is None or "RAINC" not in ds.variables:
            raise KeyError("Both RAINC and RAINNC are required")
        y_dim, x_dim = base.dims[-2:]
        time_candidates = [dim for dim in base.dims if dim not in (y_dim, x_dim)]
        if len(time_candidates) != 1:
            raise ValueError(f"Cannot identify one time dimension in RAINNC dims={base.dims}")
        time_dim = time_candidates[0]

        lat_name = first_existing(ds, ("XLAT", "XLAT_M", "lat", "latitude"))
        lon_name = first_existing(ds, ("XLONG", "XLONG_M", "lon", "longitude"))
        hgt_name = first_existing(ds, ("HGT", "HGT_M", "ter", "terrain"))
        lat_full = field_2d(ds[lat_name], y_dim, x_dim, time_dim)
        lon_full = field_2d(ds[lon_name], y_dim, x_dim, time_dim)
        hgt_full = field_2d(ds[hgt_name], y_dim, x_dim, time_dim)
        finite_geo = np.isfinite(lat_full) & np.isfinite(lon_full)
        if not np.any(finite_geo):
            raise ValueError("WRF latitude/longitude fields contain no finite cells")
        coverage = {
            "south": float(np.nanmin(lat_full)), "north": float(np.nanmax(lat_full)),
            "west": float(np.nanmin(lon_full)), "east": float(np.nanmax(lon_full)),
        }
        tolerance = 0.02
        if (coverage["south"] > DOMAIN["south"] + tolerance
                or coverage["north"] < DOMAIN["north"] - tolerance
                or coverage["west"] > DOMAIN["west"] + tolerance
                or coverage["east"] < DOMAIN["east"] - tolerance):
            raise ValueError(
                f"WRF domain does not fully cover 27–29°N, 86–89°E; "
                f"WRF centre-coordinate extent is {coverage}"
            )
        box_full = (
            finite_geo
            & (lat_full >= DOMAIN["south"]) & (lat_full <= DOMAIN["north"])
            & (lon_full >= DOMAIN["west"]) & (lon_full <= DOMAIN["east"])
        )
        yy, xx = np.where(box_full)
        if len(yy) == 0:
            raise ValueError("No WRF mass-grid centres fall inside the requested box")
        y_slice = slice(int(yy.min()), int(yy.max()) + 1)
        x_slice = slice(int(xx.min()), int(xx.max()) + 1)
        lat = lat_full[y_slice, x_slice]
        lon = lon_full[y_slice, x_slice]
        hgt = hgt_full[y_slice, x_slice]
        box = box_full[y_slice, x_slice]

        dx = float(ds.attrs.get("DX", np.nan))
        dy = float(ds.attrs.get("DY", np.nan))
        if "MAPFAC_M" in ds.variables and np.isfinite(dx) and np.isfinite(dy):
            mapfac = field_2d(ds["MAPFAC_M"], y_dim, x_dim, time_dim)[y_slice, x_slice]
            if not np.all(np.isfinite(mapfac[box]) & (mapfac[box] > 0)):
                raise ValueError("MAPFAC_M contains invalid values inside the box")
            area_km2 = (dx * dy / mapfac ** 2) / 1_000_000.0
            area_method = "DX*DY/MAPFAC_M^2"
        else:
            warnings.warn("MAPFAC_M or valid DX/DY is missing; using equal cell weights")
            area_km2 = np.ones_like(hgt, dtype="float64")
            area_method = "equal_cell_weights"
        area_km2 = np.where(box, area_km2, np.nan)

        component_names = ["RAINC", "RAINNC"]
        rainsh_status = "absent"
        if "RAINSH" in ds.variables and args.rainsh != "exclude":
            test = ds["RAINSH"].isel(
                {time_dim: time_indices, y_dim: y_slice, x_dim: x_slice}
            )
            nonzero = bool(np.nanmax(np.abs(np.asarray(test.values, dtype=float))) > 1e-10)
            if args.rainsh == "include" or nonzero:
                component_names.append("RAINSH")
                rainsh_status = "included"
            else:
                rainsh_status = "present_but_zero_excluded"
        elif "RAINSH" in ds.variables:
            rainsh_status = "excluded_by_user"

        bucket = bucket_value(ds, args.bucket_mm)
        component_totals: dict[str, np.ndarray] = {}
        component_intervals: dict[str, np.ndarray] = {}
        reset_counts: dict[str, np.ndarray] = {}
        tiny_negative_counts: dict[str, np.ndarray] = {}
        for name in component_names:
            accumulated = accumulated_component(
                ds, name, time_dim, time_indices, y_dim, x_dim,
                y_slice, x_slice, bucket,
            )
            increments, resets, tiny = interval_increments(accumulated)
            component_intervals[name] = increments
            component_totals[name] = increments.sum(axis=0)
            reset_counts[name] = resets
            tiny_negative_counts[name] = tiny

        daily_masks: dict[str, np.ndarray] = {}
        for label, day_start, day_end in zip(
            day_labels, day_starts_bjt, day_ends_bjt
        ):
            mask = np.asarray(
                (interval_starts_bjt >= day_start) & (interval_ends_bjt <= day_end),
                dtype=bool,
            )
            if not np.any(mask):
                raise ValueError(f"No WRF precipitation intervals found for {label} BJT")
            integrated_hours = float(np.sum(gaps_hours[mask]))
            if abs(integrated_hours - 24.0) > 1e-6:
                raise ValueError(
                    f"{label} contains {integrated_hours:.6f} integrated hours, not 24 h"
                )
            if interval_starts_bjt[mask][0] != day_start or interval_ends_bjt[mask][-1] != day_end:
                raise ValueError(f"{label} does not have exact 00:00 BJT day boundaries")
            daily_masks[label] = mask

        daily_component_totals: dict[str, dict[str, np.ndarray]] = {}
        daily_precip: dict[str, np.ndarray] = {}
        for label in day_labels:
            daily_component_totals[label] = {
                name: component_intervals[name][daily_masks[label]].sum(axis=0)
                for name in component_names
            }
            total = np.sum(
                np.stack(list(daily_component_totals[label].values())), axis=0
            )
            daily_precip[label] = np.where(box, total, np.nan)

        event_precip = np.sum(
            np.stack([daily_precip[label] for label in day_labels]), axis=0
        )
        event_precip = np.where(box, event_precip, np.nan)
        if np.nanmin(event_precip[box]) < -1e-8:
            raise ValueError("Final event precipitation contains negative values")
        for name in component_names:
            reconstructed = np.sum(
                np.stack(
                    [daily_component_totals[label][name] for label in day_labels]
                ),
                axis=0,
            )
            if not np.allclose(reconstructed, component_totals[name], atol=1e-8, rtol=0.0):
                raise AssertionError(f"Daily {name} totals do not reproduce the 48-h total")

        band_rows = []
        for label, day_start, day_end in zip(
            day_labels, day_starts_bjt, day_ends_bjt
        ):
            for band in BANDS:
                band_mask = (
                    box
                    & np.isfinite(hgt)
                    & (hgt >= band.lower_m)
                    & (hgt < band.upper_m)
                )
                stats = weighted_stats(
                    daily_precip[label][band_mask], area_km2[band_mask]
                )
                row = {
                    "day_label": label,
                    "day_start_bjt": str(day_start),
                    "day_end_bjt": str(day_end),
                    "band_key": band.key,
                    "elevation_lower_m_inclusive": band.lower_m,
                    "elevation_upper_m_exclusive": band.upper_m,
                    "wrf_cell_count": int(band_mask.sum()),
                    "represented_area_km2": float(np.nansum(area_km2[band_mask])),
                    "native_hgt_min_m": (
                        float(np.nanmin(hgt[band_mask])) if np.any(band_mask) else np.nan
                    ),
                    "native_hgt_max_m": (
                        float(np.nanmax(hgt[band_mask])) if np.any(band_mask) else np.nan
                    ),
                    **stats,
                }
                for name in component_names:
                    row[f"{name}_area_weighted_mean_mm"] = weighted_stats(
                        daily_component_totals[label][name][band_mask],
                        area_km2[band_mask],
                    )["area_weighted_mean_mm"]
                band_rows.append(row)
        summary = pd.DataFrame(band_rows)
        event_by_band = (
            summary.groupby("band_key", as_index=False).area_weighted_mean_mm.sum()
            .rename(columns={"area_weighted_mean_mm": "total_48h_area_weighted_mean_mm"})
        )
        summary = summary.merge(
            event_by_band, on="band_key", how="left", validate="many_to_one"
        )

        grid = pd.DataFrame(
            {
                "south_north_index": np.indices(hgt.shape)[0][box] + y_slice.start,
                "west_east_index": np.indices(hgt.shape)[1][box] + x_slice.start,
                "latitude": lat[box],
                "longitude": lon[box],
                "HGT_m": hgt[box],
                "cell_area_km2": area_km2[box],
                "event_48h_precipitation_mm": event_precip[box],
            }
        )
        grid["elevation_band"] = "outside_selected_bands"
        for band in BANDS:
            selector = (grid.HGT_m >= band.lower_m) & (grid.HGT_m < band.upper_m)
            grid.loc[selector, "elevation_band"] = band.display_name
        for label in day_labels:
            column_label = label.replace(" ", "_")
            grid[f"{column_label}_precipitation_mm"] = daily_precip[label][box]
        for name, total in component_totals.items():
            grid[f"{name}_event_48h_mm"] = total[box]

        interval_rows = []
        total_intervals = np.sum(np.stack(list(component_intervals.values())), axis=0)
        for k in range(len(selected_times) - 1):
            matching_days = [
                label for label in day_labels if bool(daily_masks[label][k])
            ]
            if len(matching_days) != 1:
                raise AssertionError(
                    f"WRF interval {k} belongs to {len(matching_days)} BJT days"
                )
            row = {
                "interval_start_wrf_time": str(selected_times[k]),
                "interval_end_wrf_time": str(selected_times[k + 1]),
                "interval_start_bjt": str(interval_starts_bjt[k]),
                "interval_end_bjt": str(interval_ends_bjt[k]),
                "bjt_day_label": matching_days[0],
                "interval_hours": float(gaps_hours[k]),
                "box_area_weighted_precipitation_mm": weighted_stats(
                    total_intervals[k][box], area_km2[box]
                )["area_weighted_mean_mm"],
            }
            for name in component_names:
                row[f"{name}_reset_cell_count"] = int(reset_counts[name][k])
                row[f"{name}_tiny_negative_cell_count"] = int(tiny_negative_counts[name][k])
            interval_rows.append(row)
        time_audit = pd.DataFrame(interval_rows)

        metadata = {
            "script_version": VERSION,
            "regional_wrf_domain": "d02",
            "regional_wrf_file": str(regional_wrf_path),
            "regional_wrf_title": str(ds.attrs.get("TITLE", "")),
            "regional_wrf_start_date_attribute": str(ds.attrs.get("START_DATE", "")),
            "wrf_time_zone_assumption": args.wrf_time_zone,
            "event_bjt_half_open": [str(start_bjt), str(end_bjt)],
            "event_wrf_time_half_open": [str(start_wrf), str(end_wrf)],
            "event_duration_hours": float((end_bjt - start_bjt).total_seconds() / 3600.0),
            "regional_event_output_intervals": int(len(selected_times) - 1),
            "regional_minimum_interval_hours": float(np.min(gaps_hours)),
            "regional_maximum_interval_hours": float(np.max(gaps_hours)),
            "domain_center_selection": DOMAIN,
            "regional_wrf_coordinate_extent": coverage,
            "regional_wrf_cells_in_box": int(box.sum()),
            "daily_groups_bjt": [
                {"label": label, "start": str(day_start), "end": str(day_end)}
                for label, day_start, day_end in zip(
                    day_labels, day_starts_bjt, day_ends_bjt
                )
            ],
            "regional_terrain_variable": hgt_name,
            "regional_terrain_definition": "native d02 WRF HGT; entire WRF cell assigned by HGT",
            "band_bounds_rule": "lower <= HGT < upper",
            "regional_cell_area_weight_method": area_method,
            "regional_components_included": component_names,
            "regional_rainsh_status": rainsh_status,
            "regional_bucket_mm_used": bucket,
            "reset_rule": "diff < -0.01 mm reconstructed as current accumulation; [-0.01,0) set to 0",
            "regional_total_reset_cells_by_component": {
                name: int(np.sum(reset_counts[name])) for name in component_names
            },
            "figure_panels": {
                "left": "Three stations per day; OBS and D04 full-height transparent bars overlap at identical x positions",
                "right": "Three separate d02 area-weighted native-HGT elevation-band bars per day",
            },
            "important_scope_note": (
                "27–29 N, 86–89 E is a rectangle, not a geomorphological north-slope mask. "
                "Boundary WRF cells are included by centre coordinate and are not fractionally clipped."
            ),
        }

    station_cells, station_summary, station_metadata = extract_station_daily_from_d04(
        station_wrf_path, args, start_bjt, end_bjt,
        day_starts_bjt, day_ends_bjt, day_labels,
    )
    obs_summary, obs_metadata = extract_obs_daily(
        obs_path, day_starts_bjt, day_labels, args.obs_hour_label
    )
    metadata["observation_source"] = obs_metadata
    metadata["station_source"] = station_metadata
    metadata["station_grid_selection"] = (
        "nearest d04 WRF mass-grid centre by great-circle distance"
    )
    metadata["station_grid_audit"] = station_cells.drop(
        columns=["local_y_index", "local_x_index"]
    ).to_dict(orient="records")
    metadata["station_grid_cells_are_unique"] = bool(
        ~station_cells.shares_grid_cell_with_another_station.any()
    )

    # Exactly the values displayed in the PNG: two BJT days, six station
    # source columns and three D02 regional elevation-band columns.
    source_data = pd.DataFrame({"time": [day.strftime("%Y-%m-%d") for day in day_starts_bjt]})
    for station in STATIONS:
        for prefix, table, field, key_column in (
            ("OBS", obs_summary, "precipitation_mm", "station"),
            ("WRF_D04", station_summary, "precipitation_mm", "station"),
        ):
            lookup = table.set_index(["day_label", key_column])[field]
            source_data[f"{prefix}_{station.key}_mm"] = [
                float(lookup.loc[(label, station.key)]) for label in day_labels
            ]
    regional_lookup = summary.set_index(["day_label", "band_key"])["area_weighted_mean_mm"]
    for band in BANDS:
        source_data[f"WRF_D02_{band.key[1:]}pm300m_mm"] = [
            float(regional_lookup.loc[(label, band.key)]) for label in day_labels
        ]
    source_data_path = output_dir / f"{OUTPUT_STEM}_source_data.csv"
    source_data.to_csv(source_data_path, index=False, float_format="%.4f", encoding="utf-8-sig")
    png = plot_bars(
        summary, station_summary, obs_summary, day_labels, output_dir, args.dpi, args.ymax
    )

    print("WRF d02-regional/d04-station precipitation extraction completed.")
    print(f"BJT event: [{start_bjt}, {end_bjt})")
    print(f"WRF time used: [{start_wrf}, {end_wrf}) ({args.wrf_time_zone})")
    print(f"d02 components: {' + '.join(component_names)}; RAINSH={rainsh_status}")
    print(
        "d04 components: "
        + " + ".join(station_metadata["components_included"])
        + f"; RAINSH={station_metadata['rainsh_status']}"
    )
    print(source_data.to_string(index=False))
    print("P5800 48-h OBS and WRF D04 totals (mm):",
          f"{source_data.OBS_P5800_mm.sum():.4f}",
          f"{source_data.WRF_D04_P5800_mm.sum():.4f}")
    for path in (png, source_data_path):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
