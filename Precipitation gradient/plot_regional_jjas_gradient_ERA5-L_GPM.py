#!/usr/bin/env python3
"""Regional JJAS monthly precipitation by DEM elevation band, 2010–2020.

All GRIB, IMERG NetCDF and DEM GeoTIFF reading and area weighting are
included in this one script. The study region is the 27–29 N, 86–89 E rectangle.
Each source remains on its own native precipitation grid. The figure uses two
monthly bar-chart panels, one per product, with a shared precipitation scale.
"""

from __future__ import annotations

import argparse
import json
import calendar
import os
import re
import struct
import warnings
import zlib
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import xarray as xr
import cfgrib

MONTHS = (6, 7, 8, 9)
MONTH_NAMES = ("June", "July", "August", "September")
YEARS = tuple(range(2010, 2021))
BAND_LIMITS = ((4900, 5500), (5500, 6100), (6200, 6800))
COLORS = {
    # Blue fills sampled from the reference image; the three GPM reds are
    # tonal variations of its red bar (the reference has only one red series).
    "ERA5-Land": ("#BAE3F5", "#3996C6", "#01688B"),
    "GPM IMERG": ("#D47D7D", "#C65151", "#B22424"),
}
BAR_EDGES = {
    "ERA5-Land": ("#9FC6D7", "#3085B0", "#005C7B"),
    "GPM IMERG": ("#B46666", "#AC3E3E", "#971D1D"),
}
STEM = "Regional_JJAS_2010_2020_ERA5Land_GPM_DEM_gradient_v12"


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--era5land-grib", type=Path, required=True)
    p.add_argument("--gpm-monthly-dir", type=Path, required=True)
    p.add_argument("--dem", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--ymax", type=float, default=400.0,
                   help="Minimum y-axis maximum in mm; axis grows if data exceed it")
    p.add_argument("--plot-only", action="store_true",
                   help="Replot the already written source_data.csv without rereading the DEM and products")
    a = p.parse_args()
    if not 100 <= a.dpi <= 1200:
        p.error("--dpi must be between 100 and 1200")
    if not np.isfinite(a.ymax) or a.ymax <= 0:
        p.error("--ymax must be positive and finite")
    if not a.plot_only:
        for name in ("era5land_grib", "dem"):
            path = getattr(a, name)
            if not path.is_file():
                p.error(f"Missing input file ({name}): {path}")
        if not a.gpm_monthly_dir.is_dir():
            p.error(f"Missing GPM directory: {a.gpm_monthly_dir}")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    return a


# ERA5-Land input, DEM and statistics
era_LAT_MIN, era_LAT_MAX = (27.0, 29.0)

era_LON_MIN, era_LON_MAX = (86.0, 89.0)

era_DOMAIN = {'south': era_LAT_MIN, 'north': era_LAT_MAX, 'west': era_LON_MIN, 'east': era_LON_MAX}

era_HISTORICAL_YEARS = tuple(range(2010, 2021))

era_MONTHS = (6, 7, 8, 9)

era_EARTH_RADIUS_M = 6371008.8

era_ELEVATION_BANDS = ({'label': 'P5200', 'display_name': '5200 ± 300 m', 'target': 5200.0, 'low': 4900.0, 'high': 5500.0, 'color': '#17325B'}, {'label': 'P5800', 'display_name': '5800 ± 300 m', 'target': 5800.0, 'low': 5500.0, 'high': 6100.0, 'color': '#E31A1C'}, {'label': 'P6500', 'display_name': '6500 ± 300 m', 'target': 6500.0, 'low': 6200.0, 'high': 6800.0, 'color': '#8EA6C0'})

era_TP_NAMES = ('tp', 'total_precipitation', 'precipitation')

era_LAT_NAMES = ('latitude', 'lat')

era_LON_NAMES = ('longitude', 'lon')

def era_find_name(names, candidates, kind: str) -> str:
    lookup = {str(name).lower(): str(name) for name in names}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise KeyError(f'Could not find {kind}; available names: {list(names)}')

def era_open_grib_groups(path: Path) -> list[xr.Dataset]:
    """Open all cfgrib hypercubes without writing persistent .idx files."""
    try:
        groups = cfgrib.open_datasets(str(path), backend_kwargs={'indexpath': ''})
    except Exception as exc:
        raise RuntimeError(f'Could not open GRIB with cfgrib: {path}\n{exc}') from exc
    if not groups:
        raise ValueError(f'No datasets found in {path}')
    return [era_standardize_spatial(ds) for ds in groups]

def era_standardize_spatial(ds: xr.Dataset) -> xr.Dataset:
    lat = era_find_name(ds.coords, era_LAT_NAMES, 'latitude coordinate')
    lon = era_find_name(ds.coords, era_LON_NAMES, 'longitude coordinate')
    rename = {}
    if lat != 'latitude':
        rename[lat] = 'latitude'
    if lon != 'longitude':
        rename[lon] = 'longitude'
    ds = ds.rename(rename)
    if float(ds.longitude.max()) > 180.0:
        ds = ds.assign_coords(longitude=(ds.longitude + 180.0) % 360.0 - 180.0)
    return ds.sortby('latitude').sortby('longitude')

def era_crop_box(da: xr.DataArray) -> xr.DataArray:
    out = da.sel(latitude=slice(era_LAT_MIN, era_LAT_MAX), longitude=slice(era_LON_MIN, era_LON_MAX))
    if out.sizes.get('latitude', 0) == 0 or out.sizes.get('longitude', 0) == 0:
        raise ValueError(f'Input does not cover {era_LAT_MIN}-{era_LAT_MAX} N, {era_LON_MIN}-{era_LON_MAX} E')
    return out

def era_collect_variable_series(groups: list[xr.Dataset], candidates: tuple[str, ...], kind: str) -> tuple[xr.DataArray, list[dict]]:
    """Collect and concatenate a variable split across cfgrib hypercubes.

    CDS GRIB files can place different years/experiments in separate cfgrib
    groups even when they belong to one request.  Selecting only the largest
    group can therefore silently retain one year and discard another.
    """
    series: list[xr.DataArray] = []
    inventory: list[dict] = []
    for ds in groups:
        lower = {str(v).lower(): str(v) for v in ds.data_vars}
        for candidate in candidates:
            if candidate in lower:
                name = lower[candidate]
                item = era_as_valid_time_series(ds[name])
                item_times = pd.DatetimeIndex(item.time.values)
                series.append(item)
                inventory.append({'variable': name, 'n_records': int(item.sizes['time']), 'time_start': str(item_times[0]), 'time_end': str(item_times[-1]), 'units': str(item.attrs.get('units', ''))})
                break
    if not series:
        group_variables = [list(ds.data_vars) for ds in groups]
        raise KeyError(f'No {kind} variable found. GRIB variables by group: {group_variables}')
    if len(series) == 1:
        combined = series[0]
    else:
        try:
            combined = xr.concat(series, dim='time', join='exact', coords='minimal', compat='override').sortby('time')
        except Exception as exc:
            raise ValueError(f'Found {len(series)} {kind} GRIB groups but their spatial grids could not be combined: {exc}') from exc
    times = pd.DatetimeIndex(combined.time.values)
    if times.duplicated().any():
        duplicate_times = times[times.duplicated(keep=False)].unique()
        warnings.warn(f'Combined {kind} groups contain duplicate valid times; retaining the first record for: {[str(t) for t in duplicate_times]}')
        combined = combined.isel(time=np.flatnonzero(~times.duplicated(keep='first')))
    return (combined, inventory)

def era_collapse_member_dims(da: xr.DataArray) -> xr.DataArray:
    """Remove ensemble/version dimensions while retaining time/step and space."""
    protected = {'time', 'valid_time', 'step', 'latitude', 'longitude'}
    for dim in list(da.dims):
        if dim in protected:
            continue
        if dim.lower() in {'expver', 'number'}:
            da = da.max(dim=dim, skipna=True)
        elif da.sizes[dim] == 1:
            da = da.isel({dim: 0}, drop=True)
        else:
            raise ValueError(f'Unsupported dimension {dim!r} (size {da.sizes[dim]}) in {da.name!r}')
    return da

def era_as_valid_time_series(da: xr.DataArray) -> xr.DataArray:
    """Return data as (time, latitude, longitude), using forecast valid time."""
    da = era_collapse_member_dims(era_crop_box(da))
    spatial = {'latitude', 'longitude'}
    record_dims = [dim for dim in da.dims if dim not in spatial]
    if not record_dims:
        raise ValueError(f'{da.name!r} has no time dimension')
    if 'valid_time' in da.coords:
        time_coord = da.coords['valid_time']
    elif 'time' in da.coords and 'step' in da.coords:
        time_coord = da.coords['time'] + da.coords['step']
    elif 'time' in da.coords:
        time_coord = da.coords['time']
    else:
        datetime_coords = [coord for coord in da.coords.values() if np.issubdtype(coord.dtype, np.datetime64)]
        if len(datetime_coords) != 1:
            raise ValueError(f'Could not identify valid time for {da.name!r}; coords={list(da.coords)}')
        time_coord = datetime_coords[0]
    record_template = da.isel(latitude=0, longitude=0, drop=True)
    time_broadcast, _ = xr.broadcast(time_coord, record_template)
    stacked = da.stack(record=record_dims).transpose('record', 'latitude', 'longitude')
    stacked_time = time_broadcast.stack(record=record_dims).values
    out = xr.DataArray(stacked.data, dims=('time', 'latitude', 'longitude'), coords={'time': pd.to_datetime(stacked_time), 'latitude': da.latitude.values, 'longitude': da.longitude.values}, name=da.name, attrs=da.attrs).sortby('time')
    times = pd.DatetimeIndex(out.time.values)
    if times.isna().any():
        raise ValueError(f'NaT values found in valid time for {da.name!r}')
    if times.duplicated().any():
        duplicate_count = int(times.duplicated().sum())
        warnings.warn(f'{da.name!r}: dropping {duplicate_count} duplicate valid-time records')
        keep = ~times.duplicated(keep='first')
        out = out.isel(time=np.flatnonzero(keep))
    return out

def era_precipitation_factor_to_mm(da: xr.DataArray, path: Path) -> tuple[float, str]:
    units = str(da.attrs.get('units', '')).strip().lower()
    if units == 'm' or 'metre' in units or 'meter' in units or ('m of water' in units):
        return (1000.0, f'{path.name}: {units!r} -> multiplied by 1000')
    if 'mm' in units or 'kg m**-2' in units or 'kg m-2' in units:
        return (1.0, f'{path.name}: {units!r} -> retained as mm')
    raise ValueError(f'Cannot safely interpret precipitation units {units!r} in {path}. Expected metres for ERA5-Land total precipitation.')

def era_load_monthly_tp(groups: list[xr.Dataset], path: Path, selected_years: tuple[int, ...]) -> tuple[xr.DataArray, dict]:
    raw, group_inventory = era_collect_variable_series(groups, era_TP_NAMES, 'monthly total precipitation')
    factor, unit_note = era_precipitation_factor_to_mm(raw, path)
    times = pd.DatetimeIndex(raw.time.values)
    select = times.year.astype(int).isin(selected_years) & times.month.astype(int).isin(era_MONTHS)
    raw = raw.isel(time=np.flatnonzero(select))
    times = pd.DatetimeIndex(raw.time.values)
    expected = {(year, month) for year in selected_years for month in era_MONTHS}
    found = list(zip(times.year.astype(int), times.month.astype(int)))
    if set(found) != expected or len(found) != len(expected):
        missing = sorted(expected - set(found))
        duplicate_count = len(found) - len(set(found))
        raise ValueError(f'Monthly GRIB must contain exactly one record for every requested JJAS month in {selected_years[0]}-{selected_years[-1]}. Missing={missing}; duplicate_count={duplicate_count}.\nFound year-month records after merging all tp groups: {found}\ncfgrib tp-group inventory: {json.dumps(group_inventory, ensure_ascii=False)}')
    days = xr.DataArray([calendar.monthrange(t.year, t.month)[1] for t in times], dims='time', coords={'time': raw.time})
    monthly_mm = (raw.astype('float64') * factor * days).load()
    monthly_mm.name = 'monthly_total_precipitation'
    monthly_mm.attrs['units'] = 'mm month-1'
    return (monthly_mm, {'path': str(path), 'variable': 'tp', 'original_units': str(group_inventory[0].get('units', '')), 'conversion': unit_note + '; multiplied by calendar days', 'selected_years': list(selected_years), 'year_month_records': [f'{y:04d}-{m:02d}' for y, m in found], 'cfgrib_tp_groups': group_inventory})

def era_monthly_field_dictionary(monthly: xr.DataArray, selected_years: tuple[int, ...]) -> dict[tuple[int, int], xr.DataArray]:
    times = pd.DatetimeIndex(monthly.time.values)
    fields: dict[tuple[int, int], xr.DataArray] = {}
    for year in selected_years:
        for month in era_MONTHS:
            index = np.flatnonzero((times.year == year) & (times.month == month))
            if len(index) != 1:
                raise ValueError(f'Expected one monthly field for {year}-{month:02d}; found {len(index)}')
            fields[year, month] = monthly.isel(time=int(index[0]), drop=True).transpose('latitude', 'longitude').load()
    return fields

def era_coordinate_edges(centers: np.ndarray, name: str) -> np.ndarray:
    centers = np.asarray(centers, dtype='float64')
    if centers.ndim != 1 or centers.size < 2 or (not np.all(np.diff(centers) > 0)):
        raise ValueError(f'{name} centers must be a strictly increasing 1-D array')
    differences = np.diff(centers)
    spacing = float(np.median(differences))
    if not np.allclose(differences, spacing, rtol=0.0, atol=1e-05):
        raise ValueError(f'{name} grid is not regularly spaced: {differences}')
    edges = np.empty(centers.size + 1, dtype='float64')
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - spacing / 2.0
    edges[-1] = centers[-1] + spacing / 2.0
    return edges

def era_spherical_cell_areas_km2(latitude_edges: np.ndarray, longitude_edges: np.ndarray) -> np.ndarray:
    latitude_term = np.abs(np.sin(np.deg2rad(latitude_edges[1:])) - np.sin(np.deg2rad(latitude_edges[:-1])))
    longitude_term = np.abs(np.diff(np.deg2rad(longitude_edges)))
    return era_EARTH_RADIUS_M ** 2 * latitude_term[:, None] * longitude_term[None, :] / 1000000.0

def era_parse_nodata_tag(value) -> float:
    if value is None:
        return -9999.0
    if isinstance(value, (tuple, list)):
        value = value[0]
    if isinstance(value, bytes):
        value = value.decode('ascii', errors='ignore')
    return float(str(value).strip().split()[0])

def era_georeference_from_tiff_tags(pixel_scale, tiepoint) -> tuple[float, float, float, float]:
    if pixel_scale is None or tiepoint is None:
        raise ValueError("GeoTIFF lacks ModelPixelScaleTag/ModelTiepointTag; export it from GEE with crs='EPSG:4326'.")
    pixel_scale = tuple((float(value) for value in pixel_scale))
    tiepoint = tuple((float(value) for value in tiepoint))
    if len(pixel_scale) < 2 or len(tiepoint) < 6:
        raise ValueError('Invalid GeoTIFF georeferencing tags')
    x_size = abs(pixel_scale[0])
    y_size = abs(pixel_scale[1])
    raster_i, raster_j = (tiepoint[0], tiepoint[1])
    map_x, map_y = (tiepoint[3], tiepoint[4])
    west = map_x - raster_i * x_size
    north = map_y + raster_j * y_size
    return (west, north, x_size, y_size)

def era_normalize_dem_block(data, expected_rows: int, expected_columns: int) -> np.ndarray:
    array = np.asarray(data, dtype='float64')
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = array[:, :, 0]
    if array.shape != (expected_rows, expected_columns):
        raise ValueError(f'DEM must contain exactly one band; expected block shape {(expected_rows, expected_columns)}, received {array.shape}')
    return array

class era_DemReader:

    def __init__(self, backend, width, height, west, north, x_size, y_size, nodata, read_rows, close):
        self.backend = str(backend)
        self.width = int(width)
        self.height = int(height)
        self.west = float(west)
        self.north = float(north)
        self.x_size = float(x_size)
        self.y_size = float(y_size)
        self.nodata = float(nodata) if nodata is not None else -9999.0
        self.read_rows = read_rows
        self.close = close

def era_tiff_lzw_decode(payload: bytes) -> bytes:
    """Decode TIFF-flavour LZW (MSB-first codes, EarlyChange=1)."""
    clear_code = 256
    end_code = 257
    bit_position = 0
    code_width = 9
    next_code = 258
    table = {code: bytes((code,)) for code in range(256)}
    previous = None
    output = bytearray()

    def read_code(width: int):
        nonlocal bit_position
        if bit_position + width > len(payload) * 8:
            return None
        value = 0
        for _ in range(width):
            byte_index, bit_index = divmod(bit_position, 8)
            value = value << 1 | payload[byte_index] >> 7 - bit_index & 1
            bit_position += 1
        return value
    while True:
        code = read_code(code_width)
        if code is None or code == end_code:
            break
        if code == clear_code:
            table = {item: bytes((item,)) for item in range(256)}
            code_width = 9
            next_code = 258
            previous = None
            continue
        if code in table:
            entry = table[code]
        elif code == next_code and previous is not None:
            entry = previous + previous[:1]
        else:
            raise ValueError(f'Invalid TIFF LZW code {code} at bit {bit_position}')
        output.extend(entry)
        if previous is not None and next_code < 4096:
            table[next_code] = previous + entry[:1]
            next_code += 1
            if next_code == (1 << code_width) - 1 and code_width < 12:
                code_width += 1
        previous = entry
    return bytes(output)

def era_tiff_packbits_decode(payload: bytes) -> bytes:
    output = bytearray()
    position = 0
    while position < len(payload):
        control = payload[position]
        position += 1
        signed_control = control if control < 128 else control - 256
        if 0 <= signed_control <= 127:
            count = signed_control + 1
            output.extend(payload[position:position + count])
            position += count
        elif -127 <= signed_control <= -1:
            if position >= len(payload):
                raise ValueError('Truncated TIFF PackBits run')
            output.extend(payload[position:position + 1] * (1 - signed_control))
            position += 1
    return bytes(output)

class era_StandardLibraryTiff:
    """Minimal one-band GeoTIFF reader used when geospatial packages are absent."""
    TYPE_FORMATS = {1: 'B', 3: 'H', 4: 'I', 6: 'b', 8: 'h', 9: 'i', 11: 'f', 12: 'd'}
    TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 6: 1, 8: 2, 9: 4, 11: 4, 12: 8}

    def __init__(self, path: Path):
        self.path = Path(path)
        self.handle = self.path.open('rb')
        marker = self.handle.read(2)
        if marker == b'II':
            self.endian = '<'
        elif marker == b'MM':
            self.endian = '>'
        else:
            self.close()
            raise ValueError('Not a TIFF file: invalid byte-order marker')
        magic = self._unpack('H', self.handle.read(2))[0]
        if magic != 42:
            self.close()
            if magic == 43:
                raise ValueError('BigTIFF is unsupported by the standard-library fallback')
            raise ValueError(f'Invalid classic-TIFF magic number: {magic}')
        ifd_offset = self._unpack('I', self.handle.read(4))[0]
        self.tags = self._read_ifd(ifd_offset)
        self.width = int(self._scalar(256))
        self.height = int(self._scalar(257))
        self.bits = int(self._scalar(258))
        self.compression = int(self._scalar(259, 1))
        self.samples_per_pixel = int(self._scalar(277, 1))
        self.predictor = int(self._scalar(317, 1))
        self.sample_format = int(self._scalar(339, 1))
        self.planar_configuration = int(self._scalar(284, 1))
        if self.samples_per_pixel != 1 or self.planar_configuration != 1:
            self.close()
            raise ValueError('Standard-library TIFF fallback requires one chunky band')
        if self.predictor not in (1, 2):
            self.close()
            raise ValueError(f'Unsupported TIFF predictor: {self.predictor}')
        dtype_codes = {(1, 8): 'u1', (1, 16): 'u2', (1, 32): 'u4', (1, 64): 'u8', (2, 8): 'i1', (2, 16): 'i2', (2, 32): 'i4', (2, 64): 'i8', (3, 32): 'f4', (3, 64): 'f8'}
        dtype_code = dtype_codes.get((self.sample_format, self.bits))
        if dtype_code is None:
            self.close()
            raise ValueError(f'Unsupported TIFF SampleFormat/BitsPerSample: {self.sample_format}/{self.bits}')
        self.dtype = np.dtype(self.endian + dtype_code)
        self.bytes_per_value = self.dtype.itemsize
        self.tile_width = self._optional_scalar(322)
        self.tile_height = self._optional_scalar(323)
        if self.tile_width is not None or self.tile_height is not None:
            if self.tile_width is None or self.tile_height is None:
                self.close()
                raise ValueError('Incomplete tiled-TIFF metadata')
            self.tile_width = int(self.tile_width)
            self.tile_height = int(self.tile_height)
            self.segment_offsets = self._as_tuple(self.tags.get(324))
            self.segment_byte_counts = self._as_tuple(self.tags.get(325))
            self.is_tiled = True
        else:
            self.rows_per_strip = int(self._scalar(278, self.height))
            self.segment_offsets = self._as_tuple(self.tags.get(273))
            self.segment_byte_counts = self._as_tuple(self.tags.get(279))
            self.is_tiled = False
        if not self.segment_offsets or not self.segment_byte_counts:
            self.close()
            raise ValueError('TIFF lacks strip/tile offsets or byte counts')
        if len(self.segment_offsets) != len(self.segment_byte_counts):
            self.close()
            raise ValueError('TIFF strip/tile offset and byte-count lengths differ')

    def _unpack(self, fmt: str, payload: bytes):
        return struct.unpack(self.endian + fmt, payload)

    def _read_ifd(self, offset: int) -> dict:
        self.handle.seek(offset)
        entry_count = self._unpack('H', self.handle.read(2))[0]
        tags = {}
        for _ in range(entry_count):
            entry = self.handle.read(12)
            if len(entry) != 12:
                raise ValueError('Truncated TIFF IFD')
            tag, value_type, count = self._unpack('HHI', entry[:8])
            size = self.TYPE_SIZES.get(value_type)
            if size is None:
                continue
            byte_count = size * count
            if byte_count <= 4:
                raw = entry[8:8 + byte_count]
            else:
                value_offset = self._unpack('I', entry[8:12])[0]
                current = self.handle.tell()
                self.handle.seek(value_offset)
                raw = self.handle.read(byte_count)
                self.handle.seek(current)
            if len(raw) != byte_count:
                raise ValueError(f'Truncated TIFF tag {tag}')
            if value_type == 2:
                value = raw.rstrip(b'\x00').decode('ascii', errors='ignore')
            else:
                fmt = self.TYPE_FORMATS[value_type]
                value = self._unpack(str(count) + fmt, raw)
                if count == 1:
                    value = value[0]
            tags[tag] = value
        return tags

    @staticmethod
    def _as_tuple(value) -> tuple:
        if value is None:
            return ()
        if isinstance(value, tuple):
            return value
        return (value,)

    def _optional_scalar(self, tag):
        value = self.tags.get(tag)
        if isinstance(value, tuple):
            return value[0]
        return value

    def _scalar(self, tag, default=None):
        value = self._optional_scalar(tag)
        if value is None:
            if default is not None:
                return default
            raise ValueError(f'TIFF required tag {tag} is missing')
        return value

    def _decode_segment(self, index: int, rows: int, columns: int) -> np.ndarray:
        self.handle.seek(int(self.segment_offsets[index]))
        compressed = self.handle.read(int(self.segment_byte_counts[index]))
        if self.compression == 1:
            decoded = compressed
        elif self.compression in (8, 32946):
            decoded = zlib.decompress(compressed)
        elif self.compression == 5:
            decoded = era_tiff_lzw_decode(compressed)
        elif self.compression == 32773:
            decoded = era_tiff_packbits_decode(compressed)
        else:
            raise ValueError(f'Unsupported TIFF compression code: {self.compression}')
        expected_values = rows * columns
        expected_bytes = expected_values * self.bytes_per_value
        if len(decoded) < expected_bytes:
            raise ValueError(f'Decompressed TIFF segment {index} is too short: {len(decoded)} < {expected_bytes} bytes')
        array = np.frombuffer(decoded[:expected_bytes], dtype=self.dtype, count=expected_values).reshape(rows, columns).copy()
        if self.predictor == 2:
            array = np.cumsum(array.astype('int64'), axis=1).astype(self.dtype)
        return array

    def read_rows(self, row_start: int, row_end: int) -> np.ndarray:
        if not 0 <= row_start < row_end <= self.height:
            raise ValueError(f'Invalid TIFF row window: {row_start}:{row_end}')
        result = np.empty((row_end - row_start, self.width), dtype=self.dtype)
        if self.is_tiled:
            tiles_across = (self.width + self.tile_width - 1) // self.tile_width
            first_tile_row = row_start // self.tile_height
            last_tile_row = (row_end - 1) // self.tile_height
            for tile_row in range(first_tile_row, last_tile_row + 1):
                tile_north = tile_row * self.tile_height
                source_row_start = max(row_start, tile_north) - tile_north
                source_row_end = min(row_end, tile_north + self.tile_height) - tile_north
                target_row_start = max(row_start, tile_north) - row_start
                target_row_end = min(row_end, tile_north + self.tile_height) - row_start
                for tile_column in range(tiles_across):
                    index = tile_row * tiles_across + tile_column
                    tile = self._decode_segment(index, self.tile_height, self.tile_width)
                    column_start = tile_column * self.tile_width
                    column_end = min(column_start + self.tile_width, self.width)
                    result[target_row_start:target_row_end, column_start:column_end] = tile[source_row_start:source_row_end, :column_end - column_start]
        else:
            first_strip = row_start // self.rows_per_strip
            last_strip = (row_end - 1) // self.rows_per_strip
            for strip_index in range(first_strip, last_strip + 1):
                strip_north = strip_index * self.rows_per_strip
                strip_rows = min(self.rows_per_strip, self.height - strip_north)
                strip = self._decode_segment(strip_index, strip_rows, self.width)
                source_start = max(row_start, strip_north) - strip_north
                source_end = min(row_end, strip_north + strip_rows) - strip_north
                target_start = max(row_start, strip_north) - row_start
                target_end = min(row_end, strip_north + strip_rows) - row_start
                result[target_start:target_end] = strip[source_start:source_end]
        return result

    def close(self):
        if not self.handle.closed:
            self.handle.close()

def era_open_dem_reader(dem_path: Path) -> era_DemReader:
    """Open a GeoTIFF with the first available existing geospatial backend."""
    errors: list[str] = []
    rasterio_source = None
    try:
        import rasterio
        from rasterio.windows import Window
        rasterio_source = rasterio.open(dem_path)
        if rasterio_source.count != 1:
            raise ValueError(f'DEM has {rasterio_source.count} bands; expected one')
        transform = rasterio_source.transform
        if abs(transform.b) > 1e-12 or abs(transform.d) > 1e-12:
            raise ValueError('Rotated GeoTIFF transforms are unsupported')
        west = transform.c
        north = transform.f
        x_size = float(transform.a)
        y_size = abs(float(transform.e))
        if x_size <= 0.0 or y_size <= 0.0:
            raise ValueError(f'Invalid GeoTIFF pixel size: {x_size}, {y_size}')
        width = rasterio_source.width
        height = rasterio_source.height

        def rasterio_read(row_start, row_end):
            return rasterio_source.read(1, window=Window(0, row_start, width, row_end - row_start), masked=False)
        return era_DemReader('rasterio', width, height, west, north, x_size, y_size, rasterio_source.nodata, rasterio_read, rasterio_source.close)
    except Exception as exc:
        if rasterio_source is not None:
            rasterio_source.close()
        errors.append(f'rasterio: {type(exc).__name__}: {exc}')
    try:
        from osgeo import gdal
        gdal_source = gdal.Open(str(dem_path), gdal.GA_ReadOnly)
        if gdal_source is None:
            raise OSError('gdal.Open returned None')
        if gdal_source.RasterCount != 1:
            raise ValueError(f'DEM has {gdal_source.RasterCount} bands; expected one')
        transform = gdal_source.GetGeoTransform()
        if abs(transform[2]) > 1e-12 or abs(transform[4]) > 1e-12:
            raise ValueError('Rotated GeoTIFF transforms are unsupported')
        west = transform[0]
        north = transform[3]
        x_size = float(transform[1])
        y_size = abs(float(transform[5]))
        if x_size <= 0.0 or y_size <= 0.0:
            raise ValueError(f'Invalid GeoTIFF pixel size: {x_size}, {y_size}')
        width = gdal_source.RasterXSize
        height = gdal_source.RasterYSize
        gdal_band = gdal_source.GetRasterBand(1)

        def gdal_read(row_start, row_end):
            return gdal_band.ReadAsArray(0, row_start, width, row_end - row_start)
        return era_DemReader('GDAL', width, height, west, north, x_size, y_size, gdal_band.GetNoDataValue(), gdal_read, lambda: None)
    except Exception as exc:
        errors.append(f'GDAL: {type(exc).__name__}: {exc}')
    try:
        import tifffile
        with tifffile.TiffFile(dem_path) as tif:
            page = tif.pages[0]
            tags = page.tags
            pixel_scale = tags[33550].value if 33550 in tags else None
            tiepoint = tags[33922].value if 33922 in tags else None
            west, north, x_size, y_size = era_georeference_from_tiff_tags(pixel_scale, tiepoint)
            nodata_value = tags[42113].value if 42113 in tags else None
            full_array = era_normalize_dem_block(page.asarray(), page.imagelength, page.imagewidth)
            width = page.imagewidth
            height = page.imagelength

        def tifffile_read(row_start, row_end):
            return full_array[row_start:row_end, :]
        return era_DemReader('tifffile', width, height, west, north, x_size, y_size, era_parse_nodata_tag(nodata_value), tifffile_read, lambda: None)
    except Exception as exc:
        errors.append(f'tifffile: {type(exc).__name__}: {exc}')
    standard_source = None
    try:
        standard_source = era_StandardLibraryTiff(dem_path)
        west, north, x_size, y_size = era_georeference_from_tiff_tags(standard_source.tags.get(33550), standard_source.tags.get(33922))
        return era_DemReader('Python-standard-library TIFF', standard_source.width, standard_source.height, west, north, x_size, y_size, era_parse_nodata_tag(standard_source.tags.get(42113)), standard_source.read_rows, standard_source.close)
    except Exception as exc:
        if standard_source is not None:
            standard_source.close()
        errors.append(f'standard-library TIFF: {type(exc).__name__}: {exc}')
    pillow_image = None
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        pillow_image = Image.open(dem_path)
        if getattr(pillow_image, 'n_frames', 1) > 1:
            pillow_image.seek(0)
        width, height = pillow_image.size
        tags = pillow_image.tag_v2
        west, north, x_size, y_size = era_georeference_from_tiff_tags(tags.get(33550), tags.get(33922))

        def pillow_read(row_start, row_end):
            return pillow_image.crop((0, row_start, width, row_end))
        return era_DemReader('Pillow', width, height, west, north, x_size, y_size, era_parse_nodata_tag(tags.get(42113)), pillow_read, pillow_image.close)
    except Exception as exc:
        if pillow_image is not None:
            pillow_image.close()
        errors.append(f'Pillow: {type(exc).__name__}: {exc}')
    raise RuntimeError('No available backend could read the DEM GeoTIFF. Tried rasterio, GDAL, tifffile, the dependency-free TIFF reader and Pillow:\n  ' + '\n  '.join(errors))

def era_dem_band_weights(dem_path: Path, template: xr.DataArray, block_rows: int=512) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame, dict]:
    """Aggregate 30 m DEM pixel areas into each ERA5-Land cell and elevation band."""
    latitudes = np.asarray(template.latitude.values, dtype='float64')
    longitudes = np.asarray(template.longitude.values, dtype='float64')
    latitude_edges = era_coordinate_edges(latitudes, 'latitude')
    longitude_edges = era_coordinate_edges(longitudes, 'longitude')
    cell_areas = era_spherical_cell_areas_km2(latitude_edges, longitude_edges)
    nlat, nlon = (len(latitudes), len(longitudes))
    output_size = nlat * nlon
    valid_area_flat = np.zeros(output_size, dtype='float64')
    valid_count_flat = np.zeros(output_size, dtype='int64')
    band_area_flat = {band['label']: np.zeros(output_size, dtype='float64') for band in era_ELEVATION_BANDS}
    band_count_flat = {band['label']: np.zeros(output_size, dtype='int64') for band in era_ELEVATION_BANDS}
    reader = era_open_dem_reader(dem_path)
    reader_backend = reader.backend
    try:
        width = reader.width
        height = reader.height
        west = reader.west
        north = reader.north
        x_size = reader.x_size
        y_size = reader.y_size
        east = west + width * x_size
        south = north - height * y_size
        nodata = reader.nodata
        tolerance = max(x_size, y_size) * 2.0
        if west > era_DOMAIN['west'] + tolerance or east < era_DOMAIN['east'] - tolerance or south > era_DOMAIN['south'] + tolerance or (north < era_DOMAIN['north'] - tolerance):
            raise ValueError(f'DEM does not cover 27-29 N, 86-89 E: bounds=({west}, {south}, {east}, {north})')
        longitude_centers = west + (np.arange(width) + 0.5) * x_size
        longitude_bins = np.searchsorted(longitude_edges, longitude_centers, side='right') - 1
        longitude_inside = (longitude_bins >= 0) & (longitude_bins < nlon) & (longitude_centers >= era_DOMAIN['west']) & (longitude_centers < era_DOMAIN['east'])
        for row_start in range(0, height, block_rows):
            row_end = min(height, row_start + block_rows)
            tile = era_normalize_dem_block(reader.read_rows(row_start, row_end), row_end - row_start, width)
            row_numbers = np.arange(row_start, row_end)
            latitude_centers = north - (row_numbers + 0.5) * y_size
            latitude_bins = np.searchsorted(latitude_edges, latitude_centers, side='right') - 1
            latitude_inside = (latitude_bins >= 0) & (latitude_bins < nlat) & (latitude_centers >= era_DOMAIN['south']) & (latitude_centers < era_DOMAIN['north'])
            spatial_inside = latitude_inside[:, None] & longitude_inside[None, :]
            valid_dem = spatial_inside & np.isfinite(tile) & ~np.isclose(tile, nodata, rtol=0.0, atol=1e-06)
            if not np.any(valid_dem):
                continue
            safe_latitude_bins = np.clip(latitude_bins, 0, nlat - 1)
            safe_longitude_bins = np.clip(longitude_bins, 0, nlon - 1)
            grid_index = safe_latitude_bins[:, None] * nlon + safe_longitude_bins[None, :]
            latitude_north = latitude_centers + y_size / 2.0
            latitude_south = latitude_centers - y_size / 2.0
            row_pixel_area = era_EARTH_RADIUS_M ** 2 * np.abs(np.sin(np.deg2rad(latitude_north)) - np.sin(np.deg2rad(latitude_south))) * np.deg2rad(x_size) / 1000000.0
            pixel_area = np.broadcast_to(row_pixel_area[:, None], tile.shape)
            valid_area_flat += np.bincount(grid_index[valid_dem], weights=pixel_area[valid_dem], minlength=output_size)
            valid_count_flat += np.bincount(grid_index[valid_dem], minlength=output_size).astype('int64')
            for band in era_ELEVATION_BANDS:
                in_band = valid_dem & (tile >= band['low']) & (tile < band['high'])
                if not np.any(in_band):
                    continue
                band_area_flat[band['label']] += np.bincount(grid_index[in_band], weights=pixel_area[in_band], minlength=output_size)
                band_count_flat[band['label']] += np.bincount(grid_index[in_band], minlength=output_size).astype('int64')
    finally:
        reader.close()
    valid_area = valid_area_flat.reshape(nlat, nlon)
    valid_count = valid_count_flat.reshape(nlat, nlon)
    band_weights = {label: values.reshape(nlat, nlon) for label, values in band_area_flat.items()}
    band_counts = {label: values.reshape(nlat, nlon) for label, values in band_count_flat.items()}
    audit_rows: list[dict] = []
    for band in era_ELEVATION_BANDS:
        weights = band_weights[band['label']]
        positive = weights > 0.0
        weight_sum = float(np.sum(weights))
        kish = weight_sum ** 2 / float(np.sum(weights ** 2)) if weight_sum > 0.0 else 0.0
        fractions = np.divide(weights, cell_areas, out=np.zeros_like(weights), where=cell_areas > 0.0)
        audit_rows.append({'band_label': band['label'], 'display_name': band['display_name'], 'elevation_lower_m_inclusive': band['low'], 'elevation_upper_m_exclusive': band['high'], 'dem_pixel_count': int(np.sum(band_counts[band['label']])), 'dem_band_area_km2': weight_sum, 'era5land_cells_intersecting_band': int(np.sum(positive)), 'full_era5land_cell_area_equivalents': float(np.sum(fractions)), 'kish_effective_era5land_cells': kish, 'maximum_band_fraction_in_one_era5land_cell': float(np.max(fractions[positive])) if np.any(positive) else 0.0})
    cell_rows: list[dict] = []
    valid_fraction = np.divide(valid_area, cell_areas, out=np.zeros_like(valid_area), where=cell_areas > 0.0)
    for iy, latitude in enumerate(latitudes):
        for ix, longitude in enumerate(longitudes):
            row = {'era5land_latitude': latitude, 'era5land_longitude': longitude, 'latitude_south': latitude_edges[iy], 'latitude_north': latitude_edges[iy + 1], 'longitude_west': longitude_edges[ix], 'longitude_east': longitude_edges[ix + 1], 'era5land_cell_area_km2': cell_areas[iy, ix], 'valid_dem_pixel_count': int(valid_count[iy, ix]), 'valid_dem_area_km2': valid_area[iy, ix], 'valid_dem_area_fraction': valid_fraction[iy, ix]}
            for band in era_ELEVATION_BANDS:
                area = band_weights[band['label']][iy, ix]
                row[f"{band['label']}_dem_pixel_count"] = int(band_counts[band['label']][iy, ix])
                row[f"{band['label']}_area_km2"] = area
                row[f"{band['label']}_cell_fraction"] = area / cell_areas[iy, ix] if cell_areas[iy, ix] > 0 else np.nan
            cell_rows.append(row)
    dem_metadata = {'path': str(dem_path), 'reader_backend': reader_backend, 'width': int(width), 'height': int(height), 'bounds_west_south_east_north': [west, south, east, north], 'pixel_size_degrees': [x_size, y_size], 'nodata': nodata, 'aggregation': '30 m DEM-pixel spherical area summed into each 0.1-degree ERA5-Land cell and elevation band'}
    return (band_weights, pd.DataFrame(audit_rows), pd.DataFrame(cell_rows), dem_metadata)

def era_weighted_band_metrics(field: xr.DataArray, weights_km2: np.ndarray) -> dict:
    """Area-weight ERA5-Land values by 30 m DEM band area in every grid cell."""
    values = np.asarray(field.values, dtype='float64')
    if values.shape != weights_km2.shape:
        raise ValueError(f'Precipitation/DEM-weight shape mismatch: {values.shape} vs {weights_km2.shape}')
    use = np.isfinite(values) & np.isfinite(weights_km2) & (weights_km2 > 0.0)
    if not np.any(use):
        raise ValueError('No 30 m DEM area contributes to an elevation band')
    data = values[use]
    weights = weights_km2[use]
    weight_sum = float(np.sum(weights))
    mean = float(np.sum(weights * data) / weight_sum)
    variance = float(np.sum(weights * (data - mean) ** 2) / weight_sum)
    kish = weight_sum ** 2 / float(np.sum(weights ** 2))
    return {'area_weighted_mean_mm': mean, 'area_weighted_sd_across_era5land_cells_mm': float(np.sqrt(max(0.0, variance))), 'era5land_cells_with_band_area_and_valid_precipitation': int(np.sum(use)), 'dem_band_area_with_valid_precipitation_km2': weight_sum, 'kish_effective_era5land_cells_for_precipitation': kish}

def era_build_historical_climatology(monthly_fields: dict[tuple[int, int], xr.DataArray], band_weights: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Calculate 2010-2020 monthly climatology and annual JJAS totals."""
    climatology_rows: list[dict] = []
    annual_rows: list[dict] = []
    plot_data = {'mean': np.full((len(era_ELEVATION_BANDS), len(era_MONTHS)), np.nan), 'interannual_sd': np.full((len(era_ELEVATION_BANDS), len(era_MONTHS)), np.nan), 'jjas_total': np.full(len(era_ELEVATION_BANDS), np.nan)}
    for band_index, band in enumerate(era_ELEVATION_BANDS):
        weights = band_weights[band['label']]
        year_month_values = np.full((len(era_HISTORICAL_YEARS), len(era_MONTHS)), np.nan)
        for year_index, year in enumerate(era_HISTORICAL_YEARS):
            for month_index, month in enumerate(era_MONTHS):
                metrics = era_weighted_band_metrics(monthly_fields[year, month], weights)
                year_month_values[year_index, month_index] = metrics['area_weighted_mean_mm']
            annual_rows.append({'year': year, 'band_label': band['label'], 'display_name': band['display_name'], 'elevation_lower_m_inclusive': band['low'], 'elevation_upper_m_exclusive': band['high'], 'jjas_total_area_weighted_mean_mm': float(np.sum(year_month_values[year_index])), **{f'month_{month:02d}_area_weighted_mean_mm': float(year_month_values[year_index, month_index]) for month_index, month in enumerate(era_MONTHS)}})
        for month_index, month in enumerate(era_MONTHS):
            values = year_month_values[:, month_index]
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1))
            plot_data['mean'][band_index, month_index] = mean
            plot_data['interannual_sd'][band_index, month_index] = sd
            climatology_rows.append({'period': '2010-2020', 'month': month, 'band_label': band['label'], 'display_name': band['display_name'], 'elevation_lower_m_inclusive': band['low'], 'elevation_upper_m_exclusive': band['high'], 'n_years': len(era_HISTORICAL_YEARS), 'climatological_monthly_mean_mm': mean, 'interannual_sd_mm': sd, 'interannual_se_mm': sd / np.sqrt(len(era_HISTORICAL_YEARS)), 'minimum_year_value_mm': float(np.min(values)), 'maximum_year_value_mm': float(np.max(values)), 'spatial_method': 'area-weighted ERA5-Land mean using 30 m DEM band area inside every ERA5-Land cell'})
        plot_data['jjas_total'][band_index] = float(np.sum(plot_data['mean'][band_index]))
    return (pd.DataFrame(climatology_rows), pd.DataFrame(annual_rows), plot_data)

# GPM IMERG input, DEM and statistics
gpm_HISTORICAL_YEARS = tuple(range(2010, 2021))

gpm_MONTHS = (6, 7, 8, 9)

gpm_DOMAIN = {'south': 27.0, 'north': 29.0, 'west': 86.0, 'east': 89.0}

gpm_EARTH_RADIUS_M = 6371008.8

gpm_PRECIPITATION_NAMES = ('precipitation', 'precipitationcal', 'precipitationuncal')

gpm_LATITUDE_NAMES = ('latitude', 'lat')

gpm_LONGITUDE_NAMES = ('longitude', 'lon')

gpm_DATE_RE = re.compile('3IMERG\\.(\\d{8})', re.IGNORECASE)

@dataclass(frozen=True)
class gpm_ElevationBand:
    label: str
    display_name: str
    lower_m: float
    upper_m: float
    color: str

gpm_BANDS = (gpm_ElevationBand('P5200', '5200 ± 300 m', 4900.0, 5500.0, '#17325B'), gpm_ElevationBand('P5800', '5800 ± 300 m', 5500.0, 6100.0, '#E31A1C'), gpm_ElevationBand('P6500', '6500 ± 300 m', 6200.0, 6800.0, '#8EA6C0'))

def gpm_netcdf_files(directory: Path) -> list[Path]:
    files = sorted({path.resolve() for pattern in ('*.nc', '*.nc4', '*.NC', '*.NC4') for path in directory.rglob(pattern) if path.is_file()})
    if not files:
        raise FileNotFoundError(f'No .nc/.nc4 files found under {directory}')
    return files

def gpm_date_from_filename(path: Path) -> pd.Timestamp | None:
    match = gpm_DATE_RE.search(path.name)
    if match is None:
        return None
    return pd.to_datetime(match.group(1), format='%Y%m%d')

def gpm_discover_historical_monthly(directory: Path) -> dict[tuple[int, int], Path]:
    """Select exactly the 44 JJAS files for 2010-2020 and ignore extras."""
    selected: dict[tuple[int, int], Path] = {}
    for path in gpm_netcdf_files(directory):
        if '3B-MO' not in path.name.upper():
            continue
        date = gpm_date_from_filename(path)
        if date is None or date.year not in gpm_HISTORICAL_YEARS or date.month not in gpm_MONTHS:
            continue
        key = (int(date.year), int(date.month))
        if key in selected:
            raise ValueError(f'Duplicate historical monthly files for {key[0]}-{key[1]:02d}:\n  {selected[key]}\n  {path}')
        selected[key] = path
    expected = {(year, month) for year in gpm_HISTORICAL_YEARS for month in gpm_MONTHS}
    missing = sorted(expected - set(selected))
    if missing:
        preview = ', '.join((f'{year}-{month:02d}' for year, month in missing[:20]))
        suffix = ' ...' if len(missing) > 20 else ''
        raise FileNotFoundError(f'Historical IMERG coverage is incomplete: expected 44 JJAS monthly files for 2010-2020, but {len(missing)} are missing: {preview}{suffix}')
    return {key: selected[key] for key in sorted(expected)}

def gpm_find_name(names, candidates: tuple[str, ...]) -> str | None:
    lookup = {str(name).lower(): str(name) for name in names}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None

def gpm_finish_spatial_field(da: xr.DataArray) -> xr.DataArray:
    """Standardize, crop and eagerly materialize a 2-D spatial field."""
    if da.latitude.ndim != 1 or da.longitude.ndim != 1:
        raise ValueError('IMERG latitude and longitude coordinates must be one-dimensional')
    for dim in list(da.dims):
        if dim in {'latitude', 'longitude'}:
            continue
        if da.sizes[dim] == 1:
            da = da.isel({dim: 0}, drop=True)
        else:
            raise ValueError(f'Unexpected non-spatial dimension {dim!r} with size {da.sizes[dim]}')
    if set(da.dims) != {'latitude', 'longitude'}:
        raise ValueError(f'Expected a 2-D precipitation field; dimensions={da.dims}')
    da = da.transpose('latitude', 'longitude')
    if float(da.longitude.max()) > 180.0:
        da = da.assign_coords(longitude=(da.longitude + 180.0) % 360.0 - 180.0)
    da = da.sortby('latitude').sortby('longitude')
    da = da.sel(latitude=slice(gpm_DOMAIN['south'], gpm_DOMAIN['north']), longitude=slice(gpm_DOMAIN['west'], gpm_DOMAIN['east']))
    if da.sizes.get('latitude', 0) < 2 or da.sizes.get('longitude', 0) < 2:
        raise ValueError('IMERG subset does not cover enough of 27-29 N, 86-89 E for interpolation')
    values = da.astype('float64')
    values = values.where(np.isfinite(values) & (values > -9000.0))
    return values.load()

def gpm_standardize_precipitation(da: xr.DataArray, ds: xr.Dataset) -> xr.DataArray:
    latitude_name = gpm_find_name(ds.variables, gpm_LATITUDE_NAMES)
    longitude_name = gpm_find_name(ds.variables, gpm_LONGITUDE_NAMES)
    if latitude_name is None or longitude_name is None:
        raise KeyError(f'Latitude/longitude not found; variables={list(ds.variables)}')
    if latitude_name not in da.coords:
        da = da.assign_coords({latitude_name: ds[latitude_name]})
    if longitude_name not in da.coords:
        da = da.assign_coords({longitude_name: ds[longitude_name]})
    rename: dict[str, str] = {}
    if latitude_name != 'latitude':
        rename[latitude_name] = 'latitude'
    if longitude_name != 'longitude':
        rename[longitude_name] = 'longitude'
    da = da.rename(rename)
    return gpm_finish_spatial_field(da)

def gpm_iter_netcdf4_groups(root):
    """Yield the root and all nested netCDF4 groups."""
    yield ('root', root)
    pending = [(name, group) for name, group in root.groups.items()]
    while pending:
        group_path, group = pending.pop(0)
        yield (group_path, group)
        pending.extend(((f'{group_path}/{name}', child) for name, child in group.groups.items()))

def gpm_direct_netcdf4_field(path: Path) -> tuple[xr.DataArray, str, str]:
    """Read IMERG eagerly with netCDF4, avoiding xarray backend locks.

    Some cluster configurations use a process-based dask scheduler globally.
    Loading an xarray object backed by netCDF4 can then attempt to pickle the
    backend's thread lock.  This reader copies the variable to a NumPy array
    while the Dataset is open, so no file handle or lock reaches xarray.
    """
    try:
        import netCDF4
    except ImportError as exc:
        raise RuntimeError('Python package netCDF4 is unavailable') from exc
    group_errors: list[str] = []
    with netCDF4.Dataset(path, mode='r') as root:
        root.set_auto_maskandscale(True)
        for group_path, group in gpm_iter_netcdf4_groups(root):
            variable_name = gpm_find_name(group.variables, gpm_PRECIPITATION_NAMES)
            if variable_name is None:
                group_errors.append(f'group={group_path}: precipitation absent; variables={list(group.variables)}')
                continue
            latitude_name = gpm_find_name(group.variables, gpm_LATITUDE_NAMES)
            longitude_name = gpm_find_name(group.variables, gpm_LONGITUDE_NAMES)
            if latitude_name is None or longitude_name is None:
                group_errors.append(f'group={group_path}: latitude/longitude absent; variables={list(group.variables)}')
                continue
            precip_var = group.variables[variable_name]
            latitude_var = group.variables[latitude_name]
            longitude_var = group.variables[longitude_name]
            if len(latitude_var.dimensions) != 1 or len(longitude_var.dimensions) != 1:
                group_errors.append(f'group={group_path}: latitude/longitude coordinates are not 1-D')
                continue
            latitude_dim = latitude_var.dimensions[0]
            longitude_dim = longitude_var.dimensions[0]
            dimensions = list(precip_var.dimensions)
            if latitude_dim not in dimensions or longitude_dim not in dimensions:
                group_errors.append(f'group={group_path}: precipitation dimensions {dimensions} do not contain {latitude_dim!r}/{longitude_dim!r}')
                continue
            raw = np.ma.asarray(precip_var[:])
            data = np.asarray(np.ma.filled(raw, np.nan), dtype='float64')
            non_spatial_axes = [axis for axis, dim in enumerate(dimensions) if dim not in {latitude_dim, longitude_dim}]
            bad_axes = [axis for axis in non_spatial_axes if data.shape[axis] != 1]
            if bad_axes:
                group_errors.append(f'group={group_path}: non-spatial dimensions must be singleton; dimensions={dimensions}, shape={data.shape}')
                continue
            if non_spatial_axes:
                data = np.squeeze(data, axis=tuple(non_spatial_axes))
            spatial_dimensions = [dim for dim in dimensions if dim in {latitude_dim, longitude_dim}]
            if data.ndim != 2 or len(spatial_dimensions) != 2:
                group_errors.append(f'group={group_path}: expected 2-D spatial data after squeeze; dimensions={dimensions}, shape={data.shape}')
                continue
            latitude_axis = spatial_dimensions.index(latitude_dim)
            longitude_axis = spatial_dimensions.index(longitude_dim)
            data = np.transpose(data, axes=(latitude_axis, longitude_axis))
            latitude = np.asarray(np.ma.filled(np.ma.asarray(latitude_var[:]), np.nan), dtype='float64').squeeze()
            longitude = np.asarray(np.ma.filled(np.ma.asarray(longitude_var[:]), np.nan), dtype='float64').squeeze()
            if data.shape != (latitude.size, longitude.size):
                group_errors.append(f'group={group_path}: coordinate/data shape mismatch; data={data.shape}, lat={latitude.size}, lon={longitude.size}')
                continue
            field = xr.DataArray(data, dims=('latitude', 'longitude'), coords={'latitude': latitude, 'longitude': longitude}, name=str(variable_name))
            source_units = str(getattr(precip_var, 'units', '')).strip()
            field = gpm_finish_spatial_field(field)
            return (field, source_units, f'netCDF4-direct,group={group_path}')
    raise RuntimeError('; '.join(group_errors))

def gpm_load_precipitation_field(path: Path) -> tuple[xr.DataArray, str, str]:
    """Open root or Grid group and return field, source units, and open mode."""
    errors: list[str] = []
    try:
        return gpm_direct_netcdf4_field(path)
    except Exception as exc:
        errors.append(f'netCDF4-direct: {type(exc).__name__}: {exc}')
    attempts = ((None, None), ('h5netcdf', None), ('netcdf4', None), (None, 'Grid'), ('h5netcdf', 'Grid'), ('netcdf4', 'Grid'))
    for engine, group in attempts:
        kwargs: dict = {'mask_and_scale': True, 'decode_times': True, 'chunks': None, 'cache': False}
        if engine is not None:
            kwargs['engine'] = engine
        if group is not None:
            kwargs['group'] = group
        ds = None
        try:
            ds = xr.open_dataset(path, **kwargs)
            variable_name = gpm_find_name(ds.data_vars, gpm_PRECIPITATION_NAMES)
            if variable_name is None:
                errors.append(f"engine={engine or 'default'}, group={group or 'root'}: precipitation absent; data_vars={list(ds.data_vars)}")
                ds.close()
                continue
            source_units = str(ds[variable_name].attrs.get('units', '')).strip()
            field = gpm_standardize_precipitation(ds[variable_name], ds)
            ds.close()
            mode = f"engine={engine or 'default'},group={group or 'root'}"
            return (field, source_units, mode)
        except Exception as exc:
            if ds is not None:
                try:
                    ds.close()
                except Exception:
                    pass
            errors.append(f"engine={engine or 'default'}, group={group or 'root'}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f'Could not read IMERG precipitation from {path}\n' + '\n'.join(errors))

def gpm_compact_units(units: str) -> str:
    return units.lower().replace(' ', '').replace('**', '').replace('−', '-').replace('per', '/')

def gpm_is_per_hour(units: str) -> bool:
    u = gpm_compact_units(units)
    return any((token in u for token in ('mm/hr', 'mm/hour', 'mmh-1', 'mmhr-1')))

def gpm_is_per_day(units: str) -> bool:
    u = gpm_compact_units(units)
    return any((token in u for token in ('mm/day', 'mmd-1', 'mmday-1')))

def gpm_to_accumulation_mm(field: xr.DataArray, source_units: str, product: str, stamp: pd.Timestamp) -> tuple[xr.DataArray, float, str]:
    u = gpm_compact_units(source_units)
    plain_mm = u in {'mm', 'millimeter', 'millimeters'}
    if product == 'monthly':
        days = calendar.monthrange(stamp.year, stamp.month)[1]
        if gpm_is_per_hour(source_units):
            factor = 24.0 * days
            note = f'monthly mean rate * 24 * {days} days'
        elif gpm_is_per_day(source_units):
            factor = float(days)
            note = f'monthly mean daily amount * {days} days'
        elif plain_mm:
            factor = 1.0
            note = 'monthly accumulation already in mm'
        else:
            raise ValueError(f'Unsupported/blank monthly precipitation units {source_units!r} in {stamp:%Y-%m}. Expected mm/hr, mm/day, or mm.')
    elif product == 'daily':
        if gpm_is_per_hour(source_units):
            factor = 24.0
            note = 'daily mean rate * 24 hours'
        elif gpm_is_per_day(source_units) or plain_mm:
            factor = 1.0
            note = 'daily accumulation'
        else:
            raise ValueError(f'Unsupported/blank daily precipitation units {source_units!r} on {stamp:%Y-%m-%d}. Expected mm/hr, mm/day, or mm.')
    elif product == 'halfhourly':
        if gpm_is_per_hour(source_units):
            factor = 0.5
            note = 'half-hour mean rate * 0.5 hour'
        elif gpm_is_per_day(source_units):
            factor = 0.5 / 24.0
            note = 'daily rate * 0.5/24 day'
        elif plain_mm:
            factor = 1.0
            note = 'half-hour accumulation already in mm'
        else:
            raise ValueError(f'Unsupported/blank half-hour precipitation units {source_units!r} at {stamp} UTC. Expected mm/hr, mm/day, or mm.')
    else:
        raise ValueError(f'Unknown product type: {product}')
    output = field * factor
    output.attrs = {'units': 'mm', 'conversion': note}
    return (output, factor, note)

def gpm_same_grid(a: xr.DataArray, b: xr.DataArray) -> bool:
    """Return True when two grids are numerically equivalent.

    NetCDF products can store the same nominal 0.1-degree coordinates with
    slightly different floating-point values.  This test is intentionally
    tolerant, but arithmetic must first snap the coordinates exactly.
    """
    return a.sizes == b.sizes and np.allclose(a.latitude.values, b.latitude.values, rtol=0.0, atol=1e-05) and np.allclose(a.longitude.values, b.longitude.values, rtol=0.0, atol=1e-05)

def gpm_snap_coordinates(field: xr.DataArray, target: xr.DataArray) -> xr.DataArray:
    """Assign the target's exact coordinate labels without changing data."""
    return field.assign_coords(latitude=np.asarray(target.latitude.values, dtype='float64').copy(), longitude=np.asarray(target.longitude.values, dtype='float64').copy())

def gpm_align_to(field: xr.DataArray, target: xr.DataArray) -> xr.DataArray:
    if gpm_same_grid(field, target):
        return gpm_snap_coordinates(field, target)
    aligned = field.interp(latitude=target.latitude, longitude=target.longitude, method='linear')
    aligned = gpm_snap_coordinates(aligned, target)
    if bool(aligned.isnull().all()):
        raise ValueError('IMERG grids do not overlap')
    return aligned

def gpm_read_historical_monthly(monthly_files: dict[tuple[int, int], Path]) -> tuple[dict[tuple[int, int], xr.DataArray], pd.DataFrame]:
    fields: dict[tuple[int, int], xr.DataArray] = {}
    inventory: list[dict] = []
    print('Reading 44 monthly IMERG files for the 2010-2020 JJAS climatology ...', flush=True)
    for (year, month), path in monthly_files.items():
        stamp = pd.Timestamp(year=year, month=month, day=1)
        field, units, mode = gpm_load_precipitation_field(path)
        field, factor, note = gpm_to_accumulation_mm(field, units, 'monthly', stamp)
        fields[year, month] = field
        inventory.append({'product': 'monthly_2010_2020_climatology_input', 'timestamp_utc': stamp, 'path': str(path), 'source_units': units, 'conversion_factor': factor, 'conversion': note, 'open_mode': mode})
    return (fields, pd.DataFrame(inventory))

def gpm_coordinate_edges(centers: np.ndarray, name: str) -> np.ndarray:
    centers = np.asarray(centers, dtype='float64')
    if centers.ndim != 1 or centers.size < 2 or (not np.all(np.diff(centers) > 0)):
        raise ValueError(f'{name} centers must be a strictly increasing 1-D array')
    differences = np.diff(centers)
    spacing = float(np.median(differences))
    if not np.allclose(differences, spacing, rtol=0.0, atol=1e-05):
        raise ValueError(f'{name} grid is not regularly spaced: {differences}')
    edges = np.empty(centers.size + 1, dtype='float64')
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - spacing / 2.0
    edges[-1] = centers[-1] + spacing / 2.0
    return edges

def gpm_spherical_cell_areas_km2(latitude_edges: np.ndarray, longitude_edges: np.ndarray) -> np.ndarray:
    latitude_term = np.abs(np.sin(np.deg2rad(latitude_edges[1:])) - np.sin(np.deg2rad(latitude_edges[:-1])))
    longitude_term = np.abs(np.diff(np.deg2rad(longitude_edges)))
    return gpm_EARTH_RADIUS_M ** 2 * latitude_term[:, None] * longitude_term[None, :] / 1000000.0

def gpm_parse_nodata_tag(value) -> float:
    if value is None:
        return -9999.0
    if isinstance(value, (tuple, list)):
        value = value[0]
    if isinstance(value, bytes):
        value = value.decode('ascii', errors='ignore')
    return float(str(value).strip().split()[0])

def gpm_georeference_from_tiff_tags(pixel_scale, tiepoint) -> tuple[float, float, float, float]:
    if pixel_scale is None or tiepoint is None:
        raise ValueError("GeoTIFF lacks ModelPixelScaleTag/ModelTiepointTag; export it from GEE with crs='EPSG:4326'.")
    pixel_scale = tuple((float(value) for value in pixel_scale))
    tiepoint = tuple((float(value) for value in tiepoint))
    if len(pixel_scale) < 2 or len(tiepoint) < 6:
        raise ValueError('Invalid GeoTIFF georeferencing tags')
    x_size = abs(pixel_scale[0])
    y_size = abs(pixel_scale[1])
    raster_i, raster_j = (tiepoint[0], tiepoint[1])
    map_x, map_y = (tiepoint[3], tiepoint[4])
    west = map_x - raster_i * x_size
    north = map_y + raster_j * y_size
    return (west, north, x_size, y_size)

def gpm_normalize_dem_block(data, expected_rows: int, expected_columns: int) -> np.ndarray:
    array = np.asarray(data, dtype='float64')
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = array[:, :, 0]
    if array.shape != (expected_rows, expected_columns):
        raise ValueError(f'DEM must contain exactly one band; expected block shape {(expected_rows, expected_columns)}, received {array.shape}')
    return array

class gpm_DemReader:

    def __init__(self, backend, width, height, west, north, x_size, y_size, nodata, read_rows, close):
        self.backend = str(backend)
        self.width = int(width)
        self.height = int(height)
        self.west = float(west)
        self.north = float(north)
        self.x_size = float(x_size)
        self.y_size = float(y_size)
        self.nodata = float(nodata) if nodata is not None else -9999.0
        self.read_rows = read_rows
        self.close = close

def gpm_tiff_lzw_decode(payload: bytes) -> bytes:
    """Decode TIFF-flavour LZW (MSB-first codes, EarlyChange=1)."""
    clear_code = 256
    end_code = 257
    bit_position = 0
    code_width = 9
    next_code = 258
    table = {code: bytes((code,)) for code in range(256)}
    previous = None
    output = bytearray()

    def read_code(width: int):
        nonlocal bit_position
        if bit_position + width > len(payload) * 8:
            return None
        value = 0
        for _ in range(width):
            byte_index, bit_index = divmod(bit_position, 8)
            value = value << 1 | payload[byte_index] >> 7 - bit_index & 1
            bit_position += 1
        return value
    while True:
        code = read_code(code_width)
        if code is None or code == end_code:
            break
        if code == clear_code:
            table = {item: bytes((item,)) for item in range(256)}
            code_width = 9
            next_code = 258
            previous = None
            continue
        if code in table:
            entry = table[code]
        elif code == next_code and previous is not None:
            entry = previous + previous[:1]
        else:
            raise ValueError(f'Invalid TIFF LZW code {code} at bit {bit_position}')
        output.extend(entry)
        if previous is not None and next_code < 4096:
            table[next_code] = previous + entry[:1]
            next_code += 1
            if next_code == (1 << code_width) - 1 and code_width < 12:
                code_width += 1
        previous = entry
    return bytes(output)

def gpm_tiff_packbits_decode(payload: bytes) -> bytes:
    output = bytearray()
    position = 0
    while position < len(payload):
        control = payload[position]
        position += 1
        signed_control = control if control < 128 else control - 256
        if 0 <= signed_control <= 127:
            count = signed_control + 1
            output.extend(payload[position:position + count])
            position += count
        elif -127 <= signed_control <= -1:
            if position >= len(payload):
                raise ValueError('Truncated TIFF PackBits run')
            output.extend(payload[position:position + 1] * (1 - signed_control))
            position += 1
    return bytes(output)

class gpm_StandardLibraryTiff:
    """Minimal one-band GeoTIFF reader used when geospatial packages are absent."""
    TYPE_FORMATS = {1: 'B', 3: 'H', 4: 'I', 6: 'b', 8: 'h', 9: 'i', 11: 'f', 12: 'd'}
    TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 6: 1, 8: 2, 9: 4, 11: 4, 12: 8}

    def __init__(self, path: Path):
        self.path = Path(path)
        self.handle = self.path.open('rb')
        marker = self.handle.read(2)
        if marker == b'II':
            self.endian = '<'
        elif marker == b'MM':
            self.endian = '>'
        else:
            self.close()
            raise ValueError('Not a TIFF file: invalid byte-order marker')
        magic = self._unpack('H', self.handle.read(2))[0]
        if magic != 42:
            self.close()
            if magic == 43:
                raise ValueError('BigTIFF is unsupported by the standard-library fallback')
            raise ValueError(f'Invalid classic-TIFF magic number: {magic}')
        ifd_offset = self._unpack('I', self.handle.read(4))[0]
        self.tags = self._read_ifd(ifd_offset)
        self.width = int(self._scalar(256))
        self.height = int(self._scalar(257))
        self.bits = int(self._scalar(258))
        self.compression = int(self._scalar(259, 1))
        self.samples_per_pixel = int(self._scalar(277, 1))
        self.predictor = int(self._scalar(317, 1))
        self.sample_format = int(self._scalar(339, 1))
        self.planar_configuration = int(self._scalar(284, 1))
        if self.samples_per_pixel != 1 or self.planar_configuration != 1:
            self.close()
            raise ValueError('Standard-library TIFF fallback requires one chunky band')
        if self.predictor not in (1, 2):
            self.close()
            raise ValueError(f'Unsupported TIFF predictor: {self.predictor}')
        dtype_codes = {(1, 8): 'u1', (1, 16): 'u2', (1, 32): 'u4', (1, 64): 'u8', (2, 8): 'i1', (2, 16): 'i2', (2, 32): 'i4', (2, 64): 'i8', (3, 32): 'f4', (3, 64): 'f8'}
        dtype_code = dtype_codes.get((self.sample_format, self.bits))
        if dtype_code is None:
            self.close()
            raise ValueError(f'Unsupported TIFF SampleFormat/BitsPerSample: {self.sample_format}/{self.bits}')
        self.dtype = np.dtype(self.endian + dtype_code)
        self.bytes_per_value = self.dtype.itemsize
        self.tile_width = self._optional_scalar(322)
        self.tile_height = self._optional_scalar(323)
        if self.tile_width is not None or self.tile_height is not None:
            if self.tile_width is None or self.tile_height is None:
                self.close()
                raise ValueError('Incomplete tiled-TIFF metadata')
            self.tile_width = int(self.tile_width)
            self.tile_height = int(self.tile_height)
            self.segment_offsets = self._as_tuple(self.tags.get(324))
            self.segment_byte_counts = self._as_tuple(self.tags.get(325))
            self.is_tiled = True
        else:
            self.rows_per_strip = int(self._scalar(278, self.height))
            self.segment_offsets = self._as_tuple(self.tags.get(273))
            self.segment_byte_counts = self._as_tuple(self.tags.get(279))
            self.is_tiled = False
        if not self.segment_offsets or not self.segment_byte_counts:
            self.close()
            raise ValueError('TIFF lacks strip/tile offsets or byte counts')
        if len(self.segment_offsets) != len(self.segment_byte_counts):
            self.close()
            raise ValueError('TIFF strip/tile offset and byte-count lengths differ')

    def _unpack(self, fmt: str, payload: bytes):
        return struct.unpack(self.endian + fmt, payload)

    def _read_ifd(self, offset: int) -> dict:
        self.handle.seek(offset)
        entry_count = self._unpack('H', self.handle.read(2))[0]
        tags = {}
        for _ in range(entry_count):
            entry = self.handle.read(12)
            if len(entry) != 12:
                raise ValueError('Truncated TIFF IFD')
            tag, value_type, count = self._unpack('HHI', entry[:8])
            size = self.TYPE_SIZES.get(value_type)
            if size is None:
                continue
            byte_count = size * count
            if byte_count <= 4:
                raw = entry[8:8 + byte_count]
            else:
                value_offset = self._unpack('I', entry[8:12])[0]
                current = self.handle.tell()
                self.handle.seek(value_offset)
                raw = self.handle.read(byte_count)
                self.handle.seek(current)
            if len(raw) != byte_count:
                raise ValueError(f'Truncated TIFF tag {tag}')
            if value_type == 2:
                value = raw.rstrip(b'\x00').decode('ascii', errors='ignore')
            else:
                fmt = self.TYPE_FORMATS[value_type]
                value = self._unpack(str(count) + fmt, raw)
                if count == 1:
                    value = value[0]
            tags[tag] = value
        return tags

    @staticmethod
    def _as_tuple(value) -> tuple:
        if value is None:
            return ()
        if isinstance(value, tuple):
            return value
        return (value,)

    def _optional_scalar(self, tag):
        value = self.tags.get(tag)
        if isinstance(value, tuple):
            return value[0]
        return value

    def _scalar(self, tag, default=None):
        value = self._optional_scalar(tag)
        if value is None:
            if default is not None:
                return default
            raise ValueError(f'TIFF required tag {tag} is missing')
        return value

    def _decode_segment(self, index: int, rows: int, columns: int) -> np.ndarray:
        self.handle.seek(int(self.segment_offsets[index]))
        compressed = self.handle.read(int(self.segment_byte_counts[index]))
        if self.compression == 1:
            decoded = compressed
        elif self.compression in (8, 32946):
            decoded = zlib.decompress(compressed)
        elif self.compression == 5:
            decoded = gpm_tiff_lzw_decode(compressed)
        elif self.compression == 32773:
            decoded = gpm_tiff_packbits_decode(compressed)
        else:
            raise ValueError(f'Unsupported TIFF compression code: {self.compression}')
        expected_values = rows * columns
        expected_bytes = expected_values * self.bytes_per_value
        if len(decoded) < expected_bytes:
            raise ValueError(f'Decompressed TIFF segment {index} is too short: {len(decoded)} < {expected_bytes} bytes')
        array = np.frombuffer(decoded[:expected_bytes], dtype=self.dtype, count=expected_values).reshape(rows, columns).copy()
        if self.predictor == 2:
            array = np.cumsum(array.astype('int64'), axis=1).astype(self.dtype)
        return array

    def read_rows(self, row_start: int, row_end: int) -> np.ndarray:
        if not 0 <= row_start < row_end <= self.height:
            raise ValueError(f'Invalid TIFF row window: {row_start}:{row_end}')
        result = np.empty((row_end - row_start, self.width), dtype=self.dtype)
        if self.is_tiled:
            tiles_across = (self.width + self.tile_width - 1) // self.tile_width
            first_tile_row = row_start // self.tile_height
            last_tile_row = (row_end - 1) // self.tile_height
            for tile_row in range(first_tile_row, last_tile_row + 1):
                tile_north = tile_row * self.tile_height
                source_row_start = max(row_start, tile_north) - tile_north
                source_row_end = min(row_end, tile_north + self.tile_height) - tile_north
                target_row_start = max(row_start, tile_north) - row_start
                target_row_end = min(row_end, tile_north + self.tile_height) - row_start
                for tile_column in range(tiles_across):
                    index = tile_row * tiles_across + tile_column
                    tile = self._decode_segment(index, self.tile_height, self.tile_width)
                    column_start = tile_column * self.tile_width
                    column_end = min(column_start + self.tile_width, self.width)
                    result[target_row_start:target_row_end, column_start:column_end] = tile[source_row_start:source_row_end, :column_end - column_start]
        else:
            first_strip = row_start // self.rows_per_strip
            last_strip = (row_end - 1) // self.rows_per_strip
            for strip_index in range(first_strip, last_strip + 1):
                strip_north = strip_index * self.rows_per_strip
                strip_rows = min(self.rows_per_strip, self.height - strip_north)
                strip = self._decode_segment(strip_index, strip_rows, self.width)
                source_start = max(row_start, strip_north) - strip_north
                source_end = min(row_end, strip_north + strip_rows) - strip_north
                target_start = max(row_start, strip_north) - row_start
                target_end = min(row_end, strip_north + strip_rows) - row_start
                result[target_start:target_end] = strip[source_start:source_end]
        return result

    def close(self):
        if not self.handle.closed:
            self.handle.close()

def gpm_open_dem_reader(dem_path: Path) -> gpm_DemReader:
    """Open a GeoTIFF with the first available existing geospatial backend."""
    errors: list[str] = []
    rasterio_source = None
    try:
        import rasterio
        from rasterio.windows import Window
        rasterio_source = rasterio.open(dem_path)
        if rasterio_source.count != 1:
            raise ValueError(f'DEM has {rasterio_source.count} bands; expected one')
        transform = rasterio_source.transform
        if abs(transform.b) > 1e-12 or abs(transform.d) > 1e-12:
            raise ValueError('Rotated GeoTIFF transforms are unsupported')
        west = transform.c
        north = transform.f
        x_size = float(transform.a)
        y_size = abs(float(transform.e))
        if x_size <= 0.0 or y_size <= 0.0:
            raise ValueError(f'Invalid GeoTIFF pixel size: {x_size}, {y_size}')
        width = rasterio_source.width
        height = rasterio_source.height

        def rasterio_read(row_start, row_end):
            return rasterio_source.read(1, window=Window(0, row_start, width, row_end - row_start), masked=False)
        return gpm_DemReader('rasterio', width, height, west, north, x_size, y_size, rasterio_source.nodata, rasterio_read, rasterio_source.close)
    except Exception as exc:
        if rasterio_source is not None:
            rasterio_source.close()
        errors.append(f'rasterio: {type(exc).__name__}: {exc}')
    try:
        from osgeo import gdal
        gdal_source = gdal.Open(str(dem_path), gdal.GA_ReadOnly)
        if gdal_source is None:
            raise OSError('gdal.Open returned None')
        if gdal_source.RasterCount != 1:
            raise ValueError(f'DEM has {gdal_source.RasterCount} bands; expected one')
        transform = gdal_source.GetGeoTransform()
        if abs(transform[2]) > 1e-12 or abs(transform[4]) > 1e-12:
            raise ValueError('Rotated GeoTIFF transforms are unsupported')
        west = transform[0]
        north = transform[3]
        x_size = float(transform[1])
        y_size = abs(float(transform[5]))
        if x_size <= 0.0 or y_size <= 0.0:
            raise ValueError(f'Invalid GeoTIFF pixel size: {x_size}, {y_size}')
        width = gdal_source.RasterXSize
        height = gdal_source.RasterYSize
        gdal_band = gdal_source.GetRasterBand(1)

        def gdal_read(row_start, row_end):
            return gdal_band.ReadAsArray(0, row_start, width, row_end - row_start)
        return gpm_DemReader('GDAL', width, height, west, north, x_size, y_size, gdal_band.GetNoDataValue(), gdal_read, lambda: None)
    except Exception as exc:
        errors.append(f'GDAL: {type(exc).__name__}: {exc}')
    try:
        import tifffile
        with tifffile.TiffFile(dem_path) as tif:
            page = tif.pages[0]
            tags = page.tags
            pixel_scale = tags[33550].value if 33550 in tags else None
            tiepoint = tags[33922].value if 33922 in tags else None
            west, north, x_size, y_size = gpm_georeference_from_tiff_tags(pixel_scale, tiepoint)
            nodata_value = tags[42113].value if 42113 in tags else None
            full_array = gpm_normalize_dem_block(page.asarray(), page.imagelength, page.imagewidth)
            width = page.imagewidth
            height = page.imagelength

        def tifffile_read(row_start, row_end):
            return full_array[row_start:row_end, :]
        return gpm_DemReader('tifffile', width, height, west, north, x_size, y_size, gpm_parse_nodata_tag(nodata_value), tifffile_read, lambda: None)
    except Exception as exc:
        errors.append(f'tifffile: {type(exc).__name__}: {exc}')
    standard_source = None
    try:
        standard_source = gpm_StandardLibraryTiff(dem_path)
        west, north, x_size, y_size = gpm_georeference_from_tiff_tags(standard_source.tags.get(33550), standard_source.tags.get(33922))
        return gpm_DemReader('Python-standard-library TIFF', standard_source.width, standard_source.height, west, north, x_size, y_size, gpm_parse_nodata_tag(standard_source.tags.get(42113)), standard_source.read_rows, standard_source.close)
    except Exception as exc:
        if standard_source is not None:
            standard_source.close()
        errors.append(f'standard-library TIFF: {type(exc).__name__}: {exc}')
    pillow_image = None
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        pillow_image = Image.open(dem_path)
        if getattr(pillow_image, 'n_frames', 1) > 1:
            pillow_image.seek(0)
        width, height = pillow_image.size
        tags = pillow_image.tag_v2
        west, north, x_size, y_size = gpm_georeference_from_tiff_tags(tags.get(33550), tags.get(33922))

        def pillow_read(row_start, row_end):
            return pillow_image.crop((0, row_start, width, row_end))
        return gpm_DemReader('Pillow', width, height, west, north, x_size, y_size, gpm_parse_nodata_tag(tags.get(42113)), pillow_read, pillow_image.close)
    except Exception as exc:
        if pillow_image is not None:
            pillow_image.close()
        errors.append(f'Pillow: {type(exc).__name__}: {exc}')
    raise RuntimeError('No available backend could read the DEM GeoTIFF. Tried rasterio, GDAL, tifffile, the dependency-free TIFF reader and Pillow:\n  ' + '\n  '.join(errors))

def gpm_dem_band_weights(dem_path: Path, template: xr.DataArray, block_rows: int=512) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame, dict]:
    """Aggregate 30 m DEM pixel areas into each GPM cell and elevation band."""
    latitudes = np.asarray(template.latitude.values, dtype='float64')
    longitudes = np.asarray(template.longitude.values, dtype='float64')
    latitude_edges = gpm_coordinate_edges(latitudes, 'latitude')
    longitude_edges = gpm_coordinate_edges(longitudes, 'longitude')
    cell_areas = gpm_spherical_cell_areas_km2(latitude_edges, longitude_edges)
    nlat, nlon = (len(latitudes), len(longitudes))
    output_size = nlat * nlon
    valid_area_flat = np.zeros(output_size, dtype='float64')
    valid_count_flat = np.zeros(output_size, dtype='int64')
    band_area_flat = {band.label: np.zeros(output_size, dtype='float64') for band in gpm_BANDS}
    band_count_flat = {band.label: np.zeros(output_size, dtype='int64') for band in gpm_BANDS}
    reader = gpm_open_dem_reader(dem_path)
    reader_backend = reader.backend
    try:
        width = reader.width
        height = reader.height
        west = reader.west
        north = reader.north
        x_size = reader.x_size
        y_size = reader.y_size
        east = west + width * x_size
        south = north - height * y_size
        nodata = reader.nodata
        tolerance = max(x_size, y_size) * 2.0
        if west > gpm_DOMAIN['west'] + tolerance or east < gpm_DOMAIN['east'] - tolerance or south > gpm_DOMAIN['south'] + tolerance or (north < gpm_DOMAIN['north'] - tolerance):
            raise ValueError(f'DEM does not cover 27-29 N, 86-89 E: bounds=({west}, {south}, {east}, {north})')
        longitude_centers = west + (np.arange(width) + 0.5) * x_size
        longitude_bins = np.searchsorted(longitude_edges, longitude_centers, side='right') - 1
        longitude_inside = (longitude_bins >= 0) & (longitude_bins < nlon)
        for row_start in range(0, height, block_rows):
            row_end = min(height, row_start + block_rows)
            tile = gpm_normalize_dem_block(reader.read_rows(row_start, row_end), row_end - row_start, width)
            row_numbers = np.arange(row_start, row_end)
            latitude_centers = north - (row_numbers + 0.5) * y_size
            latitude_bins = np.searchsorted(latitude_edges, latitude_centers, side='right') - 1
            latitude_inside = (latitude_bins >= 0) & (latitude_bins < nlat)
            spatial_inside = latitude_inside[:, None] & longitude_inside[None, :]
            valid_dem = spatial_inside & np.isfinite(tile) & ~np.isclose(tile, nodata, rtol=0.0, atol=1e-06)
            if not np.any(valid_dem):
                continue
            safe_latitude_bins = np.clip(latitude_bins, 0, nlat - 1)
            safe_longitude_bins = np.clip(longitude_bins, 0, nlon - 1)
            grid_index = safe_latitude_bins[:, None] * nlon + safe_longitude_bins[None, :]
            latitude_north = latitude_centers + y_size / 2.0
            latitude_south = latitude_centers - y_size / 2.0
            row_pixel_area = gpm_EARTH_RADIUS_M ** 2 * np.abs(np.sin(np.deg2rad(latitude_north)) - np.sin(np.deg2rad(latitude_south))) * np.deg2rad(x_size) / 1000000.0
            pixel_area = np.broadcast_to(row_pixel_area[:, None], tile.shape)
            valid_area_flat += np.bincount(grid_index[valid_dem], weights=pixel_area[valid_dem], minlength=output_size)
            valid_count_flat += np.bincount(grid_index[valid_dem], minlength=output_size).astype('int64')
            for band in gpm_BANDS:
                in_band = valid_dem & (tile >= band.lower_m) & (tile < band.upper_m)
                if not np.any(in_band):
                    continue
                band_area_flat[band.label] += np.bincount(grid_index[in_band], weights=pixel_area[in_band], minlength=output_size)
                band_count_flat[band.label] += np.bincount(grid_index[in_band], minlength=output_size).astype('int64')
    finally:
        reader.close()
    valid_area = valid_area_flat.reshape(nlat, nlon)
    valid_count = valid_count_flat.reshape(nlat, nlon)
    band_weights = {label: values.reshape(nlat, nlon) for label, values in band_area_flat.items()}
    band_counts = {label: values.reshape(nlat, nlon) for label, values in band_count_flat.items()}
    audit_rows: list[dict] = []
    for band in gpm_BANDS:
        weights = band_weights[band.label]
        positive = weights > 0.0
        weight_sum = float(np.sum(weights))
        kish = weight_sum ** 2 / float(np.sum(weights ** 2)) if weight_sum > 0.0 else 0.0
        fractions = np.divide(weights, cell_areas, out=np.zeros_like(weights), where=cell_areas > 0.0)
        audit_rows.append({'band_label': band.label, 'display_name': band.display_name, 'elevation_lower_m_inclusive': band.lower_m, 'elevation_upper_m_exclusive': band.upper_m, 'dem_pixel_count': int(np.sum(band_counts[band.label])), 'dem_band_area_km2': weight_sum, 'gpm_cells_intersecting_band': int(np.sum(positive)), 'full_gpm_cell_area_equivalents': float(np.sum(fractions)), 'kish_effective_gpm_cells': kish, 'maximum_band_fraction_in_one_gpm_cell': float(np.max(fractions[positive])) if np.any(positive) else 0.0})
    cell_rows: list[dict] = []
    valid_fraction = np.divide(valid_area, cell_areas, out=np.zeros_like(valid_area), where=cell_areas > 0.0)
    for iy, latitude in enumerate(latitudes):
        for ix, longitude in enumerate(longitudes):
            row = {'gpm_latitude': latitude, 'gpm_longitude': longitude, 'latitude_south': latitude_edges[iy], 'latitude_north': latitude_edges[iy + 1], 'longitude_west': longitude_edges[ix], 'longitude_east': longitude_edges[ix + 1], 'gpm_cell_area_km2': cell_areas[iy, ix], 'valid_dem_pixel_count': int(valid_count[iy, ix]), 'valid_dem_area_km2': valid_area[iy, ix], 'valid_dem_area_fraction': valid_fraction[iy, ix]}
            for band in gpm_BANDS:
                area = band_weights[band.label][iy, ix]
                row[f'{band.label}_dem_pixel_count'] = int(band_counts[band.label][iy, ix])
                row[f'{band.label}_area_km2'] = area
                row[f'{band.label}_cell_fraction'] = area / cell_areas[iy, ix] if cell_areas[iy, ix] > 0 else np.nan
            cell_rows.append(row)
    dem_metadata = {'path': str(dem_path), 'reader_backend': reader_backend, 'width': int(width), 'height': int(height), 'bounds_west_south_east_north': [west, south, east, north], 'pixel_size_degrees': [x_size, y_size], 'nodata': nodata, 'aggregation': '30 m DEM-pixel spherical area summed into each 0.1-degree GPM cell and elevation band'}
    return (band_weights, pd.DataFrame(audit_rows), pd.DataFrame(cell_rows), dem_metadata)

def gpm_harmonize_monthly_dictionary(fields: dict[tuple[int, int], xr.DataArray], template: xr.DataArray) -> dict[tuple[int, int], xr.DataArray]:
    """Place historical monthly fields on the same native IMERG grid."""
    return {key: gpm_align_to(field, template) for key, field in fields.items()}

def gpm_weighted_band_metrics(field: xr.DataArray, weights_km2: np.ndarray) -> dict:
    values = np.asarray(field.values, dtype='float64')
    if values.shape != weights_km2.shape:
        raise ValueError(f'Precipitation/DEM-weight shape mismatch: {values.shape} vs {weights_km2.shape}')
    use = np.isfinite(values) & np.isfinite(weights_km2) & (weights_km2 > 0.0)
    if not np.any(use):
        raise ValueError('No finite GPM cells contribute to an elevation band')
    weights = weights_km2[use]
    data = values[use]
    weight_sum = float(np.sum(weights))
    mean = float(np.sum(weights * data) / weight_sum)
    variance = float(np.sum(weights * (data - mean) ** 2) / weight_sum)
    kish = weight_sum ** 2 / float(np.sum(weights ** 2))
    return {'area_weighted_mean_mm': mean, 'area_weighted_sd_across_gpm_cells_mm': np.sqrt(max(0.0, variance)), 'minimum_contributing_gpm_cell_mm': float(np.min(data)), 'maximum_contributing_gpm_cell_mm': float(np.max(data)), 'contributing_gpm_cells': int(np.sum(use)), 'represented_dem_band_area_km2': weight_sum, 'kish_effective_gpm_cells': kish}

def gpm_build_historical_climatology(monthly_fields: dict[tuple[int, int], xr.DataArray], band_weights: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Calculate the 2010-2020 monthly climatology and annual JJAS totals."""
    climatology_rows: list[dict] = []
    annual_rows: list[dict] = []
    plot_data = {'mean': np.full((len(gpm_BANDS), len(gpm_MONTHS)), np.nan), 'interannual_sd': np.full((len(gpm_BANDS), len(gpm_MONTHS)), np.nan), 'jjas_total': np.full(len(gpm_BANDS), np.nan)}
    for band_index, band in enumerate(gpm_BANDS):
        weights = band_weights[band.label]
        year_month_values = np.full((len(gpm_HISTORICAL_YEARS), len(gpm_MONTHS)), np.nan)
        for year_index, year in enumerate(gpm_HISTORICAL_YEARS):
            for month_index, month in enumerate(gpm_MONTHS):
                metrics = gpm_weighted_band_metrics(monthly_fields[year, month], weights)
                year_month_values[year_index, month_index] = metrics['area_weighted_mean_mm']
            annual_rows.append({'year': year, 'band_label': band.label, 'display_name': band.display_name, 'elevation_lower_m_inclusive': band.lower_m, 'elevation_upper_m_exclusive': band.upper_m, 'jjas_total_area_weighted_mean_mm': float(np.sum(year_month_values[year_index])), **{f'month_{month:02d}_area_weighted_mean_mm': float(year_month_values[year_index, month_index]) for month_index, month in enumerate(gpm_MONTHS)}})
        for month_index, month in enumerate(gpm_MONTHS):
            values = year_month_values[:, month_index]
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1))
            plot_data['mean'][band_index, month_index] = mean
            plot_data['interannual_sd'][band_index, month_index] = sd
            climatology_rows.append({'period': '2010-2020', 'month': month, 'band_label': band.label, 'display_name': band.display_name, 'elevation_lower_m_inclusive': band.lower_m, 'elevation_upper_m_exclusive': band.upper_m, 'n_years': len(gpm_HISTORICAL_YEARS), 'climatological_monthly_mean_mm': mean, 'interannual_sd_mm': sd, 'interannual_se_mm': sd / np.sqrt(len(gpm_HISTORICAL_YEARS)), 'minimum_year_value_mm': float(np.min(values)), 'maximum_year_value_mm': float(np.max(values)), 'spatial_method': 'area-weighted IMERG mean using 30 m DEM band area inside every native IMERG cell'})
        plot_data['jjas_total'][band_index] = float(np.sum(plot_data['mean'][band_index]))
    return (pd.DataFrame(climatology_rows), pd.DataFrame(annual_rows), plot_data)

def check_bands() -> None:
    era_limits = tuple((int(b["low"]), int(b["high"])) for b in era_ELEVATION_BANDS)
    gpm_limits = tuple((int(b.lower_m), int(b.upper_m)) for b in gpm_BANDS)
    if era_limits != BAND_LIMITS or gpm_limits != BAND_LIMITS:
        raise ValueError(f"Inconsistent DEM bands: {era_limits}; {gpm_limits}")
    if era_HISTORICAL_YEARS != YEARS or gpm_HISTORICAL_YEARS != YEARS:
        raise ValueError("Historical years must be 2010–2020")


def era_climatology(grib: Path, dem: Path):
    print("Reading historical ERA5-Land monthly GRIB ...", flush=True)
    groups = era_open_grib_groups(grib)
    try:
        monthly, metadata = era_load_monthly_tp(groups, grib, YEARS)
        fields = era_monthly_field_dictionary(monthly, YEARS)
        template = fields[(2010, 6)]
        print("Aggregating DEM area onto ERA5-Land grid ...", flush=True)
        weights, audit, _, dem_meta = era_dem_band_weights(dem, template)
        summary, annual, _ = era_build_historical_climatology(fields, weights)
    finally:
        for group in groups:
            group.close()
    return summary, annual, audit, {"monthly": metadata, "dem": dem_meta}


def gpm_climatology(directory: Path, dem: Path):
    print("Discovering 44 historical GPM IMERG monthly files ...", flush=True)
    files = gpm_discover_historical_monthly(directory)
    fields, inventory = gpm_read_historical_monthly(files)
    template = fields[(2010, 6)]
    fields = gpm_harmonize_monthly_dictionary(fields, template)
    print("Aggregating DEM area onto GPM grid ...", flush=True)
    weights, audit, _, dem_meta = gpm_dem_band_weights(dem, template)
    summary, annual, _ = gpm_build_historical_climatology(fields, weights)
    return summary, annual, audit, {"monthly_file_inventory": inventory.to_dict("records"), "dem": dem_meta}


def combine(era_summary: pd.DataFrame, gpm_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for product, table in (("ERA5-Land", era_summary), ("GPM IMERG", gpm_summary)):
        lookup = table.set_index(["month", "elevation_lower_m_inclusive"])
        if len(lookup) != 12 or lookup.index.has_duplicates:
            raise ValueError(f"Unexpected {product} climatology rows")
        for low, high in BAND_LIMITS:
            result = {"product": product, "elevation_band_m": f"{low}-{high}"}
            for month, name in zip(MONTHS, MONTH_NAMES):
                row = lookup.loc[(month, low)]
                if int(row["elevation_upper_m_exclusive"]) != high or int(row["n_years"]) != 11:
                    raise ValueError(f"Incorrect {product} band or year count at {month}, {low}")
                mean = float(row["climatological_monthly_mean_mm"])
                if not np.isfinite(mean) or mean < 0:
                    raise ValueError(f"Invalid {product} value at {month}, {low}: {mean}")
                result[f"{name}_mm"] = mean
            rows.append(result)
    return pd.DataFrame(rows)


def plot(table: pd.DataFrame, path: Path, dpi: int, ymax: float) -> None:
    """Two monthly small multiples: shared y-scale and three touching bars/month."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 11, "font.weight": "normal",
        "axes.labelweight": "normal", "axes.unicode_minus": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    products = ("ERA5-Land", "GPM IMERG")
    maximum = float(table[[f"{name}_mm" for name in MONTH_NAMES]].to_numpy().max())
    upper = max(ymax, np.ceil(maximum / 50.0) * 50.0 + 50.0)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.0), sharey=True)
    # Both panels have the same scale so corresponding band heights can be compared.
    fig.subplots_adjust(left=0.105, right=0.98, bottom=0.17, top=0.91,
                        wspace=0.11)
    centers = np.arange(len(MONTHS), dtype=float)
    width = 0.15
    offsets = (np.arange(len(BAND_LIMITS)) - 1) * width
    for panel_index, (ax, product) in enumerate(zip(axes, products)):
        for band_index, (low, high) in enumerate(BAND_LIMITS):
            match = table.loc[(table["product"] == product) &
                              (table["elevation_band_m"] == f"{low}-{high}")]
            if len(match) != 1:
                raise ValueError(f"Expected one {product} row for {low}-{high} m")
            values = match[[f"{name}_mm" for name in MONTH_NAMES]].iloc[0].to_numpy(dtype=float)
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError(f"Invalid JJAS monthly values for {product}, {low}-{high} m")
            ax.bar(centers + offsets[band_index], values, width=width,
                   color=COLORS[product][band_index],
                   edgecolor=BAR_EDGES[product][band_index],
                   linewidth=0.28, zorder=3)
        ax.set_xlim(-0.35, centers[-1] + 0.35)
        ax.set_ylim(0, upper)
        ax.set_xticks(centers)
        ax.set_xticklabels(MONTH_NAMES, fontsize=11)
        handles = [Patch(facecolor=COLORS[product][i],
                         edgecolor=BAR_EDGES[product][i], linewidth=0.28,
                         label=f"{(low + high) // 2} ± 300 m")
                   for i, (low, high) in enumerate(BAND_LIMITS)]
        ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=10,
                  handlelength=1.05, handletextpad=0.5, labelspacing=0.3,
                  borderaxespad=0.65)
        ax.tick_params(axis="x", direction="out", length=4, width=0.75, pad=5)
        ax.tick_params(axis="y", direction="out", length=5, width=0.75,
                       labelsize=11, labelleft=panel_index == 0)
        ax.grid(False)
        for side in ("left", "bottom", "top", "right"):
            ax.spines[side].set_color("#888888")
            ax.spines[side].set_linewidth(0.75)
    axes[0].set_ylabel("Monthly precipitation (mm)", fontsize=13, labelpad=8)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = arguments()
    out = args.output_dir
    png = out / f"{STEM}.png"
    source = out / f"{STEM}_source_data.csv"
    if args.plot_only:
        if not source.is_file():
            raise FileNotFoundError(f"Cannot replot: source data CSV is missing: {source}")
        table = pd.read_csv(source)
        required = {"product", "elevation_band_m", *(f"{name}_mm" for name in MONTH_NAMES)}
        if not required.issubset(table.columns) or len(table) != 6:
            raise ValueError(f"Expected six rows with {sorted(required)} in {source}")
        expected_rows = {(product, f"{low}-{high}")
                         for product in ("ERA5-Land", "GPM IMERG")
                         for low, high in BAND_LIMITS}
        actual_rows = set(zip(table["product"], table["elevation_band_m"]))
        if actual_rows != expected_rows:
            raise ValueError("Source CSV elevation bands differ from this script; rerun without --plot-only")
        plot(table, png, args.dpi, args.ymax)
        print(f"PNG: {png}\nReused source data: {source}", flush=True)
        return
    check_bands()
    era_summary, _, _, _ = era_climatology(
        args.era5land_grib, args.dem)
    gpm_summary, _, _, _ = gpm_climatology(
        args.gpm_monthly_dir, args.dem)
    table = combine(era_summary, gpm_summary)
    table.to_csv(source, index=False, float_format="%.6f")
    plot(table, png, args.dpi, args.ymax)
    print(f"PNG: {png}\nFigure source data: {source}", flush=True)


if __name__ == "__main__":
    main()
