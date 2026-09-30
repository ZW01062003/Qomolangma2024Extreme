#!/usr/bin/env python3
"""
ERA5 hourly anomaly diagnostics for the 2024-09-28 Everest event using a
±3-day climatological window and 1991-2020 hourly samples.

Default target: 2024-09-28 16:00 BJT.
For each target BJT hour, the climatology consists of the same BJT hour on
7 calendar days (target day ±3 days) in each of 30 years (1991-2020):
7 × 30 = 210 samples.

Anomaly:
    2024 event hour - mean(210 climatological hourly samples)

Historical central-95% range:
    At each grid cell, compare the single 2024 event value with the 2.5th and
    97.5th percentiles of the 210 historical same-hour samples. Stippling marks
    values outside that range, not a Student's t-test of means or field-wide
    statistical significance. Anomalies are relative to the 210-sample mean.

Panels:
    (a) 200 hPa HGT anomaly + UV anomaly
    (b) 500 hPa specific humidity anomaly
    (c) 500 hPa HGT anomaly + UV anomaly
    (d) 300-500 hPa mean -omega anomaly (positive = anomalous ascent)
    (e) 850 hPa HGT anomaly + UV anomaly
    (f) vertically integrated moisture flux anomaly + moisture divergence anomaly

Panels a/c/e use the July 2026 reference blue-white-red height palette and
purple wind arrows where the historical range is exceeded. Panel b uses its short-wave radiation palette,
and panel d uses its vertical-velocity palette. Panel f retains its scalar
palette and uses red IVT arrows sampled from the user's reference where the
historical range is exceeded. All other finite arrows are black.
Scalar historical-95% exceedance areas are stippled with black dots.

The script writes exactly one PNG; sample counts are checked in memory.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np

G = 9.80665
BJT_OFFSET_HOURS = 8

VAR_CANDIDATES = {
    "z": ["z", "geopotential"],
    "u": ["u", "u_component_of_wind"],
    "v": ["v", "v_component_of_wind"],
    "w": ["w", "vertical_velocity"],
    "q": ["q", "specific_humidity"],
    "sp": ["sp", "surface_pressure"],
    "ivte": ["viwve", "p71.162", "vertical_integral_of_eastward_water_vapour_flux"],
    "ivtn": ["viwvn", "p72.162", "vertical_integral_of_northward_water_vapour_flux"],
    "vimd": ["vimd", "vimdf", "p84.162", "vertical_integral_of_divergence_of_moisture_flux"],
}

EVEREST_LON = 86 + 55/60 + 39.51/3600
EVEREST_LAT = 27 + 59/60 + 15.85/3600

def parse_extent(text: str) -> tuple[float, float, float, float]:
    vals = [float(v.strip()) for v in text.split(",")]
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("extent must be W,E,S,N")
    w, e, s, n = vals
    if e <= w or n <= s:
        raise argparse.ArgumentTypeError("extent must satisfy E>W and N>S")
    return w, e, s, n


def parse_levels(text: str) -> list[float]:
    vals = [float(v.strip()) for v in text.split(",")]
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("levels must be min,max,step")
    start, stop, step = vals
    if step <= 0 or stop <= start:
        raise argparse.ArgumentTypeError("levels require max>min and step>0")
    out = []
    x = start
    while x <= stop + 1e-10:
        out.append(round(x, 10))
        x += step
    return out


def parse_bjt(text: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H"):
        try:
            return datetime.strptime(text.strip(), fmt)
        except ValueError:
            pass
    raise argparse.ArgumentTypeError("target BJT must look like '2024-09-28 16:00'")


def _open_dataset(path: Path):
    import xarray as xr
    errors = []
    for engine in ("netcdf4", "h5netcdf", "scipy"):
        try:
            return xr.open_dataset(path, engine=engine)
        except Exception as exc:
            errors.append(f"{engine}: {exc}")
    raise RuntimeError(f"Cannot open {path}\n" + "\n".join(errors))


def normalize_coords(ds):
    ren = {}
    if "valid_time" in ds.coords and "time" not in ds.coords and "time" not in ds.dims:
        ren["valid_time"] = "time"
    if "date" in ds.coords and "time" not in ds.coords and "time" not in ds.dims:
        ren["date"] = "time"
    for name in ("pressure_level", "isobaricInhPa", "plev"):
        if (name in ds.coords or name in ds.dims) and "level" not in ds.coords and "level" not in ds.dims:
            ren[name] = "level"
            break
    if "lon" in ds.coords and "longitude" not in ds.coords:
        ren["lon"] = "longitude"
    if "lat" in ds.coords and "latitude" not in ds.coords:
        ren["lat"] = "latitude"
    if ren:
        ds = ds.rename(ren)

    # Squeeze only harmless singleton auxiliary dimensions.
    for dim in list(ds.dims):
        if dim not in {"time", "level", "latitude", "longitude"} and ds.sizes.get(dim, 0) == 1:
            ds = ds.isel({dim: 0}, drop=True)

    if "longitude" in ds.coords:
        lon = ds["longitude"]
        if float(lon.max()) > 180:
            ds = ds.assign_coords(longitude=(((lon + 180) % 360) - 180)).sortby("longitude")
        else:
            ds = ds.sortby("longitude")
    if "latitude" in ds.coords:
        ds = ds.sortby("latitude")
    if "time" not in ds.coords and "time" not in ds.dims:
        raise ValueError("No time coordinate found after normalization")
    return ds


def find_var(ds, logical: str) -> str:
    for name in VAR_CANDIDATES[logical]:
        if name in ds.data_vars:
            return name
    raise KeyError(f"Cannot find {logical}; available={list(ds.data_vars)}")


def bjt_times_from_utc(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values).astype("datetime64[ns]")
    return arr + np.timedelta64(BJT_OFFSET_HOURS, "h")


def year_window_datetimes(target_bjt: datetime, year: int, half_window: int) -> list[datetime]:
    center = datetime(year, target_bjt.month, target_bjt.day, target_bjt.hour, target_bjt.minute)
    return [center + timedelta(days=d) for d in range(-half_window, half_window + 1)]


def files_for_group(directory: Path, pattern: str) -> list[Path]:
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")
    return files


def select_climatology_group(
    directory: Path,
    pattern: str,
    target_bjt: datetime,
    start_year: int,
    end_year: int,
    half_window: int,
):
    """Load only the required same-hour ±window samples into memory."""
    import xarray as xr

    expected_per_year = 2 * half_window + 1
    all_parts = []
    counts: dict[int, int] = {y: 0 for y in range(start_year, end_year + 1)}

    for path in files_for_group(directory, pattern):
        ds = normalize_coords(_open_dataset(path))
        try:
            bjt = bjt_times_from_utc(ds["time"].values)
            mask = np.zeros(bjt.shape, dtype=bool)
            bjt_hour = bjt.astype("datetime64[h]")
            for year in range(start_year, end_year + 1):
                wanted = year_window_datetimes(target_bjt, year, half_window)
                for dt in wanted:
                    mask |= bjt_hour == np.datetime64(dt, "h")
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            sub = ds.isel(time=idx).load()
            sub = sub.assign_coords(time=("time", bjt[idx]))
            for tv in bjt[idx]:
                py = int(str(tv.astype("datetime64[Y]"))[:4])
                if py in counts:
                    counts[py] += 1
            all_parts.append(sub)
        finally:
            ds.close()

    if not all_parts:
        raise RuntimeError(f"No climatology samples selected from {directory}/{pattern}")
    out = xr.concat(all_parts, dim="time", data_vars="minimal", coords="minimal", compat="override").sortby("time")

    expected_total = expected_per_year * (end_year - start_year + 1)
    actual_total = int(out.sizes["time"])
    bad = {y: n for y, n in counts.items() if n != expected_per_year}
    if actual_total != expected_total or bad:
        raise RuntimeError(
            f"Climatology sample audit failed for {pattern}: expected {expected_total} total "
            f"({expected_per_year}/year), got {actual_total}; bad years={bad}"
        )
    return out, counts


def select_event_group(directory: Path, pattern: str, target_bjt: datetime):
    import xarray as xr

    target_hour = np.datetime64(target_bjt, "h")
    matches = []
    for path in files_for_group(directory, pattern):
        ds = normalize_coords(_open_dataset(path))
        try:
            bjt = bjt_times_from_utc(ds["time"].values)
            idx = np.flatnonzero(bjt.astype("datetime64[h]") == target_hour)
            if idx.size:
                sub = ds.isel(time=idx).load()
                sub = sub.assign_coords(time=("time", bjt[idx]))
                matches.append(sub)
        finally:
            ds.close()
    if not matches:
        raise RuntimeError(f"Target event time {target_bjt} BJT not found in {directory}/{pattern}")
    out = xr.concat(matches, dim="time", data_vars="minimal", coords="minimal", compat="override").sortby("time")
    if int(out.sizes["time"]) != 1:
        raise RuntimeError(f"Expected exactly one event sample for {target_bjt} BJT, got {out.sizes['time']}")
    return out.isel(time=0, drop=True)


def da_level(ds, logical: str, level: int | None = None):
    name = find_var(ds, logical)
    da = ds[name]
    if level is not None and ("level" in da.coords or "level" in da.dims):
        try:
            da = da.sel(level=level)
        except Exception:
            da = da.sel(level=str(level))
    return da


def empirical_anomaly_sig(event, clim, scale=1.0, min_valid_fraction=0.90):
    """Return anomaly and two-sided historical-95% exceedance mask.

    For each grid point:
      1) compute the climatological mean from the same-hour ±3-day samples;
      2) form the central historical interval [P2.5, P97.5] for each grid cell;
      3) flag a single event value below P2.5 or above P97.5.

    This is a pointwise historical-range check, not a t-test of group means.
    In particular, one event hour is not a second independent sample group.
    """
    n_total = int(clim.sizes["time"])
    mean = clim.mean("time", skipna=True)
    lower = clim.quantile(0.025, dim="time", skipna=True)
    upper = clim.quantile(0.975, dim="time", skipna=True)
    if "quantile" in lower.dims:
        lower = lower.squeeze("quantile", drop=True)
    if "quantile" in upper.dims:
        upper = upper.squeeze("quantile", drop=True)
    if "quantile" in lower.coords:
        lower = lower.reset_coords("quantile", drop=True)
    if "quantile" in upper.coords:
        upper = upper.reset_coords("quantile", drop=True)
    nvalid = clim.count("time")
    enough = (nvalid >= math.ceil(n_total * min_valid_fraction)) & np.isfinite(event)
    raw_anom = event - mean
    anom = raw_anom * scale
    sig = ((event < lower) | (event > upper)) & enough
    return anom, sig, enough, nvalid


def smooth_field(da, window: int):
    if window <= 1:
        return da
    return da.rolling(latitude=window, longitude=window, center=True, min_periods=1).mean()


def subset(da, extent):
    w, e, s, n = extent
    return da.sel(longitude=slice(w, e), latitude=slice(s, n))


def mesh(da):
    return np.meshgrid(da["longitude"].values, da["latitude"].values)


def terrain_masks(sp_clim, sp_event, pressure_hpa: int):
    threshold = pressure_hpa * 100.0
    clim_valid = sp_clim >= threshold
    event_valid = sp_event >= threshold
    return clim_valid, event_valid


def softened_cmap(name: str, blend: float = 0.24):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    base = plt.get_cmap(name)
    colors = base(np.linspace(0, 1, 256))
    colors[:, :3] = colors[:, :3] * (1.0 - blend) + blend
    return LinearSegmentedColormap.from_list(f"{name}_soft_{blend:.2f}", colors)


def reference_palettes():
    """Exact control points from everest_heat_20260701_05_sixpanel_v16."""
    from matplotlib.colors import LinearSegmentedColormap
    height = LinearSegmentedColormap.from_list(
        "reference_height_blue_red",
        ["#092747", "#185a91", "#60a9d5", "#cae9f4", "#fcfcfc",
         "#f7d7c8", "#eb947b", "#cf4c43", "#8c1232"], N=256)
    ascent = LinearSegmentedColormap.from_list(
        "reference_omega_blue_red",
        ["#1023e9", "#5c71ff", "#bbc1ff", "#fefefe",
         "#ffb9bc", "#fa5967", "#f21226"], N=256)
    radiation = LinearSegmentedColormap.from_list(
        "reference_olr_purple_orange",
        ["#290d4c", "#58367d", "#917bc2", "#cec7e6", "#faf9fe",
         "#fff8ee", "#f9d3a2", "#f39a37", "#b34b00"], N=256)
    return height, radiation, ascent


def setup_axes(ax, extent, projection, title: str):
    import cartopy.crs as ccrs
    ax.set_extent(extent, crs=projection)
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.0, color="none",
        x_inline=False, y_inline=False,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}
    # Panel titles: place them just OUTSIDE the axes at the upper-left corner,
    # matching the user-indicated reference position.
    ax.text(
        0.0, 1.045, title,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=11,
        fontweight="normal",
        color="black",
        clip_on=False,
        zorder=100,
    )


def geometry_bounds(geometries):
    bounds = [geom.bounds for geom in geometries if geom is not None and not geom.is_empty]
    if not bounds:
        raise ValueError("boundary has no drawable geometries")
    west = min(item[0] for item in bounds)
    south = min(item[1] for item in bounds)
    east = max(item[2] for item in bounds)
    north = max(item[3] for item in bounds)
    return float(west), float(south), float(east), float(north)


def bounds_look_lonlat(bounds):
    west, south, east, north = bounds
    return (-360.0 <= west <= 360.0 and -360.0 <= east <= 360.0
            and -90.0 <= south <= 90.0 and -90.0 <= north <= 90.0)


def geometry_type_counts(geometries):
    counts = {}
    for geom in geometries:
        if geom is None or geom.is_empty:
            continue
        key = getattr(geom, "geom_type", "Unknown")
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(f"{key}:{value}" for key, value in sorted(counts.items())) or "none"


BOUNDARY_CACHE = {}


def load_vector_geometries(boundary: Path, label: str = "boundary"):
    """Load the user's actual shapefile without requiring geopandas.

    Cartopy itself is already required by the plotting script, and its
    shapereader can read ordinary ESRI shapefiles through the lightweight
    pyshp backend.  This avoids adding geopandas/fiona as an extra server
    dependency.

    The supplied shapefiles are expected to be geographic lon/lat data.
    If their numerical bounds do not look like longitude/latitude, fail
    loudly rather than drawing a misplaced boundary.
    """
    cache_key = (str(boundary.resolve()), label)
    if cache_key in BOUNDARY_CACHE:
        return BOUNDARY_CACHE[cache_key]

    if boundary.suffix.lower() != ".shp":
        raise ValueError(f"{label} must be an ESRI .shp file: {boundary}")

    # A shapefile is a multi-file dataset.  .shp and .shx are essential;
    # .dbf is normally present as well.  Check explicitly so failures are clear.
    required_sidecars = [boundary.with_suffix(".shx")]
    missing = [str(item) for item in required_sidecars if not item.exists()]
    if missing:
        raise FileNotFoundError(
            f"{label} shapefile sidecar missing: {', '.join(missing)}"
        )

    from cartopy.io import shapereader

    try:
        reader = shapereader.Reader(str(boundary))
        geometries = [geom for geom in reader.geometries()
                      if geom is not None and not geom.is_empty]
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read {label} with Cartopy shapereader: {boundary}\n{exc}"
        ) from exc

    if not geometries:
        raise ValueError(f"{label} file is empty or has no drawable geometries: {boundary}")

    bounds = geometry_bounds(geometries)
    if not bounds_look_lonlat(bounds):
        raise ValueError(
            f"{label} bounds do not look like WGS84 lon/lat: {bounds}. "
            "Please provide a geographic lon/lat shapefile (EPSG:4326)."
        )

    # Read the .prj text for an informative console message.
    prj = boundary.with_suffix(".prj")
    if prj.exists():
        try:
            prj_text = prj.read_text(encoding="utf-8", errors="ignore").strip()
            crs_note = prj_text[:180] + ("..." if len(prj_text) > 180 else "")
        except Exception:
            crs_note = "PRJ present"
    else:
        crs_note = "PRJ missing; treated as lon/lat because bounds are geographic"

    info = {"geometries": geometries, "crs": crs_note, "bounds": bounds}
    BOUNDARY_CACHE[cache_key] = info
    return info


def plot_xy(ax, coords, projection, color: str, linewidth: float, zorder: int):
    if len(coords) < 2:
        return
    xs, ys = zip(*[(xy[0], xy[1]) for xy in coords])
    ax.plot(
        xs, ys,
        color=color,
        linewidth=linewidth,
        transform=projection,
        zorder=zorder,
        solid_capstyle="round",
        solid_joinstyle="round",
    )


def plot_geometry_boundary(ax, geom, projection, color: str, linewidth: float, zorder: int):
    if geom is None or geom.is_empty:
        return
    geom_type = getattr(geom, "geom_type", "")
    if geom_type == "Polygon":
        plot_xy(ax, list(geom.exterior.coords), projection, color, linewidth, zorder)
    elif geom_type == "MultiPolygon":
        for part in geom.geoms:
            plot_geometry_boundary(ax, part, projection, color, linewidth, zorder)
    elif geom_type in {"LineString", "LinearRing"}:
        plot_xy(ax, list(geom.coords), projection, color, linewidth, zorder)
    elif geom_type == "MultiLineString":
        for part in geom.geoms:
            plot_geometry_boundary(ax, part, projection, color, linewidth, zorder)
    elif geom_type == "GeometryCollection":
        for part in geom.geoms:
            plot_geometry_boundary(ax, part, projection, color, linewidth, zorder)


def add_vector_boundary(ax, boundary_path: str, projection, *, color: str,
                        linewidth: float, zorder: int):
    boundary = Path(boundary_path).expanduser()
    if not boundary.exists():
        raise FileNotFoundError(f"boundary file not found: {boundary}")

    info = load_vector_geometries(boundary, "country borders")
    for geom in info["geometries"]:
        plot_geometry_boundary(ax, geom, projection, color, linewidth, zorder)


def finish_axes(ax, projection, country_shapefile):
    add_vector_boundary(
        ax,
        country_shapefile,
        projection,
        color="0.35",
        linewidth=0.45,
        zorder=70,
    )
    ax.scatter(EVEREST_LON, EVEREST_LAT, marker="^", s=28,
               facecolors="black", edgecolors="none", linewidths=0,
               transform=projection, zorder=85)


def print_boundary_info(path_text: str | None, label: str):
    if not path_text:
        print(f"[boundary] {label}: not provided")
        return
    path = Path(path_text).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    info = load_vector_geometries(path, label)
    print(
        f"[boundary] {label}: crs={info['crs']}; bounds={info['bounds']}; "
        f"geometry_types={geometry_type_counts(info['geometries'])}"
    )

def add_stipple(ax, sig, extent, projection, step=4, size=1.4, alpha=0.55):
    ss = subset(sig, extent).isel(latitude=slice(None, None, step), longitude=slice(None, None, step))
    x, y = mesh(ss)
    m = np.asarray(ss.values, dtype=bool)
    ax.scatter(x[m], y[m], s=size, c="k", marker=".", alpha=alpha, linewidths=0,
               transform=projection, zorder=32)


def add_terrain_mask(ax, valid_event, extent, projection):
    invalid = subset(~valid_event, extent)
    x, y = mesh(invalid)
    arr = np.where(np.asarray(invalid.values, dtype=bool), 1.0, np.nan)
    ax.contourf(x, y, arr, levels=[0.5, 1.5], colors=["0.88"], alpha=0.42,
                transform=projection, zorder=5)


def add_threshold_vectors(ax, u, v, exceedance, extent, projection, step,
                          reference, label, exceed_color, *, width,
                          key_x, key_y, labelpos, key_fontsize, scale_factor=12.0):
    """Black arrows inside the historical range, colored arrows outside it."""
    uu = subset(u, extent).isel(latitude=slice(None, None, step), longitude=slice(None, None, step))
    vv = subset(v, extent).isel(latitude=slice(None, None, step), longitude=slice(None, None, step))
    ss = subset(exceedance, extent).isel(latitude=slice(None, None, step),
                                        longitude=slice(None, None, step))
    x, y = mesh(uu)
    finite = np.isfinite(uu.values) & np.isfinite(vv.values)
    if not np.any(finite):
        return
    flagged = finite & np.asarray(ss.values, dtype=bool)
    ordinary = finite & ~flagged
    artists = []
    for mask, color, zorder in ((ordinary, "#000000", 34),
                                (flagged, exceed_color, 36)):
        if np.any(mask):
            artists.append((color, ax.quiver(
                x[mask], y[mask], uu.values[mask], vv.values[mask],
                color=color, alpha=0.96, scale=reference * scale_factor, width=width,
                headwidth=3.7, headlength=4.4, headaxislength=4.0,
                transform=projection, zorder=zorder)))
    key_artist = next((artist for color, artist in artists if color == exceed_color),
                      artists[0][1])
    ax.quiverkey(key_artist, key_x, key_y, reference, label,
                 labelpos=labelpos, coordinates="axes", color=exceed_color,
                 labelcolor="0.15", fontproperties={"size": key_fontsize})


def main():
    parser = argparse.ArgumentParser(description="ERA5 hourly ±3-day-window anomaly figure with a central historical-95% range.")
    parser.add_argument("--pre-clim", default="/home/phadcloud86gn14506/928/data/pre/19912020")
    parser.add_argument("--pre-event", default="/home/phadcloud86gn14506/928/data/pre/2024")
    parser.add_argument("--single-clim", default="/home/phadcloud86gn14506/928/data/single/19912020")
    parser.add_argument("--single-event", default="/home/phadcloud86gn14506/928/data/single/2024")
    parser.add_argument("--target-bjt", type=parse_bjt, default=parse_bjt("2024-09-28 16:00"))
    parser.add_argument("--start-year", type=int, default=1991)
    parser.add_argument("--end-year", type=int, default=2020)
    parser.add_argument("--window-days", type=int, default=3)
    parser.add_argument("--extent", type=parse_extent, default=parse_extent("65,105,10,40"))
    parser.add_argument("--country-shapefile", required=True)
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--quiver-step", type=int, default=10)
    parser.add_argument("--hgt-quiver-step", type=int, default=6,
                        help="Arrow grid step for panels a/c/e")
    parser.add_argument("--hgt-quiver-scale", type=float, default=40.0,
                        help="Arrow scale multiplier for panels a/c/e; higher means shorter")
    parser.add_argument("--hgt200-levels", type=parse_levels, default=parse_levels("-320,320,40"))
    parser.add_argument("--hgt500-levels", type=parse_levels, default=parse_levels("-260,260,40"))
    parser.add_argument("--hgt850-levels", type=parse_levels, default=parse_levels("-220,220,40"))
    parser.add_argument("--q-levels", type=parse_levels, default=parse_levels("-6,6,1"))
    parser.add_argument("--omega-levels", type=parse_levels, default=parse_levels("-0.4,0.4,0.05"))
    parser.add_argument("--mdiv-levels", type=parse_levels, default=parse_levels("-80,80,10"))
    parser.add_argument("--mdiv-plot-scale", type=float, default=1e5)
    parser.add_argument("--panel-f-reference", type=float, default=400.0)
    parser.add_argument("--figsize", default="10.2,10.9")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--out", required=True)
    # argparse treats comma-separated negative ranges such as "-320,320,40"
    # as option-like tokens when passed after a space. Accept both that form
    # and the conventional --hgt200-levels=-320,320,40 form.
    level_options = {
        "--hgt200-levels", "--hgt500-levels", "--hgt850-levels",
        "--q-levels", "--omega-levels", "--mdiv-levels",
    }
    argv = sys.argv[1:]
    i = 0
    while i < len(argv) - 1:
        if argv[i] in level_options and argv[i + 1].startswith("-") and "," in argv[i + 1]:
            argv[i:i + 2] = [f"{argv[i]}={argv[i + 1]}"]
        i += 1
    args = parser.parse_args(argv)

    if Path(args.out).suffix.lower() != ".png":
        parser.error("--out must be a .png file")

    if args.end_year < args.start_year:
        parser.error("end-year must be >= start-year")
    if args.quiver_step < 1:
        parser.error("quiver-step must be positive")
    if args.hgt_quiver_step < 1 or args.hgt_quiver_scale <= 0:
        parser.error("hgt-quiver-step and hgt-quiver-scale must be positive")
    expected_n = (2*args.window_days + 1) * (args.end_year - args.start_year + 1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import xarray as xr

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
    })

    pre_clim = Path(args.pre_clim)
    pre_event = Path(args.pre_event)
    single_clim = Path(args.single_clim)
    single_event = Path(args.single_event)

    print(f"[target] BJT={args.target_bjt}; UTC={args.target_bjt - timedelta(hours=8)}")
    print(f"[climatology] years={args.start_year}-{args.end_year}; ±{args.window_days} days; expected n={expected_n}")
    print("[threshold] one event hour compared with pointwise historical P2.5–P97.5 interval; not a Student t-test")

    # Load single-level first because surface pressure is required for terrain masks.
    print("[load] climatology single-level samples")
    sl_clim, _ = select_climatology_group(single_clim, "era5_sl_flux_sp_*.nc", args.target_bjt,
                                                       args.start_year, args.end_year, args.window_days)
    print("[load] event single-level sample")
    sl_event = select_event_group(single_event, "era5_sl_flux_sp_*_event.nc", args.target_bjt)

    print("[load] climatology Z/U/V samples")
    zuv_clim, _ = select_climatology_group(pre_clim, "era5_pl_zuv_*.nc", args.target_bjt,
                                            args.start_year, args.end_year, args.window_days)
    print("[load] event Z/U/V sample")
    zuv_event = select_event_group(pre_event, "era5_pl_zuv_*_event.nc", args.target_bjt)

    print("[load] climatology q500 samples")
    q_clim_ds, _ = select_climatology_group(pre_clim, "era5_pl_q500_*.nc", args.target_bjt,
                                             args.start_year, args.end_year, args.window_days)
    print("[load] event q500 sample")
    q_event_ds = select_event_group(pre_event, "era5_pl_q500_*_event.nc", args.target_bjt)

    print("[load] climatology omega samples")
    om_clim_ds, _ = select_climatology_group(pre_clim, "era5_pl_omega_*.nc", args.target_bjt,
                                              args.start_year, args.end_year, args.window_days)
    print("[load] event omega sample")
    om_event_ds = select_event_group(pre_event, "era5_pl_omega_*_event.nc", args.target_bjt)

    # Cross-check all selected groups have identical sample counts.
    for label, ds in [("single", sl_clim), ("zuv", zuv_clim), ("q500", q_clim_ds), ("omega", om_clim_ds)]:
        n = int(ds.sizes.get("time", 0))
        if n != expected_n:
            raise RuntimeError(f"{label} climatology has n={n}, expected {expected_n}")
        print(f"[audit] {label}: n={n}")

    sp_clim = da_level(sl_clim, "sp")
    sp_event = da_level(sl_event, "sp")

    def hgt_uv_fields(level: int):
        clim_valid, event_valid = terrain_masks(sp_clim, sp_event, level)
        zc = da_level(zuv_clim, "z", level).where(clim_valid)
        ze = da_level(zuv_event, "z", level).where(event_valid)
        uc = da_level(zuv_clim, "u", level).where(clim_valid)
        ue = da_level(zuv_event, "u", level).where(event_valid)
        vc = da_level(zuv_clim, "v", level).where(clim_valid)
        ve = da_level(zuv_event, "v", level).where(event_valid)
        za, zsig, zvalid, _ = empirical_anomaly_sig(ze, zc, scale=1.0/G)
        ua, usig, uvalid, _ = empirical_anomaly_sig(ue, uc)
        va, vsig, vvalid, _ = empirical_anomaly_sig(ve, vc)
        valid = zvalid & uvalid & vvalid & event_valid
        return (
            smooth_field(za.where(valid), args.smooth_window),
            zsig & valid,
            smooth_field(ua.where(valid), args.smooth_window),
            smooth_field(va.where(valid), args.smooth_window),
            (usig | vsig) & valid,
            valid,
            event_valid,
        )

    a_z, a_zsig, a_u, a_v, a_uvsig, a_valid, a_event_valid = hgt_uv_fields(200)
    c_z, c_zsig, c_u, c_v, c_uvsig, c_valid, c_event_valid = hgt_uv_fields(500)
    e_z, e_zsig, e_u, e_v, e_uvsig, e_valid, e_event_valid = hgt_uv_fields(850)

    # 500-hPa specific humidity.
    q_clim_valid, q_event_valid = terrain_masks(sp_clim, sp_event, 500)
    q_c = da_level(q_clim_ds, "q", 500).where(q_clim_valid)
    q_e = da_level(q_event_ds, "q", 500).where(q_event_valid)
    b_q, b_qsig, b_valid, _ = empirical_anomaly_sig(q_e, q_c, scale=1000.0)
    b_q = smooth_field(b_q.where(b_valid & q_event_valid), args.smooth_window)
    b_valid = b_valid & q_event_valid
    b_qsig = b_qsig & b_valid

    # 300-500 hPa mean omega. Use the 500-hPa validity mask for all three levels.
    om_clim_valid, om_event_valid = terrain_masks(sp_clim, sp_event, 500)
    wc_list = [da_level(om_clim_ds, "w", lev).where(om_clim_valid) for lev in (300, 400, 500)]
    we_list = [da_level(om_event_ds, "w", lev).where(om_event_valid) for lev in (300, 400, 500)]
    wc = xr.concat(wc_list, dim="omega_level").mean("omega_level", skipna=True)
    we = xr.concat(we_list, dim="omega_level").mean("omega_level", skipna=True)
    d_om, d_sig, d_valid, _ = empirical_anomaly_sig(we, wc)
    d_om = -smooth_field(d_om.where(d_valid & om_event_valid), args.smooth_window)
    d_valid = d_valid & om_event_valid
    d_sig = d_sig & d_valid

    # Panel F: IVT and VIMD, no pressure-level terrain mask.
    ivte_a, ivte_sig, ivte_valid, _ = empirical_anomaly_sig(da_level(sl_event, "ivte"), da_level(sl_clim, "ivte"))
    ivtn_a, ivtn_sig, ivtn_valid, _ = empirical_anomaly_sig(da_level(sl_event, "ivtn"), da_level(sl_clim, "ivtn"))
    vimd_a, vimd_sig, vimd_valid, _ = empirical_anomaly_sig(da_level(sl_event, "vimd"), da_level(sl_clim, "vimd"))
    f_u = smooth_field(ivte_a.where(ivte_valid), args.smooth_window)
    f_v = smooth_field(ivtn_a.where(ivtn_valid), args.smooth_window)
    f_uvsig = (ivte_sig | ivtn_sig) & ivte_valid & ivtn_valid
    f_div = smooth_field(vimd_a.where(vimd_valid), args.smooth_window) * args.mdiv_plot_scale

    # Plot.
    projection = ccrs.PlateCarree()
    # Validate the country-border shapefile before rendering any panel.
    print_boundary_info(args.country_shapefile, "country borders")
    fw, fh = [float(x.strip()) for x in args.figsize.split(",", 1)]
    fig, axes = plt.subplots(3, 2, figsize=(fw, fh), subplot_kw={"projection": projection}, constrained_layout=False)
    # Layout tuned to match the approved v38 reference style more closely:
    # larger panel area, slightly tighter row/column spacing, and more top room
    # so the first row and quiver-key labels are not clipped.
    fig.subplots_adjust(left=0.052, right=0.985, top=0.965, bottom=0.055, hspace=0.24, wspace=0.14)

    hgt_cmap, q_cmap, om_cmap = reference_palettes()
    moist_cmap = softened_cmap("BrBG_r", 0.24)
    # Moisture transport keeps its approved layout. The upper-air wind uses a
    # slightly wider grid and much shorter shafts so neighboring arrows remain
    # distinct even where wind anomalies are several times the key value.
    ivt_vector_step = max(1, int(round(args.quiver_step * 0.5)))

    def plot_hgt(ax, z, zsig, u, v, uvsig, valid, event_valid, title, levels, reference):
        setup_axes(ax, args.extent, projection, title)
        zp = subset(z, args.extent); x, y = mesh(zp)
        cf = ax.contourf(x, y, zp.values, levels=levels, cmap=hgt_cmap, extend="both", transform=projection)
        add_stipple(ax, zsig, args.extent, projection, step=4)
        add_terrain_mask(ax, event_valid, args.extent, projection)
        add_threshold_vectors(ax, u, v, uvsig, args.extent, projection,
                              args.hgt_quiver_step, reference,
                              rf"${reference:g}\ \mathrm{{m\,s^{{-1}}}}$",
                              "#713b9c", width=0.0015,
                              key_x=0.77, key_y=1.035, labelpos="E", key_fontsize=7.8,
                              scale_factor=args.hgt_quiver_scale)
        finish_axes(ax, projection, args.country_shapefile)
        return cf

    cf_a = plot_hgt(axes[0,0], a_z, a_zsig, a_u, a_v, a_uvsig, a_valid, a_event_valid,
                    "(a) 200 hPa HGT & UV", args.hgt200_levels, 10.0)

    setup_axes(axes[0,1], args.extent, projection, "(b) 500 hPa q")
    bp = subset(b_q, args.extent); x, y = mesh(bp)
    cf_b = axes[0,1].contourf(x, y, bp.values, levels=args.q_levels, cmap=q_cmap, extend="both", transform=projection)
    add_stipple(axes[0,1], b_qsig, args.extent, projection, step=4)
    add_terrain_mask(axes[0,1], q_event_valid, args.extent, projection)
    finish_axes(axes[0,1], projection, args.country_shapefile)

    cf_c = plot_hgt(axes[1,0], c_z, c_zsig, c_u, c_v, c_uvsig, c_valid, c_event_valid,
                    "(c) 500 hPa HGT & UV", args.hgt500_levels, 5.0)

    setup_axes(axes[1,1], args.extent, projection, "(d) 300-500 hPa -ω")
    dp = subset(d_om, args.extent); x, y = mesh(dp)
    cf_d = axes[1,1].contourf(x, y, dp.values, levels=args.omega_levels, cmap=om_cmap, extend="both", transform=projection)
    add_stipple(axes[1,1], d_sig, args.extent, projection, step=4)
    add_terrain_mask(axes[1,1], om_event_valid, args.extent, projection)
    finish_axes(axes[1,1], projection, args.country_shapefile)

    cf_e = plot_hgt(axes[2,0], e_z, e_zsig, e_u, e_v, e_uvsig, e_valid, e_event_valid,
                    "(e) 850 hPa HGT & UV", args.hgt850_levels, 5.0)

    setup_axes(axes[2,1], args.extent, projection, "(f) Moisture Flux & Div")
    fp = subset(f_div, args.extent); x, y = mesh(fp)
    cf_f = axes[2,1].contourf(x, y, fp.values, levels=args.mdiv_levels, cmap=moist_cmap, extend="both", transform=projection)
    add_stipple(axes[2,1], vimd_sig & vimd_valid, args.extent, projection, step=6, size=1.0, alpha=0.35)
    # Only IVT values outside the historical range are red.
    ref = args.panel_f_reference
    ref_label = rf"${ref:g}\ \mathrm{{kg\,m^{{-1}}\,s^{{-1}}}}$"
    add_threshold_vectors(axes[2,1], f_u, f_v, f_uvsig, args.extent, projection,
                          ivt_vector_step, ref, ref_label, "#D7191C", width=0.0018,
                          key_x=0.80, key_y=1.022, labelpos="N", key_fontsize=7.1)
    finish_axes(axes[2,1], projection, args.country_shapefile)

    cbar_specs = [
        (cf_a, axes[0,0], "gpm"), (cf_b, axes[0,1], r"$\mathrm{g\,kg^{-1}}$"),
        (cf_c, axes[1,0], "gpm"), (cf_d, axes[1,1], r"$-\omega\ (\mathrm{Pa\,s^{-1}})$"),
        (cf_e, axes[2,0], "gpm"),
        (cf_f, axes[2,1], r"$10^{-5}\ \mathrm{kg\,m^{-2}\,s^{-1}}$"),
    ]
    for cf, ax, label in cbar_specs:
        cb = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.085, fraction=0.047, aspect=34)
        cb.ax.tick_params(labelsize=7, length=2)
        cb.set_label(label, fontsize=7, labelpad=1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor="white")
    plt.close(fig)
    print(f"[saved] figure: {out}")


if __name__ == "__main__":
    main()
