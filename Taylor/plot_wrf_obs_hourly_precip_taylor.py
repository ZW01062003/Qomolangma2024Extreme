#!/usr/bin/env python3
"""Everest 928: Taylor diagram for hourly OBS and 200 m WRF precipitation.

One PNG with two 90-degree Taylor panels (27 and 28 September BJT) and one
paired-hour CSV. Blue, yellow and gray points denote P5200/P5800/P6500.
Each panel has its own clearly labeled standard-deviation scale so a day
with SD ratio >8 cannot crowd the other day's ticks. Black dashed arcs show
normalized centered RMS error (cRMSE). WRF Times are UTC; OBS is already BJT.
No phase shift is applied. Hourly amounts have the END timestamp, so
(start, end] contains 48 hourly bins. The CSV preserves the paired hourly
input series. The correlation angle is rescaled to fit R=-0.2..1 inside a
90-degree panel. This is an angularly compressed Taylor-style diagram, not
the conventional arccos(R) geometry. Both station markers and centered-RMS
contours use the same angular projection.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STATIONS = {
    "P5200": (28.128911, 86.859978),
    "P5800": (28.085467, 86.914133),
    "P6500": (28.030633, 86.940464),
}
SITE_COLORS = {
    "P5200": "#0073C2",  # blue in the supplied openair example
    "P5800": "#EFC000",  # yellow
    "P6500": "#868686",  # gray
}
MIN_CORRELATION = -0.2  # Contains P5200 at roughly -0.13 on 28 September.
ANGLE_SCALE = (np.pi / 2) / np.arccos(MIN_CORRELATION)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wrf-file", required=True, type=Path,
                   help="One multi-time wrfout_d04 NetCDF file")
    p.add_argument("--obs-hourly-file", required=True, type=Path,
                   help="OBS CSV in mm per hour; naive timestamps are already BJT")
    p.add_argument("--out", required=True, type=Path, help="Output PNG")
    p.add_argument("--out-csv", type=Path,
                   help="Output CSV; default is the PNG path with a .csv suffix")
    p.add_argument("--start-bjt", default="2024-09-27 00:00")
    p.add_argument("--end-bjt", default="2024-09-29 00:00")
    p.add_argument("--dpi", type=int, default=600)
    a = p.parse_args()
    if a.out.suffix.lower() != ".png":
        p.error("--out must have a .png suffix")
    if a.out_csv is None:
        a.out_csv = a.out.with_suffix(".csv")
    if a.out_csv.suffix.lower() != ".csv":
        p.error("--out-csv must have a .csv suffix")
    if a.dpi < 1:
        p.error("--dpi must be positive")
    for key in ("start_bjt", "end_bjt"):
        setattr(a, key, pd.Timestamp(getattr(a, key)))
    if a.start_bjt >= a.end_bjt:
        p.error("--start-bjt must precede --end-bjt")
    if (a.end_bjt - a.start_bjt) / pd.Timedelta(hours=1) % 1:
        p.error("The event start/end must be separated by whole hours")
    return a


def wrf_utc_times(nc):
    if "Times" not in nc.variables:
        raise ValueError("The WRF file has no Times variable")
    strings = []
    for row in nc.variables["Times"][:]:
        if hasattr(row, "tobytes"):
            raw = row.tobytes().decode("ascii", errors="ignore")
        else:
            raw = b"".join(row).decode("ascii", errors="ignore")
        strings.append(raw.strip("\x00 "))
    return pd.DatetimeIndex(pd.to_datetime(strings, format="%Y-%m-%d_%H:%M:%S"))


def nearest_cells(nc):
    lat = np.asarray(nc.variables["XLAT"][0], dtype=float)
    lon = np.asarray(nc.variables["XLONG"][0], dtype=float)
    if lat.shape != lon.shape or lat.ndim != 2:
        raise ValueError("XLAT/XLONG must contain 2-D horizontal grids")
    cells = {}
    for station, (site_lat, site_lon) in STATIONS.items():
        distance2 = (lat - site_lat) ** 2 + (
            (lon - site_lon) * np.cos(np.deg2rad(site_lat))
        ) ** 2
        j, i = np.unravel_index(np.nanargmin(distance2), distance2.shape)
        km = 111.2 * np.sqrt(distance2[j, i])
        print(f"[grid] {station}: j={j}, i={i}, "
              f"lat={lat[j, i]:.6f}, lon={lon[j, i]:.6f}, distance={km:.3f} km")
        cells[station] = (int(j), int(i))
    return cells


def cumulative_precip(nc, record, j, i):
    total = 0.0
    for name in ("RAINNC", "RAINC", "RAINSH"):
        if name in nc.variables:
            raw = nc.variables[name][record, j, i]
            if np.ma.is_masked(raw):
                return np.nan
            total += float(raw)
    return total


def extract_wrf(path, start, end):
    try:
        from netCDF4 import Dataset
    except ImportError as exc:
        raise RuntimeError("The Python environment needs netCDF4") from exc
    with Dataset(path) as nc:
        if "RAINNC" not in nc.variables:
            raise ValueError("RAINNC is missing from the WRF file")
        times = wrf_utc_times(nc) + pd.Timedelta(hours=8)
        # A restart produced one repeated WRF hour: the later record wins.
        lookup = pd.Series(np.arange(len(times)), index=times)
        duplicate_count = int(lookup.index.duplicated().sum())
        lookup = lookup[~lookup.index.duplicated(keep="last")].sort_index()
        hours = pd.date_range(start + pd.Timedelta(hours=1), end, freq="h")
        previous = hours[0] - pd.Timedelta(hours=1)
        needed = pd.DatetimeIndex([previous]).append(hours)
        missing = needed.difference(lookup.index)
        if len(missing):
            raise ValueError(
                f"WRF lacks {len(missing)} required BJT times, "
                f"including {str(missing[:5].tolist())}. "
                "A previous hour is required to difference cumulative rain."
            )
        print(f"[time] WRF UTC -> BJT +8 h; {duplicate_count} duplicate record(s), "
              f"keeping the last; {len(hours)} right-closed hourly amounts; no lag")
        cells = nearest_cells(nc)
        result = pd.DataFrame(index=hours)
        result.index.name = "time_bjt"
        for station, (j, i) in cells.items():
            cumulative = np.array(
                [cumulative_precip(nc, int(lookup.loc[t]), j, i) for t in needed],
                dtype=float,
            )
            increments = np.diff(cumulative)
            reset = np.isfinite(increments) & (increments < -1e-4)
            if reset.any():
                reset_times = hours[reset]
                warnings.warn(
                    f"{station}: WRF rain accumulator decreases at "
                    f"{reset_times.tolist()}; these hourly values become missing"
                )
                increments[reset] = np.nan
            tiny_negative = np.isfinite(increments) & (increments < 0)
            increments[tiny_negative] = 0.0
            result[station] = increments
        return result


def obs_column(columns, station):
    names = [station.lower()]
    if station == "P6500":
        names.append("p6400")
    normalized = {c: str(c).lower().replace(" ", "").replace("_", "")
                  for c in columns}
    for name in names:
        exact = [c for c in columns if normalized[c] == name]
        if exact:
            if name == "p6400":
                warnings.warn("OBS column P6400 is interpreted as P6500")
            return exact[0]
        candidates = [c for c in columns if name in normalized[c] and
                      any(term in normalized[c] for term in ("precip", "rain", "obs", "mm"))]
        if candidates:
            if name == "p6400":
                warnings.warn("OBS column P6400 is interpreted as P6500")
            return candidates[0]
    raise ValueError(f"Cannot identify OBS column {station}; found {list(columns)}")


def read_obs(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.empty:
        raise ValueError("OBS CSV is empty")
    time_col = next((c for c in df.columns if any(
        token in str(c).lower() for token in ("time", "date", "bjt")
    )), df.columns[0])
    time = pd.to_datetime(df[time_col], errors="raise")
    if time.dt.tz is not None:
        time = time.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    if time.duplicated().any():
        raise ValueError("OBS has duplicate BJT timestamps")
    obs = pd.DataFrame(index=pd.DatetimeIndex(time))
    obs.index.name = "time_bjt"
    for station in STATIONS:
        col = obs_column(df.columns, station)
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if np.any(values[np.isfinite(values)] < 0):
            raise ValueError(f"OBS has negative precipitation at {station}")
        obs[station] = values
    return obs.sort_index()


def valid_stats(observed, modeled):
    both = np.isfinite(observed) & np.isfinite(modeled)
    x, y = observed[both], modeled[both]
    if len(x) == 0:
        raise ValueError("No exact paired OBS–WRF hours")
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    bias = float(np.mean(y - x))  # WRF minus OBS, mm per hourly bin
    r = float(np.corrcoef(x, y)[0, 1]) if (
        len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0
    ) else np.nan
    return x, y, r, rmse, bias


def panel_axis_max(points):
    """Round up only enough to show every WRF standard deviation ratio."""
    largest = max(point[4] for point in points)
    for limit in (1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0,
                  10.0, 12.0, 15.0, 20.0, 25.0, 30.0):
        if largest <= limit * 0.94:
            return limit
    return float(np.ceil(largest / 5.0) * 5.0 + 5.0)


def panel_tick_step(limit):
    if limit <= 2:
        return 0.5
    if limit <= 5:
        return 1.0
    if limit <= 12:
        return 2.0
    return 5.0


def crms_levels(limit):
    if limit <= 2:
        return (0.5, 1.0, 1.5)
    if limit <= 5:
        return tuple(np.arange(1.0, limit, 1.0))
    if limit <= 12:
        return tuple(level for level in (2.0, 5.0, 8.0) if level < limit)
    return tuple(np.arange(5.0, limit, 5.0))


def display_angle(correlation):
    """Map true R=-0.2..1 monotonically onto the visible 90-degree angle."""
    return np.arccos(np.clip(correlation, -1.0, 1.0)) * ANGLE_SCALE


def draw_taylor_panel(ax, points, limit, show_y_labels=True):
    """Two radial contour families in the same compressed-angle projection."""
    theta = np.linspace(0.0, np.pi / 2.0, 500)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.13 * limit, 1.16 * limit)
    ax.set_ylim(-0.14 * limit, 1.18 * limit)
    ax.set_axis_off()

    # Fine correlation guides. All have the same mapping as the observations.
    correlations = (-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                    0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)
    for corr in (-0.1, 0.0, 0.2, 0.4, 0.6, 0.8, 0.95):
        angle = display_angle(corr)
        ax.plot([0, limit * np.cos(angle)],
                [0, limit * np.sin(angle)],
                color="#DADADA", linewidth=0.40, zorder=0)

    # First contour family: normalized standard deviation, centered at (0, 0).
    major_step = panel_tick_step(limit)
    major_ticks = np.arange(0, limit + major_step * 0.4, major_step)
    for value in major_ticks:
        if (value > 0 and value < limit and
                not np.isclose(value, 1.0) and
                not (limit <= 2 and limit - value < 0.2)):
            ax.plot(value * np.cos(theta), value * np.sin(theta),
                    color="#D2D2D2", linewidth=0.48, zorder=1)
    # The OBS reference SD has radius one and is the only heavy inner arc.
    ax.plot(np.cos(theta), np.sin(theta),
            color="#252525", linewidth=1.35, zorder=3)

    # Second contour family: centered RMS error about REF=(1, 0).
    # Project the true circles point by point, just as the station R values are.
    phase = np.linspace(0, 2 * np.pi, 1800)
    for level in crms_levels(limit):
        true_x = 1.0 + level * np.cos(phase)
        true_y = level * np.sin(phase)
        radius = np.hypot(true_x, true_y)
        valid = ((radius > 0) & (true_y >= 0) & (radius <= limit) &
                 (true_x >= MIN_CORRELATION * radius))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.divide(true_x, radius, out=np.zeros_like(true_x),
                             where=radius > 0)
        projected_angle = display_angle(corr)
        px = radius * np.cos(projected_angle)
        py = radius * np.sin(projected_angle)
        ax.plot(np.where(valid, px, np.nan), np.where(valid, py, np.nan),
                color="#CBCBCB", linewidth=0.58, linestyle=(0, (3, 3)),
                zorder=2)

    # Boundary and orthogonal SD axes. REF remains at (1, 0).
    ax.plot(limit * np.cos(theta), limit * np.sin(theta),
            color="#242424", linewidth=0.90, zorder=5)
    ax.plot([0, limit], [0, 0], color="#242424", linewidth=0.90, zorder=5)
    ax.plot([0, 0], [0, limit], color="#242424", linewidth=0.90, zorder=5)

    for corr in correlations:
        angle = display_angle(corr)
        ux, uy = np.cos(angle), np.sin(angle)
        ax.plot([0.983 * limit * ux, limit * ux],
                [0.983 * limit * uy, limit * uy],
                color="#333333", linewidth=0.55, zorder=6)
        label = f"{corr:.2f}" if 0.95 <= corr < 1 else f"{corr:.1f}"
        ax.text(1.062 * limit * ux, 1.062 * limit * uy,
                label, fontsize=6.55, color="#282828", ha="center", va="center",
                rotation=-np.degrees(angle), rotation_mode="anchor")

    for value in major_ticks:
        if value > 0:
            ax.plot([value, value], [0, 0.018 * limit],
                    color="#333333", linewidth=0.55, zorder=6)
            ax.plot([0, 0.018 * limit], [value, value],
                    color="#333333", linewidth=0.55, zorder=6)
        tick_label = "REF" if np.isclose(value, 1.0) else f"{value:g}"
        ax.text(value, -0.065 * limit, tick_label,
                fontsize=6.8, color="#282828", ha="center", va="top")
        if value > 0 and show_y_labels:
            label_y = value - 0.045 * limit if np.isclose(value, limit) else value
            ax.text(-0.045 * limit, label_y, tick_label,
                    fontsize=6.8, color="#282828", ha="right", va="center")

    # The left panel uses two-unit ticks: add the radius-one REF to both axes.
    if not np.any(np.isclose(major_ticks, 1.0)):
        ax.text(1.0, -0.065 * limit, "REF", fontsize=6.7,
                color="#282828", ha="center", va="top")
        ax.text(-0.045 * limit, 1.0, "REF", fontsize=6.7,
                color="#282828", ha="right", va="center")

    ax.text(0.68 * limit, 0.88 * limit, "correlation",
            rotation=-45, ha="center", va="center", fontsize=6.7,
            color="#282828")

    ax.plot([1.0], [0], marker="o", markersize=4.2,
            markeredgewidth=0, color="#9259BD", linestyle="none",
            clip_on=False, zorder=7)
    for number, station, n, r, ratio, crmse, rmse, bias in points:
        if r < MIN_CORRELATION or r > 1.0:
            raise ValueError(f"{station}: R={r:.3f} is outside the angular scale")
        angle = display_angle(r)
        ax.plot([ratio * np.cos(angle)], [ratio * np.sin(angle)],
                marker="o", markersize=4.7, markeredgewidth=0,
                color=SITE_COLORS[station], linestyle="none", zorder=9)


def draw_station_legend(fig, axes):
    for i, station in enumerate(STATIONS):
        y = 0.744 - i * 0.056
        axes[0].plot([0.708], [y + 0.006], marker="o", markersize=4.3,
                     markeredgewidth=0, color=SITE_COLORS[station],
                     linestyle="none", transform=fig.transFigure,
                     clip_on=False, zorder=20)
        fig.text(0.725, y, station, fontsize=7.0, ha="left",
                 color="#282828", fontweight="normal")


def make_figure(obs, wrf, args):
    plt.rcParams.update({
        # Match the reference's light sans lettering on the computing node.
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Nimbus Sans", "Liberation Sans",
                            "DejaVu Sans"],
        "font.size": 8.0,
        "font.weight": "normal",
        "axes.labelweight": "normal",
        "text.color": "#282828",
        "axes.unicode_minus": False,
    })
    first_day = args.start_bjt
    midpoint = first_day + pd.Timedelta(days=1)
    if args.end_bjt - first_day != pd.Timedelta(days=2):
        raise ValueError("The two panels require exactly 48 BJT hours")
    hours = wrf.index
    csv_rows = []
    for station in STATIONS:
        csv_rows.append(pd.DataFrame({
            "time_bjt": hours.strftime("%Y-%m-%d %H:%M:%S"),
            "station": station,
            "obs_precip_mm": obs[station].reindex(hours).to_numpy(dtype=float),
            "wrf_precip_mm": wrf[station].to_numpy(dtype=float),
        }))

    points_by_panel = []
    for panel, lower, upper in ((0, first_day, midpoint),
                                (1, midpoint, args.end_bjt)):
        panel_hours = hours[(hours > lower) & (hours <= upper)]
        if len(panel_hours) != 24:
            raise ValueError(f"Day {panel + 1} needs 24 BJT hourly bins")
        points = []
        for number, station in enumerate(STATIONS, 1):
            x = obs[station].reindex(panel_hours).to_numpy(dtype=float)
            y = wrf[station].reindex(panel_hours).to_numpy(dtype=float)
            paired_obs, paired_wrf, r, rmse, bias = valid_stats(x, y)
            if len(paired_obs) < 3 or not np.isfinite(r):
                raise ValueError(f"Day {panel + 1} {station}: at least 3 "
                                 "variable paired hours are required")
            obs_sd = float(np.std(paired_obs, ddof=1))
            wrf_sd = float(np.std(paired_wrf, ddof=1))
            ratio = wrf_sd / obs_sd
            crmse = float(np.sqrt(max(0.0, 1 + ratio ** 2 - 2 * ratio * r)))
            points.append((number, station, len(paired_obs), r,
                           ratio, crmse, rmse, bias))
            print(f"[stats] {lower.strftime('%d Sep')} {station}: "
                  f"n={len(paired_obs)}, R={r:.4f}, "
                  f"SD(WRF)/SD(OBS)={ratio:.4f}, cRMSE/SD(OBS)={crmse:.4f}, "
                  f"RMSE={rmse:.4f} mm h-1, bias={bias:+.4f} mm h-1")
        points_by_panel.append(points)

    limits = [panel_axis_max(points) for points in points_by_panel]
    print(f"[scale] {first_day.strftime('%d Sep')} outer SD ratio={limits[0]:g}; "
          f"{midpoint.strftime('%d Sep')} outer SD ratio={limits[1]:g}; "
          "sparse major ticks")
    fig = plt.figure(figsize=(8.6, 4.45), facecolor="white")
    axes = [fig.add_axes([0.055, 0.155, 0.322, 0.690]),
            fig.add_axes([0.405, 0.155, 0.322, 0.690])]
    for ax, points, limit in zip(axes, points_by_panel, limits):
        draw_taylor_panel(ax, points, limit)

    # Each panel has its own nearby axis titles; the scales differ by day.
    for panel in range(2):
        fig.text(0.050 + 0.350 * panel, 0.515,
                 "Normalized standard deviation", fontsize=7.4,
                 color="#282828", fontweight="normal",
                 rotation=90, ha="center", va="center")
        fig.text(0.216 + 0.350 * panel, 0.173,
                 "Normalized standard deviation", fontsize=7.4,
                 color="#282828", fontweight="normal",
                 ha="center", va="center")
    draw_station_legend(fig, axes)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor="white",
                bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    print(f"[saved] {args.out}")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(csv_rows, ignore_index=True).to_csv(
        args.out_csv, index=False, encoding="utf-8-sig", float_format="%.6f",
    )
    print(f"[saved] {args.out_csv} ({sum(len(df) for df in csv_rows)} rows)")

def main():
    args = parse_args()
    for path in (args.wrf_file, args.obs_hourly_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    wrf = extract_wrf(args.wrf_file, args.start_bjt, args.end_bjt)
    obs = read_obs(args.obs_hourly_file)
    make_figure(obs, wrf, args)


if __name__ == "__main__":
    main()
