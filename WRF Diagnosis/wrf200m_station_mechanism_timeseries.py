#!/usr/bin/env python3
"""Six-panel station diagnostic PNG and its plotted values as CSV.

WRF Times are UTC and become BJT only by adding eight hours. WRF and OBS are
then compared at identical BJT timestamps, with no empirical time delay.
The six plotted quantities match the station-series panel of the user's v15
diagnostic: hourly precipitation, q, RH, W, upslope wind, and q*upslope wind.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


G = 9.80665
EPS = 0.622
RCP = 0.2854
STATIONS = {
    "P5200": (28.128911, 86.859978),
    "P5800": (28.085467, 86.914133),
    "P6500": (28.030633, 86.940464),
}

PRECIP_STYLE = {
    "P5200": "#111111",
    "P5800": "#d62728",
    "P6500": "#858585",
}
MET_STYLE = {
    "P5200": ("#111111", "-"),
    "P5800": ("#d62728", "-"),
    "P6500": ("#111111", "--"),
}


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wrf-file", required=True, type=Path)
    p.add_argument("--obs-hourly-file", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path, help="Output PNG path")
    p.add_argument("--csv", type=Path,
                   help="Plotted hourly values; default: --out with a .csv suffix")
    p.add_argument("--start-bjt", default="2024-09-27 00:00")
    p.add_argument("--end-bjt", default="2024-09-29 00:00")
    p.add_argument("--low-levels", type=int, default=3)
    p.add_argument("--dx", type=float, default=None)
    p.add_argument("--dy", type=float, default=None)
    p.add_argument("--dpi", type=int, default=600)
    a = p.parse_args()
    if a.out.suffix.lower() != ".png":
        p.error("--out must end in .png")
    if a.csv is None:
        a.csv = a.out.with_suffix(".csv")
    if a.csv.suffix.lower() != ".csv":
        p.error("--csv must end in .csv")
    if a.low_levels < 1 or a.dpi < 1:
        p.error("--low-levels and --dpi must be positive")
    a.start_bjt = pd.Timestamp(a.start_bjt)
    a.end_bjt = pd.Timestamp(a.end_bjt)
    if a.start_bjt >= a.end_bjt:
        p.error("--start-bjt must precede --end-bjt")
    return a


def decode_wrf_times(nc):
    values = nc.variables["Times"][:]
    strings = []
    for row in values:
        if hasattr(row, "tobytes"):
            raw = row.tobytes().decode("ascii", errors="ignore")
        else:
            raw = b"".join(row).decode("ascii", errors="ignore")
        strings.append(raw.strip("\x00 "))
    return pd.DatetimeIndex(pd.to_datetime(strings, format="%Y-%m-%d_%H:%M:%S"))


def station_grid(nc):
    lat = np.asarray(nc.variables["XLAT"][0], dtype=float)
    lon = np.asarray(nc.variables["XLONG"][0], dtype=float)
    if "HGT" in nc.variables:
        hgt = np.asarray(nc.variables["HGT"][0], dtype=float)
    else:
        hgt = np.asarray(nc.variables["HGT_M"][0], dtype=float)
    locations = {}
    for name, (slat, slon) in STATIONS.items():
        distance2 = (lat - slat)**2 + ((lon - slon) * np.cos(np.deg2rad(slat)))**2
        j, i = np.unravel_index(np.nanargmin(distance2), distance2.shape)
        locations[name] = (int(j), int(i))
        print(f"[grid] {name}: ({float(lat[j,i]):.5f}, {float(lon[j,i]):.5f}), "
              f"terrain {float(hgt[j,i]):.0f} m")
    return locations, hgt


def observation_time_column(df):
    for column in df.columns:
        name = str(column).lower()
        if "time" in name or "date" in name or "bjt" in name:
            return column
    return df.columns[0]


def observation_precip_column(df, station):
    station_names = [station.lower()]
    if station == "P6500":
        station_names.append("p6400")  # Older OBS files use this column name.
    for candidate in station_names:
        matching = [column for column in df.columns
                    if candidate in str(column).lower() and
                    any(s in str(column).lower() for s in ("precip", "rain", "obs", "mm"))]
        if matching:
            return matching[0]
        matching = [column for column in df.columns if str(column).lower() == candidate]
        if matching:
            return matching[0]
    raise ValueError(f"OBS column for {station} not found; available={list(df.columns)}")


def read_observations(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    time_col = observation_time_column(df)
    time = pd.to_datetime(df[time_col], errors="raise")
    # Timezone-aware input can be converted; naive input is already BJT.
    if time.dt.tz is not None:
        time = time.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    obs = pd.DataFrame({"time_bjt": time})
    for station in STATIONS:
        col = observation_precip_column(df, station)
        obs[station] = pd.to_numeric(df[col], errors="coerce")
    if obs["time_bjt"].duplicated().any():
        raise ValueError("OBS has duplicate BJT timestamps; fix these before pairing")
    return obs.set_index("time_bjt").sort_index()


def cumulative_rain_at(nc, tidx, j, i):
    total = float(nc.variables["RAINNC"][tidx, j, i])
    for name in ("RAINC", "RAINSH"):
        if name in nc.variables:
            total += float(nc.variables[name][tidx, j, i])
    return total


def station_at_time(nc, tidx, j, i, low_levels, slope_x, slope_y):
    q = np.asarray(nc.variables["QVAPOR"][tidx, :low_levels, j, i], dtype=float)
    p = (np.asarray(nc.variables["P"][tidx, :low_levels, j, i], dtype=float) +
         np.asarray(nc.variables["PB"][tidx, :low_levels, j, i], dtype=float))
    theta = np.asarray(nc.variables["T"][tidx, :low_levels, j, i], dtype=float) + 300.0
    temp_c = theta * (p / 100000.0)**RCP - 273.15
    es = 611.2 * np.exp(17.67 * temp_c / (temp_c + 243.5))
    qvs = EPS * es / np.maximum(p - es, 1.0)
    rh = np.clip(100.0 * q / np.maximum(qvs, 1e-12), 0.0, 150.0)

    u_edges = np.asarray(nc.variables["U"][tidx, :low_levels, j, i:i+2], dtype=float)
    v_edges = np.asarray(nc.variables["V"][tidx, :low_levels, j:j+2, i], dtype=float)
    w_edges = np.asarray(nc.variables["W"][tidx, :low_levels+1, j, i], dtype=float)
    u_low = float(np.mean(0.5 * (u_edges[:, 0] + u_edges[:, 1])))
    v_low = float(np.mean(0.5 * (v_edges[:, 0] + v_edges[:, 1])))
    w_low = float(np.mean(0.5 * (w_edges[:-1] + w_edges[1:])))
    mag = float(np.hypot(slope_x, slope_y))
    ex, ey = (slope_x / mag, slope_y / mag) if mag > 1e-8 else (0.0, 0.0)
    upslope = u_low * ex + v_low * ey
    q_gkg = float(np.mean(q) * 1000.0)
    return {
        "q_gkg": q_gkg,
        "rh_percent": float(np.mean(rh)),
        "w_low_ms": w_low,
        "upslope_wind_ms": upslope,
        "moist_upslope_gkg_ms": q_gkg * upslope,
    }


def extract(nc, args):
    times_utc = decode_wrf_times(nc)
    times_bjt = times_utc + pd.Timedelta(hours=8)
    locations, terrain = station_grid(nc)
    dx = float(args.dx if args.dx is not None else nc.DX)
    dy = float(args.dy if args.dy is not None else nc.DY)
    dhdy, dhdx = np.gradient(terrain, dy, dx)

    # WRF restart may duplicate a timestamp. Retain the last record of that
    # physical hour. Precipitation uses the previous distinct physical hour.
    unique_time = pd.Series(np.arange(len(times_bjt)), index=times_bjt)
    unique_time = unique_time[~unique_time.index.duplicated(keep="last")].sort_index()
    wanted = unique_time[(unique_time.index > args.start_bjt) &
                         (unique_time.index <= args.end_bjt)]
    if len(wanted) < 2:
        raise RuntimeError("Fewer than two WRF hours in the requested BJT interval")
    if wanted.index.to_series().diff().dropna().ne(pd.Timedelta(hours=1)).any():
        raise RuntimeError("WRF event hours are not continuous after duplicate removal")

    rows = []
    for time_bjt, index in wanted.items():
        prev_time = time_bjt - pd.Timedelta(hours=1)
        prev_index = int(unique_time.loc[prev_time]) if prev_time in unique_time.index else None
        for station, (j, i) in locations.items():
            now = cumulative_rain_at(nc, int(index), j, i)
            previous = cumulative_rain_at(nc, prev_index, j, i) if prev_index is not None else np.nan
            rain = now - previous
            if rain < 0:  # Accumulator reset: do not turn a reset into rainfall.
                rain = np.nan
            metrics = station_at_time(nc, int(index), j, i, args.low_levels,
                                      dhdx[j, i], dhdy[j, i])
            rows.append({"time_bjt": time_bjt, "station": station,
                         "wrf_precip_mm_h": rain, **metrics})
    print(f"[time] {len(wanted)} WRF hours from {wanted.index[0]} to {wanted.index[-1]} BJT; "
          "no additional WRF/OBS shift")
    return pd.DataFrame(rows)


def make_figure(ts, obs, args):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9.5,
        "axes.linewidth": 0.8, "axes.labelsize": 10.0,
        "xtick.labelsize": 9.0, "ytick.labelsize": 9.0,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    panels = [
        ("wrf_precip_mm_h", "Precipitation\n(mm h$^{-1}$)"),
        ("q_gkg", "q\n(g kg$^{-1}$)"),
        ("rh_percent", "RH\n(%)"),
        ("w_low_ms", "W\n(m s$^{-1}$)"),
        ("upslope_wind_ms", "$\\mathbf{V}_{\\mathbf{slope}}$\n(m s$^{-1}$)"),
        ("moist_upslope_gkg_ms", "$\\mathbf{qV}_{\\mathbf{slope}}$\n(g kg$^{-1}$ m s$^{-1}$)"),
    ]
    fig, axes = plt.subplots(6, 1, figsize=(10.5, 10.5), sharex=True)
    fig.subplots_adjust(left=0.15, right=0.94, top=0.965, bottom=0.085, hspace=0.18)
    for idx, (ax, (field, label)) in enumerate(zip(axes, panels)):
        for station in STATIONS:
            series = ts.loc[ts.station == station].sort_values("time_bjt")
            # Older Matplotlib tries Series[:, None], which recent pandas
            # forbids. Pass plain arrays to every plot call.
            x = pd.to_datetime(series["time_bjt"]).to_numpy()
            values = pd.to_numeric(series[field], errors="coerce").to_numpy(dtype=float)
            if idx == 0:
                color = PRECIP_STYLE[station]
                ax.plot(x, values, color=color, ls="--", lw=1.35, zorder=2)
                y_obs = obs[station].reindex(pd.DatetimeIndex(x)).to_numpy()
                ax.plot(x, y_obs, color=color, ls="-", lw=1.6, zorder=4)
            else:
                color, style = MET_STYLE[station]
                ax.plot(x, values, color=color, ls=style, lw=1.4,
                        dash_capstyle="round", zorder=3 if station == "P5800" else 2)
        ax.set_ylabel(label, fontweight="bold" if idx in (4, 5) else "normal")
        ax.text(0.006, 0.97, f"({chr(97+idx)})", transform=ax.transAxes,
                va="top", ha="left", fontsize=9.4, fontweight="bold")
        ax.grid(axis="y", color="0.86", lw=0.7)
        ax.grid(axis="x", color="0.92", lw=0.45)
        ax.set_axisbelow(True)
        ax.tick_params(direction="out", length=3.0, width=0.7)
        if idx in (3, 4, 5):
            ax.axhline(0, color="0.65", lw=0.65, zorder=1)
        if idx == 0:
            ax.set_ylim(bottom=0)
        if idx == 1:
            # Reserve a quiet strip under the q traces for its in-panel legend.
            lower, upper = ax.get_ylim()
            ax.set_ylim(lower - 0.35 * (upper - lower), upper)
    axes[-1].set_xlim(args.start_bjt, args.end_bjt)
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    axes[-1].set_xlabel("Time")

    precip_handles = [
        Line2D([], [], color=PRECIP_STYLE[station], lw=1.6, ls=style,
               label=f"{station} {source}")
        for station in STATIONS for source, style in (("OBS", "-"), ("WRF", "--"))
    ]
    met_handles = [
        Line2D([], [], color=MET_STYLE[station][0], lw=1.4,
               ls=MET_STYLE[station][1], label=f"{station} WRF")
        for station in STATIONS
    ]
    # Column-major ordering in Matplotlib gives one OBS/WRF pair per row.
    precip_handles = [precip_handles[i] for i in (0, 2, 4, 1, 3, 5)]
    axes[0].legend(handles=precip_handles, ncol=2, loc="upper left",
                   bbox_to_anchor=(0.038, 0.94), frameon=False,
                   fontsize=7.4, handlelength=1.65, handletextpad=0.4,
                   columnspacing=0.85, labelspacing=0.15,
                   borderaxespad=0.0)
    axes[1].legend(handles=met_handles, ncol=1, loc="lower left",
                   bbox_to_anchor=(0.01, 0.015), frameon=False,
                   fontsize=7.4, handlelength=1.65, handletextpad=0.4,
                   labelspacing=0.12, borderaxespad=0.0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor="white")
    plt.close(fig)
    print(f"[saved] {args.out}")


def save_figure_data_csv(ts, obs, args):
    """Write the six panels' values at the exact BJT timestamps used for plotting."""
    times = pd.DatetimeIndex(sorted(ts["time_bjt"].unique()))
    by_station = {}
    for station in STATIONS:
        series = ts.loc[ts.station == station].set_index("time_bjt")
        if not series.index.is_unique or not series.index.sort_values().equals(times):
            raise ValueError(f"Incomplete or repeated plotted hours for {station}")
        by_station[station] = series.reindex(times)

    columns = {"TIME": times.strftime("%Y-%m-%d %H:%M:%S")}
    for station in STATIONS:
        columns[f"OBS_{station}_precip_mm_h"] = obs[station].reindex(times).to_numpy(dtype=float)
        columns[f"WRF_{station}_precip_mm_h"] = by_station[station]["wrf_precip_mm_h"].to_numpy(dtype=float)
    for field in ("q_gkg", "rh_percent", "w_low_ms", "upslope_wind_ms",
                  "moist_upslope_gkg_ms"):
        for station in STATIONS:
            columns[f"WRF_{station}_{field}"] = by_station[station][field].to_numpy(dtype=float)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns).to_csv(args.csv, index=False, float_format="%.10g", na_rep="")
    print(f"[saved] {args.csv}; {len(times)} plotted hourly timestamps (BJT)")


def main():
    args = arguments()
    from netCDF4 import Dataset
    if not args.wrf_file.is_file() or not args.obs_hourly_file.is_file():
        raise FileNotFoundError("Check --wrf-file and --obs-hourly-file paths")
    obs = read_observations(args.obs_hourly_file)
    with Dataset(args.wrf_file) as nc:
        ts = extract(nc, args)
    comparison_hours = pd.DatetimeIndex(ts.time_bjt.unique())
    for station in STATIONS:
        paired = int(obs[station].reindex(comparison_hours).notna().sum())
        print(f"[OBS] {station}: {paired}/{len(comparison_hours)} exact BJT hours with precipitation")
        if paired == 0:
            raise ValueError(f"No exact OBS-WRF BJT precipitation pairs for {station}")
    make_figure(ts, obs, args)
    save_figure_data_csv(ts, obs, args)


if __name__ == "__main__":
    main()
