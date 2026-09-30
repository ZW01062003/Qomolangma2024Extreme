#!/usr/bin/env python3
"""One WRF-only 928 station-following section and moisture-convergence profiles.

The section passes through the nearest WRF grid cells to Everest, P6500,
P5800 and P5200; profiles use the same station grid cells as the section.
Both panels show the same output time, default 28 Sep 2024 13:00 BJT.
Times in WRF are UTC; BJT = UTC + 8 h. No empirical time lag.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates


STATIONS = {
    "P6500": (28.030633, 86.940464),
    "P5800": (28.085467, 86.914133),
    "P5200": (28.128911, 86.859978),
}
STATION_ELEV_M = {"P6500": 6409.0, "P5800": 5800.0, "P5200": 5221.0}
COLORS = {"P6500": "#777777", "P5800": "#c9282e", "P5200": "#191919"}
EVEREST = (27.9877361, 86.9276417)
G = 9.80665


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wrf-file", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path, help="Output PNG path")
    p.add_argument("--csv", type=Path,
                   help="Right-panel profile CSV; default: same basename as --out with .csv")
    p.add_argument("--time-bjt", default="2024-09-28 13:00",
                   help="Exact WRF output time in Beijing time (UTC+8)")
    p.add_argument("--dx", type=float, default=None, help="WRF grid spacing, m; default: file DX")
    p.add_argument("--dy", type=float, default=None, help="WRF grid spacing, m; default: file DY")
    p.add_argument("--section-step-m", type=float, default=200.0,
                   help="Distance between samples along the station-following section, m")
    p.add_argument("--vertical-step-m", type=float, default=60.0)
    p.add_argument("--top-km", type=float, default=10.5)
    p.add_argument("--profile-top-km", type=float, default=2.0)
    p.add_argument("--mfc-max", type=float, default=None,
                   help="Symmetric convergence scale, 10^-5 s^-1; default: 98th percentile")
    p.add_argument("--quiver-scale", type=float, default=10.0)
    p.add_argument("--dpi", type=int, default=600)
    a = p.parse_args()
    if a.out.suffix.lower() != ".png":
        p.error("--out must be a PNG path")
    if a.csv is None:
        a.csv = a.out.with_suffix(".csv")
    if a.csv.suffix.lower() != ".csv" or a.csv == a.out:
        p.error("--csv must be a CSV path distinct from --out")
    if min(a.section_step_m, a.vertical_step_m, a.top_km, a.profile_top_km,
           a.quiver_scale, a.dpi) <= 0 or (a.mfc_max is not None and a.mfc_max <= 0):
        p.error("sampling, heights, scales and dpi must be positive")
    a.time_bjt = pd.Timestamp(a.time_bjt)
    if a.time_bjt.minute or a.time_bjt.second:
        p.error("--time-bjt must coincide with a whole-hour WRF output time")
    return a


def as_float(values):
    return np.asarray(np.ma.filled(values, np.nan), dtype=np.float64)


def decode_times(nc):
    strings = [row.tobytes().decode("ascii", "ignore").strip("\x00 ")
               for row in nc.variables["Times"][:]]
    return pd.DatetimeIndex(pd.to_datetime(strings, format="%Y-%m-%d_%H:%M:%S"))


def hour_to_plot(nc, target):
    times_bjt = decode_times(nc) + pd.Timedelta(hours=8)
    # Restart may repeat an hour; use the last record at that BJT timestamp.
    last_index = {t: i for i, t in enumerate(times_bjt)}
    if target not in last_index:
        raise ValueError(f"Requested WRF output {target} BJT (UTC {target - pd.Timedelta(hours=8)}) "
                         f"is missing; available BJT: {times_bjt.min()} to {times_bjt.max()}")
    return last_index[target], sum(t == target for t in times_bjt)


def nearest(lat2d, lon2d, coord):
    lat, lon = coord
    d2 = (lat2d - lat)**2 + ((lon2d - lon) * np.cos(np.deg2rad(lat)))**2
    j, i = np.unravel_index(np.nanargmin(d2), d2.shape)
    return int(j), int(i)


def make_section(lat2d, lon2d, dx, dy, step_m):
    e = np.asarray(EVEREST)
    south = tuple(e + .50 * (e - np.asarray(STATIONS["P6500"])))
    north = tuple(np.asarray(STATIONS["P5200"]) +
                  .45 * (np.asarray(STATIONS["P5200"]) - np.asarray(STATIONS["P5800"])))
    coords = [south, EVEREST, STATIONS["P6500"], STATIONS["P5800"],
              STATIONS["P5200"], north]
    nodes = [nearest(lat2d, lon2d, coord) for coord in coords]
    if any(nodes[k] == nodes[k + 1] for k in range(len(nodes) - 1)):
        raise ValueError("Adjacent section control points fell into one WRF grid cell")
    ii, jj, distances, marks = [], [], [], {}
    total = 0.0
    for seg, ((j0, i0), (j1, i1)) in enumerate(zip(nodes[:-1], nodes[1:])):
        length = float(np.hypot((i1 - i0) * dx, (j1 - j0) * dy))
        n = max(2, int(np.ceil(length / step_m)))
        frac = np.linspace(0, 1, n + 1)
        if seg:
            frac = frac[1:]
        ii.extend(i0 + (i1 - i0) * frac)
        jj.extend(j0 + (j1 - j0) * frac)
        distances.extend(total + length * frac)
        total += length
        if seg == 0:
            marks["Everest"] = total / 1000.0
        elif seg <= 3:
            marks[("P6500", "P5800", "P5200")[seg - 1]] = total / 1000.0
    ii, jj = np.asarray(ii), np.asarray(jj)
    distance = np.asarray(distances) / 1000.0
    tangent_e = np.gradient(ii * dx, distance * 1000)
    tangent_n = np.gradient(jj * dy, distance * 1000)
    length = np.hypot(tangent_e, tangent_n)
    station_cells = dict(zip(STATIONS, nodes[2:5]))
    for name, (j, i) in station_cells.items():
        k = int(np.argmin(np.abs(distance - marks[name])))
        if abs(jj[k] - j) > 1e-6 or abs(ii[k] - i) > 1e-6:
            raise ValueError(f"Section does not pass through the {name} WRF grid cell")
    return jj, ii, distance, tangent_e / length, tangent_n / length, marks, station_cells


def sample_section(field, ylocal, xlocal):
    coords = np.vstack((ylocal, xlocal))
    return np.stack([map_coordinates(level, coords, order=1, mode="nearest")
                     for level in field])


def height_interpolate(height_km, field, heights_km, terrain_km):
    out = np.full((len(heights_km), height_km.shape[1]), np.nan)
    for col in range(height_km.shape[1]):
        h, f = height_km[:, col], field[:, col]
        valid = np.isfinite(h) & np.isfinite(f)
        if valid.sum() >= 2:
            hh, ff = h[valid], f[valid]
            order = np.argsort(hh)
            hh, ff = hh[order], ff[order]
            unique = np.r_[True, np.diff(hh) > 1e-7]
            if unique.sum() >= 2:
                out[:, col] = np.interp(heights_km, hh[unique], ff[unique],
                                         left=np.nan, right=np.nan)
    out[heights_km[:, None] <= terrain_km[None, :] + .02] = np.nan
    return out


def terrain_following_interpolate(height_km, field, terrain_km, agl_km):
    """Display interpolation; extend the lowest mass-level value to the surface.

    WRF does not have a mass-level value exactly at ground. This nearest-level
    extension closes the blank strip without creating a new model diagnosis.
    """
    out = np.full((len(agl_km), height_km.shape[1]), np.nan)
    for col in range(height_km.shape[1]):
        ground = terrain_km[col]
        h, f = height_km[:, col], field[:, col]
        valid = np.isfinite(h) & np.isfinite(f) & (h > ground)
        if valid.sum() < 2:
            continue
        order = np.argsort(h[valid])
        hh, ff = h[valid][order], f[valid][order]
        unique = np.r_[True, np.diff(hh) > 1e-7]
        hh, ff = hh[unique], ff[unique]
        if len(hh) < 2:
            continue
        targets = ground + agl_km
        out[:, col] = np.interp(targets, np.r_[ground, hh],
                                np.r_[ff[0], ff], left=np.nan, right=np.nan)
    return out


def read_one_hour(nc, tidx, box):
    j0, j1, i0, i1 = box
    def read(key, js=slice(j0, j1), xs=slice(i0, i1)):
        return as_float(nc.variables[key][tidx, :, js, xs])
    p = read("P") + read("PB")
    temp = (read("T") + 300) * np.power(p / 100000.0, .2854)
    q = read("QVAPOR")
    tc = temp - 273.15
    es = 611.2 * np.exp(17.67 * tc / (tc + 243.5))
    qs = .622 * es / np.maximum(p - es, 1.0)
    rh = np.clip(100 * q / np.maximum(qs, 1e-12), 0, 150)
    zstag = (read("PH") + read("PHB")) / G
    z = .5 * (zstag[:-1] + zstag[1:])
    wstag = read("W")
    w = .5 * (wstag[:-1] + wstag[1:])
    us = read("U", xs=slice(i0, i1 + 1))
    vs = read("V", js=slice(j0, j1 + 1))
    u = .5 * (us[:, :, :-1] + us[:, :, 1:])
    v = .5 * (vs[:, :-1, :] + vs[:, 1:, :])
    if not (u.shape == v.shape == w.shape == z.shape == q.shape == rh.shape):
        raise ValueError("Staggered and mass-grid WRF variables have inconsistent shapes")
    return u, v, w, z, q, rh


def extract(nc, args):
    lat = as_float(nc.variables["XLAT"][0]); lon = as_float(nc.variables["XLONG"][0])
    terrain = as_float(nc.variables["HGT"][0] if "HGT" in nc.variables else nc.variables["HGT_M"][0])
    dx = args.dx if args.dx is not None else float(nc.DX)
    dy = args.dy if args.dy is not None else float(nc.DY)
    tidx, repeated = hour_to_plot(nc, args.time_bjt)
    yline, xline, distance, tangent_e, tangent_n, marks, station_cells = \
        make_section(lat, lon, dx, dy, args.section_step_m)
    station_js = [j for j, _ in station_cells.values()]
    station_is = [i for _, i in station_cells.values()]
    # Include station cells and a 2-cell halo for horizontal derivatives.
    j0 = max(0, int(np.floor(min(yline.min(), *station_js))) - 2)
    j1 = min(lat.shape[0], int(np.ceil(max(yline.max(), *station_js))) + 3)
    i0 = max(0, int(np.floor(min(xline.min(), *station_is))) - 2)
    i1 = min(lat.shape[1], int(np.ceil(max(xline.max(), *station_is))) + 3)
    box = (j0, j1, i0, i1)
    yl, xl = yline - j0, xline - i0
    terrain_line = map_coordinates(terrain[j0:j1, i0:i1], np.vstack((yl, xl)), order=1)
    terrain_km = terrain_line / 1000.0
    low_km = np.floor((terrain_km.min() - .30) * 10) / 10
    if low_km >= args.top_km - .4:
        raise ValueError("--top-km is below the section's terrain")
    heights = np.arange(0, args.top_km - low_km + .001,
                        args.vertical_step_m / 1000)  # height above local WRF terrain
    agl = np.arange(0, args.profile_top_km + .0001, .05)
    for name, (j, i) in station_cells.items():
        print(f"[station] {name}: grid ({j},{i}), WRF terrain {terrain[j,i]:.0f} m, "
              f"observed altitude {STATION_ELEV_M[name]:.0f} m, "
              f"grid lat/lon {lat[j,i]:.5f}/{lon[j,i]:.5f}; "
              f"cross-section position {marks[name]:.2f} km")
    print(f"[time] {args.time_bjt} BJT = {args.time_bjt - pd.Timedelta(hours=8)} UTC; "
          f"WRF record {tidx}; {repeated} matching record(s), last used")
    print(f"[transect] station-following path: south, Everest, P6500, P5800, "
          f"P5200, north; {distance[-1]:.1f} km, {len(distance)} samples")

    u, v, w, z, q, rh = read_one_hour(nc, tidx, box)
    # Same horizontal moisture-flux convergence proxy as the earlier WRF
    # diagnostics: q is WRF QVAPOR (kg/kg). Positive values mean convergence.
    # Grid derivatives are computed before interpolation to the transect.
    dqu_dy, dqu_dx = np.gradient(q * u, dy, dx, axis=(1, 2))
    dqv_dy, dqv_dx = np.gradient(q * v, dy, dx, axis=(1, 2))
    mfc = -(dqu_dx + dqv_dy) * 1e5  # plotted in 10^-5 s^-1
    section_z = sample_section(z, yl, xl) / 1000
    native_fields = {
        "mfc": sample_section(mfc, yl, xl),
        "w": sample_section(w, yl, xl),
        "rh": sample_section(rh, yl, xl),
        "along": sample_section(u, yl, xl) * tangent_e[None, :]
                 + sample_section(v, yl, xl) * tangent_n[None, :],
    }
    for name, (j, i) in station_cells.items():
        k = int(np.argmin(np.abs(distance - marks[name])))
        section_values = native_fields["mfc"][:, k]
        station_values = mfc[:, j - j0, i - i0]
        if not np.allclose(section_values, station_values, rtol=1e-7,
                           atol=1e-7, equal_nan=True):
            raise ValueError(f"{name} section and profile sample different WRF grid cells")
    fields = {key: terrain_following_interpolate(section_z, value, terrain_km, heights)
              for key, value in native_fields.items()}
    profiles = {}
    for name, (j, i) in station_cells.items():
        jj, ii = j - j0, i - i0
        z_agl = (z[:, jj, ii] - terrain[j, i]) / 1000
        profiles[name] = height_interpolate(z_agl[:, None], mfc[:, jj, ii, None],
                                            agl, np.array([0.]))[:, 0]
        low3 = min(3, mfc.shape[0])
        print(f"[lowest {low3} WRF levels] {name}: moisture-flux convergence "
              f"{np.nanmean(mfc[:low3, jj, ii]):+.3g} × 10^-5 s^-1")
    return distance, heights, terrain_km, marks, agl, fields, profiles


def glacier_cmap():
    return LinearSegmentedColormap.from_list("glacier_moisture_convergence", [
        (0.00, "#687488"), (0.23, "#c9ccdc"), (0.49, "#f9f9fb"),
        (0.53, "#f0efff"), (0.68, "#d3cfff"), (0.80, "#a7a1f7"),
        (0.90, "#6562ee"), (1.00, "#2637d2")])


def draw(args, distance, heights, terrain, marks, agl, fields, profiles):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 10.4, "axes.linewidth": .85,
        "axes.labelsize": 11, "xtick.labelsize": 9.2, "ytick.labelsize": 9.2,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    mfc, w, rh, along = (fields[key] for key in ("mfc", "w", "rh", "along"))
    finite = np.isfinite(mfc)
    if not np.any(finite):
        raise ValueError("Cross section contains no finite moisture-flux convergence values")
    mfcmax = args.mfc_max or max(.05, float(np.nanpercentile(np.abs(mfc[finite]), 98)))
    zmin = float(np.floor((np.min(terrain) - .3) * 10) / 10)
    fig = plt.figure(figsize=(12, 6.4), facecolor="white")
    ax = fig.add_axes([.08, .20, .665, .735])
    right = fig.add_axes([.79, .20, .17, .735])
    cax = fig.add_axes([.20, .105, .425, .022])
    xx = np.broadcast_to(distance, mfc.shape)
    zz = terrain[None, :] + np.broadcast_to(heights[:, None], mfc.shape)
    # An 8 m overlap below the gray terrain removes antialiasing seams.
    zz[0, :] = terrain - .008
    cf = ax.contourf(xx, zz, mfc, levels=np.linspace(-mfcmax, mfcmax, 17),
                     cmap=glacier_cmap(), norm=Normalize(vmin=-mfcmax, vmax=mfcmax),
                     extend="both", zorder=1)
    if np.any(np.isfinite(rh)):
        rhmin, rhmax = np.nanmin(rh), np.nanmax(rh)
        # Relative humidity (%) contours only: 90% and 95%.
        for level, color, lw in ((90, "#4e5358", .9), (95, "#232629", 1.25)):
            if rhmin < level < rhmax:
                cs = ax.contour(xx, zz, rh, levels=[level], colors=[color],
                                linewidths=[lw], zorder=4)
                ax.clabel(cs, fmt=lambda _: f"{level}%", fontsize=8.5,
                          inline=True, inline_spacing=5)
    ax.fill_between(distance, zmin, terrain, color="#b7babd",
                    linewidth=0, zorder=10)
    xq = np.arange(distance[0] + 1.2, distance[-1] - .1, 2.7)
    zq = np.arange(zmin + .4, args.top_km - .1, .65)
    ix = np.clip(np.searchsorted(distance, xq), 0, len(distance) - 1)
    XQ, ZQ = np.meshgrid(distance[ix], zq)
    XI = np.broadcast_to(ix[None, :], XQ.shape)
    above_ground = (ZQ - terrain[XI]) / (args.vertical_step_m / 1000)
    ZI = np.clip(np.rint(above_ground).astype(int), 0, len(heights) - 1)
    uq, wq = along[ZI, XI], w[ZI, XI]
    above = ZQ > terrain[XI] + .25
    labels_corner = ((XQ < distance[0] + 2.5) |
                     (XQ > distance[-1] - 2.5)) & (ZQ > args.top_km - .9)
    good = above & ~labels_corner & np.isfinite(uq) & np.isfinite(wq)
    ax.quiver(XQ[good], ZQ[good], uq[good], wq[good], angles="xy", scale_units="xy",
              scale=args.quiver_scale, width=.0022, headwidth=3.1, headlength=4.0,
              headaxislength=3.8, color="#30343a", alpha=.87, zorder=12)
    for name in ("P6500", "P5800", "P5200"):
        sx = marks[name]
        sy = float(np.interp(sx, distance, terrain))
        station_y = STATION_ELEV_M[name] / 1000.0
        color = COLORS[name]
        if abs(station_y - sy) > .025:
            ax.plot([sx, sx], [sy, station_y], lw=.8, color=color,
                    alpha=.8, zorder=14)
        ax.plot([sx, sx], [max(station_y, sy) + .06,
                           min(max(station_y, sy) + .96, args.top_km - .15)],
                ls=(0, (3, 3)), lw=.95, color=color, alpha=.72, zorder=13)
        ax.scatter([sx], [station_y], s=40, marker="^", color=color,
                   edgecolor="white", linewidth=.65, zorder=15)
        ty = station_y - .22 if name == "P5800" else station_y + .14
        ax.text(sx + .2, ty, name, color=color, weight="bold", size=9.4, zorder=16)
    ax.text(.035, .973, "South", transform=ax.transAxes, va="top", color="#41464a", size=8.7)
    ax.text(.875, .973, "North", transform=ax.transAxes, va="top", color="#41464a", size=8.7)
    ax.annotate("", xy=(.975, .963), xytext=(.944, .963), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=.9, color="#41464a"))
    ax.set(xlim=(distance[0], distance[-1]), ylim=(zmin, args.top_km),
           xlabel="Distance along north-slope transect (km)",
           ylabel="Altitude (km a.s.l.)")
    ax.tick_params(length=3.6, width=.85, direction="out")
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal",
                      ticks=[-mfcmax, -mfcmax / 2, 0, mfcmax / 2, mfcmax])
    cb.set_label("Moisture-flux convergence ($10^{-5}$ s$^{-1}$)", labelpad=5)
    cb.outline.set_linewidth(.65)

    all_profile = np.concatenate([p[np.isfinite(p)] for p in profiles.values()])
    if not len(all_profile):
        raise ValueError("No valid station moisture-flux convergence profiles")
    lo = min(-.5, float(np.min(all_profile)) * 1.12)
    hi = max(.5, float(np.max(all_profile)) * 1.12)
    margin = .04 * (hi - lo)
    right.axvline(0, color="#9da2a7", lw=.8, zorder=1)
    for name in ("P5200", "P5800", "P6500"):
        right.plot(profiles[name], agl, color=COLORS[name], lw=2.05, label=name, zorder=3)
    right.set(xlim=(lo - margin, hi + margin), ylim=(0, args.profile_top_km),
              xlabel="Moisture-flux convergence\n($10^{-5}$ s$^{-1}$)",
              ylabel="Height above ground (km)")
    right.grid(axis="y", color="#e6e7e8", lw=.65)
    right.tick_params(length=3.6, width=.85, direction="out")
    right.legend(frameon=False, loc="upper right", fontsize=8.5,
                 handlelength=1.8, labelspacing=.25)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor="white")
    plt.close(fig)
    print(f"[saved] {args.out}; convergence color range ±{mfcmax:.3g} × 10^-5 s^-1")


def save_profiles_csv(args, agl, profiles):
    """Export exactly the height grid and convergence values drawn at right."""
    names = ("P5200", "P5800", "P6500")
    if any(len(profiles[name]) != len(agl) for name in names):
        raise ValueError("Profile length differs from the plotted height grid")
    table = pd.DataFrame({
        "TIME": [args.time_bjt.strftime("%Y-%m-%d %H:%M:%S")] * len(agl),
        "HEIGHT_AGL_KM": agl,
        **{f"{name}_MFC_1E-5_S-1": profiles[name] for name in names},
    })
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.csv, index=False, float_format="%.10g", na_rep="")
    print(f"[saved] {args.csv}; {len(table)} heights, TIME in BJT, "
          "HEIGHT_AGL_KM above local WRF terrain, MFC in 10^-5 s^-1")


def main():
    args = arguments()
    if not args.wrf_file.is_file():
        raise FileNotFoundError(args.wrf_file)
    from netCDF4 import Dataset
    with Dataset(args.wrf_file) as nc:
        data = extract(nc, args)
    draw(args, *data)
    save_profiles_csv(args, data[4], data[6])


if __name__ == "__main__":
    main()
