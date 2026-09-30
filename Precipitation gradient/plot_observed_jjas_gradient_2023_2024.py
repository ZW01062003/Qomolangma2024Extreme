#!/usr/bin/env python3
"""North-slope station JJAS monthly precipitation, 2023 versus 2024.

Inputs are dated observations in Beijing time, in mm per record. The 2024
In September 2024, the solid part excludes 27–29 September and the hatched
top part shows precipitation on those dates; together they give the full
monthly total. No gridded products are read by this script.
"""

from __future__ import annotations

import argparse
import calendar
import re
import warnings
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


MONTHS = (6, 7, 8, 9)
MONTH_NAMES = ("June", "July", "August", "September")
STATIONS = ("P5200", "P5800", "P6500")
COLORS = {
    2023: ("#BAE3F5", "#3996C6", "#01688B"),
    2024: ("#D47D7D", "#C65151", "#B22424"),
}
EDGES = {
    2023: ("#9FC6D7", "#3085B0", "#005C7B"),
    2024: ("#B46666", "#AC3E3E", "#971D1D"),
}
STEM = "Observed_JJAS_2023_2024_monthly_gradient_v3"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obs-2023", type=Path, required=True)
    parser.add_argument("--obs-2024", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--ymax", type=float, default=400,
                        help="Minimum common y-axis maximum in mm")
    args = parser.parse_args()
    for year in (2023, 2024):
        path = getattr(args, f"obs_{year}")
        if not path.is_file():
            parser.error(f"Missing {year} observation workbook: {path}")
    if not 100 <= args.dpi <= 1200 or not np.isfinite(args.ymax) or args.ymax <= 0:
        parser.error("Use --dpi 100–1200 and a positive --ymax")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def normalized(value: object) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", str(value).lower())


def station_in_header(value: object) -> str | None:
    """Accept P5200 or 5200 labels, including the obsolete P6400 alias."""
    name = normalized(value)
    if re.search(r"(?<!\d)(?:p)?(?:6500|6400)(?!\d)", name):
        return "P6500"
    for code in ("5200", "5800"):
        if re.search(rf"(?<!\d)(?:p)?{code}(?!\d)", name):
            return f"P{code}"
    return None


def parse_dates(series: pd.Series) -> pd.Series:
    """Read Excel date cells, common date strings and numeric Excel serials."""
    if pd.api.types.is_numeric_dtype(series):
        numbers = pd.to_numeric(series, errors="coerce")
        if numbers.dropna().between(20000101, 20991231).all():
            dates = pd.to_datetime(numbers.astype("Int64").astype(str),
                                   format="%Y%m%d", errors="coerce")
        else:
            dates = pd.to_datetime(series, unit="D", origin="1899-12-30", errors="coerce")
    else:
        dates = pd.to_datetime(series, errors="coerce")
        # Mixed date strings and genuine Excel serials in one column.
        missing = dates.isna() & pd.to_numeric(series, errors="coerce").between(40000, 60000)
        if missing.any():
            serials = pd.to_numeric(series[missing], errors="coerce")
            dates.loc[missing] = pd.to_datetime(serials, unit="D", origin="1899-12-30")
    if getattr(dates.dt, "tz", None) is not None:
        raise ValueError("Timezone-aware dates detected. Convert timestamps to Beijing time in Excel.")
    return dates


def find_date_column(frame: pd.DataFrame, year: int) -> tuple[object, pd.Series]:
    candidates: list[tuple[float, object, pd.Series]] = []
    for column in frame.columns:
        key = normalized(column)
        # Avoid treating precipitation numbers as Excel serial dates.
        if station_in_header(column) or any(s in key for s in ("降水", "precip", "rain")):
            continue
        named_date = any(s in key for s in ("date", "time", "日期", "时间", "年月日", "北京时间", "bjt"))
        if not named_date and not pd.api.types.is_datetime64_any_dtype(frame[column]):
            continue
        parsed = parse_dates(frame[column])
        n = int((parsed.dt.year == year).sum())
        if n:
            candidates.append((n / max(len(frame), 1) + (0.01 if named_date else 0), column, parsed))
    if not candidates:
        # Some saved Excel indices are called "Unnamed: 0".
        for column in frame.columns[:3]:
            if station_in_header(column):
                continue
            parsed = parse_dates(frame[column])
            n = int((parsed.dt.year == year).sum())
            if n >= 20 and n / max(frame[column].notna().sum(), 1) > 0.8:
                candidates.append((n / max(len(frame), 1), column, parsed))
    if not candidates:
        raise ValueError("Could not find a dated column. Expected a Date/Time/日期 column in BJT.")
    _, column, dates = max(candidates, key=lambda item: item[0])
    return column, dates


def read_observations(path: Path, year: int) -> pd.DataFrame:
    """Find a worksheet with daily or finer dated, wide-format station data."""
    try:
        sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    except ImportError as exc:
        raise RuntimeError("The Python environment needs pandas and openpyxl") from exc
    issues = []
    for sheet_name, frame in sheets.items():
        try:
            frame = frame.dropna(how="all").copy()
            if frame.empty:
                continue
            date_column, dates = find_date_column(frame, year)
            station_columns: dict[str, object] = {}
            for column in frame.columns:
                if column == date_column:
                    continue
                station = station_in_header(column)
                if station is not None:
                    if station in station_columns:
                        raise ValueError(f"Multiple columns found for {station}: "
                                         f"{station_columns[station]!r}, {column!r}")
                    station_columns[station] = column
            missing = set(STATIONS) - set(station_columns)
            if missing:
                raise ValueError(f"Missing station columns: {sorted(missing)}")
            if any("6400" in normalized(c) for c in station_columns.values()):
                warnings.warn(f"{path.name}: treating P6400 as P6500", stacklevel=2)
            keep = dates.dt.year.eq(year) & dates.dt.month.isin(MONTHS)
            result = pd.DataFrame({"date_bjt": dates.loc[keep]})
            if result.empty:
                raise ValueError(f"No JJAS {year} dated records")
            for station in STATIONS:
                source = frame.loc[keep, station_columns[station]]
                values = pd.to_numeric(source, errors="coerce")
                bad = source.notna() & values.isna()
                if bad.any():
                    raise ValueError(f"Non-numeric precipitation for {station} at "
                                     f"{dates.loc[bad.index[bad]].iloc[0]}")
                if values.lt(0).any():
                    raise ValueError(f"Negative precipitation found in {station}; "
                                     "convert missing-value sentinels to blank cells")
                result[station] = values
            result = result.dropna(subset=list(STATIONS), how="all")
            if result.empty:
                raise ValueError("All three station columns contain only blanks")
            result = result.sort_values("date_bjt").reset_index(drop=True)
            # A daily table with duplicate calendar dates is ambiguous: it may
            # be a duplicated row rather than genuine subdaily increments.
            if (result.date_bjt.dt.normalize().duplicated().any()
                    and result.date_bjt.dt.time.nunique() == 1):
                raise ValueError("Duplicate daily dates with no hour field; check rows before summing")
            print(f"{year}: sheet {sheet_name!r}, {len(result)} dated records, "
                  f"station columns {[str(station_columns[s]) for s in STATIONS]}", flush=True)
            return result
        except (ValueError, TypeError, AttributeError) as exc:
            issues.append(f"{sheet_name!r}: {exc}")
    raise ValueError(f"No usable sheet in {path}. " + " | ".join(issues))


def monthly_table(data_by_year: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for year, data in data_by_year.items():
        for station in STATIONS:
            row: dict[str, object] = {"year": year, "station": station}
            for month, name in zip(MONTHS, MONTH_NAMES):
                subset = data.loc[data.date_bjt.dt.month.eq(month)]
                observed = subset.loc[subset[station].notna(), "date_bjt"]
                if observed.empty:
                    raise ValueError(f"{year} {name} {station} has no observations")
                n_days = observed.dt.normalize().nunique()
                if n_days < calendar.monthrange(year, month)[1]:
                    warnings.warn(f"{year} {name} {station}: {n_days} observed dates; "
                                  "monthly total is not adjusted for missing dates", stacklevel=2)
                row[f"{name}_mm"] = float(subset[station].sum(min_count=1))
            if year == 2024:
                september = data.loc[data.date_bjt.dt.month.eq(9)]
                for day in (27, 28, 29):
                    dated = september.loc[september.date_bjt.dt.day.eq(day), station]
                    if not dated.notna().any():
                        raise ValueError(f"2024-09-{day:02d} {station} is missing; "
                                         "cannot calculate excluded-event total")
                without_event = september.loc[~september.date_bjt.dt.day.isin((27, 28, 29)), station]
                row["September_excluding_27_29_mm"] = float(without_event.sum(min_count=1))
            else:
                row["September_excluding_27_29_mm"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def plot(table: pd.DataFrame, path: Path, dpi: int, min_ymax: float) -> None:
    available = {font.name for font in mpl.font_manager.fontManager.ttflist}
    font = next((name for name in ("Times New Roman", "Liberation Serif", "DejaVu Serif")
                 if name in available), "DejaVu Serif")
    mpl.rcParams.update({
        "font.family": font,
        "font.size": 11,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    maximum = float(table[[f"{m}_mm" for m in MONTH_NAMES]].to_numpy().max())
    upper = max(min_ymax, np.ceil(maximum * 1.35 / 50) * 50)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.0), sharey=True)
    fig.subplots_adjust(left=0.105, right=0.98, bottom=0.17, top=0.91, wspace=0.11)
    centers = np.arange(4, dtype=float)
    width = 0.15

    def pale(hex_color: str, white_fraction: float = 0.70) -> tuple[float, float, float]:
        rgb = np.array(mpl.colors.to_rgb(hex_color))
        return tuple(rgb * (1 - white_fraction) + white_fraction)

    for panel_index, (ax, year) in enumerate(zip(axes, (2023, 2024))):
        handles = []
        for i, station in enumerate(STATIONS):
            row = table.loc[(table.year == year) & (table.station == station)]
            if len(row) != 1:
                raise ValueError(f"Expected one table row for {year} {station}")
            position = centers + (i - 1) * width
            values = row[[f"{m}_mm" for m in MONTH_NAMES]].iloc[0].to_numpy(dtype=float)
            if year == 2024:
                excluded = float(row["September_excluding_27_29_mm"].iloc[0])
                event = float(values[3] - excluded)
                if not 0 <= excluded <= values[3] or event < 0:
                    raise ValueError(f"Invalid 27–29 September component for {station}")
                values[3] = excluded
            ax.bar(position, values, width=width, color=COLORS[year][i],
                   edgecolor=EDGES[year][i], linewidth=0.28, zorder=3)
            handles.append(Patch(facecolor=COLORS[year][i], edgecolor=EDGES[year][i],
                                 linewidth=0.28, label=station))
            if year == 2024:
                ax.bar(position[3], event, bottom=excluded, width=width,
                       facecolor=pale(COLORS[year][i]),
                       edgecolor=EDGES[year][i], linewidth=0.8,
                       hatch="///", zorder=4)
        if year == 2024:
            handles.append(Patch(facecolor=pale(COLORS[2024][1]),
                                 edgecolor=EDGES[2024][1], linewidth=0.8,
                                 hatch="///", label="27-29 Sep 2024"))
        ax.set_xlim(-0.35, 3.35)
        ax.set_ylim(0, upper)
        # The cluster has an older Matplotlib: set_xticks cannot take labels
        # or fontsize there, so configure the ticks and text separately.
        ax.set_xticks(centers)
        ax.set_xticklabels(MONTH_NAMES, fontsize=11)
        ax.text(0, 1.018, str(year), transform=ax.transAxes,
                ha="left", va="bottom", fontsize=13)
        ax.legend(handles=handles, loc="upper right", frameon=False,
                  fontsize=9.6, handlelength=1.35, handletextpad=0.5,
                  labelspacing=0.3, borderaxespad=0.65)
        ax.tick_params(axis="x", direction="out", length=4, width=0.75, pad=5)
        ax.tick_params(axis="y", direction="out", length=5, width=0.75,
                       labelsize=11, labelleft=(panel_index == 0))
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_color("#888888")
            spine.set_linewidth(0.75)
    axes[0].set_ylabel("Monthly precipitation (mm)", fontsize=13, labelpad=8)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = arguments()
    observations = {year: read_observations(getattr(args, f"obs_{year}"), year)
                    for year in (2023, 2024)}
    table = monthly_table(observations)
    csv_path = args.output_dir / f"{STEM}_source_data.csv"
    png_path = args.output_dir / f"{STEM}.png"
    table.to_csv(csv_path, index=False, float_format="%.6f")
    plot(table, png_path, args.dpi, args.ymax)
    print(f"PNG: {png_path}\nFigure source data: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
