from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


def parse_depot(s: Optional[str]) -> Optional[tuple[float, float]]:
    if not s:
        return None
    try:
        a, b = s.split(",")
        return float(a.strip()), float(b.strip())
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid --depot '{s}': {e}")


def parse_window(s: str) -> tuple[str, str]:
    try:
        a, b = s.split("-")
        # validate HH:MM
        datetime.strptime(a.strip(), "%H:%M")
        datetime.strptime(b.strip(), "%H:%M")
        return a.strip(), b.strip()
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid --window '{s}': {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare VRPTW-ready CSV (hello schema) from a geocoded CSV")
    ap.add_argument("input", type=Path, help="Geocoded CSV with at least site_name, lat, lon")
    ap.add_argument("--output", type=Path, required=True, help="Output CSV path (hello schema)")
    ap.add_argument("--depot", type=str, default=None, help='Depot as "lat,lon" (default: centroid of input)')
    ap.add_argument("--window", type=str, default="08:00-17:00", help='Default window for all rows (e.g., "08:00-17:00")')
    ap.add_argument("--service-min", type=int, default=20, help="Default service minutes for customers (depot forced to 0)")
    ap.add_argument("--team-col", type=str, default=None, help="Optional team column to carry through")
    ap.add_argument("--take", type=int, default=None, help="Take first N customers (plus depot) to make a subset")

    args = ap.parse_args()

    df = pd.read_csv(args.input)
    for col in ("site_name", "lat", "lon"):
        if col not in df.columns:
            raise KeyError(f"Input missing required column: {col}")
    df = df.dropna(subset=["lat", "lon"]).copy()
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)

    # Determine depot
    depot_latlon = parse_depot(args.depot)
    if depot_latlon is None:
        depot_lat = float(df["lat"].mean())
        depot_lon = float(df["lon"].mean())
    else:
        depot_lat, depot_lon = depot_latlon

    # Build hello schema rows
    start, end = parse_window(args.window)

    # Customers from input (optionally limited)
    customers = df.copy()
    if args.take is not None and args.take > 0:
        customers = customers.head(args.take).copy()

    out_cols = [
        "site_name",
        "lat",
        "lon",
        "is_depot",
        "window_start",
        "window_end",
        "service_min",
    ]
    if args.team_col and args.team_col in df.columns:
        out_cols.append("team")

    rows = []
    # Depot row first
    rows.append(
        {
            "site_name": "Depot",
            "lat": depot_lat,
            "lon": depot_lon,
            "is_depot": 1,
            "window_start": start,
            "window_end": end,
            "service_min": 0,
        }
    )

    # Customer rows
    for _, r in customers.iterrows():
        row = {
            "site_name": str(r.get("site_name", "")),
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "is_depot": 0,
            "window_start": start,
            "window_end": end,
            "service_min": int(args.service_min),
        }
        if args.team_col and args.team_col in df.columns:
            row["team"] = r.get(args.team_col)
        rows.append(row)

    out_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote VRPTW-ready CSV to {args.output} (rows={len(out_df)})")


if __name__ == "__main__":
    main()
