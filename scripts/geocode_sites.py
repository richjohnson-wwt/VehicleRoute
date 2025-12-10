from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter


CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS geocache (
    key TEXT PRIMARY KEY,
    lat REAL,
    lon REAL,
    display_name TEXT,
    status TEXT,
    provider TEXT,
    ts INTEGER
);
"""


@dataclass
class GeoResult:
    lat: Optional[float]
    lon: Optional[float]
    display_name: str
    status: str  # ok | not_found | error
    provider: str = "nominatim"


def make_key(address: str, city: str, state: str, zip_code: str) -> str:
    parts = [address.strip(), city.strip(), state.strip(), zip_code.strip()]
    return " | ".join(p for p in parts if p)


def open_cache(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(CACHE_SCHEMA)
    return conn


def cache_get(conn: sqlite3.Connection, key: str) -> Optional[GeoResult]:
    cur = conn.execute(
        "SELECT lat, lon, display_name, status, provider FROM geocache WHERE key = ?",
        (key,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    lat, lon, display_name, status, provider = row
    return GeoResult(lat=lat, lon=lon, display_name=display_name or "", status=status or "ok", provider=provider or "nominatim")


def cache_put(conn: sqlite3.Connection, key: str, result: GeoResult) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO geocache(key, lat, lon, display_name, status, provider, ts) VALUES(?,?,?,?,?,?,?)",
        (
            key,
            result.lat,
            result.lon,
            result.display_name,
            result.status,
            result.provider,
            int(time.time()),
        ),
    )


def build_query(address: str, city: str, state: str, zip_code: str) -> str:
    parts = [address, city, state, zip_code]
    return ", ".join([p for p in parts if p])


def geocode_row(
    geocode: RateLimiter,
    address: str,
    city: str,
    state: str,
    zip_code: str,
) -> GeoResult:
    query = build_query(address, city, state, zip_code)
    try:
        loc = geocode(query)
        if not loc:
            return GeoResult(lat=None, lon=None, display_name="", status="not_found")
        return GeoResult(lat=float(loc.latitude), lon=float(loc.longitude), display_name=str(getattr(loc, "address", "")), status="ok")
    except Exception as e:
        return GeoResult(lat=None, lon=None, display_name=str(e), status="error")


def run(
    input_csv: Path,
    output_csv: Path,
    cache_db: Path,
    user_agent: str,
    email: Optional[str],
    limit: Optional[int],
) -> Tuple[int, int, int]:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input not found: {input_csv}")

    conn = open_cache(cache_db)
    df = pd.read_csv(input_csv)

    required_cols = ["site_name", "address", "city", "state", "zip"]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"Missing required column '{c}' in {input_csv}")

    # Configure Nominatim with a respectful User-Agent including contact
    ua = user_agent
    if email:
        ua = f"{user_agent} ({email})"
    geocoder = Nominatim(user_agent=ua, timeout=10)
    # Rate limit to 1 request per second; error_wait_true handles backoff on exceptions
    geocode = RateLimiter(geocoder.geocode, min_delay_seconds=1.0, swallow_exceptions=False)

    results: list[GeoResult] = []
    hits = 0
    misses = 0
    errors = 0

    rows: Iterable[pd.Series]
    if limit is not None:
        rows = df.head(limit).itertuples(index=False)
    else:
        rows = df.itertuples(index=False)

    # Map tuple indices for speed
    col_idx = {name: i for i, name in enumerate(df.columns)}

    out_rows = []
    processed = 0

    for row in rows:
        address = str(row[col_idx["address"]]) if pd.notna(row[col_idx["address"]]) else ""
        city = str(row[col_idx["city"]]) if pd.notna(row[col_idx["city"]]) else ""
        state = str(row[col_idx["state"]]) if pd.notna(row[col_idx["state"]]) else ""
        zip_code = str(row[col_idx["zip"]]) if pd.notna(row[col_idx["zip"]]) else ""
        site_name = str(row[col_idx["site_name"]]) if pd.notna(row[col_idx["site_name"]]) else ""

        key = make_key(address, city, state, zip_code)
        cached = cache_get(conn, key)
        if cached is not None and cached.status == "ok":
            res = cached
            hits += 1
        else:
            res = geocode_row(geocode, address, city, state, zip_code)
            cache_put(conn, key, res)
            if res.status == "ok":
                misses += 1
            elif res.status == "not_found":
                errors += 1
            else:
                errors += 1
            conn.commit()

        out_rows.append(
            {
                "site_name": site_name,
                "address": address,
                "city": city,
                "state": state,
                "zip": zip_code,
                "lat": res.lat,
                "lon": res.lon,
                "geocode_status": res.status,
                "geocode_display_name": res.display_name,
                "geocode_provider": res.provider,
            }
        )
        processed += 1
        if processed % 50 == 0:
            print(f"Processed {processed} rows... (cache hits={hits}, new={misses}, errors={errors})")

    out_df = pd.DataFrame(out_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"Done. Wrote {len(out_df)} rows to {output_csv}. Cache hits={hits}, new={misses}, errors={errors}")
    conn.close()
    return hits, misses, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Geocode sites CSV using Nominatim with local cache and rate limit (1 rps)")
    parser.add_argument("input", type=Path, help="Input CSV with columns: site_name,address,city,state,zip")
    parser.add_argument("-o", "--output", type=Path, default=Path("data/sites_geocoded.csv"), help="Output CSV path")
    parser.add_argument("--cache", type=Path, default=Path("data/cache/geocache.sqlite"), help="Path to SQLite cache DB")
    parser.add_argument("--user-agent", type=str, default="VehicleRoute-Geocoder/0.1", help="User-Agent string (include your contact)")
    parser.add_argument("--email", type=str, default=None, help="Contact email to include in User-Agent as per Nominatim policy")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for testing")

    args = parser.parse_args()

    try:
        run(args.input, args.output, args.cache, args.user_agent, args.email, args.limit)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
