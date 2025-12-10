from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Optional YAML support (installed via pyyaml)
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

# ---------- Helpers ----------

ALIAS_MAP: Dict[str, List[str]] = {
    "site_name": ["site", "site_name", "location", "location_name", "facility", "name"],
    "address": ["address", "street", "street_address", "addr1", "address1"],
    "city": ["city", "town"],
    "state": ["state", "st", "province"],
    "zip": ["zip", "zip_code", "zipcode", "postal", "postal_code"],
}

US_STATE_ABBR: Dict[str, str] = {
    # Lowercase keys for easy compare
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS", "missouri": "MO",
    "montana": "MT", "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH",
    "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "district of columbia": "DC", "washington dc": "DC", "dc": "DC",
}

RE_MULTI_SPACE = re.compile(r"\s+")
RE_ZIP9 = re.compile(r"^(\d{5})(?:[- ]?(\d{4}))?$")
RE_IDISH = re.compile(r"\b(id|code|number|no)\b", re.IGNORECASE)


def _normalize_colname(name: str) -> str:
    return RE_MULTI_SPACE.sub(" ", name.strip().lower().replace("_", " "))


def _score_column(norm_name: str, alias_set: set[str], field: str) -> int:
    score = 0
    # Exact alias match
    if norm_name in alias_set:
        score += 100
    # Startswith / contains
    if any(norm_name.startswith(a) for a in alias_set):
        score += 30
    if any(a in norm_name for a in alias_set):
        score += 20
    # Field-specific heuristics
    if field == "site_name":
        if "name" in norm_name:
            score += 25
        if RE_IDISH.search(norm_name) or norm_name.endswith(" id") or norm_name.endswith("id"):
            score -= 60
        if "code" in norm_name:
            score -= 40
    return score


def _find_column(source_cols: List[str], aliases: List[str], field: str) -> Optional[str]:
    norm_cols = {c: _normalize_colname(c) for c in source_cols}
    alias_set = {a.strip().lower().replace("_", " ") for a in aliases}
    best_col: Optional[str] = None
    best_score = -10**9
    for original, norm in norm_cols.items():
        base = norm.replace("  ", " ")
        s = _score_column(base, alias_set, field)
        if s > best_score:
            best_score = s
            best_col = original
    return best_col


def _clean_str(val: object) -> str:
    if pd.isna(val):
        return ""
    s = str(val).strip()
    s = RE_MULTI_SPACE.sub(" ", s)
    return s


def _normalize_state(state: str) -> str:
    s = _clean_str(state).upper()
    if not s:
        return ""
    if len(s) == 2 and s.isalpha():
        return s
    # try full name to abbr
    abbr = US_STATE_ABBR.get(s.lower())
    return abbr or s[:2]


def _normalize_zip(zipcode: str) -> str:
    z = _clean_str(zipcode)
    if not z:
        return ""
    m = RE_ZIP9.match(z)
    if not m:
        # Keep original text, but try to strip non-digits and re-check
        digits = re.sub(r"\D", "", z)
        if len(digits) == 4:
            # Likely lost a leading zero in Excel; pad to 5
            return digits.zfill(5)
        if len(digits) == 5:
            return digits
        if len(digits) == 9:
            return f"{digits[:5]}-{digits[5:]}"
        return z
    five, four = m.group(1), m.group(2)
    return five if not four else f"{five}-{four}"


def _select_columns(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, Dict[str, str]]:
    cols = list(df.columns)
    site_col = _find_column(cols, ALIAS_MAP["site_name"], "site_name") or cols[0]
    addr_col = _find_column(cols, ALIAS_MAP["address"], "address") or cols[1 if len(cols) > 1 else 0]
    city_col = _find_column(cols, ALIAS_MAP["city"], "city") or cols[2 if len(cols) > 2 else 0]
    state_col = _find_column(cols, ALIAS_MAP["state"], "state") or cols[3 if len(cols) > 3 else 0]
    zip_col = _find_column(cols, ALIAS_MAP["zip"], "zip") or cols[4 if len(cols) > 4 else 0]

    mapping = {
        "site_name": site_col,
        "address": addr_col,
        "city": city_col,
        "state": state_col,
        "zip": zip_col,
    }

    return (
        df[site_col],
        df[addr_col],
        df[city_col],
        df[state_col],
        df[zip_col],
        mapping,
    )


def parse_xlsx(
    input_path: Path,
    sheet: Optional[str] = None,
    header_row: int = 0,
    all_sheets: bool = False,
    debug: bool = False,
    config: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (clean_df, errors_df)
    clean_df columns: site_name, address, city, state, zip
    errors_df columns: row_index, issue
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    def process_one(
        df: pd.DataFrame,
        sheet_name: Optional[str],
        explicit_mapping: Optional[Dict[str, str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if explicit_mapping:
            missing_cols = [src for src in explicit_mapping.values() if src not in df.columns]
            if missing_cols:
                raise KeyError(
                    f"Missing expected columns in sheet '{sheet_name}': {missing_cols}"
                )
            mapping = explicit_mapping
            site = df[mapping["site_name"]]
            addr = df[mapping["address"]]
            city = df[mapping["city"]]
            state = df[mapping["state"]]
            zip_series = df[mapping["zip"]]
        else:
            site, addr, city, state, zip_series, mapping = _select_columns(df)
        if debug:
            print(f"Column mapping for sheet '{sheet_name}': {mapping}")

        out = pd.DataFrame(
            {
                "site_name": site.apply(_clean_str),
                "address": addr.apply(_clean_str),
                "city": city.apply(_clean_str),
                "state": state.apply(_normalize_state),
                "zip": zip_series.astype("string").apply(_normalize_zip),
            }
        )

        # Drop rows fully empty
        out.replace("", pd.NA, inplace=True)
        out.dropna(how="all", inplace=True)

        # Validate
        issues: List[Tuple[int, str]] = []
        for idx, row in out.iterrows():
            missing = [c for c in ["address", "city", "state", "zip"] if pd.isna(row[c]) or row[c] == ""]
            if missing:
                issues.append((idx, f"Missing required fields: {', '.join(missing)}"))
            # basic zip validity
            z = row.get("zip")
            if isinstance(z, str) and not RE_ZIP9.match(z) and not re.fullmatch(r"\d{5}", z):
                digits = re.sub(r"\D", "", z)
                if len(digits) not in (5, 9):
                    issues.append((idx, f"Suspicious ZIP: {z}"))
            # state format
            s = row.get("state")
            if isinstance(s, str) and len(s) != 2:
                issues.append((idx, f"Non-2-letter state: {s}"))

        errors_df = (
            pd.DataFrame(issues, columns=["row_index", "issue"]) if issues else pd.DataFrame(columns=["row_index", "issue"]) 
        )
        if sheet_name is not None:
            out["source_sheet"] = sheet_name
            if not errors_df.empty:
                errors_df.insert(0, "sheet", sheet_name)
        return out, errors_df

    # Config-driven path
    if config is not None:
        if yaml is None:
            raise RuntimeError(
                "YAML support not available. Please install pyyaml (uv add pyyaml) to use --config."
            )
        cfg = yaml.safe_load(Path(config).read_text())
        sheets_cfg = cfg.get("sheets")
        if not isinstance(sheets_cfg, list) or not sheets_cfg:
            raise ValueError("Config must define a non-empty 'sheets' list")
        outs: List[pd.DataFrame] = []
        errs: List[pd.DataFrame] = []
        for s in sheets_cfg:
            s_name = s.get("name")
            s_header = int(s.get("header_row", header_row))
            cols_map = s.get("columns", {})
            # Normalize keys for canonical fields
            explicit_map = {
                "site_name": cols_map.get("site_name"),
                "address": cols_map.get("address"),
                "city": cols_map.get("city"),
                "state": cols_map.get("state"),
                "zip": cols_map.get("zip"),
            }
            if not all(explicit_map.values()):
                raise ValueError(
                    f"Config sheet '{s_name}' must specify columns for site_name, address, city, state, zip"
                )
            sdf = pd.read_excel(input_path, sheet_name=s_name, header=s_header, dtype={})
            o, e = process_one(sdf, s_name, explicit_map)
            outs.append(o)
            if not e.empty:
                errs.append(e)
        out_df = pd.concat(outs, ignore_index=True)
        err_df = pd.concat(errs, ignore_index=True) if errs else pd.DataFrame(columns=["sheet","row_index","issue"])
        return out_df, err_df

    # Heuristic sheets path
    if all_sheets or (isinstance(sheet, str) and sheet.lower() == "all"):
        book = pd.read_excel(input_path, sheet_name=None, header=header_row, dtype={})
        outs: List[pd.DataFrame] = []
        errs: List[pd.DataFrame] = []
        for sname, sdf in book.items():
            o, e = process_one(sdf, sname)
            outs.append(o)
            if not e.empty:
                errs.append(e)
        out_df = pd.concat(outs, ignore_index=True) if outs else pd.DataFrame(columns=["site_name","address","city","state","zip","source_sheet"]) 
        err_df = pd.concat(errs, ignore_index=True) if errs else pd.DataFrame(columns=["sheet","row_index","issue"]) 
        return out_df, err_df
    else:
        # Single sheet (index 0 when None)
        df_raw = pd.read_excel(
            input_path,
            sheet_name=(sheet if sheet is not None else 0),
            header=header_row,
            dtype={},
        )
        if isinstance(df_raw, dict):
            df = next(iter(df_raw.values()))
            sheet_name = next(iter(df_raw.keys()))
        else:
            df = df_raw
            sheet_name = str(sheet) if sheet is not None else None
        return process_one(df, sheet_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse client site XLSX into canonical sites.csv")
    parser.add_argument("input", type=Path, help="Path to input .xlsx file")
    parser.add_argument("-o", "--output", type=Path, default=Path("data/sites.csv"), help="Output CSV path")
    parser.add_argument("--sheet", type=str, default=None, help="Sheet name (default: first sheet). Use 'all' to parse all sheets.")
    parser.add_argument("--header-row", type=int, default=0, help="Header row index (0-based)")
    parser.add_argument("--errors", type=Path, default=Path("data/sites_errors.csv"), help="Optional errors report CSV path")
    parser.add_argument("--all-sheets", action="store_true", help="Parse all sheets in the workbook")
    parser.add_argument("--debug", action="store_true", help="Print column mapping info")
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config to lock sheets and exact columns")

    args = parser.parse_args()

    clean, errors = parse_xlsx(
        args.input,
        sheet=args.sheet,
        header_row=args.header_row,
        all_sheets=args.all_sheets,
        debug=args.debug,
        config=args.config,
    )

    # Ensure destination directories exist
    args.output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.output, index=False, encoding="utf-8")

    if not errors.empty:
        args.errors.parent.mkdir(parents=True, exist_ok=True)
        errors.to_csv(args.errors, index=False, encoding="utf-8")
        print(f"Wrote {len(errors)} validation issues to {args.errors}")

    print(f"Wrote {len(clean)} rows to {args.output}")


if __name__ == "__main__":
    main()
