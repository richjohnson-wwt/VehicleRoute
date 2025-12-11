from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def parse_hhmm(s: str) -> int:
    s = str(s).strip()
    t = datetime.strptime(s, "%H:%M")
    base = datetime(t.year, t.month, t.day)
    return int((t - base).total_seconds() // 60)


def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    R = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def build_time_matrix(lats: List[float], lons: List[float], speed_kmph: float) -> List[List[int]]:
    n = len(lats)
    minutes_per_km = 60.0 / max(1e-6, speed_kmph)
    coords = list(zip(lats, lons))
    mat: List[List[int]] = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i][j] = 0
            else:
                km = haversine_km(coords[i], coords[j])
                mat[i][j] = int(round(km * minutes_per_km))
    return mat


essential_cols = [
    "site_name",
    "lat",
    "lon",
    "is_depot",
    "window_start",
    "window_end",
    "service_min",
]


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in essential_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df = df.copy()
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)
    df["is_depot"] = df["is_depot"].astype(int)
    df["service_min"] = df["service_min"].fillna(0).astype(int)
    return df


def solve_from_csv(
    csv_path: Path,
    teams: int,
    speed_kmph: float,
    team_col: Optional[str] = None,
    no_return: bool = False,
    horizon_min: int = 24 * 60,
    ignore_windows: bool = False,
    time_limit_sec: int = 10,
    diagnose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = load_csv(csv_path)

    # Identify depot (first is_depot == 1)
    depot_idx = df.index[df["is_depot"] == 1]
    if len(depot_idx) == 0:
        raise ValueError("CSV must contain a depot row with is_depot=1")
    depot_pos = int(depot_idx[0])

    # Build arrays in csv order
    names = df["site_name"].astype(str).tolist()
    lats = df["lat"].astype(float).tolist()
    lons = df["lon"].astype(float).tolist()
    service = df["service_min"].astype(int).tolist()
    # Ensure depot has zero service time
    if 0 <= depot_pos < len(service):
        service[depot_pos] = 0

    # Time windows per row
    windows: List[Tuple[int, int]] = []
    for _, row in df.iterrows():
        if ignore_windows:
            ws = 0
            we = int(horizon_min)
        else:
            ws = parse_hhmm(row["window_start"]) if pd.notna(row["window_start"]) else 0
            we = parse_hhmm(row["window_end"]) if pd.notna(row["window_end"]) else 24 * 60
        if we < ws:
            raise ValueError(f"window_end < window_start for site {row.get('site_name')}")
        windows.append((ws, we))

    # Manager and routing are instantiated after optional open-route handling in main(),
    # so we return necessary components upward. We keep construction here for clarity.

    # Fixed team assignment (optional)
    label_to_vehicle: Optional[dict[str, int]] = None
    if team_col is not None and team_col in df.columns:
        cust_df = df[df["is_depot"] == 0]
        raw_labels = [str(x).strip() if pd.notna(x) else "" for x in cust_df[team_col].tolist()]
        uniq = sorted({x for x in raw_labels if x})
        if uniq:
            if teams != len(uniq):
                print(
                    f"[WARN] teams argument ({teams}) != unique labels in {team_col} ({len(uniq)}): {uniq}; auto-adjusting teams={len(uniq)}"
                )
                teams = len(uniq)
            label_to_vehicle = {lab: i for i, lab in enumerate(uniq)}

    n0 = len(lats)
    time_matrix = build_time_matrix(lats, lons, speed_kmph)
    dummy_indices: List[int] = []
    if no_return:
        # Add one dummy end node per vehicle: free to end (0 inbound), impossible to leave (BIG outbound)
        BIG = 10**6
        for v in range(teams):
            dummy_idx = n0 + len(dummy_indices)
            dummy_indices.append(dummy_idx)
            # Append columns (0 cost) to existing rows
            for row in time_matrix:
                row.append(0)
            # Append new dummy row (BIG to all), then set self to 0
            time_matrix.append([BIG] * (len(time_matrix[0])))
            # After appending columns above, len(time_matrix[0]) equals new width
            time_matrix[dummy_idx][dummy_idx] = 0
            # Extend attributes for completeness
            lats.append(lats[depot_pos])
            lons.append(lons[depot_pos])
            names.append(f"END_{v}")
            service.append(0)
        n = n0 + len(dummy_indices)
    else:
        n = n0

    # Create manager/routing with appropriate starts/ends
    if no_return and dummy_indices:
        starts = [depot_pos] * teams
        ends = dummy_indices
        manager = pywrapcp.RoutingIndexManager(n, teams, starts, ends)
    else:
        manager = pywrapcp.RoutingIndexManager(n, teams, depot_pos)
    routing = pywrapcp.RoutingModel(manager)

    # Transit = travel + service at from node
    def transit_cb(from_index: int, to_index: int) -> int:
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return time_matrix[i][j] + service[i]

    cb_index = routing.RegisterTransitCallback(transit_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(cb_index)

    # Time dimension
    routing.AddDimension(
        cb_index,
        slack_max=int(horizon_min),
        capacity=int(horizon_min),
        fix_start_cumul_to_zero=False,
        name="Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")

    # Apply fixed vehicle assignments if provided (customers only)
    if label_to_vehicle is not None:
        for node in range(n):
            if node == depot_pos:
                continue
            label = df.iloc[node][team_col]
            if pd.isna(label):
                continue
            v = label_to_vehicle.get(str(label))
            if v is not None:
                routing.SetAllowedVehiclesForIndex([v], manager.NodeToIndex(node))

    # Apply windows to all customers (skip depot and any dummies)
    dummy_set = set(dummy_indices)
    for node in range(n):
        if node == depot_pos or node in dummy_set:
            continue
        index = manager.NodeToIndex(node)
        ws, we = windows[node]
        time_dim.CumulVar(index).SetRange(ws, we)

    # Apply depot window at each vehicle start; set end windows
    depot_ws, depot_we = windows[depot_pos]
    for v in range(teams):
        s_idx = routing.Start(v)
        e_idx = routing.End(v)
        time_dim.CumulVar(s_idx).SetRange(depot_ws, depot_we)
        if no_return:
            time_dim.CumulVar(e_idx).SetRange(0, int(horizon_min))
        else:
            time_dim.CumulVar(e_idx).SetRange(depot_ws, depot_we)

    # Minimize start and end times to help feasibility
    for v in range(teams):
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.Start(v)))
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.End(v)))

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(int(time_limit_sec))

    if diagnose:
        # quick stats
        import itertools
        coords = list(zip(lats, lons))
        def dist_min(i, j):
            km = haversine_km(coords[i], coords[j])
            return km * (60.0 / max(1e-6, speed_kmph))
        max_pair = 0.0
        for i, j in itertools.combinations(range(len(coords)), 2):
            max_pair = max(max_pair, dist_min(i, j))
        total_service = sum(service) - service[depot_pos]
        print(f"[DIAG] rows={len(df)} teams={teams} horizon_min={horizon_min}")
        print(f"[DIAG] total_service_min={total_service:.1f} max_pair_travel_min~={max_pair:.1f}")

    solution = routing.SolveWithParameters(params)
    if not solution:
        raise RuntimeError("No solution found for hello CSV.")

    # Extract
    rows = []
    summary = []
    for v in range(teams):
        if not routing.IsVehicleUsed(solution, v):
            continue
        index = routing.Start(v)
        seq = 0
        total_time = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            arr = solution.Value(time_dim.CumulVar(index))
            dep = arr + service[node]
            rows.append(
                {
                    "team_id": v,
                    "seq": seq,
                    "node": node,
                    "site_name": names[node],
                    "lat": lats[node],
                    "lon": lons[node],
                    "arrival_min": arr,
                    "depart_min": dep,
                }
            )
            prev = index
            index = solution.Value(routing.NextVar(index))
            if not routing.IsEnd(index):
                i = manager.IndexToNode(prev)
                j = manager.IndexToNode(index)
                total_time += time_matrix[i][j] + service[i]
            seq += 1
        # end node
        node = manager.IndexToNode(index)
        arr = solution.Value(time_dim.CumulVar(index))
        dep = arr + service[node]
        rows.append(
            {
                "team_id": v,
                "seq": seq,
                "node": node,
                "site_name": names[node],
                "lat": lats[node],
                "lon": lons[node],
                "arrival_min": arr,
                "depart_min": dep,
            }
        )
        summary.append({"team_id": v, "total_time_min": arr})

    return pd.DataFrame(rows), pd.DataFrame(summary)


def main() -> None:
    ap = argparse.ArgumentParser(description="Hello VRPTW from small CSV")
    ap.add_argument("input", type=Path, help="CSV with site_name,lat,lon,is_depot,window_start,window_end,service_min")
    ap.add_argument("--teams", type=int, default=2)
    ap.add_argument("--speed-kmph", type=float, default=50.0)
    ap.add_argument("--output", type=Path, default=Path("data/hello/vrptw_hello_routes.csv"))
    ap.add_argument("--summary", type=Path, default=Path("data/hello/vrptw_hello_summary.csv"))
    ap.add_argument("--team-col", type=str, default=None, help="Optional column name for fixed team assignment")
    ap.add_argument("--no-return", action="store_true", help="Allow vehicles to end at last stop (open routes)")
    ap.add_argument("--horizon-min", type=int, default=24 * 60, help="Time dimension horizon in minutes (default 1440)")
    ap.add_argument("--ignore-windows", action="store_true", help="Ignore per-site windows and use [0,horizon] for all")
    ap.add_argument("--time-limit-sec", type=int, default=10, help="Search time limit in seconds (default 10)")
    ap.add_argument("--diagnose", action="store_true", help="Print quick dataset diagnostics before solving")
    args = ap.parse_args()

    routes, summary = solve_from_csv(
        args.input,
        teams=args.teams,
        speed_kmph=args.speed_kmph,
        team_col=args.team_col,
        no_return=args.no_return,
        horizon_min=args.horizon_min,
        ignore_windows=args.ignore_windows,
        time_limit_sec=args.time_limit_sec,
        diagnose=args.diagnose,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    routes.to_csv(args.output, index=False)
    summary.to_csv(args.summary, index=False)
    print(f"Wrote routes: {args.output}\nWrote summary: {args.summary}")


if __name__ == "__main__":
    main()
