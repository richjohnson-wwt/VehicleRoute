from __future__ import annotations

import argparse
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def parse_time_window(s: str) -> Tuple[int, int]:
    """Parse HH:MM-HH:MM to minutes since day start."""
    try:
        start_s, end_s = s.split("-")
        t0 = datetime.strptime(start_s.strip(), "%H:%M")
        t1 = datetime.strptime(end_s.strip(), "%H:%M")
        base = datetime(t0.year, t0.month, t0.day)
        start = int((t0 - base).total_seconds() // 60)
        end = int((t1 - base).total_seconds() // 60)
        if end < start:
            raise ValueError("End must be after start")
        return start, end
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid time window '{s}': {e}")


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


def load_sites(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"site_name", "lat", "lon"}
    if not required.issubset(df.columns):
        raise KeyError(f"Input CSV must include columns: {sorted(required)}")
    # Filter rows with valid coordinates
    df = df.dropna(subset=["lat", "lon"]).copy()
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)
    return df


def build_time_matrix(lats: List[float], lons: List[float], speed_kmph: float) -> List[List[int]]:
    n = len(lats)
    coords = list(zip(lats, lons))
    minutes_per_km = 60.0 / max(1e-6, speed_kmph)
    mat: List[List[int]] = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i][j] = 0
            else:
                km = haversine_km(coords[i], coords[j])
                minutes = km * minutes_per_km
                mat[i][j] = int(round(minutes))
    return mat


def solve_vrptw(
    df: pd.DataFrame,
    teams: int,
    depot_lat: Optional[float],
    depot_lon: Optional[float],
    workday: Tuple[int, int],
    service_min: int,
    speed_kmph: float,
    no_return: bool = False,
    vehicle_label_map: Optional[dict[str, int]] = None,
    team_labels: Optional[list[Optional[str]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Build nodes: depot (index 0) + all sites 1..N
    if depot_lat is None or depot_lon is None:
        depot_lat = float(df["lat"].mean())
        depot_lon = float(df["lon"].mean())
    depot = (depot_lat, depot_lon)

    lats = [depot[0]] + df["lat"].astype(float).tolist()
    lons = [depot[1]] + df["lon"].astype(float).tolist()
    names = ["DEPOT"] + df["site_name"].astype(str).tolist()

    n_nodes = len(lats)
    time_mat = build_time_matrix(lats, lons, speed_kmph)

    # Service times (min): 0 for depot, service_min for sites
    service_times = [0] + [int(service_min)] * (n_nodes - 1)

    # Implement open routes by adding a distinct dummy end node per vehicle
    dummy_end_indices: list[int] = []
    if no_return:
        BIG = 10**6
        for v in range(teams):
            dummy_idx = n_nodes
            dummy_end_indices.append(dummy_idx)
            # Extend lats/lons/names for completeness (not used downstream)
            lats.append(depot[0])
            lons.append(depot[1])
            names.append(f"END_{v}")
            # Extend service times
            service_times.append(0)
            # Extend existing rows with 0 to dummy (free to end)
            for row in time_mat:
                row.append(0)
            # Add new dummy row initialized to BIG (disallow leaving dummy)
            time_mat.append([BIG] * (n_nodes + 1))
            n_nodes += 1

    starts = [0] * teams
    if no_return and dummy_end_indices:
        ends = dummy_end_indices
    else:
        ends = [0] * teams
    manager = pywrapcp.RoutingIndexManager(n_nodes, teams, starts, ends)
    routing = pywrapcp.RoutingModel(manager)

    # Apply fixed team assignments if provided
    orig_n_sites = len(df)  # customers correspond to nodes 1..orig_n_sites
    if vehicle_label_map is not None and team_labels is not None:
        for i in range(1, orig_n_sites + 1):
            label = team_labels[i - 1]
            if label is None:
                continue
            v = vehicle_label_map.get(str(label))
            if v is None:
                continue
            index = manager.NodeToIndex(i)
            routing.SetAllowedVehiclesForIndex([v], index)

    # Transit = travel time + service time at 'from' node
    def transit_cb(from_index: int, to_index: int) -> int:
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return time_mat[i][j] + service_times[i]

    transit_index = routing.RegisterTransitCallback(transit_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)

    # Time dimension per OR-Tools sample: allow waiting, 24h horizon
    routing.AddDimension(
        transit_index,
        slack_max=24 * 60,
        capacity=24 * 60,
        fix_start_cumul_to_zero=True,
        name="Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")

    # Global customer windows, depot windows on starts/ends; skip dummy ends
    start_min, end_min = workday
    for v in range(teams):
        s_idx = routing.Start(v)
        e_idx = routing.End(v)
        time_dim.CumulVar(s_idx).SetRange(0, end_min)
        # keep end within horizon; we can allow full day to avoid tightness
        time_dim.CumulVar(e_idx).SetRange(0, 24 * 60)

    dummy_set = set(dummy_end_indices)
    for node in range(1, n_nodes):
        if node in dummy_set:
            continue
        index = manager.NodeToIndex(node)
        time_dim.CumulVar(index).SetRange(start_min, end_min)

    # Search parameters
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(60)

    solution = routing.SolveWithParameters(params)
    if not solution:
        raise RuntimeError("No VRPTW solution found.")

    # Extract routes
    rows = []
    summary = []
    for v in range(teams):
        index = routing.Start(v)
        seq = 0
        total_travel = 0
        route_nodes: List[int] = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route_nodes.append(node)
            arr = solution.Value(time_dim.CumulVar(index))
            dep = arr + service_times[node]
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
            prev_index = index
            index = solution.Value(routing.NextVar(index))
            if not routing.IsEnd(index):
                i = manager.IndexToNode(prev_index)
                j = manager.IndexToNode(index)
                total_travel += time_mat[i][j]
            seq += 1
        # add end depot
        node = manager.IndexToNode(index)
        arr = solution.Value(time_dim.CumulVar(index))
        dep = arr + service_times[node]
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
        total_duration = arr  # since start is 0
        summary.append(
            {
                "team_id": v,
                "stops_including_depot": len(route_nodes) + 1,
                "total_travel_min": total_travel,
                "total_duration_min": total_duration,
            }
        )

    route_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summary)
    return route_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="VRPTW solver (PoC) over geocoded CSV using OR-Tools")
    parser.add_argument("input", type=Path, help="Input CSV with site_name, lat, lon")
    parser.add_argument("--teams", type=int, required=True, help="Number of vehicles/teams")
    parser.add_argument("--depot", type=str, default=None, help='Depot as "lat,lon" (default: centroid)')
    parser.add_argument("--workday", type=str, default="08:00-17:00", help="Global time window (e.g., 08:00-17:00)")
    parser.add_argument("--service-min", type=int, default=20, help="Service time per site in minutes")
    parser.add_argument("--speed-kmph", type=float, default=50.0, help="Assumed travel speed (km/h) for time estimation")
    parser.add_argument("--output", type=Path, default=Path("data/vrptw_routes.csv"), help="Output CSV for detailed route")
    parser.add_argument("--summary", type=Path, default=Path("data/vrptw_summary.csv"), help="Output CSV summary per team")
    parser.add_argument("--no-return", action="store_true", help="Allow vehicles to end at last stop (approximate)")
    parser.add_argument("--diagnose", action="store_true", help="Print capacity summary and rough lower bounds before solving")
    parser.add_argument("--no-windows", action="store_true", help="Disable time windows and time dimension; solve pure VRP by distance")
    parser.add_argument("--no-service", action="store_true", help="Set service time to 0 minutes for testing")
    parser.add_argument("--routes-file", type=Path, default=None, help="Optional precomputed team routes CSV (team,seq,lat,lon) to derive team assignments")
    parser.add_argument("--team-col", type=str, default=None, help="If provided, use this column from input CSV for team labels")

    args = parser.parse_args()

    df = load_sites(args.input)

    depot_lat = depot_lon = None
    if args.depot:
        try:
            lat_s, lon_s = args.depot.split(",")
            depot_lat = float(lat_s.strip())
            depot_lon = float(lon_s.strip())
        except Exception as e:
            raise argparse.ArgumentTypeError(f"Invalid --depot value: {e}")

    workday = parse_time_window(args.workday)

    if args.diagnose:
        n_sites = len(df)
        w_minutes = workday[1] - workday[0]
        total_service = n_sites * args.service_min
        # crude lower bound for travel: average nearest-neighbor hop from centroid
        coords = list(zip(df["lat"].astype(float).tolist(), df["lon"].astype(float).tolist()))
        if coords:
            centroid = (float(df["lat"].mean()), float(df["lon"].mean()))
            avg_km = sum(haversine_km(centroid, c) for c in coords) / max(1, len(coords))
            minutes_per_km = 60.0 / max(1e-6, args.speed_kmph)
            rough_travel_min = int(avg_km * minutes_per_km * n_sites * 0.5)
        else:
            rough_travel_min = 0
        cap_minutes = args.teams * w_minutes
        print("[DIAG] sites=", n_sites, "teams=", args.teams, "workday_min=", w_minutes)
        print("[DIAG] total_service_min=", total_service, "rough_travel_min>=", rough_travel_min)
        print("[DIAG] capacity_minutes=", cap_minutes)

    # Apply --no-service by forcing service_min=0 for this run if requested
    service_min = 0 if args.no_service else args.service_min

    # Team assignment (optional): from routes-file or team-col
    team_labels = None
    vehicle_map = None
    if args.routes_file is not None and args.routes_file.exists():
        r = pd.read_csv(args.routes_file)
        if {"team", "lat", "lon"}.issubset(r.columns):
            # Map by exact lat/lon
            assign = {(float(row.lat), float(row.lon)): str(row.team) for _, row in r.iterrows()}
            labs = []
            for _, row in df.iterrows():
                labs.append(assign.get((float(row.lat), float(row.lon)), None))
            team_labels = labs
    if team_labels is None and args.team_col and args.team_col in df.columns:
        team_labels = [str(x) if pd.notna(x) else None for x in df[args.team_col].tolist()]

    if team_labels is not None:
        uniq = sorted({t for t in team_labels if t is not None})
        vehicle_map = {label: i for i, label in enumerate(uniq)}
        if len(uniq) != args.teams:
            print(f"[WARN] teams argument ({args.teams}) != unique assigned teams ({len(uniq)}); using teams={len(uniq)}")
            args.teams = len(uniq)

    routes, summary = solve_vrptw(
        df,
        teams=args.teams,
        depot_lat=depot_lat,
        depot_lon=depot_lon,
        workday=workday,
        service_min=service_min,
        speed_kmph=args.speed_kmph,
        no_return=args.no_return,
        # pass vehicle assignment map and labels
        vehicle_label_map=vehicle_map,
        team_labels=team_labels,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    routes.to_csv(args.output, index=False)
    summary.to_csv(args.summary, index=False)
    print(f"Wrote {len(routes)} route rows to {args.output}")
    print(f"Wrote summary per team to {args.summary}")


if __name__ == "__main__":
    main()
