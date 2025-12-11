from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.cluster import KMeans
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


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


def build_dist_matrix(lats: List[float], lons: List[float]) -> List[List[int]]:
    n = len(lats)
    coords = list(zip(lats, lons))
    mat: List[List[int]] = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i][j] = 0
            else:
                # meters
                mat[i][j] = int(haversine_km(coords[i], coords[j]) * 1000)
    return mat


def tsp_order(lats: List[float], lons: List[float], time_limit_s: int = 5) -> List[int]:
    n = len(lats)
    if n <= 1:
        return list(range(n))

    # choose start as point closest to centroid to stabilize route
    centroid = (sum(lats) / n, sum(lons) / n)
    start = min(range(n), key=lambda i: haversine_km((lats[i], lons[i]), centroid))

    manager = pywrapcp.RoutingIndexManager(n, 1, [start], [start])  # closed tour starting/ending at start
    routing = pywrapcp.RoutingModel(manager)

    dist = build_dist_matrix(lats, lons)

    def dist_cb(from_index: int, to_index: int) -> int:
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return dist[i][j]

    cb = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(cb)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(int(time_limit_s))

    solution = routing.SolveWithParameters(params)
    if not solution:
        # fallback to trivial order
        return list(range(n))

    index = routing.Start(0)
    order: List[int] = []
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        order.append(node)
        index = solution.Value(routing.NextVar(index))
    order.append(manager.IndexToNode(index))  # return to start
    return order


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster-first team TSP routing (KMeans + per-team TSP)")
    parser.add_argument("input", type=Path, help="Geocoded CSV with columns: site_name, lat, lon")
    parser.add_argument("--teams", type=int, required=True, help="Number of teams (KMeans clusters)")
    parser.add_argument("--time-limit", type=int, default=5, help="Per-team TSP time limit (seconds)")
    parser.add_argument("--output-routes", type=Path, default=Path("data/cluster_routes.csv"), help="Output CSV for team routes")
    parser.add_argument("--output-teams", type=Path, default=None, help="Optional CSV path to write team labels per site")

    args = parser.parse_args()

    df = pd.read_csv(args.input)
    required = {"site_name", "lat", "lon"}
    if not required.issubset(df.columns):
        raise KeyError(f"Input must include columns: {sorted(required)}")
    df = df.dropna(subset=["lat", "lon"]).copy()
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)

    if len(df) == 0:
        raise ValueError("No geocoded rows found.")

    # KMeans assignment
    k = max(1, int(args.teams))
    coords = df[["lat", "lon"]].to_numpy()
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(coords)
    df["team"] = [f"T{int(i)+1}" for i in labels]

    # Optional write of teams
    if args.output_teams is not None:
        outp = Path(args.output_teams)
        outp.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(outp, index=False)
        print(f"Wrote team labels to {outp}")

    # Per-team TSP
    rows = []
    for team, g in df.groupby("team"):
        lats = g["lat"].tolist()
        lons = g["lon"].tolist()
        order = tsp_order(lats, lons, time_limit_s=int(args.time_limit))
        # order contains indices into lats/lons; map back to original rows (by position)
        g_reset = g.reset_index(drop=True)
        seq = 0
        for idx in order:
            r = g_reset.loc[idx]
            rows.append(
                {
                    "team": team,
                    "seq": seq,
                    "site_name": r.get("site_name", ""),
                    "lat": float(r["lat"]),
                    "lon": float(r["lon"]),
                }
            )
            seq += 1

    routes_df = pd.DataFrame(rows)
    args.output_routes.parent.mkdir(parents=True, exist_ok=True)
    routes_df.to_csv(args.output_routes, index=False)
    print(f"Wrote {len(routes_df)} route rows to {args.output_routes}")


if __name__ == "__main__":
    main()
