from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import folium
import pandas as pd
from folium.plugins import MarkerCluster
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import math
from sklearn.cluster import KMeans


def load_points(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Expect columns: site_name,address,city,state,zip,lat,lon
    if not {"lat", "lon"}.issubset(df.columns):
        raise KeyError("CSV must contain 'lat' and 'lon' columns")
    return df.dropna(subset=["lat", "lon"])  # keep only geocoded


def make_map(
    df: pd.DataFrame,
    output_html: Path,
    center_lat: Optional[float] = None,
    center_lon: Optional[float] = None,
    zoom_start: int = 5,
    cluster: bool = True,
    team_col: Optional[str] = None,
    draw_routes: bool = False,
    route_point_cap: int = 50,
    route_time_limit_sec: int = 3,
    routes_file: Optional[Path] = None,
) -> Path:
    if df.empty:
        raise ValueError("No points to plot")

    if center_lat is None or center_lon is None:
        center_lat = float(df["lat"].astype(float).mean())
        center_lon = float(df["lon"].astype(float).mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles="OpenStreetMap")

    if cluster:
        mc = MarkerCluster(name="Sites", disableClusteringAtZoom=12)
        mc.add_to(m)
        target = mc
    else:
        target = m

    # Color palette for teams (cycled)
    team_colors = [
        "blue","red","green","purple","orange","darkred","lightred","beige",
        "darkblue","darkgreen","cadetblue","darkpurple","white","pink","lightblue",
        "lightgreen","gray","black","lightgray",
    ]
    def color_for(team: str) -> str:
        if not team:
            return "blue"
        idx = abs(hash(team)) % len(team_colors)
        return team_colors[idx]

    for _, row in df.iterrows():
        name = str(row.get("site_name", ""))
        addr = str(row.get("address", ""))
        city = str(row.get("city", ""))
        state = str(row.get("state", ""))
        zip_code = str(row.get("zip", ""))
        lat = float(row["lat"])  # type: ignore[arg-type]
        lon = float(row["lon"])  # type: ignore[arg-type]
        team = str(row.get(team_col, "")) if team_col and team_col in df.columns else ""
        popup = folium.Popup(
            folium.IFrame(
                html=f"<b>{name}</b><br/>{addr}<br/>{city}, {state} {zip_code}",
                width=240,
                height=120,
            ),
            max_width=260,
        )
        icon = folium.Icon(color=color_for(team)) if team else None
        if icon:
            folium.Marker([lat, lon], popup=popup, icon=icon).add_to(target)
        else:
            folium.Marker([lat, lon], popup=popup).add_to(target)

    # Optional: draw per-team routes from a precomputed routes file (preferred)
    if routes_file is not None:
        rf = Path(routes_file)
        if rf.exists():
            r = pd.read_csv(rf)
            # Normalize expected columns: allow team_id->team, sequence->seq
            if "team" not in r.columns and "team_id" in r.columns:
                r = r.rename(columns={"team_id": "team"})
            if "seq" not in r.columns and "sequence" in r.columns:
                r = r.rename(columns={"sequence": "seq"})
            required = {"team", "seq", "lat", "lon"}
            if required.issubset(r.columns):
                # Filter out synthetic END nodes if present
                if "site_name" in r.columns:
                    r = r[~r["site_name"].astype(str).str.startswith("END_")]
                # Ensure numeric and sorted
                r["seq"] = pd.to_numeric(r["seq"], errors="coerce")
                r = r.dropna(subset=["seq", "lat", "lon"]).copy()
                for team, g in r.groupby("team"):
                    g = g.sort_values("seq")
                    coords = list(zip(g["lat"].astype(float), g["lon"].astype(float)))
                    if len(coords) >= 2:
                        folium.PolyLine(
                            locations=coords,
                            color=color_for(str(team)),
                            weight=3,
                            opacity=0.8,
                            tooltip=f"Team {team} precomputed route (n={len(coords)})",
                        ).add_to(m)
        else:
            print(f"[WARN] routes file not found: {routes_file}")

    # Or compute on the fly: per-team routes using OR-Tools TSP over capped points
    elif draw_routes and (team_col and team_col in df.columns):
        for team, g in df.groupby(team_col):
            # Cap to avoid heavy runs; take first N by appearance
            sub = g.head(route_point_cap).copy()
            if len(sub) < 2:
                continue
            lats = sub["lat"].astype(float).tolist()
            lons = sub["lon"].astype(float).tolist()
            order = _tsp_order(lats, lons, time_limit_sec=route_time_limit_sec)
            if not order:
                continue
            coords = [(lats[i], lons[i]) for i in order]
            folium.PolyLine(
                locations=coords,
                color=color_for(str(team)),
                weight=3,
                opacity=0.8,
                tooltip=f"Team {team} route (n={len(sub)})",
            ).add_to(m)

    folium.LayerControl().add_to(m)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_html))
    return output_html


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    R = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    h = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*R*math.asin(math.sqrt(h))


def _tsp_order(lats: list[float], lons: list[float], time_limit_sec: int = 3) -> list[int]:
    n = len(lats)
    if n < 2:
        return list(range(n))
    coords = list(zip(lats, lons))
    dist = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i][j] = 0
            else:
                dist[i][j] = int(_haversine_km(coords[i], coords[j]) * 1000)

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_cb(from_index: int, to_index: int) -> int:
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return dist[i][j]

    transit_index = routing.RegisterTransitCallback(distance_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(int(time_limit_sec))

    solution = routing.SolveWithParameters(params)
    if not solution:
        return []
    order: list[int] = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        order.append(node)
        index = solution.Value(routing.NextVar(index))
    order.append(manager.IndexToNode(index))
    return order


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an OSM map from a geocoded CSV using Folium")
    parser.add_argument("input", type=Path, help="Input geocoded CSV (must include lat,lon)")
    parser.add_argument("-o", "--output", type=Path, default=Path("data/map.html"), help="Output HTML map path")
    parser.add_argument("--no-cluster", action="store_true", help="Disable marker clustering")
    parser.add_argument("--zoom", type=int, default=5, help="Initial zoom level (default 5)")
    parser.add_argument("--team-col", type=str, default=None, help="Optional column name for team assignment to color markers")
    parser.add_argument("--routes", action="store_true", help="Overlay per-team TSP routes (capped)")
    parser.add_argument("--routes-file", type=Path, default=None, help="CSV with precomputed routes: team,seq,lat,lon")
    parser.add_argument("--route-limit", type=int, default=50, help="Max points per team for route overlay (default 50)")
    parser.add_argument("--route-timeout", type=int, default=3, help="Route solver time limit per team in seconds (default 3)")
    parser.add_argument("--kmeans-teams", type=int, default=None, help="If provided and team-col is absent, auto-assign KMeans clusters as teams")

    args = parser.parse_args()

    df = load_points(args.input)
    # Optional KMeans assignment to create a team column
    team_col = args.team_col
    if team_col is None and args.kmeans_teams and args.kmeans_teams > 0:
        # Use lat/lon for clustering; simple KMeans, no scaling
        coords = df[["lat", "lon"]].astype(float).to_numpy()
        k = max(1, int(args.kmeans_teams))
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = kmeans.fit_predict(coords)
        df = df.copy()
        df["team"] = [f"T{int(i)+1}" for i in labels]
        team_col = "team"
    make_map(
        df,
        output_html=args.output,
        zoom_start=args.zoom,
        cluster=(not args.no_cluster),
        team_col=team_col,
        draw_routes=args.routes,
        route_point_cap=args.route_limit,
        route_time_limit_sec=args.route_timeout,
        routes_file=args.routes_file,
    )

    print(f"Wrote map to {args.output}")


if __name__ == "__main__":
    main()
