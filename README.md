
uv add pandas openpyxl pyqt6 pyyaml geopy matplotlib ortools folium scikit-learn

* openpyxl is for parsing the XLSX files
* pyqt6 is for the GUI
* pandas is for data manipulation
* pyyaml is for per-client config files
* matplotlib is for visualizing the routes
* ortools is for routing
* folium is for visualizing the routes on a map
* scikit-learn is for clustering

# Workflow

1. Parse the client's site list into a CSV file
2. Geocode the sites
3. Cluster the sites into teams
4. Route the teams
5. Visualize the routes
6. Export the routes to a CSV file

# Scripts

scripts/parse_sites.py is a small utility that parses a client's Excel .xlsx site list into a CSV file with columns: site_name,address,city,state,zip,source_sheet

scripts/geocode_sites.py uses nominatim to geocode the sites from addresses. It also caches the results in a SQLite database to avoid re-geocoding.

scripts/cluster_tsp.py is our “cluster-first, TSP-second” fallback router.Clusters sites into K teams using KMeans on lat/lon.
    Writes two outputs:
    A sites-with-teams CSV (adds team labels to each site).
    A routes CSV (team, seq, lat, lon, site_name…) suitable for Folium overlays via scripts/make_map.py --routes-file.

scripts/prepare_vrptw_input.py is a small utility that converts a geocoded sites CSV into a “VRPTW-ready” CSV in the hello schema that our solver expects.

scripts/vrptw_from_csv.py is our main VRPTW solver (OR-Tools). Supports: --teams, --team-col, --no-return, --horizon-min, --ignore-windows, --time-limit-sec, --diagnose.

scripts/make_map.py overlays routes on a Folium map; accepts --routes-file with columns (team or team_id, seq, lat, lon).

# Workflow Overview

1) Parse client Excel → canonical CSV
2) Geocode canonical CSV → lat/lon CSV (with cache)
3) Optional: Cluster to seed team labels (cluster_tsp.py), or skip and let VRPTW assign
4) Prepare VRPTW input (insert depot, uniform windows/service) via prepare_vrptw_input.py
5) Solve VRPTW with vrptw_from_csv.py (tune teams, windows, horizon, open routes)
6) Visualize routes with make_map.py (--routes-file) or via the UI

VRPTW quickstart

- Create a small VRPTW-ready subset (15 customers, 08:00–17:00, service 20):
  uv run python scripts/prepare_vrptw_input.py data/ascension_sites_geocoded.csv --output data/asc_test/asc_subset_vrptw.csv --window "08:00-17:00" --service-min 20 --take 15

- Run solver (let OR-Tools assign vehicles):
  uv run python scripts/vrptw_from_csv.py data/asc_test/asc_subset_vrptw.csv --teams 3 --speed-kmph 50 --output data/asc_test/routes.csv --summary data/asc_test/summary.csv

- Fixed teams (if the CSV carries a team column):
  uv run python scripts/vrptw_from_csv.py data/asc_test/asc_subset_vrptw.csv --teams 3 --team-col team --speed-kmph 50 --output data/asc_test/routes_fixed.csv --summary data/asc_test/summary_fixed.csv

- Open routes and multi-day horizon (sanity check for long distances):
  uv run python scripts/vrptw_from_csv.py data/asc_test/asc_subset_vrptw.csv --teams 3 --no-return --ignore-windows --horizon-min 4320 --time-limit-sec 60 --diagnose --output data/asc_test/routes_md.csv --summary data/asc_test/summary_md.csv

- Visualize routes on a map:
  uv run python scripts/make_map.py data/asc_test/asc_subset_vrptw.csv --routes-file data/asc_test/routes_md.csv

# Next Steps

- Regionalize Ascension subsets (e.g., FL-only, KS-only) and validate VRPTW per region
- Regenerate team labels geographically (KMeans) to avoid cross-state teams, then re-run VRPTW
- Tighten from multi-day/ignore-windows toward realistic (08:00–17:00, service 15–20 min)
- Consider compacting team IDs in outputs and adding legends to Folium maps
- Wire a "Load Routes CSV" action in the PyQt visualizer to open make_map with overlays
- Document known limits and recommended flags in this README (speed, horizon, timeouts)


scripts/prepare_vrptw_input.py is a small utility that converts a geocoded sites CSV into a “VRPTW-ready” CSV in the hello schema that our solver expects.

scripts/vrptw_from_csv.py is our main solver. It takes a VRPTW-ready CSV and solves it using OR-Tools.

scripts/make_map.py is a small utility that takes a routes CSV and a geocoded sites CSV and makes a Folium map with the routes overlaid.

# Parse
uv run python scripts/parse_sites.py "data/Ascension Health Site List.xlsx" --config configs/ascension.yaml --output data/ascension_sites_config.csv --errors data/ascension_sites_config_errors.csv --debug

uv run python scripts/parse_sites.py "data/PNCSiteLists_MapTest.xlsx" --config configs/pnc.yaml --output data/pnc_sites_config.csv --errors data/pnc_sites_config_errors.csv --debug


# Geocode
uv run python scripts/geocode_sites.py data/ascension_sites_config.csv --output data/ascension_sites_geocoded_sample.csv --cache data/cache/geocache.sqlite --limit 5 --user-agent "VehicleRoute-Geocoder/0.1"



uv run python scripts/geocode_sites.py data/ascension_sites_config.csv --output data/ascension_sites_geocoded.csv --cache data/cache/geocache.sqlite --user-agent "VehicleRoute-Geocoder/0.1" --email "rich.johnson@wwt.com"


# VRPTW - this worked
uv run python scripts/vrptw.py data/ascension/List_sites_with_teams.csv --teams 7 --team-col team --no-windows --no-return --output data/ascension/vrp_routes_fixedteams.csv --summary data/ascension/vrp_summary_fixedteams.csv

# UI apps

uv run python main.py - UI app for parsing and geocoding Excel data. Runs scripts' parse_sites.py and geocode_sites.py.
uv run python visualizer.py - UI app for visualizing routes. Runs scripts' cluster_tsp.py and make_map.py.

# OR-Tools Sample output for VRPTW

    uv run python scripts/vrptw_hello.py 
    Objective: 71
    Route for vehicle 0:
    0 Time(0,0) -> 9 Time(2,3) -> 14 Time(7,8) -> 16 Time(11,11) -> 0 Time(18,18)
    Time of the route: 18min

    Route for vehicle 1:
    0 Time(0,0) -> 7 Time(2,4) -> 1 Time(7,11) -> 4 Time(10,13) -> 3 Time(16,16) -> 0 Time(24,24)
    Time of the route: 24min

    Route for vehicle 2:
    0 Time(0,0) -> 12 Time(4,4) -> 13 Time(6,6) -> 15 Time(11,11) -> 11 Time(14,14) -> 0 Time(20,20)
    Time of the route: 20min

    Route for vehicle 3:
    0 Time(0,0) -> 5 Time(3,3) -> 8 Time(5,5) -> 6 Time(7,7) -> 2 Time(10,10) -> 10 Time(14,14) -> 0 Time(20,20)
    Time of the route: 20min

    Total time of all routes: 82min