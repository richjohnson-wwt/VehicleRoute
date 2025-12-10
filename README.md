
uv add pandas openpyxl pyqt6 pyyaml geopy

* openpyxl is for parsing the XLSX files
* pyqt6 is for the GUI
* pandas is for data manipulation
* pyyaml is for per-client config files

# Parse
uv run python scripts/parse_sites.py "data/Ascension Health Site List.xlsx" --config configs/ascension.yaml --output data/ascension_sites_config.csv --errors data/ascension_sites_config_errors.csv --debug

uv run python scripts/parse_sites.py "data/PNCSiteLists_MapTest.xlsx" --config configs/pnc.yaml --output data/pnc_sites_config.csv --errors data/pnc_sites_config_errors.csv --debug


# Geocode
uv run python scripts/geocode_sites.py data/ascension_sites_config.csv --output data/ascension_sites_geocoded_sample.csv --cache data/cache/geocache.sqlite --limit 5 --user-agent "VehicleRoute-Geocoder/0.1"



uv run python scripts/geocode_sites.py data/ascension_sites_config.csv --output data/ascension_sites_geocoded.csv --cache data/cache/geocache.sqlite --user-agent "VehicleRoute-Geocoder/0.1" --email "rich.johnson@wwt.com"