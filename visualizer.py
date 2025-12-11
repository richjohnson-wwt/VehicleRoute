import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QHBoxLayout,
    QSpinBox,
    QCheckBox,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import math
from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtCore import QProcess


class PlotCanvas(FigureCanvas):
    def __init__(self, parent: Optional[QWidget] = None):
        fig = Figure(figsize=(6, 4), tight_layout=True)
        super().__init__(fig)
        self.setParent(parent)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Geocoded Sites")
        self.ax.set_xlabel("Longitude")
        self.ax.set_ylabel("Latitude")

    def plot_points(self, lons, lats, title: str = "Geocoded Sites") -> None:
        self.ax.clear()
        self.ax.scatter(lons, lats, s=10, c="#1f77b4", alpha=0.8, edgecolors="none")
        self.ax.set_title(title)
        self.ax.set_xlabel("Longitude")
        self.ax.set_ylabel("Latitude")
        if len(lons) and len(lats):
            pad_x = max(0.1, (max(lons) - min(lons)) * 0.05)
            pad_y = max(0.1, (max(lats) - min(lats)) * 0.05)
            self.ax.set_xlim(min(lons) - pad_x, max(lons) + pad_x)
            self.ax.set_ylim(min(lats) - pad_y, max(lats) + pad_y)
        self.ax.grid(True, alpha=0.3)
        self.draw()


class VisualizerWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Site Visualizer")

        central = QWidget(self)
        layout = QVBoxLayout(central)

        self.btn_open = QPushButton("Open geocoded CSV", self)
        self.btn_route = QPushButton("Compute OR-Tools Route", self)
        # OSM/Folium controls
        osm_row = QHBoxLayout()
        self.btn_osm = QPushButton("Open OSM Map (Browser)", self)
        self.spin_teams = QSpinBox(self)
        self.spin_teams.setRange(0, 100)
        self.spin_teams.setValue(0)
        self.spin_teams.setToolTip("Number of teams for KMeans auto-assignment (0 = no auto teams)")
        self.chk_routes = QCheckBox("Overlay Routes", self)
        self.chk_routes.setChecked(False)
        osm_row.addWidget(self.btn_osm)
        osm_row.addWidget(self.spin_teams)
        osm_row.addWidget(self.chk_routes)
        self.canvas = PlotCanvas(self)

        layout.addWidget(self.btn_open)
        layout.addWidget(self.btn_route)
        layout.addLayout(osm_row)
        layout.addWidget(self.canvas)
        self.setCentralWidget(central)

        self.btn_open.clicked.connect(self.on_open_clicked)
        self.btn_route.clicked.connect(self.on_route_clicked)
        self.btn_osm.clicked.connect(self.on_osm_clicked)

        # Data placeholders for currently loaded points
        self._lons: list[float] = []
        self._lats: list[float] = []
        self._proc: Optional[QProcess] = None

    def on_open_clicked(self) -> None:
        csv_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select geocoded CSV (with lat,lon)",
            str(Path.cwd() / "data"),
            "CSV Files (*.csv);;All Files (*)",
        )
        if not csv_path:
            return
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read CSV:\n{e}")
            return

        # Expect columns: site_name,address,city,state,zip,lat,lon,geocode_status,...
        if not {"lat", "lon"}.issubset(df.columns):
            QMessageBox.warning(self, "Missing columns", "CSV must contain 'lat' and 'lon' columns.")
            return

        # Filter valid coordinates
        valid = df.dropna(subset=["lat", "lon"]).copy()
        title = f"Geocoded Sites: {len(valid)} points"
        try:
            lons = valid["lon"].astype(float).tolist()
            lats = valid["lat"].astype(float).tolist()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Invalid coordinate values:\n{e}")
            return

        if not lons or not lats:
            QMessageBox.information(self, "No points", "No valid coordinates found to plot.")
            return

        # Save to instance for routing later
        self._lons = lons
        self._lats = lats
        self.canvas.plot_points(lons, lats, title)

    # ---- Generate Folium OSM map in browser ----
    def on_osm_clicked(self) -> None:
        csv_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select geocoded CSV (with lat,lon)",
            str(Path.cwd() / "data"),
            "CSV Files (*.csv);;All Files (*)",
        )
        if not csv_path:
            return

        suggested_out = Path(csv_path).with_suffix("")
        suggested_out = suggested_out.parent / (suggested_out.name + "_map.html")
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save OSM map as",
            str(suggested_out),
            "HTML Files (*.html)",
        )
        if not out_path:
            return

        args = [
            sys.executable,
            str(Path(__file__).resolve().parent / "scripts" / "make_map.py"),
            csv_path,
            "--output",
            out_path,
        ]
        teams = int(self.spin_teams.value())
        if teams > 0:
            args += ["--kmeans-teams", str(teams)]
        if self.chk_routes.isChecked():
            args += ["--routes"]

        # Run generator without freezing UI
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
            self._proc = None
        self._proc = QProcess(self)
        self._proc.finished.connect(lambda code, status: self._on_osm_finished(out_path, code))
        self._proc.start(args[0], args[1:])

    def _on_osm_finished(self, out_path: str, code: int) -> None:
        if code != 0:
            QMessageBox.warning(self, "OSM Map", "Failed to generate map. See console for details.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(out_path))

    # ---- OR-Tools routing over current points ----
    def on_route_clicked(self) -> None:
        if not self._lons or not self._lats:
            QMessageBox.information(self, "No data", "Load a geocoded CSV first.")
            return
        if len(self._lons) < 2:
            QMessageBox.information(self, "Not enough points", "Need at least 2 points to compute a route.")
            return

        # Build a symmetric distance matrix using haversine distance (km)
        coords = list(zip(self._lats, self._lons))  # (lat, lon)
        n = len(coords)
        dist = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    dist[i][j] = 0
                else:
                    dist[i][j] = int(self._haversine_km(coords[i], coords[j]) * 1000)  # meters as int

        # OR-Tools TSP: single vehicle, start/end at 0
        manager = pywrapcp.RoutingIndexManager(n, 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_cb(from_index: int, to_index: int) -> int:
            i = manager.IndexToNode(from_index)
            j = manager.IndexToNode(to_index)
            return dist[i][j]

        transit_index = routing.RegisterTransitCallback(distance_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_index)

        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_params.time_limit.FromSeconds(5)

        solution = routing.SolveWithParameters(search_params)
        if not solution:
            QMessageBox.warning(self, "Routing", "Failed to compute a route.")
            return

        # Extract order
        index = routing.Start(0)
        order: list[int] = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            order.append(node)
            index = solution.Value(routing.NextVar(index))
        order.append(manager.IndexToNode(index))  # end node

        # Plot points and the route as a polyline
        lons = self._lons
        lats = self._lats
        self.canvas.ax.clear()
        self.canvas.ax.scatter(lons, lats, s=12, c="#1f77b4", alpha=0.9, edgecolors="none")
        # draw route
        route_lons = [lons[i] for i in order]
        route_lats = [lats[i] for i in order]
        self.canvas.ax.plot(route_lons, route_lats, "-", c="#d62728", linewidth=1.5, alpha=0.9)
        self.canvas.ax.set_title(f"Route over {len(lons)} points (start=0)")
        self.canvas.ax.set_xlabel("Longitude")
        self.canvas.ax.set_ylabel("Latitude")
        pad_x = max(0.1, (max(lons) - min(lons)) * 0.05)
        pad_y = max(0.1, (max(lats) - min(lats)) * 0.05)
        self.canvas.ax.set_xlim(min(lons) - pad_x, max(lons) + pad_x)
        self.canvas.ax.set_ylim(min(lats) - pad_y, max(lats) + pad_y)
        self.canvas.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    @staticmethod
    def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
        # Returns great-circle distance between two (lat, lon) in km
        lat1, lon1 = a
        lat2, lon2 = b
        R = 6371.0088
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        h = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        return 2*R*math.asin(math.sqrt(h))


def main() -> None:
    app = QApplication(sys.argv)
    win = VisualizerWindow()
    win.resize(900, 600)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
