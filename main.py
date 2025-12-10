import sys
from pathlib import Path
from PyQt6.QtCore import Qt, QProcess
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QTextEdit,
    QFileDialog,
    QMessageBox,
)


PROJECT_ROOT = Path(__file__).resolve().parent
EMAIL_CONTACT = "rich.johnson@wwt.com"  # Used for Nominatim contact per policy


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("VehicleRoute")

        self.process: QProcess | None = None

        # UI
        central = QWidget(self)
        layout = QVBoxLayout(central)

        self.btn_parse = QPushButton("Parse XLSX → canonical CSV", self)
        self.btn_geocode = QPushButton("Geocode CSV (Nominatim)", self)
        self.log = QTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Logs will appear here...")

        layout.addWidget(self.btn_parse)
        layout.addWidget(self.btn_geocode)
        layout.addWidget(self.log)
        self.setCentralWidget(central)

        # Connect
        self.btn_parse.clicked.connect(self.on_parse_clicked)
        self.btn_geocode.clicked.connect(self.on_geocode_clicked)

    # ----- Utilities -----
    def append_log(self, text: str) -> None:
        self.log.append(text)
        self.log.ensureCursorVisible()

    def set_busy(self, busy: bool) -> None:
        self.btn_parse.setEnabled(not busy)
        self.btn_geocode.setEnabled(not busy)

    def run_command(self, cmd: list[str]) -> None:
        # Kill any previous process if still around
        if self.process is not None:
            try:
                self.process.kill()
            except Exception:
                pass
            self.process = None

        self.process = QProcess(self)
        self.process.setProgram(cmd[0])
        self.process.setArguments(cmd[1:])
        self.process.setWorkingDirectory(str(PROJECT_ROOT))

        # Capture output
        self.process.readyReadStandardOutput.connect(
            lambda: self.append_log(bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="ignore"))
        )
        self.process.readyReadStandardError.connect(
            lambda: self.append_log(bytes(self.process.readAllStandardError()).decode("utf-8", errors="ignore"))
        )
        self.process.finished.connect(self.on_process_finished)

        self.append_log(f"Running: {' '.join(cmd)}")
        self.set_busy(True)
        self.process.start()

    def on_process_finished(self, exitCode: int, exitStatus) -> None:  # type: ignore[override]
        self.append_log(f"\nProcess finished with code {exitCode}")
        self.set_busy(False)

    # ----- Actions -----
    def on_parse_clicked(self) -> None:
        xlsx_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Excel file to parse",
            str(PROJECT_ROOT / "data"),
            "Excel Files (*.xlsx *.xls);;All Files (*)",
        )
        if not xlsx_path:
            return

        # Optional: select a YAML config (recommended)
        use_cfg = QMessageBox.question(
            self,
            "Use Config?",
            "Do you want to use a YAML config to lock sheets/columns?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        cfg_arg: list[str] = []
        if use_cfg == QMessageBox.StandardButton.Yes:
            cfg_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select YAML config (optional)",
                str(PROJECT_ROOT / "configs"),
                "YAML Files (*.yaml *.yml);;All Files (*)",
            )
            if cfg_path:
                cfg_arg = ["--config", cfg_path]

        # Choose output CSV
        suggested_out = PROJECT_ROOT / "data" / (Path(xlsx_path).stem + "_sites_config.csv")
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save canonical CSV as",
            str(suggested_out),
            "CSV Files (*.csv)",
        )
        if not out_path:
            return
        # Errors sidecar
        errors_path = str(Path(out_path).with_name(Path(out_path).stem + "_errors.csv"))

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "parse_sites.py"),
            xlsx_path,
            "--output",
            out_path,
            "--errors",
            errors_path,
            "--debug",
        ] + cfg_arg

        self.run_command(cmd)

    def on_geocode_clicked(self) -> None:
        csv_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select canonical sites CSV to geocode",
            str(PROJECT_ROOT / "data"),
            "CSV Files (*.csv);;All Files (*)",
        )
        if not csv_path:
            return

        suggested_out = PROJECT_ROOT / "data" / (Path(csv_path).stem + "_geocoded.csv")
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save geocoded CSV as",
            str(suggested_out),
            "CSV Files (*.csv)",
        )
        if not out_path:
            return

        cache_db = str(PROJECT_ROOT / "data" / "cache" / "geocache.sqlite")
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "geocode_sites.py"),
            csv_path,
            "--output",
            out_path,
            "--cache",
            cache_db,
            "--user-agent",
            "VehicleRoute-Geocoder/0.1",
            "--email",
            EMAIL_CONTACT,
        ]

        self.run_command(cmd)


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(600, 400)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
