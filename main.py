import sys
from PyQt6.QtWidgets import QApplication, QLabel
from PyQt6.QtCore import Qt


def main() -> None:
    """Launch a minimal PyQt6 window."""
    app = QApplication(sys.argv)

    label = QLabel("Hello, PyQt6!")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.resize(360, 200)
    label.setWindowTitle("PyQt6 Hello World")
    label.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
