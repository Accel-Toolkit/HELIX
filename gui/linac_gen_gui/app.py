"""Application entry point."""
import sys
from PyQt6.QtWidgets import QApplication
from linac_gen_gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Linac_Gen")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
