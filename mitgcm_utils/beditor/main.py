# main.py
import logging
import sys

from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from .bathymetry_model import BathymetryModel
from .command_manager import CommandManager
from .heatmap_canvas import HeatmapCanvas
from .lake_selection_command import LakeSelectionCommand
from .logger import setup_logging
from .nearest_neighbor_average_command import NearestNeighborAverageTransformCommand


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bathymetry Editor with NetCDF Support")
        self.resize(800, 600)

        # Initialize model and command manager
        self.model = BathymetryModel()
        self.cmd_manager = CommandManager()

        # Initialize UI components
        self.canvas = HeatmapCanvas(self.model)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.create_actions()
        self.create_menus()

    def create_actions(self):
        # File menu actions
        self.open_act = QAction("&Open NetCDF...", self)
        self.open_act.setShortcut("Ctrl+O")
        self.open_act.triggered.connect(self.open_file)

        self.exit_act = QAction("&Exit", self)
        self.exit_act.setShortcut("Ctrl+Q")
        self.exit_act.triggered.connect(self.exit_app)

        # Edit menu actions
        self.lake_selection_act = QAction("&Lake Selection", self)
        self.lake_selection_act.setShortcut("Ctrl+L")
        self.lake_selection_act.triggered.connect(self.perform_lake_selection)

        self.nearest_neighbor_avg_act = QAction("&Nearest Neighbor Average", self)
        self.nearest_neighbor_avg_act.setShortcut("Ctrl+A")
        self.nearest_neighbor_avg_act.triggered.connect(
            self.perform_nearest_neighbor_average
        )

        self.undo_act = QAction("&Undo", self)
        self.undo_act.setShortcut("Ctrl+Z")
        self.undo_act.triggered.connect(self.undo)

        self.redo_act = QAction("&Redo", self)
        self.redo_act.setShortcut("Ctrl+Y")
        self.redo_act.triggered.connect(self.redo)

    def exit_app(self):
        self.close()

    def create_menus(self):
        menubar = self.menuBar()  # type: ignore
        # File Menu
        file_menu = menubar.addMenu("&File")  # type: ignore
        file_menu.addAction(self.open_act)  # type: ignore
        file_menu.addSeparator()  # type: ignore
        file_menu.addAction(self.exit_act)  # type: ignore

        # Edit Menu
        edit_menu = menubar.addMenu("&Edit")  # type: ignore
        edit_menu.addAction(self.lake_selection_act)  # type: ignore
        edit_menu.addAction(self.nearest_neighbor_avg_act)  # type: ignore
        edit_menu.addSeparator()  # type: ignore
        edit_menu.addAction(self.undo_act)  # type: ignore
        edit_menu.addAction(self.redo_act)  # type: ignore

    def open_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly  # type: ignore
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open NetCDF File",
            "",
            "NetCDF Files (*.nc *.nc4);;All Files (*)",
            options=options,  # type: ignore
        )
        if file_path:
            try:
                self.model.load_from_netcdf(file_path)
                QMessageBox.information(
                    self, "Success", f"Loaded NetCDF file: {file_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load NetCDF file:\n{e}")

    def perform_lake_selection(self):
        cmd = LakeSelectionCommand(self.model)
        self.cmd_manager.execute_command(cmd)
        # Update the view
        self.canvas.update_selection_view()

    def perform_nearest_neighbor_average(self):
        if not self.model.selected_points:
            QMessageBox.warning(
                self, "No Selection", "No points selected for transformation."
            )
            return
        cmd = NearestNeighborAverageTransformCommand(self.model)
        self.cmd_manager.execute_command(cmd)
        # Update the view
        self.canvas.update_heatmap()

    def undo(self):
        self.cmd_manager.undo()
        self.canvas.update_heatmap()
        self.canvas.update_selection_view()

    def redo(self):
        self.cmd_manager.redo()
        self.canvas.update_heatmap()
        self.canvas.update_selection_view()


def main():
    setup_logging(logging.DEBUG)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
