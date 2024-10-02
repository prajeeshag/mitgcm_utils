# heatmap_canvas.py
import numpy as np
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,  # type: ignore
)
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure

from .bathymetry_model import BathymetryModel


class HeatmapCanvas(FigureCanvas):  # type: ignore
    def __init__(self, model: BathymetryModel):
        self.model = model
        self.fig = Figure()
        super().__init__(self.fig)  # type: ignore
        self.plot_heatmap()
        self.model.data_loaded.connect(self.plot_heatmap)
        self.model.grid_changed.connect(self.update_heatmap)
        self.model.selection_changed.connect(self.update_selection_view)
        self.mpl_connect("button_press_event", self.on_click)  # type: ignore

    def plot_heatmap(self):
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)  # type: ignore
        self.ax.clear()
        # Create meshgrid for pcolormesh
        x = np.arange(self.model.shape[1] + 1)
        y = np.arange(self.model.shape[0] + 1)
        X, Y = np.meshgrid(x, y)
        # Plot using pcolormesh
        self.pmesh: QuadMesh = self.ax.pcolormesh(  # type: ignore
            X, Y, self.model.grid, cmap="coolwarm", shading="auto"
        )
        self.fig.colorbar(self.pmesh, ax=self.ax, label="Bathymetry Value")  # type: ignore
        self.ax.set_xlabel("X")  # type: ignore
        self.ax.set_ylabel("Y")  # type: ignore
        self.ax.set_title("Bathymetry Heatmap")  # type: ignore
        self.ax.invert_yaxis()  # To match array indexing
        self.draw()

    def update_heatmap(self):
        # Update pcolormesh with new grid data
        self.pmesh.set_array(self.model.grid[:-1, :-1].ravel())
        self.pmesh.set_clim(
            float(np.min(self.model.grid)),
            float(np.max(self.model.grid)),
        )
        self.draw()

    def update_selection_view(self):
        # Remove existing scatter
        if self.scatter:
            self.scatter.remove()
            self.scatter = None
        if not self.model.selected_points:
            self.draw()
            return
        # Extract x and y coordinates
        x = [p.y + 0.5 for p in self.model.selected_points]  # +0.5 to center the marker
        y = [p.x + 0.5 for p in self.model.selected_points]
        self.scatter = self.ax.scatter(  # type: ignore
            x, y, facecolors="none", edgecolors="yellow", s=100, linewidths=2
        )
        self.draw()

    def on_click(self, event: MouseEvent) -> None:
        if event.inaxes != self.ax:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        # Convert to grid indices
        grid_x = int(np.floor(y))
        grid_y = int(np.floor(x))
        if 0 <= grid_x < self.model.shape[0] and 0 <= grid_y < self.model.shape[1]:
            print(grid_x, grid_y)
