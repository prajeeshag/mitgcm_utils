# bathymetry_model.py
import logging
from typing import Any, cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from PyQt5.QtCore import QObject, pyqtSignal


class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    # For easy comparison in selections
    def __eq__(self, other: Any):
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y


class BathymetryModel(QObject):
    selection_changed = pyqtSignal()
    grid_changed = pyqtSignal()
    data_loaded = pyqtSignal()

    def __init__(self, width: int = 100, height: int = 100):
        self.logger = logging.getLogger(__name__)
        super().__init__()
        self._grid: NDArray[np.float32] = np.zeros((width, height), dtype=np.float32)
        self._selected_points: list[Point] = []
        self._latitude = None
        self._longitude = None

    @property
    def grid(self) -> NDArray[np.float32]:
        return self._grid

    @property
    def shape(self) -> tuple[int, int]:
        return self._grid.shape

    @property
    def selected_points(self) -> list[Point]:
        return self._selected_points

    # Selection methods
    def select_negative_bathymetry(self):
        self._selected_points = []
        rows, cols = np.where(self._grid < 0)
        for x, y in zip(rows, cols):
            self._selected_points.append(Point(x, y))
        self.selection_changed.emit()

    def get_selected_points(self) -> list[Point]:
        return self._selected_points

    def set_selected_points(self, points: list[Point]):
        self._selected_points = points
        self.selection_changed.emit()

    # Transformation methods
    def average_selected_points(self):
        for point in self._selected_points:
            neighbors = self.get_neighbors(point.x, point.y)
            if neighbors:
                avg = np.mean([self._grid[p.x, p.y] for p in neighbors])
                self._grid[point.x, point.y] = avg
        self.grid_changed.emit()

    def get_selected_values(self) -> list[float]:
        return [self._grid[point.x, point.y] for point in self._selected_points]

    def set_selected_values(self, values: list[float]):
        for point, value in zip(self._selected_points, values):
            self._grid[point.x, point.y] = value
        self.grid_changed.emit()

    def get_neighbors(self, x: int, y: int) -> list[Point]:
        neighbors: list[Point] = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self._grid.shape[0] and 0 <= ny < self._grid.shape[1]:
                neighbors.append(Point(nx, ny))
        return neighbors

    # NetCDF Loading Method
    def load_from_netcdf(
        self,
        file_path: str,
    ):
        try:
            ds = xr.open_dataset(file_path)  # type: ignore
            data_vars = list(ds.data_vars)
            if len(data_vars) != 1:
                raise ValueError("NetCDF file must contain exactly one variable.")
            self.logger.debug(f"data variables: {data_vars}")
            arr: xr.DataArray = cast(xr.DataArray, ds[data_vars[0]])
            self._grid = cast(NDArray[np.float32], arr.to_numpy())  # type: ignore
            # Ensure grid is 2D
            if len(self._grid.shape) != 2:
                raise ValueError("Bathymetry data must be 2-dimensional.")

            # Emit data_loaded signal
            self.data_loaded.emit()

        except Exception as e:
            raise e
