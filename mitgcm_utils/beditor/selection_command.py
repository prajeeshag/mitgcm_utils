# selection_commands.py
from .bathymetry_model import BathymetryModel, Point
from .command import Command


class SelectionCommand(Command):
    def __init__(self, model: BathymetryModel):
        self._model = model
        self._previous_selection: list[Point] = []

    def undo(self):
        self._model.set_selected_points(self._previous_selection)
