from .bathymetry_model import BathymetryModel
from .command import Command


class TransformCommand(Command):
    def __init__(self, model: BathymetryModel):
        self._model = model
        self._previous_values: list[float] = []

    def undo(self):
        self._model.set_selected_values(self._previous_values)
