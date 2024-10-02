# lake_selection_command.py
from .selection_command import SelectionCommand


class LakeSelectionCommand(SelectionCommand):
    def execute(self):
        # Save current selection for undo
        self._previous_selection = self._model.get_selected_points().copy()
        # Perform lake selection: select points with negative bathymetry
        self._model.select_negative_bathymetry()
