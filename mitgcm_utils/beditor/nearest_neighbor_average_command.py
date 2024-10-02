# nearest_neighbor_average_command.py
from .transform_commands import TransformCommand


class NearestNeighborAverageTransformCommand(TransformCommand):
    def execute(self):
        # Save current values of selected points for undo
        self._previous_values = self._model.get_selected_values().copy()
        # Perform nearest neighbor averaging
        self._model.average_selected_points()
