# command_manager.py
from collections import deque

from .command import Command


class CommandManager:
    def __init__(self, max_history: int = 100):
        self.undo_stack = deque[Command](maxlen=max_history)
        self.redo_stack = deque[Command](maxlen=max_history)

    def execute_command(self, command: Command):
        command.execute()
        self.undo_stack.append(command)
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack:
            return
        command = self.undo_stack.pop()
        command.undo()
        self.redo_stack.append(command)

    def redo(self):
        if not self.redo_stack:
            return
        command = self.redo_stack.pop()
        command.execute()
        self.undo_stack.append(command)
