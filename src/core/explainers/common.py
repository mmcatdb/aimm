from abc import ABC, abstractmethod
from core.nn_operator import NnOperator

# We use the rich library for the color formatting. The ANSI color codes would break it's internal line length calculations.
HAS_ALL_TEXT = '[bright_blue]'
HAS_TYPE_TEXT = '[bright_magenta]'
HAS_NOTHING_TEXT = '[bright_red]'
RESET_TEXT = '[reset]'

class OperatorNameFormatter:

    def __init__(self, allow_custom_names: bool):
        self.__allow_custom_names = allow_custom_names
        self.__operator_keys = set[str]()
        self.__operator_types = set[str]()

        self.__missing_keys = set[str]()
        self.__missing_types = set[str]()

    def add(self, operator: NnOperator) -> None:
        self.__operator_keys.add(operator.key())
        self.__operator_types.add(operator.type)

    def format(self, type: str, num_children: int, custom_name: str | None = None) -> str:
        """If the custom_name isn't provided, the type will be used instead."""
        key = NnOperator.compute_key(type, num_children)
        name_part = custom_name if self.__allow_custom_names and custom_name else type
        full_name = f'{name_part}_{num_children}'

        if key in self.__operator_keys:
            color = HAS_ALL_TEXT
        elif type in self.__operator_types:
            color = HAS_TYPE_TEXT
            self.__missing_keys.add(key)
        else:
            color = HAS_NOTHING_TEXT
            self.__missing_types.add(type)

        return f'{color}{full_name}{RESET_TEXT}'

    def get_missing_keys(self) -> list[str]:
        return list(self.__missing_keys)

    def get_missing_types(self) -> list[str]:
        return list(self.__missing_types)

class TreeRenderer(ABC):

    def __init__(self, operators: OperatorNameFormatter | None):
        self._operators = operators

    @abstractmethod
    def _node_label(self, node: dict) -> str:
        pass

    @abstractmethod
    def _get_node_children(self, node: dict) -> list[dict]:
        pass

    def render_tree(self, node: dict) -> list[str]:
        """Recursively render the plan tree into a list of lines."""
        return self._render_tree(node, '', True)

    def _render_tree(self, node: dict, prefix: str, is_last: bool) -> list[str]:
        connector = '└─ ' if is_last else '├─ '
        lines = [prefix + connector + self._node_label(node)]

        child_prefix = prefix + ('   ' if is_last else '│  ')

        children: list[dict] = self._get_node_children(node)
        for i, child in enumerate(children):
            is_last_child = i == len(children) - 1
            lines.extend(self._render_tree(child, child_prefix, is_last_child))

        return lines
