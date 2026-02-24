from typing import Any, Generic, Literal, Protocol, TypeVar
import dataclasses, json
from contextlib import contextmanager

# Add small epsilon to avoid division by zero
EPSILON = 1e-8

class JsonEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            dataclass: Any = o
            return dataclasses.asdict(dataclass)
        return super().default(o)

class Closeable(Protocol):
    def close(self) -> None:
        pass

@contextmanager
def auto_close(closeable: Closeable):
    try:
        yield closeable
    except Exception:
        import sys
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        closeable.close()

def pretty_print_int(value: int) -> str:
    abs_val = abs(value)
    if abs_val < 1000:
        return str(value)
    return f"{value:.2e}"

def pretty_print_double(value: float) -> str:
    abs_val = abs(value)
    if abs_val < 0.01:
        return f"{value:.3g}" if value == 0 else f"{value:.2e}"
    if abs_val < 1000:
        return f"{value:.3g}"
    return f"{value:.2e}"

TUnit = TypeVar('TUnit', bound=str)

class Quantity(Generic[TUnit]):
    def __init__(self, units: tuple[TUnit, ...], thresholds: tuple[int, ...], is_base_integer: bool):
        self.units = units
        self.thresholds = thresholds
        self.is_base_integer = is_base_integer

    def define_units(self, from_unit: TUnit | None = None, to_unit: TUnit | None = None) -> list[TUnit]:
        from_index = self.units.index(from_unit) if from_unit != None else 0
        to_index = self.units.index(to_unit) if to_unit != None else len(self.units)
        return list(self.units[from_index:to_index])

    def pretty_print(self, value: float, unit: TUnit | None = None, is_integer: bool | None = None) -> str:
        if not unit:
            value, unit = self.find_unit(value)
        else:
            value = self.from_base(value, unit)

        omit_decimal = (is_integer if is_integer is not None else self.is_base_integer) and unit == self.units[0]
        number_part = str(int(value)) if omit_decimal else f"{value:.2f}"
        return f"{number_part} {unit}"

    def find_unit(self, value_in_base: float) -> tuple[float, TUnit]:
        index = 0
        value = value_in_base
        while value >= self.thresholds[index] and index < len(self.units) - 1:
            value /= self.thresholds[index]
            index += 1
        return (value, self.units[index])

    def from_base(self, value_in_base: float, to_unit: TUnit) -> float:
        value = value_in_base
        for i in range(len(self.units)):
            if to_unit == self.units[i]:
                return value
            value /= self.thresholds[i]
        raise ValueError("Impossibruh")

    def to_base(self, value: float, from_unit: TUnit) -> float:
        base_value = value
        for i in range(len(self.units)):
            if from_unit == self.units[i]:
                break
            base_value *= self.thresholds[i]
        raise ValueError("Impossibruh")

DataSizeUnit = Literal["B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
DATA_SIZE_UNITS = ("B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
data_size_quantity = Quantity[DataSizeUnit](
    DATA_SIZE_UNITS,
    (1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024),
    True,
)

TimeUnit = Literal["ms", "s", "min", "h", "d", "y"]
TIME_UNITS = ("ms", "s", "min", "h", "d", "y")
time_quantity = Quantity[TimeUnit](
    TIME_UNITS,
    (1000, 60, 60, 24, 365),
    False,
)
