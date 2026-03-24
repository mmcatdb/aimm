import sys
import time
from typing import Any, Generic, Literal, NoReturn, Protocol, TypeVar
import dataclasses, json
from contextlib import contextmanager
import textwrap

# Add small epsilon to avoid division by zero or log(0).
EPSILON = 1e-8

class JsonEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            dataclass: Any = o
            return dataclasses.asdict(dataclass)
        return super().default(o)

class Closeable(Protocol):
    def close(self) -> None: ...

@contextmanager
def auto_close(closeable: Closeable):
    """Use this in each top-level script that uses some resources which should be closed."""
    try:
        yield closeable
    except Exception as e:
        exit_with_exception(e)
    finally:
        closeable.close()

def exit_with_exception(exception: Exception) -> NoReturn:
    """Use this in each top-level script in a catch block to print unexpected terminal exceptions."""
    import traceback

    print(exception, file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

ERROR_TEXT = '\033[91m'    # bright red
WARNING_TEXT = '\033[95m'  # bright magenta
INFO_TEXT = '\033[94m'     # bright blue
BOLD_TEXT = '\033[1m'
RESET_BOLD_TEXT = '\033[22m'
RESET_TEXT = '\033[0m'
CLEAR_TEXT_LINE = '\033[K'

def exit_with_error(message: str, exception: Exception | None = None) -> NoReturn:
    """Use this to exit whenever a terminal yet expected error is encountered. If the expected error is a result of an unexpected exception, provide it as well."""
    print(f'{ERROR_TEXT}{BOLD_TEXT}Error:{RESET_BOLD_TEXT} {message}{RESET_TEXT}', file=sys.stderr)
    if exception is not None:
        exit_with_exception(exception)
    else:
        sys.exit(1)

def print_warning(message: str, exception: Exception | None = None) -> None:
    """Use this to print a warning message without exiting."""
    print(f'{WARNING_TEXT}{BOLD_TEXT}Warning:{RESET_BOLD_TEXT} {message}{RESET_TEXT}', file=sys.stderr)
    if exception is not None:
        import traceback

        print(exception, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

def print_info(message: str) -> None:
    """Use this to print an informational message."""
    # Loggin should also go to stderr to avoid messing with the actual output of the script, which might be piped to another command or file.
    print(f'{INFO_TEXT}{BOLD_TEXT}Info:{RESET_BOLD_TEXT} {message}{RESET_TEXT}', file=sys.stderr)

def pretty_print_int(value: int) -> str:
    abs_val = abs(value)
    if abs_val < 1000:
        return str(value)
    return f'{value:.2e}'

def pretty_print_double(value: float) -> str:
    abs_val = abs(value)
    if abs_val < 0.01:
        return f'{value:.3g}' if value == 0 else f'{value:.2e}'
    if abs_val < 1000:
        return f'{value:.3g}'
    return f'{value:.2e}'

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
        number_part = str(int(value)) if omit_decimal else f'{value:.2f}'
        return f'{number_part} {unit}'

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
        raise ValueError('Impossibruh')

    def to_base(self, value: float, from_unit: TUnit) -> float:
        base_value = value
        for i in range(len(self.units)):
            if from_unit == self.units[i]:
                return base_value
            base_value *= self.thresholds[i]
        raise ValueError('Impossibruh')

DataSizeUnit = Literal['B', 'kB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
DATA_SIZE_UNITS = ('B', 'kB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB')
data_size_quantity = Quantity[DataSizeUnit](
    DATA_SIZE_UNITS,
    (1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024),
    True,
)

TimeUnit = Literal['ms', 's', 'min', 'h', 'd', 'y']
TIME_UNITS = ('ms', 's', 'min', 'h', 'd', 'y')
time_quantity = Quantity[TimeUnit](
    TIME_UNITS,
    (1000, 60, 60, 24, 365),
    False,
)

# def trim_to_block(text: str, tabsize: int = 4) -> str:
def trim_to_block(text: str) -> str:
    lines = text.splitlines()

    # remove leading/trailing blank lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    # strip trailing whitespace
    lines = [l.rstrip() for l in lines]

    text = '\n'.join(lines)

    # text = text.expandtabs(tabsize)
    text = textwrap.dedent(text)

    return text

MIN_REPORT_INTERVAL_S = 0.2

class ProgressTracker:
    def __init__(self, start_interval: int, growth: float, show_percents_from_total: int | None):
        self.start_interval = start_interval
        self.growth = growth
        self.show_percents_from_total = show_percents_from_total

        self.next_report: int
        self.interval: int

        self.prefix: str
        self.count: int
        self.start_time: float
        self.last_time: float

    @staticmethod
    def unlimited(start_interval=1000, growth=1.5) -> 'ProgressTracker':
        """Use this when you don't know the total number of items in advance. It will report at exponentially increasing intervals."""
        return ProgressTracker(start_interval, growth, None)

    @staticmethod
    def limited(total: int, steps=100, show_percents=True) -> 'ProgressTracker':
        """Use this when you know the total number of items in advance. It will try to report after a fixed interval."""
        start_interval = max(1, total // steps)
        show_percents_from_total = total if show_percents else None
        return ProgressTracker(start_interval, 1, show_percents_from_total)

    def start(self, prefix: str):
        """Starts the tracker and sets the prefix. Also prints the first message so make sure to call this before anything that might cause an exception."""
        self.prefix = prefix
        self.count = 0
        self.next_report = self.start_interval
        self.interval = self.start_interval
        self.start_time = time.time()
        self.last_time = self.start_time

        sys.stdout.write(self.prefix)
        sys.stdout.flush()

    def track(self, amount = 1):
        self.count += amount

        if self.count >= self.next_report:
            self._print()
            self.interval = int(self.interval * self.growth)
            self.next_report += self.interval

    def finish(self):
        self._print(True)
        sys.stdout.write('\n')
        sys.stdout.flush()

    def _print(self, force=False):
        now = time.time()
        if not force and now - self.last_time < MIN_REPORT_INTERVAL_S:
            # Don't report too frequently to avoid spamming the console and hurting performance. This can happen if the growth factor is too small or if the total number of items is small.
            return
        self.last_time = now

        elapsed = now - self.start_time
        rate = self.count / elapsed if elapsed > 0 else 0
        percents = f', {self.count / self.show_percents_from_total * 100:.1f}%' if self.show_percents_from_total is not None else ''
        message = f'{self.prefix}{self.count:,} ({rate:,.0f}/s{percents})'
        sys.stdout.write('\r' + message)
        sys.stdout.flush()
