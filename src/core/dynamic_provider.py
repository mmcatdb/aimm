import importlib.util
import os
from pathlib import Path
from enum import Enum
import sys
from types import ModuleType
from typing import TypeVar
from core.data_generator import DataGenerator
from core.loaders.base_loader import BaseLoader
from core.query import SchemaName, DriverType, QueryRegistry
from core.utils import exit_with_error

class ScriptFile(Enum):
    DATA_GENERATOR = 'data_generator.py'
    LOADER = 'loader.py'
    QUERY_REGISTRY = 'query_registry.py'

    def module_name(self):
        return self.value[:-3]

SRC_DIRECTORY = Path(__file__).resolve().parents[1]
PROJECT_ROOT_DIRECTORY = SRC_DIRECTORY.parent
DYNAMIC_MODULE_NAME = 'dynamic'
DYNAMIC_ROOT_DIRECTORY = Path(os.getenv('DYNAMIC_ROOT_DIRECTORY', PROJECT_ROOT_DIRECTORY / DYNAMIC_MODULE_NAME)).resolve()
NO_DRIVER_DIRECTORY = 'common'
FACTORY_FUNCTION_NAME = 'export'

TClass = TypeVar('TClass', DataGenerator, BaseLoader, QueryRegistry)

def get_dynamic_class_instance(clazz: type[TClass], driver: DriverType | None, schema: SchemaName) -> TClass:
    """Returns a driver-and-schema-specific implementation of the given class, if it exists.

    All dynamic class instances should be created through this function. They should never be imported directly.
    """
    try:
        script = _get_script_for_class(clazz)
        module = _get_or_import_module(script, driver, schema)

        if not hasattr(module, FACTORY_FUNCTION_NAME):
            exit_with_error(f'Module "{module.__name__}" does not contain a function "{FACTORY_FUNCTION_NAME}".')

        factory = getattr(module, FACTORY_FUNCTION_NAME)
        if not callable(factory):
            exit_with_error(f'"{FACTORY_FUNCTION_NAME}" in module "{module.__name__}" is not callable.')

        instance = factory()
        if not isinstance(instance, clazz):
            exit_with_error(f'"{FACTORY_FUNCTION_NAME}" in module "{module.__name__}" does not produce an instance of "{clazz.__name__}".')

        return instance
    except Exception as e:
        driver_info = f'driver "{driver.value}" and ' if driver else ''
        exit_with_error(f'Failed to load class "{clazz.__name__}" for {driver_info}schema "{schema}".', e)

def _get_script_for_class(clazz: type) -> ScriptFile:
    if clazz == DataGenerator:
        return ScriptFile.DATA_GENERATOR
    elif clazz == BaseLoader:
        return ScriptFile.LOADER
    elif clazz == QueryRegistry:
        return ScriptFile.QUERY_REGISTRY
    else:
        exit_with_error(f'Unsupported class type "{clazz}" in dynamic loader.')

def _get_or_import_module(script: ScriptFile, driver: DriverType | None, schema: SchemaName) -> ModuleType:
    _ensure_dynamic_parent_on_path()

    driver_directory = driver.value if driver else NO_DRIVER_DIRECTORY
    module_name = _get_module_name(script, driver_directory, schema)

    module = sys.modules.get(module_name)
    if not module:
        file_path = _find_script_path(script, driver_directory, schema)
        module = _import_module_from_path(module_name, file_path)

    return module

def _get_module_name(script: ScriptFile, driver_directory: str, schema: SchemaName) -> str:
    return f'{DYNAMIC_MODULE_NAME}.{driver_directory}.{schema}.{script.module_name()}'

def _find_script_path(script: ScriptFile, driver_directory: str, schema: SchemaName) -> Path:
    """Finds the dynamic script for the requested driver/schema pair."""
    driver_path = DYNAMIC_ROOT_DIRECTORY / driver_directory
    if not driver_path.is_dir():
        exit_with_error(f'Database driver directory not found: {driver_path}')

    file_path = driver_path / str(schema) / script.value
    if not file_path.is_file():
        available_schemas = _get_available_schemas(script, driver_path, schema)
        exit_with_error(f'Script is not available for schema "{schema}". Make sure the path "{file_path}" exists.\nAvailable schemas: {", ".join(available_schemas)}.')

    return file_path

def _get_available_schemas(script: ScriptFile, driver_path: Path, schema: SchemaName) -> list[str]:
    output = list[str]()

    for schema_directory in driver_path.iterdir():
        if not schema_directory.is_dir() or schema_directory.name == str(schema):
            continue

        file_path = schema_directory / script.value
        if not file_path.is_file():
            continue

        output.append(schema_directory.name)

    return output

def _import_module_from_path(module_name: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if not spec or not spec.loader:
        exit_with_error(f'Module "{module_name}" could not be loaded from "{file_path}".')

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module

def _ensure_dynamic_parent_on_path() -> None:
    parent_directory = str(DYNAMIC_ROOT_DIRECTORY.parent)
    if parent_directory not in sys.path:
        sys.path.insert(0, parent_directory)
