import os
import json
import yaml
import joblib
from box import ConfigBox
from box.exceptions import BoxValueError
from pathlib import Path
from typing import Any, List
from typeguard import typechecked
from RED_WINE.logging.logger import logger


@typechecked
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads yaml file and returns ConfigBox object."""
    try:
        if not path_to_yaml.exists():
            raise FileNotFoundError(f"YAML file not found at: {path_to_yaml}")

        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)

            if content is None:
                raise ValueError("YAML file is empty")

            if not isinstance(content, dict):
                raise ValueError(f"YAML file {path_to_yaml} must contain a dictionary at the top level")

            logger.info(f"YAML file loaded successfully from: {path_to_yaml}")
            return ConfigBox(content)

    except BoxValueError:
        raise ValueError("YAML file is empty or invalid format")
    except Exception as e:
        logger.error(f"Error while reading YAML file: {e}")
        raise



@typechecked
def create_directories(path_to_directories: List[Path], verbose: bool = True):
    """Create list of directories."""
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")


@typechecked
def save_json(path: Path, data: dict):
    """Save dictionary as JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved at: {path}")


@typechecked
def load_json(path: Path) -> ConfigBox:
    """Load JSON file and return as ConfigBox."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found at: {path}")

    with open(path) as f:
        content = json.load(f)

    logger.info(f"JSON file loaded successfully from: {path}")
    return ConfigBox(content)


@typechecked
def save_bin(data: Any, path: Path):
    """Save data as binary using joblib."""
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")


@typechecked
def load_bin(path: Path) -> Any:
    """Load binary file using joblib."""
    if not path.exists():
        raise FileNotFoundError(f"Binary file not found at: {path}")

    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data


@typechecked
def get_size(path: Path) -> str:
    """Get file size in KB."""
    if not path.exists():
        raise FileNotFoundError(f"File not found at: {path}")

    size_in_kb = round(path.stat().st_size / 1024)
    return f"~ {size_in_kb} KB"
