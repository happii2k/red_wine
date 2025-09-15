import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO , format='%(asctime)s - %(levelname)s - %(message)s')

Project_Name = "RED_WINE"

list_of_files = [
    f"src/{Project_Name}/__init__.py",
    f"src/{Project_Name}/components/__init__.py",
    f"src/{Project_Name}/utils/__init__.py",
    f"src/{Project_Name}/config/__init__.py",
    f"src/{Project_Name}/config/configuration.py",
    f"src/{Project_Name}/pipeline/__init__.py",
    f"src/{Project_Name}/pipeline/training_pipeline.py",
    f"src/{Project_Name}/entity/__init__.py",
    f"src/{Project_Name}/entity/artifact_entity.py",
    f"src/{Project_Name}/entity/config_entity.py",
    f"src/{Project_Name}/constants/__init__.py",
    f"src/{Project_Name}/constants/constant.py",
    f"src/{Project_Name}/logging/__init__.py",
    f"src/{Project_Name}/logging/logger.py",
    f"src/{Project_Name}/exception/__init__.py",
    f"src/{Project_Name}/exception/custom_exception.py",
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    "main.py",
    "app.py",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html"
    ]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory: {filedir}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
        logging.info(f"Created empty file: {filepath}")
    else:
        logging.info(f"File already exists and is not empty: {filepath}")
        