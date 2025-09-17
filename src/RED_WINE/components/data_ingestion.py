from RED_WINE.entity.config_entity import DataIngestionConfig
import urllib.request as request
import zipfile
from pathlib import Path
from RED_WINE.logging.logger import logger
from RED_WINE.utils.utils import get_size


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self) -> Path:
        """Download data from source URL to local file."""
        logger.info("Starting data download...")
        local_file = Path(self.config.local_data_file)

        if not local_file.exists():
            local_file.parent.mkdir(parents=True, exist_ok=True)
            filename, _ = request.urlretrieve(
                self.config.source_URL, str(local_file)
            )
            logger.info(
                f"Data downloaded successfully: {filename} with size {get_size(local_file)}"
            )
        else:
            logger.info("Data file already exists. Skipping download.")
        
        return local_file

    def extract_zip_file(self, zip_file_path: Path):
        """Extract the downloaded zip file to the specified directory."""
        logger.info("Starting data extraction...")
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(self.config.unzip_dir)
        logger.info(f"Data extracted successfully to: {self.config.unzip_dir}")
