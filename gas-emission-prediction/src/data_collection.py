# src/data_collection.py

import logging
import requests
import pandas as pd
from datetime import datetime
from typing import Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentinelDataCollector:
    """
    Handles downloading emission data from NASA Sentinel-5P API.
    """

    BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

    def __init__(self, api_key: str, output_dir: str = "data/raw") -> None:
        self.api_key = api_key
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def download_data(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        location_name: str,
    ) -> pd.DataFrame:
        """
        Download Sentinel-5P data for given location and date range.

        Args:
            latitude: Latitude of location
            longitude: Longitude of location
            start_date: YYYY-MM-DD
            end_date: YYYY-MM-DD
            location_name: Name for saving file

        Returns:
            DataFrame containing emission data
        """

        params = {
            "MAP_KEY": self.api_key,
            "start_date": start_date,
            "end_date": end_date,
            "lat": latitude,
            "lon": longitude,
        }

        try:
            logger.info(f"Requesting data for {location_name}...")
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()

            df = pd.read_csv(pd.compat.StringIO(response.text))

            if df.empty:
                raise ValueError("Downloaded dataset is empty.")

            file_path = self.output_path / f"{location_name}_emissions.csv"
            df.to_csv(file_path, index=False)

            logger.info(f"Data saved to {file_path}")
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise