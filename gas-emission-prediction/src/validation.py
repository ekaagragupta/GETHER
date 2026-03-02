# src/validation.py

import pandas as pd
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def validate_raw_data(df: pd.DataFrame) -> Dict[str, float]:
    """
    Validate raw emission dataset.

    Returns:
        Dictionary containing validation statistics.
    """

    results = {}

    results["total_rows"] = len(df)
    results["duplicate_rows"] = df.duplicated().sum()
    results["missing_percentage"] = df.isnull().mean().mean() * 100

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        results["time_span_days"] = (
            (df["timestamp"].max() - df["timestamp"].min()).days
        )

    logger.info(f"Validation Results: {results}")
    return results