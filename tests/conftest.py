"""Pytest fixtures for SDCpy app tests."""

from pathlib import Path

import pandas as pd
import pytest

# Path to example dataset
EXAMPLE_DATASET_PATH = Path(__file__).parent.parent / "data" / "oni_temp_sa.csv"

# Default test parameters matching user specification
DEFAULT_TEST_PARAMS = {
    "ts1": "oni_anomaly",
    "ts2": "temp_anomaly_sa",
    "date_column": "date",
    "method": "pearson",
    "min_lag": -24,
    "max_lag": 24,
    "window": 24,
    "n_permutations": 0,  # Skip permutations for faster tests
    "labels_fontsize": 6,
    "plot_dpi": 150,  # Lower DPI for faster tests
    "alpha": 0.05,
    "show_colorbar": True,
    "show_ts2": True,
    "plot_width": 7.0,
    "plot_height": 7.0,
    "plot_min_lag": None,
    "plot_max_lag": None,
}


@pytest.fixture
def example_dataframe() -> pd.DataFrame:
    """Load the example ONI/temperature dataset."""
    return pd.read_csv(EXAMPLE_DATASET_PATH)


@pytest.fixture
def example_data_payload(example_dataframe: pd.DataFrame) -> dict:
    """Serialize example DataFrame in the format expected by run_sdc_analysis."""
    df = example_dataframe.copy()
    return df.to_dict("list")


@pytest.fixture
def default_params() -> dict:
    """Default analysis parameters for testing."""
    return DEFAULT_TEST_PARAMS.copy()
