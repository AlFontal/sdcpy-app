"""Integration tests for app.py - Dash application."""

import base64

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAppInitialization:
    """Test that the Dash app initializes correctly."""

    def test_app_loads(self):
        """Verify the Dash app can be imported and initialized."""
        # Import triggers app creation
        import app as dash_app

        assert dash_app.app is not None
        assert dash_app.app.title == "SDCpy app"
        assert dash_app.server is not None

    def test_default_params_defined(self):
        """Verify default parameters are properly defined."""
        import app as dash_app

        params = dash_app.DEFAULT_PARAMS
        assert "method" in params
        assert "labels_fontsize" in params
        assert "plot_dpi" in params
        assert params["method"] == "pearson"


class TestSanitizeDataframe:
    """Test the sanitize_dataframe function."""

    def test_sanitize_preserves_data(self, example_dataframe: pd.DataFrame):
        """Test that sanitization preserves valid data."""
        import app as dash_app

        sanitized, report = dash_app.sanitize_dataframe(example_dataframe.copy())

        assert len(sanitized) == len(example_dataframe)
        assert "oni_anomaly" in sanitized.columns
        assert "temp_anomaly_sa" in sanitized.columns

    def test_sanitize_detects_dates(self, example_dataframe: pd.DataFrame):
        """Test that date columns are detected and converted."""
        import app as dash_app

        sanitized, report = dash_app.sanitize_dataframe(example_dataframe.copy())

        # Date column should be converted
        assert pd.api.types.is_datetime64_any_dtype(sanitized["date"])

        # Report should document the conversion
        date_cols = report.get("date_columns", [])
        date_col_names = [c["column"] for c in date_cols]
        assert "date" in date_col_names

    def test_sanitize_drops_empty_columns(self):
        """Test that entirely empty columns are dropped."""
        import app as dash_app

        df = pd.DataFrame(
            {"a": [1, 2, 3], "b": [None, None, None], "c": ["x", "y", "z"]}
        )

        sanitized, report = dash_app.sanitize_dataframe(df)

        assert "b" not in sanitized.columns
        assert "b" in report.get("dropped_columns", [])


class TestParseContents:
    """Test CSV file parsing."""

    def test_parse_csv_contents(self, example_dataframe: pd.DataFrame):
        """Test that CSV content can be parsed correctly."""
        import app as dash_app

        # Simulate file upload by encoding CSV content
        csv_content = example_dataframe.to_csv(index=False)
        encoded = base64.b64encode(csv_content.encode()).decode()
        content_string = f"data:text/csv;base64,{encoded}"

        result = dash_app.parse_contents(content_string, "test.csv")

        # parse_contents returns (sanitized_df, error, report, original_df)
        sanitized_df, error, report, original_df = result

        assert error is None, "Should not have an error"
        assert isinstance(sanitized_df, pd.DataFrame)
        assert isinstance(original_df, pd.DataFrame)
        assert len(sanitized_df) == len(example_dataframe)
        assert list(original_df.columns) == list(example_dataframe.columns)
