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


class TestWeirdDateFormats:
    """Test handling of unusual date formats."""

    def test_sanitize_european_date_format(self):
        """Test DD/MM/YYYY format (European style)."""
        import app as dash_app

        df = pd.DataFrame(
            {
                "date": ["15/01/1998", "15/02/1998", "15/03/1998"],
                "value": [1.0, 2.0, 3.0],
            }
        )

        sanitized, report = dash_app.sanitize_dataframe(df)

        # Date should be detected (pandas may parse it)
        date_cols = [c["column"] for c in report.get("date_columns", [])]
        assert "date" in date_cols

    def test_sanitize_us_date_format(self):
        """Test MM/DD/YYYY format (US style)."""
        import app as dash_app

        df = pd.DataFrame(
            {
                "date": ["01/15/1998", "02/15/1998", "03/15/1998"],
                "value": [1.0, 2.0, 3.0],
            }
        )

        sanitized, report = dash_app.sanitize_dataframe(df)

        date_cols = [c["column"] for c in report.get("date_columns", [])]
        assert "date" in date_cols

    def test_sanitize_written_month_format(self):
        """Test '15 Jan 1998' format with written month names."""
        import app as dash_app

        df = pd.DataFrame(
            {
                "date": ["15 Jan 1998", "15 Feb 1998", "15 Mar 1998"],
                "value": [1.0, 2.0, 3.0],
            }
        )

        sanitized, report = dash_app.sanitize_dataframe(df)

        date_cols = [c["column"] for c in report.get("date_columns", [])]
        assert "date" in date_cols

    def test_sanitize_unix_timestamp_like(self):
        """Test that numeric-only columns are NOT detected as dates."""
        import app as dash_app

        df = pd.DataFrame(
            {
                "timestamp": [883612800, 886291200, 888710400],  # Unix timestamps
                "value": [1.0, 2.0, 3.0],
            }
        )

        sanitized, report = dash_app.sanitize_dataframe(df)

        # Unix timestamps should NOT be auto-detected as dates
        date_cols = [c["column"] for c in report.get("date_columns", [])]
        assert "timestamp" not in date_cols

    def test_format_override_applied_correctly(self):
        """
        Test that user-specified format overrides are applied correctly.

        This simulates what happens when user enters a custom format in the
        date review modal and clicks Confirm.
        """
        # Create data with ambiguous European format (DD/MM/YYYY)
        raw_df = pd.DataFrame(
            {
                "date": ["15/01/1998", "15/02/1998", "15/03/1998"],
                "value": [1.0, 2.0, 3.0],
            }
        )

        # Apply the user's format override
        user_format = "%d/%m/%Y"
        parsed = pd.to_datetime(raw_df["date"], format=user_format, errors="coerce")

        assert parsed.notna().all(), "All dates should parse with the format"

        # Verify the parsed dates are correct (15th of each month)
        assert parsed.iloc[0].day == 15
        assert parsed.iloc[0].month == 1
        assert parsed.iloc[1].month == 2
        assert parsed.iloc[2].month == 3

        # Verify normalization to ISO format (as done in app.py line 1231)
        normalized = parsed.dt.strftime("%Y-%m-%d %H:%M:%S")
        assert normalized.iloc[0] == "1998-01-15 00:00:00"
        assert normalized.iloc[1] == "1998-02-15 00:00:00"
        assert normalized.iloc[2] == "1998-03-15 00:00:00"


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

    def test_parse_csv_with_weird_dates(self):
        """Test parsing CSV with non-standard date format."""
        import app as dash_app

        # Create CSV with European dates
        csv_content = """date,value
15/01/1998,1.0
15/02/1998,2.0
15/03/1998,3.0"""
        encoded = base64.b64encode(csv_content.encode()).decode()
        content_string = f"data:text/csv;base64,{encoded}"

        result = dash_app.parse_contents(content_string, "test.csv")
        sanitized_df, error, report, original_df = result

        assert error is None
        assert len(sanitized_df) == 3
        # Date column should be detected
        date_cols = [c["column"] for c in report.get("date_columns", [])]
        assert "date" in date_cols
