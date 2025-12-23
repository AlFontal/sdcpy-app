"""Unit tests for tasks.py - SDC analysis background tasks."""

import base64
import time

import pandas as pd

# Import the module under test
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tasks


class TestSerializationRoundtrip:
    """Test serialization/deserialization functions."""

    def test_serialize_deserialize_series(self, example_dataframe: pd.DataFrame):
        """Test that series can be serialized and deserialized correctly."""
        original = example_dataframe["oni_anomaly"]
        original.index = pd.to_datetime(example_dataframe["date"])

        serialized = tasks._serialize_series(original)
        deserialized = tasks._deserialize_series(serialized)

        assert len(deserialized) == len(original)
        pd.testing.assert_series_equal(
            deserialized.reset_index(drop=True),
            original.reset_index(drop=True),
            check_names=False,
        )


class TestRunSDCAnalysis:
    """Test the main run_sdc_analysis function."""

    def test_run_sdc_analysis_full(
        self, example_data_payload: dict, default_params: dict
    ):
        """
        Run a complete SDC analysis with example data and verify output structure.

        This is the main integration test that exercises the full analysis pipeline.
        """
        result = tasks.run_sdc_analysis(
            data=example_data_payload,
            ts1=default_params["ts1"],
            ts2=default_params["ts2"],
            date_column=default_params["date_column"],
            method=default_params["method"],
            min_lag=default_params["min_lag"],
            max_lag=default_params["max_lag"],
            window=default_params["window"],
            n_permutations=default_params["n_permutations"],
            labels_fontsize=default_params["labels_fontsize"],
            plot_dpi=default_params["plot_dpi"],
            alpha=default_params["alpha"],
            show_colorbar=default_params["show_colorbar"],
            show_ts2=default_params["show_ts2"],
            plot_width=default_params["plot_width"],
            plot_height=default_params["plot_height"],
        )

        # Verify result structure
        assert "image" in result, "Result should contain 'image' (base64 PNG)"
        assert "excel" in result, "Result should contain 'excel' (base64 Excel)"
        assert "analysis" in result, "Result should contain 'analysis' payload"
        assert "summary" in result, "Result should contain 'summary'"
        assert "plot_settings" in result, "Result should contain 'plot_settings'"

        # Verify image is valid base64
        image_data = result["image"]
        assert image_data, "Image should not be empty"
        decoded = base64.b64decode(image_data)
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n", "Image should be a valid PNG"

        # Verify excel is valid base64
        excel_data = result["excel"]
        assert excel_data, "Excel data should not be empty"
        decoded_excel = base64.b64decode(excel_data)
        # Excel files start with PK (zip format)
        assert decoded_excel[:2] == b"PK", "Excel should be a valid XLSX file"

        # Verify analysis payload structure
        analysis = result["analysis"]
        assert "ts1" in analysis, "Analysis should contain ts1"
        assert "ts2" in analysis, "Analysis should contain ts2"
        assert "sdc_df" in analysis, "Analysis should contain sdc_df"
        assert "method" in analysis, "Analysis should contain method"

        # Verify summary
        summary = result["summary"]
        assert summary["method"] == "pearson"
        assert summary["window"] == 24
        assert summary["min_lag"] == -24
        assert summary["max_lag"] == 24

    def test_run_sdc_analysis_with_permutations(
        self, example_data_payload: dict, default_params: dict
    ):
        """Test analysis with permutations enabled (slower but tests full functionality)."""
        params = default_params.copy()
        params["n_permutations"] = 9  # Small number for faster test

        result = tasks.run_sdc_analysis(
            data=example_data_payload,
            ts1=params["ts1"],
            ts2=params["ts2"],
            date_column=params["date_column"],
            method=params["method"],
            min_lag=params["min_lag"],
            max_lag=params["max_lag"],
            window=params["window"],
            n_permutations=params["n_permutations"],
        )

        assert result["summary"]["n_permutations"] == 9
        assert result["summary"]["permutations"] is True

    def test_run_sdc_analysis_benchmark(
        self, example_data_payload: dict, default_params: dict
    ):
        """
        Benchmark test - measures execution time.

        This test can be used to track performance regressions.
        """
        start_time = time.perf_counter()

        result = tasks.run_sdc_analysis(
            data=example_data_payload,
            ts1=default_params["ts1"],
            ts2=default_params["ts2"],
            date_column=default_params["date_column"],
            method=default_params["method"],
            min_lag=default_params["min_lag"],
            max_lag=default_params["max_lag"],
            window=default_params["window"],
            n_permutations=0,  # No permutations for benchmark
        )

        elapsed = time.perf_counter() - start_time

        assert result is not None
        print(f"\n📊 Benchmark: SDC analysis completed in {elapsed:.2f} seconds")

        # Performance assertion - should complete in reasonable time
        assert elapsed < 120, f"Analysis took too long: {elapsed:.2f}s (max: 120s)"


class TestHelperFunctions:
    """Test helper functions in tasks.py."""

    def test_build_analysis_dataframe(self, example_data_payload: dict):
        """Test DataFrame construction from payload."""
        df = tasks._build_analysis_dataframe(example_data_payload, "date")

        assert isinstance(df, pd.DataFrame)
        assert "oni_anomaly" in df.columns
        assert "temp_anomaly_sa" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df.index)
        assert len(df) == 300  # Example dataset has 300 data rows
