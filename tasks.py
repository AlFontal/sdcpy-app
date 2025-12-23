"""Background tasks for executing SDC analysis with progress reporting."""

from __future__ import annotations

import io
import math
import base64
import numpy as np
import pandas as pd

# Force non-interactive backend for faster rendering in worker processes
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Optimize matplotlib for performance
plt.rcParams["path.simplify"] = True
plt.rcParams["path.simplify_threshold"] = 1.0
plt.rcParams["agg.path.chunksize"] = 10000

import sdcpy.scale_dependent_correlation as sdc  # noqa: E402
import sdcpy.core as sdc_core  # noqa: E402

from typing import Any, Dict, Optional  # noqa: E402
from pandas import ExcelWriter  # noqa: E402
from rq import get_current_job  # noqa: E402

metric_maps = {
    "pearson": "Pearson's r",
    "spearman": "Spearman's $\\rho$",
}


def _serialize_series(series: pd.Series) -> Dict[str, Any]:
    return {
        "json": series.to_json(date_format="iso"),
        "name": series.name,
        "index_name": series.index.name,
    }


def _deserialize_series(payload: Dict[str, Any]) -> pd.Series:
    series = pd.read_json(io.StringIO(payload["json"]), typ="series")
    series.index = pd.to_datetime(series.index)
    if payload.get("name") is not None:
        series.name = payload["name"]
    if payload.get("index_name") is not None:
        series.index.name = payload["index_name"]
    return series


def _serialize_analysis(
    analysis: sdc.SDCAnalysis,
    ts1_label: str,
    ts2_label: str,
    min_lag: float,
    max_lag: float,
) -> Dict[str, Any]:
    min_lag_serial = None if not math.isfinite(min_lag) else float(min_lag)
    max_lag_serial = None if not math.isfinite(max_lag) else float(max_lag)
    return {
        "fragment_size": analysis.fragment_size,
        "method": analysis.method,
        "min_lag": min_lag_serial,
        "max_lag": max_lag_serial,
        "n_permutations": getattr(analysis, "n_permutations", None),
        "permutations": bool(getattr(analysis, "permutations", False)),
        "ts1": _serialize_series(analysis.ts1),
        "ts2": _serialize_series(analysis.ts2),
        "ts1_label": ts1_label,
        "ts2_label": ts2_label,
        "sdc_df": analysis.sdc_df.to_json(orient="split", date_format="iso"),
    }


def _deserialize_analysis(payload: Dict[str, Any]) -> sdc.SDCAnalysis:
    ts1 = _deserialize_series(payload["ts1"])
    ts2 = _deserialize_series(payload["ts2"])
    sdc_df = pd.read_json(io.StringIO(payload["sdc_df"]), orient="split")
    for column in ("date_1", "date_2"):
        if column in sdc_df.columns:
            sdc_df[column] = pd.to_datetime(sdc_df[column])

    min_lag = payload.get("min_lag")
    max_lag = payload.get("max_lag")
    min_lag = -np.inf if min_lag is None else float(min_lag)
    max_lag = np.inf if max_lag is None else float(max_lag)

    n_permutations = payload.get("n_permutations")
    if n_permutations is None:
        n_permutations = 99

    return sdc.SDCAnalysis(
        ts1=ts1,
        ts2=ts2,
        fragment_size=payload["fragment_size"],
        method=payload["method"],
        min_lag=min_lag,
        max_lag=max_lag,
        sdc_df=sdc_df,
        n_permutations=n_permutations,
        permutations=payload.get("permutations", False),
    )


def _progress_wrapper():
    """Monkey patch tqdm in sdc module to surface progress via RQ meta."""

    job = get_current_job()
    original_tqdm = sdc_core.tqdm

    if job is None:
        return original_tqdm, original_tqdm

    def reporting_tqdm(*args: Any, **kwargs: Any):
        bar = original_tqdm(*args, **kwargs)

        job.meta["progress"] = {
            "current": 0,
            "total": bar.total,
            "description": kwargs.get("desc", getattr(bar, "desc", "")),
        }
        job.save_meta()

        original_update = bar.update

        def update(n: int = 1) -> None:
            original_update(n)
            job.meta["progress"] = {
                "current": bar.n,
                "total": bar.total,
                "description": getattr(bar, "desc", ""),
            }
            job.save_meta()

        bar.update = update  # type: ignore[assignment]
        return bar

    return reporting_tqdm, original_tqdm


def _build_analysis_dataframe(data: Dict[str, Any], date_column: str) -> pd.DataFrame:
    df = pd.DataFrame(data)
    return df.assign(date=lambda dd: pd.to_datetime(dd[date_column])).set_index("date")


def _encode_plot(
    analysis: sdc.SDCAnalysis,
    ts1: str,
    ts2: str,
    min_lag: float,
    max_lag: float,
    *,
    labels_fontsize: int = 12,
    show_colorbar: bool = True,
    show_ts2: bool = True,
    plot_dpi: int = 150,
    figsize: tuple[float, float] = (7, 7),
    alpha: float = 0.05,
) -> tuple[str, str]:
    png_buffer = io.BytesIO()
    fig = analysis.combi_plot(
        xlabel=ts1,
        ylabel=ts2,
        wspace=0.15,
        hspace=0.15,
        date_fmt="%Y-%m",
        alpha=alpha,
        label_fontsize=labels_fontsize,
        max_lag=max_lag,
        min_lag=min_lag,
        metric_label=metric_maps.get(analysis.method, analysis.method),
        figsize=figsize,
        title=None,
        show_colorbar=show_colorbar,
        show_ts2=show_ts2,
        dpi=plot_dpi,
    )
    fig.savefig(png_buffer, format="png", dpi=plot_dpi, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(png_buffer.getvalue()).decode("utf-8"), None


def render_sdc_plot_from_payload(
    payload: Dict[str, Any],
    *,
    labels_fontsize: int = 12,
    show_colorbar: bool = True,
    show_ts2: bool = True,
    plot_dpi: int = 300,
    figsize: tuple[float, float] = (7, 7),
    min_lag_override: Optional[int] = None,
    max_lag_override: Optional[int] = None,
    alpha: float = 0.05,
) -> tuple[str, str]:
    analysis = _deserialize_analysis(payload)
    min_lag = payload.get("min_lag")
    max_lag = payload.get("max_lag")
    min_lag = -np.inf if min_lag is None else float(min_lag)
    max_lag = np.inf if max_lag is None else float(max_lag)
    if min_lag_override is not None:
        try:
            min_lag = int(min_lag_override)
        except (TypeError, ValueError):
            pass
    if max_lag_override is not None:
        try:
            max_lag = int(max_lag_override)
        except (TypeError, ValueError):
            pass
    return _encode_plot(
        analysis,
        payload.get("ts1_label", "Time Series 1"),
        payload.get("ts2_label", "Time Series 2"),
        min_lag,
        max_lag,
        labels_fontsize=labels_fontsize,
        show_colorbar=show_colorbar,
        show_ts2=show_ts2,
        plot_dpi=plot_dpi,
        figsize=figsize,
        alpha=alpha,
    )


def _build_excel_payload(analysis: sdc.SDCAnalysis) -> bytes:
    buffer = io.BytesIO()
    with ExcelWriter(buffer, engine="xlsxwriter") as writer:
        (
            analysis.sdc_df.dropna()
            .pivot(index="start_1", columns="start_2", values="r")
            .to_excel(writer, sheet_name="rs")
        )
        (
            analysis.sdc_df.dropna()
            .pivot(index="start_1", columns="start_2", values="p_value")
            .to_excel(writer, sheet_name="p_values")
        )

        (
            pd.concat(
                [
                    analysis.ts1.rename("ts1")
                    .reset_index()
                    .reset_index()
                    .rename(columns={"index": "start_1"}),
                    analysis.ts2.rename("ts2")
                    .reset_index()
                    .reset_index()
                    .rename(columns={"index": "start_2"}),
                ],
                axis=1,
            ).to_excel(writer, sheet_name="time_series", index=False)
        )

        pd.DataFrame(
            {
                "fragment_size": analysis.fragment_size,
                "n_permutations": analysis.n_permutations,
                "method": analysis.method,
            },
            index=[1],
        ).to_excel(writer, sheet_name="config", index=False)

    return buffer.getvalue()


def run_sdc_analysis(
    data: Dict[str, Any],
    ts1: str,
    ts2: str,
    date_column: str,
    method: str,
    min_lag: int,
    max_lag: int,
    window: int,
    n_permutations: Optional[int] = None,
    labels_fontsize: Optional[int] = None,
    plot_dpi: Optional[int] = None,
    alpha: Optional[float] = None,
    show_colorbar: bool = True,
    show_ts2: bool = True,
    plot_width: Optional[float] = None,
    plot_height: Optional[float] = None,
    plot_min_lag: Optional[int] = None,
    plot_max_lag: Optional[int] = None,
) -> Dict[str, Any]:
    job = get_current_job()
    if job is not None:
        job.meta["progress"] = {
            "current": 0,
            "total": 1,
            "description": "Preparing data",
        }
        job.save_meta()

    patched_tqdm, original_tqdm = _progress_wrapper()
    sdc_core.tqdm = patched_tqdm

    try:
        fragment_size = int(window)
        min_lag = -np.inf if min_lag is None else int(min_lag)
        max_lag = np.inf if max_lag is None else int(max_lag)
        labels_fontsize = int(labels_fontsize) if labels_fontsize else 12
        plot_dpi = int(plot_dpi) if plot_dpi else 150
        show_colorbar = bool(show_colorbar)
        show_ts2 = bool(show_ts2)
        try:
            n_permutations = int(n_permutations) if n_permutations is not None else 99
        except (TypeError, ValueError):
            n_permutations = 99
        if n_permutations < 0:
            n_permutations = 0
        permutations_enabled = n_permutations > 0

        try:
            alpha = float(alpha) if alpha not in (None, "") else 0.05
        except (TypeError, ValueError):
            alpha = 0.05
        if not 0 < alpha <= 1:
            alpha = 0.05

        plot_width = float(plot_width) if plot_width else 7.0
        plot_height = float(plot_height) if plot_height else 7.0
        if plot_width <= 0:
            plot_width = 7.0
        if plot_height <= 0:
            plot_height = 7.0

        def _coerce_lag(value, fallback):
            if value is None or value == "":
                return fallback
            try:
                return int(value)
            except (TypeError, ValueError):
                return fallback

        plot_min_lag = _coerce_lag(plot_min_lag, min_lag)
        plot_max_lag = _coerce_lag(plot_max_lag, max_lag)

        df = _build_analysis_dataframe(data, date_column)
        analysis = sdc.SDCAnalysis(
            ts1=df[ts1],
            ts2=df[ts2],
            method=method,
            min_lag=min_lag,
            max_lag=max_lag,
            fragment_size=fragment_size,
            n_permutations=n_permutations,
            permutations=permutations_enabled,
        )

        if job is not None:
            progress = job.meta.get("progress", {})
            total = progress.get("total")
            job.meta["progress"] = {
                "current": total,
                "total": total,
                "description": "Generating plot...",
            }
            job.save_meta()

        analysis_payload = _serialize_analysis(analysis, ts1, ts2, min_lag, max_lag)
        plot_png, plot_svg = _encode_plot(
            analysis,
            ts1,
            ts2,
            plot_min_lag,
            plot_max_lag,
            labels_fontsize=labels_fontsize,
            show_colorbar=show_colorbar,
            show_ts2=show_ts2,
            plot_dpi=plot_dpi,
            figsize=(plot_width, plot_height),
            alpha=alpha,
        )
        excel_bytes = _build_excel_payload(analysis)

        if job is not None:
            progress = job.meta.get("progress", {})
            total = progress.get("total")
            job.meta["progress"] = {
                "current": total,
                "total": total,
                "description": "Completed",
            }
            job.save_meta()

        summary_min = None if not math.isfinite(min_lag) else min_lag
        summary_max = None if not math.isfinite(max_lag) else max_lag
        plot_settings = {
            "label_fontsize": labels_fontsize,
            "dpi": plot_dpi,
            "show_colorbar": show_colorbar,
            "show_ts2": show_ts2,
            "width": plot_width,
            "height": plot_height,
            "min_lag": None if not math.isfinite(plot_min_lag) else plot_min_lag,
            "max_lag": None if not math.isfinite(plot_max_lag) else plot_max_lag,
            "n_permutations": n_permutations,
            "permutations": permutations_enabled,
            "alpha": alpha,
        }

        return {
            "image": plot_png,
            "image_svg": plot_svg,
            "excel": base64.b64encode(excel_bytes).decode("utf-8"),
            "analysis": analysis_payload,
            "plot_settings": plot_settings,
            "summary": {
                "method": method,
                "window": fragment_size,
                "min_lag": summary_min,
                "max_lag": summary_max,
                "n_permutations": n_permutations,
                "permutations": permutations_enabled,
                "alpha": alpha,
                "plot": plot_settings,
            },
        }

    finally:
        sdc_core.tqdm = original_tqdm
