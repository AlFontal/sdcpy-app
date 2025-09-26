"""Background tasks for executing SDC analysis with progress reporting."""

from __future__ import annotations

import io
import math
import base64
import re
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import sdcpy.scale_dependent_correlation as sdc

from typing import Any, Dict, Optional
from pandas import ExcelWriter
from rq import get_current_job

metric_maps = {
    "pearson": "Pearson's r",
    "spearman": "Spearman's $\\rho$",
}


def _serialize_series(series: pd.Series) -> Dict[str, Any]:
    return {
        "json": series.to_json(date_format='iso'),
        "name": series.name,
        "index_name": series.index.name,
    }


def _deserialize_series(payload: Dict[str, Any]) -> pd.Series:
    series = pd.read_json(io.StringIO(payload["json"]), typ='series')
    series.index = pd.to_datetime(series.index)
    if payload.get("name") is not None:
        series.name = payload["name"]
    if payload.get("index_name") is not None:
        series.index.name = payload["index_name"]
    return series


def _serialize_analysis(analysis: sdc.SDCAnalysis, ts1_label: str, ts2_label: str,
                        min_lag: float, max_lag: float) -> Dict[str, Any]:
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
        "sdc_df": analysis.sdc_df.to_json(orient='split', date_format='iso'),
    }


def _deserialize_analysis(payload: Dict[str, Any]) -> sdc.SDCAnalysis:
    ts1 = _deserialize_series(payload["ts1"])
    ts2 = _deserialize_series(payload["ts2"])
    sdc_df = pd.read_json(io.StringIO(payload["sdc_df"]), orient='split')
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
        permutations=payload.get("permutations", False)
    )


def _progress_wrapper():
    """Monkey patch tqdm in sdc module to surface progress via RQ meta."""

    job = get_current_job()
    original_tqdm = sdc.tqdm

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


def combi_plot(self, alpha: float = .05, xlabel: str = '', ylabel: str = '', title: str = None, max_r: float = None,
                   date_fmt: str = '%m-%d', align: str = 'center', max_lag: int = np.inf, min_lag: int = -np.inf,
                   labels_fontsize: int = 12, wspace: float = 1., hspace: float = 1., show_colorbar: bool = True,
                   metric_label: str = None, show_ts2: bool = True,
                   **kwargs):
        # Setting up parameters
        frequency = pd.infer_freq(self.ts1.index)
        if frequency:
            if re.match(r'^[0-9]*D', frequency):
                freq_str = 'days'
                freq_unit = 'D'
                if re.match(r'^[0-9]+D', frequency):
                    freq_mult = int(re.match(r'^([0-9]+)D', frequency).groups()[0])
                else:
                    freq_mult = 1
            # capture first alphabetic (non-numeric) letter and check if it's W
            elif re.match(r'^[0-9]*W', frequency):
                freq_str = 'weeks'
                freq_unit = 'W'
                if re.match(r'^[0-9]+W', frequency):
                    freq_mult = int(re.match(r'^([0-9]+)W', frequency).groups()[0])
                else:
                    freq_mult = 1
            elif frequency[0] == 'M':
                freq_str = 'months'
                freq_unit = 'W'
                if re.match(r'^[0-9]+M', frequency):
                    freq_mult = int(re.match(r'^([0-9]+)M', frequency).groups()[0]) * 4.33
                else:
                    freq_mult = 4.33
            else: 
                freq_str = ''
        else:
            raise warnings.warn('Could not infer frequency of the time-series. Axis labels may be incorrect.')
            freq_str = ''
        
        title = f'' if title is None else title
        align = align.lower()
        metric_label = metric_label if metric_label is not None else self.method

        if align not in ['left', 'center', 'right']:
            warnings.warn(f'Alignment method "{align}" not recognized, defaulting to center alignment.')
            align = 'center'
        offset = self.fragment_size // 2 if align == 'center' else self.fragment_size
        left_offset = 0 if align == 'left' else offset
        right_offset = 0 if align == 'right' else offset

        date_format = mdates.DateFormatter(date_fmt)
        sdc_df = self.sdc_df.copy()
        fig = plt.figure(**kwargs)
        # We are organizing the grid in a 5 x 5 matrix so that (TT=Title, HM: Heatmap, TS1/TS2: Time-Series 1/2,
        # MC: Max Correlations):
        # TT TT TT TT TT
        # NA TS1 TS1 NA NA
        # TS2 HM HM MC2 CB
        # TS2 HM HM MC2 CB
        # NA MC1 MC1 NA NA
        # gs = fig.add_gridspec(4, 4, height_ratios=[.15, .7, 2, 2], width_ratios=[.7, 2, 2, .2])
        if min_lag < 0 < max_lag:
            gs = fig.add_gridspec(5, 5, height_ratios=[.15, 1, 2, 2, 1], width_ratios=[1, 2, 2, 1, .2])
        elif min_lag < 0:
            gs = fig.add_gridspec(5, 4, height_ratios=[.15, 1, 2, 2, 1], width_ratios=[1, 2, 2, .3])
        elif max_lag > 0:
            gs = fig.add_gridspec(4, 5, height_ratios=[.15, 1, 2, 1], width_ratios=[1, 2, 2, 1, .2])
        else:
            raise ValueError('Range of lags to be considered should be bigger than 1')
        
        # Time series 1
        ts1 = fig.add_subplot(gs[1, 1:3])
        ts1.plot(self.ts1, color='black', linewidth=1)
        # Time series 2
        if show_ts2:
            ts2 = fig.add_subplot(gs[2:4, 0])
            plt.plot(self.ts2, self.ts2.reset_index()['date_2'], color='black', linewidth=1)
        
        # Heat map
        hm = fig.add_subplot(gs[2:4, 1:3])

        if self.method in ['pearson', 'spearman']:
            (sdc_df
            .loc[lambda dd: (dd.lag <= max_lag) & (dd.lag >= min_lag)]
            .pipe(lambda dd: sns.heatmap(dd.pivot(index='date_2', columns='date_1', values='r'), cbar=False,
                                        mask=dd.pivot(index='date_2', columns='date_1', values='p_value') >= alpha,
                                        cmap='RdBu_r', ax=hm))
            )

        elif self.method == 'custom':
            (sdc_df
            .loc[lambda dd: (dd.lag <= max_lag) & (dd.lag >= min_lag)]
            .pipe(lambda dd: sns.heatmap(dd.pivot(index='date_2', columns='date_1', values='r'), cbar=False,
                                        mask=dd.pivot(index='date_2', columns='date_1', values='p_value') >= alpha,
                                        cmap='plasma', ax=hm))
            )
        # Add identity line to ease shift visualization
        identity_len = min(len(self.ts1), len(self.ts2)) - self.fragment_size + 1
        plt.plot(range(identity_len), range(identity_len), linestyle=':', 
                 color='black', alpha=.4, linewidth=1)
        # Correct and format ticks, labels, grids
        # Hide Heatmap labels and ticks
        hm.set_xlabel('')
        hm.set_ylabel('')
        plt.setp(hm.get_yticklabels(), visible=False)
        plt.setp(hm.get_xticklabels(), visible=False)
        hm.tick_params(axis='both', which='both', length=0)
        # Each dot in the heatmap represents a `fragment_size` long region of the time-series, so we need to choose how
        # to represent each dot because the heatmap axis are `fragment_size` shorter than the time-series axis.
        # Alignment parameter comes then into play:
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        max_r = max_r if max_r is not None else self.sdc_df.r.abs().max()
        hm.set_xlim(xmin - left_offset, xmax + right_offset)
        hm.set_ylim(ymin + right_offset, ymax - left_offset)
        trans_x = hm.get_xaxis_transform()
        trans_y = hm.get_yaxis_transform()

        hm.plot([-self.fragment_size / 2, self.fragment_size / 2], [1.0, 1.0],
                color="k", transform=trans_x, clip_on=False, linewidth=5, solid_capstyle='butt')
        hm.plot([0, 0], [-self.fragment_size / 2, self.fragment_size / 2],
                color='k', transform=trans_y, clip_on=False, linewidth=5, solid_capstyle='butt')

        hm.annotate(f'$s={self.fragment_size}$ {freq_str}', xy=(self.fragment_size / 2 + 5, .99),
                    xycoords=trans_x, fontsize=labels_fontsize)

        # Handle TS1 labels and ticks
        ts1.xaxis.set_major_formatter(date_format)
        ts1.xaxis.set_label_position('top')
        ts1.set_xlim(self.ts1.index[0], self.ts1.index[-1])
        ts1.grid(True, which='major', axis='x', linestyle='--', alpha=.5)
        # ts1.xaxis.set_major_locator(ticker.MultipleLocator(len(self.ts1) * 3))
        ts1.set_xlabel(xlabel, fontsize=labels_fontsize + 2)
        ts1.tick_params(
             axis='x',
             top=True,
             labeltop=True, 
             labelbottom=False,
             bottom=False,
             labelsize=labels_fontsize,
             )

        # Handle TS2 labels and ticks
        if show_ts2:
            ts2.yaxis.set_major_formatter(date_format)
            ts2.set_ylim(self.ts2.index[0], self.ts2.index[-1])
            ts2.grid(True, which='major', axis='y', linestyle='--', alpha=.5)
            ts2.invert_xaxis()
            ts2.invert_yaxis()

            ts2.set_ylabel(ylabel, fontsize=labels_fontsize + 2)
            plt.setp(ts2.get_yticklabels(), visible=True, rotation=90, va='center')
            ts2.tick_params(
                axis='y',
                top=True,
                labeltop=True, 
                labelbottom=False,
                bottom=False,
                labelsize=labels_fontsize,
                )
        gs.update(wspace=wspace, hspace=hspace)

        # Colorbar
        if show_colorbar:
            cax = fig.add_subplot(gs[2:4, -1])
        color_mesh = hm.get_children()[0]
        lims = -1, 1 if self.method in ['pearson', 'spearman'] else 0, max_r
        color_mesh.set_clim(lims[0], lims[1])
        if show_colorbar:
            fig.colorbar(color_mesh, cax=cax, label=metric_label, pad=0.05)
        
        colors = {'Max $r$': '#A81529', 'Min $r$ (abs)': '#144E8A'}
        if min_lag < 0:
            mc1 = fig.add_subplot(gs[-1, 1:3])
            (self.sdc_df
            .loc[lambda dd: dd.p_value < alpha]
            .loc[lambda dd: (dd.lag <= max_lag) & (dd.lag >= min_lag)]
            .groupby('date_1')
            .agg(r_max=('r', lambda x: x.where(x > 0).max()), r_min=('r', lambda x: abs(x.where(x < 0).min())))
            .rename(columns={'r_max': 'Max $r$', 'r_min': 'Min $r$ (abs)'})
            .reset_index()
            .melt('date_1')
            .assign(date_1=lambda dd: dd.date_1 + pd.to_timedelta(left_offset * freq_mult, unit=freq_unit))
            .assign(color=lambda dd: dd.variable.apply(lambda x: colors[x]))
            .plot.scatter(x='date_1', y='value', c='color', ax=mc1, alpha=.7, colorbar=False, s=10)
            )
            plt.setp(mc1.get_xticklabels(), visible=False)
            mc1.set_xlabel('')
            mc1.set_ylabel('Max |corr|')
            mc1.yaxis.set_label_position('right')
            mc1.set_xlim(self.ts1.index[0], self.ts1.index[-1])
            mc1.set_ylim(0, 1.05)
            mc1.grid(True, which='major')
            mc1.set_yticks([0, .5, 1])
        if max_lag > 0:
            mc2 = fig.add_subplot(gs[2:4, 3])
            (self.sdc_df
            .loc[lambda dd: dd.p_value < alpha]
            .loc[lambda dd: (dd.lag <= max_lag) & (dd.lag >= min_lag)]
            .groupby('date_2')
            .agg(r_max=('r', lambda x: x.where(x > 0).max()), r_min=('r', lambda x: abs(x.where(x < 0).min())))
            .rename(columns={'r_max': 'Max $r$', 'r_min': 'Min $r$ (abs)'})
            .reset_index()
            .melt('date_2')
            .assign(date_2=lambda dd: dd.date_2 + pd.to_timedelta(left_offset * freq_mult, unit=freq_unit))
            .assign(color=lambda dd: dd.variable.apply(lambda x: colors[x]))
            .plot.scatter(x='value', y='date_2', c='color', ax=mc2, alpha=.7, colorbar=False, s=10)
            )
            plt.setp(mc2.get_yticklabels(), visible=False)
            mc2.set_xlabel('Max |corr|')
            mc2.set_ylabel('')
            mc2.grid(True, which='major')
            mc2.set_xlim(1.05, 0)
            mc2.yaxis.set_label_position('left')
            mc2.set_ylim(self.ts2.index[-1], self.ts2.index[0])


        fig.suptitle(title)

        return fig


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
) -> str:
    buffer = io.BytesIO()
    fig = combi_plot(
        analysis,
        xlabel=ts1,
        ylabel=ts2,
        wspace=.15,
        hspace=.15,
        date_fmt="%Y-%m",
        alpha=alpha,
        labels_fontsize=labels_fontsize,
        max_lag=max_lag,
        min_lag=min_lag,
        metric_label=metric_maps.get(analysis.method, analysis.method),
        figsize=figsize,
        title=None,
        show_colorbar=show_colorbar,
        show_ts2=show_ts2,
    )
    fig.savefig(buffer, format="png", dpi=plot_dpi, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


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
) -> str:
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
                    analysis.ts1.rename("ts1").reset_index().reset_index().rename(columns={"index": "start_1"}),
                    analysis.ts2.rename("ts2").reset_index().reset_index().rename(columns={"index": "start_2"}),
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
        job.meta["progress"] = {"current": 0, "total": 1, "description": "Preparing data"}
        job.save_meta()

    patched_tqdm, original_tqdm = _progress_wrapper()
    sdc.tqdm = patched_tqdm

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
            alpha = float(alpha) if alpha not in (None, '') else 0.05
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

        analysis_payload = _serialize_analysis(analysis, ts1, ts2, min_lag, max_lag)
        plot_b64 = _encode_plot(
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
            "image": plot_b64,
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
        sdc.tqdm = original_tqdm
