"""
Lightweight Dash app to emulate the SDC `combi_plot` in an interactive plotly format.

Run locally (outside Docker):
    python sdc_heatmap_app.py

It serves on http://0.0.0.0:8051
"""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Patch
from plotly.subplots import make_subplots
from dash.dependencies import Input, Output, State

FILEPATH = 'data/oni_sdc_results_example.xlsx'



def build_figure(grid: pd.DataFrame,
                pvals: pd.DataFrame,
                ts_df: pd.DataFrame,
                fragment_size: int,
                method: str = 'pearson',
                alpha: float = 0.05,
                min_lag: float | None = None,
                max_lag: float | None = None,
                ) -> go.Figure:
    
    # Reserve a 3x3 layout: heatmap in center; top=ts1, left=ts2, bottom/right=max r
    offset = fragment_size // 2
    fig = make_subplots(
        rows=3,
        cols=3,
        row_heights=[0.13, 0.74, 0.13],
        column_widths=[0.13, 0.74, 0.13],
        specs=[
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "heatmap"}, {"type": "xy"}],
            [None,        {"type": "xy"},    None],
        ],
        horizontal_spacing=0.03,
        vertical_spacing=0.03,
        shared_xaxes=True,
        shared_yaxes=True,
    )

    # Align and mask by p-value < alpha
    z = grid
    z.index = z.index + offset
    z.columns = [int(c) + offset for c in z.columns]
    p = pvals.copy()
    p.index = p.index + offset
    p.columns = [int(c) + offset for c in p.columns]
    # Align
    p = p.reindex(index=z.index, columns=z.columns)
    z = z.where(p < alpha)
    lags = z.index.to_numpy()[:, None] - z.columns.to_numpy()[None, :]
    mask = np.ones_like(lags, dtype=bool)
    if min_lag is not None:
        mask &= lags >= min_lag
    if max_lag is not None:
        mask &= lags <= max_lag
    z = z.where(mask)
    p = p.where(mask)


    # Central heatmap
    fig.add_trace(
        go.Heatmap(
            z=z.values,
            x=z.columns,
            y=z.index,
            text=p.values,
            colorscale='RdBu_r',
            zmin=-1, 
            zmax=1, 
            zmid=0,
            colorbar=dict(
                title="Pearson's r",
                lenmode='fraction',
                len=0.6,
                outlinewidth=2,
                outlinecolor='black',
            ),
            hovertemplate=
            'start_1=%{x}<br>start_2=%{y}<br>r=%{z:.3f}<br>p=%{text:.4f}<extra></extra>',
        )
        ,
        row=2, col=2,
    )

    # Top: TS1 
    ts1 = ts_df.dropna(subset=['start_1', 'ts1'])
    ts1_ymin, ts1_ymax = ts1['ts1'].min(), ts1['ts1'].max()
    full_range_ts1 = ts1_ymax - ts1_ymin
    ts1_offset = 0.1 * full_range_ts1
    fig.add_trace(
        go.Scatter(x=ts1['start_1'], y=ts1['ts1'], mode='lines',
                   line=dict(color='black', width=2), name='ts1', showlegend=False),
            row=1, col=2,
        )

    # Left: TS2
    ts2 = ts_df.dropna(subset=['start_2', 'ts2'])
    ts2_ymin, ts2_ymax = ts2['ts2'].min(), ts2['ts2'].max()
    full_range_ts2 = ts2_ymax - ts2_ymin
    ts2_offset = 0.1 * full_range_ts2
    fig.add_trace(
        go.Scatter(x=ts2['ts2'], y=ts2['start_2'], mode='lines',
                   line=dict(color='black', width=2), name='ts2', showlegend=False),
        row=2, col=1,
    )

    # Right-most: per-row max correlation over start_2
    row_max = np.nanmax(z.where(z > 0).values, axis=1)
    row_antimax = np.abs(np.nanmin(z.where(z < 0).values, axis=1))
    fig.add_trace(
        go.Scatter(x=row_max, y=z.index, mode='lines',
                   line=dict(color='#A81529', width=2),
                    name='Max corr', showlegend=False),
        row=2, col=3)
    fig.add_trace(
        go.Scatter(x=row_antimax, y=z.index, mode='lines',
                   line=dict(color='#1E4E79', width=2),
                    name='Max Anti corr', showlegend=False),
        row=2, col=3)

    # Bottom: per-column max correlation over start_1
    col_max = np.nanmax(z.where(z > 0).values, axis=0)
    col_antimax = np.abs(np.nanmin(z.where(z < 0).values, axis=0))

    fig.add_trace(
        go.Scatter(x=z.columns, y=col_max, mode='lines',
                   line=dict(color='#A81529', width=2), 
                   name='Max corr', showlegend=False),
        row=3, col=2,
    )
    fig.add_trace(
        go.Scatter(x=z.columns, y=col_antimax, mode='lines',
                   line=dict(color='#1E4E79', width=2), 
                   name='Max anti-corr', showlegend=False),
        row=3, col=2,
    )

    # Axis + layout styling
    fig.update_yaxes(row=2, col=1, autorange='reversed')
    fig.update_xaxes(title='', showgrid=False, row=1, col=2)
    fig.update_yaxes(title='', row=1, col=2, range=[-2, 2])

    fig.update_xaxes(title='', showgrid=False, row=2, col=2)
    fig.update_yaxes(title='', showgrid=False, autorange='reversed', row=2, col=2)

    fig.update_xaxes(title='Max r', range=[1, 0], row=2, col=3)
    fig.update_yaxes(title='', showgrid=False, row=2, col=3)

    fig.update_xaxes(title='', showgrid=False, row=2, col=1)
    fig.update_yaxes(title='', showgrid=False, row=2, col=1)

    fig.update_xaxes(title='', showgrid=False, row=3, col=2)
    fig.update_yaxes(title='Max r', showgrid=False, row=3, col=2, range=[0, 1])

    fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=40),
        template='simple_white',
        hovermode='closest',
        height=800,
                            
    )

    for row in [1, 2, 3]:
        for col in [1, 2, 3]:
            if col == 2 and row == 2:
                # no border or ticks on heatmap
                fig.update_xaxes(showline=False, linewidth=0, linecolor='black', 
                                mirror=False, row=row, col=col, ticks='')
                fig.update_yaxes(showline=False, linewidth=0, linecolor='black', 
                                mirror=False, row=row, col=col, ticks='')
            else:
                # pass
                fig.update_xaxes(showline=True, linewidth=1.5, linecolor='black', 
                                mirror=True, row=row, col=col)
                fig.update_yaxes(showline=True, linewidth=1.5, linecolor='black', 
                                mirror=True, row=row, col=col)

    return fig


app = Dash(__name__,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            '/assets/custom.css',
            'https://fonts.googleapis.com/css2?family=Play:wght@400;700&display=swap',
        ],
        suppress_callback_exceptions=True,
        )
app.title = 'SDC Heatmap Preview'

app.layout = html.Div(
    [
        html.H2('SDC Result Preview (heatmap + margins)'),
        html.Div([
            html.Span('Alpha: ', style={'marginRight': '8px'}),
            dcc.Input(id='alpha-input', type='number', value=0.05, min=0.0, max=1.0, step=0.01,
                      style={'width': '100px'}),
            html.Span('Min lag: ', style={'marginRight': '8px'}),
            dcc.Input(id='min-lag-input', type='number', value=None, step=1,
                      style={'width': '100px'}),
            html.Span('Max lag: ', style={'marginRight': '8px'}),
            dcc.Input(id='max-lag-input', type='number', value=None, step=1,
                      style={'width': '100px'}),
            html.Span(id='data-status', style={'marginLeft': '16px'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '12px', 'marginBottom': '8px'}),
        dcc.Graph(id='sdc-heatmap', style={'height': '720px'}),
        dcc.Store(id='fragment-size-store'),
    ],
    style={'maxWidth': '1120px', 'margin': '0 auto', 'padding': '24px'}
)


@app.callback(
    Output('sdc-heatmap', 'figure'),
    Output('data-status', 'children'),
    Output('fragment-size-store', 'data'),
    Input('alpha-input', 'value'),
    Input('min-lag-input', 'value'),
    Input('max-lag-input', 'value'),
    prevent_initial_call=False
)
def _load(alpha, min_lag, max_lag):
    # Read with index_col=0 to recover the grid index
    try:
        rs_grid = pd.read_excel(FILEPATH, sheet_name=0, index_col=0)
        pv_grid = pd.read_excel(FILEPATH, sheet_name=1, index_col=0)
        ts_df = pd.read_excel(FILEPATH, sheet_name=2)
        config = pd.read_excel(FILEPATH, sheet_name=3)
    except Exception as e:
        return go.Figure(), f'Failed to load {FILEPATH}: {e}'

    try:
        alpha_val = float(alpha) if alpha is not None else 0.05
    except (TypeError, ValueError):
        alpha_val = 0.05
    min_lag_val = None
    if min_lag is not None:
        try:
            min_lag_val = float(min_lag)
        except (TypeError, ValueError):
            min_lag_val = None
    max_lag_val = None
    if max_lag is not None:
        try:
            max_lag_val = float(max_lag)
        except (TypeError, ValueError):
            max_lag_val = None
    s = config['fragment_size'].values[0]
    method = config['method'].values[0]

    fig = build_figure(rs_grid, pvals=pv_grid, ts_df=ts_df, alpha=alpha_val,
                      fragment_size=s, method=method, min_lag=min_lag_val, max_lag=max_lag_val)
    status = f'Loaded grid {rs_grid.shape[0]}×{rs_grid.shape[1]}, alpha={alpha_val:.3f}'
    lag_terms = []
    if min_lag_val is not None:
        lag_terms.append(f'lag≥{min_lag_val:g}')
    if max_lag_val is not None:
        lag_terms.append(f'lag≤{max_lag_val:g}')
    if lag_terms:
        status += ', ' + ', '.join(lag_terms)
    return fig, status, int(s)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8051)
