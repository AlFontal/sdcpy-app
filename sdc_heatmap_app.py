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
from dash.exceptions import PreventUpdate

FILEPATH = 'data/oni_sdc_results_example.xlsx'
METHOD_LABELS = {
    'pearson': "Pearson's r",
    'spearman': "Spearman's rho",
}


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
            [None, {"type": "xy"}, None],
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
                title=METHOD_LABELS[method],
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
    ts1_range = ts1_ymax - ts1_ymin

    fig.add_trace(
        go.Scatter(x=ts1['start_1'], y=ts1['ts1'], mode='lines',
                   line=dict(color='black', width=2), name='ts1', showlegend=False),
            row=1, col=2,
        )
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode='lines',
            line=dict(color="#016B8B", width=3),
            name='ts1-highlight',
            opacity=1,
            hoverinfo='skip',
            showlegend=False,
        ),
        row=1, col=2,
    )

    # Left: TS2
    ts2 = ts_df.dropna(subset=['start_2', 'ts2'])
    ts2_min, ts2_max = ts2['ts2'].min(), ts2['ts2'].max()
    ts2_range = ts2_max - ts2_min
    fig.add_trace(
        go.Scatter(x=ts2['ts2'], y=ts2['start_2'], mode='lines',
                   line=dict(color='black', width=2), name='ts2', showlegend=False),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode='lines',
            line=dict(color="#016B8B", width=3),
            opacity=1,
            name='ts2-highlight',
            hoverinfo='skip',
            showlegend=False,
        ),
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

    # Annotations: fragment size, lag=0 diagonal
    cols_numeric = pd.to_numeric(pd.Index(z.columns), errors='coerce').to_numpy()
    rows_numeric = pd.to_numeric(pd.Index(z.index), errors='coerce').to_numpy()
    cols_clean = cols_numeric[~np.isnan(cols_numeric)]
    rows_clean = rows_numeric[~np.isnan(rows_numeric)]
    diag_vals = np.intersect1d(cols_clean, rows_clean)
    if diag_vals.size > 1:
        fig.add_trace(
            go.Scatter(
                x=diag_vals,
                y=diag_vals,
                mode='lines',
                line=dict(color='black', dash='dot', width=1.5),
                hoverinfo='skip',
                showlegend=False,
                name='Lag 0',
            ),
            row=2, col=2,
        )
    if cols_clean.size and rows_clean.size:

        fig.add_shape(
            type='line',
            x0=0,
            y0=0,
            x1=fragment_size,
            y1=0,
            line=dict(color='black', width=6),
            opacity=1,
            row=2,
            col=2,
        )
        fig.add_shape(
            type='line',
            x0=0,
            y0=0,
            x1=0,
            y1=fragment_size,
            line=dict(color='black', width=6),
            opacity=1,
            row=2,
            col=2,
        )
        fig.add_annotation(
            x=.07 * len(ts1),
            y=.03 * len(ts2),
            text=f's = {fragment_size}',
            showarrow=False,
            font=dict(color='black', size=14),
            row=2, col=2,
        )
    highlight_fill = 'white'
    fig.add_shape(
        type='rect',
        name='ts1-mask-left',
        x0=0,
        y0=0,
        x1=0,
        y1=0,
        fillcolor=highlight_fill,
        opacity=.8,
        line=dict(width=0),
        visible=False,
        layer='above',
        row=1, col=2,
    )
    fig.add_shape(
        type='rect',
        name='ts1-mask-right',
        x0=0,
        y0=0,
        x1=0,
        y1=0,
        fillcolor=highlight_fill,
        line=dict(width=0),
        opacity=.8,
        visible=False,
        layer='above',
        row=1, col=2,
    )
    fig.add_shape(
        type='rect',
        name='ts2-mask-bottom',
        x0=0,
        y0=0,
        x1=0,
        y1=0,
        fillcolor=highlight_fill,
        opacity=.8,
        line=dict(width=0),
        visible=False,
        layer='above',
        row=2, col=1,
    )
    fig.add_shape(
        type='rect',
        name='ts2-mask-top',
        x0=0,
        y0=0,
        x1=0,
        y1=0,
        fillcolor=highlight_fill,
        opacity=.8,
        line=dict(width=0),
        visible=False,
        layer='above',
        row=2, col=1,
    )

    # Axis + layout styling
    side_panel_kwargs = dict(
        showline=True, 
        linewidth=1.5, 
        linecolor='black', 
        mirror=True, 
        showgrid=False,
        )
    for row in range(1, 4):
        for col in range(1, 4):
            if not (row==2 and col==2):
                fig.update_xaxes(row=row, col=col, **side_panel_kwargs)
                fig.update_yaxes(row=row, col=col, **side_panel_kwargs)
    

    hm_kwargs = dict(showline=False, linewidth=0, linecolor='black', mirror=False, ticks='')
    fig.update_xaxes(row=2, col=2, **hm_kwargs)
    fig.update_yaxes(row=2, col=2, **hm_kwargs)

    fig.update_xaxes(row=1, col=2, side='top', showticklabels=True)
    fig.update_yaxes(row=1, col=2, range=[ts1_ymin - ts1_range * 0.1, ts1_ymax + ts1_range * 0.1])
    fig.update_yaxes(row=2, col=1, range=[len(ts2) * 1.05,  - 0.05 * len(ts2)])
    fig.update_xaxes(row=2, col=1, autotickangles=[0], range=[ts2_max + ts2_range * 0.1,
                                                             ts2_min - ts2_range * 0.1,])
    
    fig.update_xaxes(row=2, col=3, range=[1.01, -0.01])
    fig.update_yaxes(row=2, col=3, ticks='')
    fig.update_yaxes(row=3, col=2, range=[-0.01, 1.01])
    fig.update_xaxes(row=3, col=2, range=[-len(ts1) * .05, len(ts1) * 1.05], 
                    showticklabels=False, ticks='')

    fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=40),
        template='simple_white',                            
    )

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
        dcc.Graph(id='sdc-heatmap', style={'width': '100%', 'aspectRatio': '1.2', 'maxWidth': '900px'}),
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


@app.callback(
    Output('sdc-heatmap', 'figure', allow_duplicate=True),
    Input('sdc-heatmap', 'hoverData'),
    State('fragment-size-store', 'data'),
    State('sdc-heatmap', 'figure'),
    prevent_initial_call=True,
)
def _highlight_hover(hover_data, fragment_size, figure_state):
    if not figure_state:
        raise PreventUpdate

    data = figure_state.get('data', [])
    layout = figure_state.get('layout', {})

    def _trace_idx(target):
        return next((i for i, trace in enumerate(data) if trace.get('name') == target), None)

    ts1_idx = _trace_idx('ts1')
    ts1_high_idx = _trace_idx('ts1-highlight')
    ts2_idx = _trace_idx('ts2')
    ts2_high_idx = _trace_idx('ts2-highlight')
    if None in (ts1_idx, ts1_high_idx, ts2_idx, ts2_high_idx):
        raise PreventUpdate

    shape_indices = {shape.get('name'): idx for idx, shape in enumerate(layout.get('shapes', []))}
    mask_names = ['ts1-mask-left', 'ts1-mask-right', 'ts2-mask-bottom', 'ts2-mask-top']
    if not all(name in shape_indices for name in mask_names):
        raise PreventUpdate

    patch = Patch()

    def _clear():
        patch['data'][ts1_high_idx]['x'] = []
        patch['data'][ts1_high_idx]['y'] = []
        patch['data'][ts2_high_idx]['x'] = []
        patch['data'][ts2_high_idx]['y'] = []
        for name in mask_names:
            patch['layout']['shapes'][shape_indices[name]]['visible'] = False
        return patch

    if hover_data is None or fragment_size is None:
        return _clear()

    points = hover_data.get('points') or []
    point = points[0] if points else None
    if not point or point.get('curveNumber') != 0:
        return _clear()

    try:
        center_x = float(point.get('x'))
        center_y = float(point.get('y'))
        span = float(fragment_size)
    except (TypeError, ValueError):
        return _clear()

    if span <= 0:
        return _clear()

    half = span / 2.0
    try:
        ts1_x = np.asarray(data[ts1_idx]['x'], dtype=float)
        ts1_y = np.asarray(data[ts1_idx]['y'], dtype=float)
        ts2_x = np.asarray(data[ts2_idx]['x'], dtype=float)
        ts2_y = np.asarray(data[ts2_idx]['y'], dtype=float)
    except (TypeError, ValueError):
        return _clear()

    ts1_valid = np.isfinite(ts1_x) & np.isfinite(ts1_y)
    ts2_valid = np.isfinite(ts2_x) & np.isfinite(ts2_y)
    if not ts1_valid.any() or not ts2_valid.any():
        return _clear()

    ts1_x = ts1_x[ts1_valid]
    ts1_y = ts1_y[ts1_valid]
    ts2_x = ts2_x[ts2_valid]
    ts2_y = ts2_y[ts2_valid]

    ts1_low_raw = center_x - half
    ts1_high_raw = center_x + half
    ts2_low_raw = center_y - half
    ts2_high_raw = center_y + half

    ts1_min, ts1_max = float(np.min(ts1_x)), float(np.max(ts1_x))
    ts2_min, ts2_max = float(np.min(ts2_y)), float(np.max(ts2_y))

    ts1_low = max(ts1_low_raw, ts1_min)
    ts1_high = min(ts1_high_raw, ts1_max)
    ts2_low = max(ts2_low_raw, ts2_min)
    ts2_high = min(ts2_high_raw, ts2_max)

    ts1_mask = (ts1_x >= ts1_low) & (ts1_x <= ts1_high)
    ts2_mask = (ts2_y >= ts2_low) & (ts2_y <= ts2_high)
    if not ts1_mask.any() and not ts2_mask.any():
        return _clear()

    patch['data'][ts1_high_idx]['x'] = ts1_x[ts1_mask].tolist()
    patch['data'][ts1_high_idx]['y'] = ts1_y[ts1_mask].tolist()
    patch['data'][ts2_high_idx]['x'] = ts2_x[ts2_mask].tolist()
    patch['data'][ts2_high_idx]['y'] = ts2_y[ts2_mask].tolist()

    ts1_y_min, ts1_y_max = float(np.min(ts1_y)), float(np.max(ts1_y))
    ts2_x_min, ts2_x_max = float(np.min(ts2_x)), float(np.max(ts2_x))

    left_shape = patch['layout']['shapes'][shape_indices['ts1-mask-left']]
    ts1_y_range = ts1_y_max - ts1_y_min
    ts2_x_range = ts2_x_max - ts2_x_min
    if ts1_low_raw > ts1_min:
        left_x0 = ts1_min
        left_x1 = min(ts1_low_raw, ts1_max)
        if left_x1 > left_x0:
            left_shape['visible'] = True
            left_shape['x0'] = left_x0 - 1
            left_shape['x1'] = left_x1
            left_shape['y0'] = ts1_y_min - ts1_y_range * 0.025
            left_shape['y1'] = ts1_y_max + ts1_y_range * 0.025
        else:
            left_shape['visible'] = False
    else:
        left_shape['visible'] = False

    right_shape = patch['layout']['shapes'][shape_indices['ts1-mask-right']]
    if ts1_high_raw < ts1_max:
        right_x0 = max(ts1_high_raw, ts1_min)
        right_x1 = ts1_max
        if right_x1 > right_x0:
            right_shape['visible'] = True
            right_shape['x0'] = right_x0
            right_shape['x1'] = right_x1 + 1 
            right_shape['y0'] = ts1_y_min - ts1_y_range * 0.025
            right_shape['y1'] = ts1_y_max + ts1_y_range * 0.025
        else:
            right_shape['visible'] = False
    else:
        right_shape['visible'] = False

    bottom_shape = patch['layout']['shapes'][shape_indices['ts2-mask-bottom']]
    if ts2_low_raw > ts2_min:
        bottom_y0 = ts2_min
        bottom_y1 = min(ts2_low_raw, ts2_max)
        if bottom_y1 > bottom_y0:
            bottom_shape['visible'] = True
            bottom_shape['x0'] = ts2_x_min
            bottom_shape['x1'] = ts2_x_max
            bottom_shape['y0'] = bottom_y0 - 1
            bottom_shape['y1'] = bottom_y1
        else:
            bottom_shape['visible'] = False
    else:
        bottom_shape['visible'] = False

    top_shape = patch['layout']['shapes'][shape_indices['ts2-mask-top']]
    if ts2_high_raw < ts2_max:
        top_y0 = max(ts2_high_raw, ts2_min)
        top_y1 = ts2_max
        if top_y1 > top_y0:
            top_shape['visible'] = True
            top_shape['x0'] = ts2_x_min
            top_shape['x1'] = ts2_x_max
            top_shape['y0'] = top_y0
            top_shape['y1'] = top_y1 + 1
        else:
            top_shape['visible'] = False
    else:
        top_shape['visible'] = False
    return patch


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8051)
