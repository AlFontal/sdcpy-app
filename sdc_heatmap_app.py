"""
Lightweight Dash app to preview an SDC-like layout from a results table.

- Reads `data/oni_sdc_results_example.xlsx` (example data included).
- 4 sheets: results grid, p-values grid, time series, config
- Renders a central heatmap (start_1 on x, start_2 on y, r as color).
- Reserves top and right panels as line-plot placeholders for future wiring.

Run locally (outside Docker):
    python sdc_heatmap_app.py

It serves on http://0.0.0.0:8051
"""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from dash import Dash, dcc, html, Patch
from plotly.subplots import make_subplots
from dash.dependencies import Input, Output, State

FILEPATH = 'data/oni_sdc_results_example.xlsx'



def build_figure(grid: pd.DataFrame,
                pvals: pd.DataFrame | None = None,
                ts_df: pd.DataFrame | None = None,
                fragment_size: int = 0,
                method: str = 'pearson',
                alpha: float = 0.05,
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
    if pvals is not None:
        p = pvals.copy()

        # Match orientation
        # p = p.transpose()
        p.index = p.index + offset
        p.columns = [int(c) + offset for c in p.columns]
        # Align
        p = p.reindex(index=z.index, columns=z.columns)
        z = z.where(p < alpha)


    # Central heatmap
    fig.add_trace(
        go.Heatmap(
            z=z.values,
            x=z.columns,
            y=z.index,
            colorscale='RdBu_r',
            zmin=-1, zmax=1, zmid=0,
            colorbar=dict(
                title="Pearson's r",
                lenmode='fraction',
                len=0.6,
                outlinewidth=2,
    outlinecolor='black',

                          
                          ),
            hovertemplate='start_1=%{x}<br>start_2=%{y}<br>r=%{z:.3f}<br>p={p:.3f}<extra></extra>',
        )
        ,
        row=2, col=2,
    )

    # Top: ts1 if present (start_1 vs ts1), else faint placeholder
    if ts_df is not None and {'start_1', 'ts1'}.issubset(ts_df.columns):
        ts1 = ts_df.dropna(subset=['start_1', 'ts1'])
        fig.add_trace(
            go.Scatter(x=ts1['start_1'], y=ts1['ts1'], mode='lines',
                       line=dict(color='black', width=2), name='ts1', showlegend=False),
            row=1, col=2,
        )

    # Left: ts2 if present (start_2 vs ts2 as vertical line plot)
    if ts_df is not None and {'start_2', 'ts2'}.issubset(ts_df.columns):
        ts2 = ts_df.dropna(subset=['start_2', 'ts2'])
        fig.add_trace(
            go.Scatter(x=ts2['ts2'], y=ts2['start_2'], mode='lines',
                       line=dict(color='black', width=2), name='ts2', showlegend=False),
            row=2, col=1,
        )

    # Right-most: per-row max correlation over start_2
    if len(z.index):
        row_max = np.nanmax(z.abs().values, axis=1)
        fig.add_trace(
            go.Scatter(x=row_max, y=z.index, mode='lines',
                       line=dict(color='#A81529', width=2), name='max r rows', showlegend=False),
            row=2, col=3,
        )

    # Bottom: per-column max correlation over start_1
    if len(z.columns):
        col_max = np.nanmax(z.values, axis=0)
        fig.add_trace(
            go.Scatter(x=z.columns, y=col_max, mode='lines',
                       line=dict(color='#A81529', width=2), name='max r cols', showlegend=False),
            row=3, col=2,
        )

    # Axis + layout styling
    fig.update_yaxes(row=2, col=1, autorange='reversed')
    fig.update_xaxes(title='', showgrid=False, row=1, col=2)
    fig.update_yaxes(title='', showgrid=False, row=1, col=2)

    fig.update_xaxes(title='', showgrid=False, row=2, col=2)
    fig.update_yaxes(title='', showgrid=False, autorange='reversed', row=2, col=2)

    fig.update_xaxes(title='Max r', autorange='reversed', row=2, col=3)
    fig.update_yaxes(title='', showgrid=False, row=2, col=3)

    fig.update_xaxes(title='', showgrid=False, row=2, col=1)
    fig.update_yaxes(title='', showgrid=False, row=2, col=1)

    fig.update_xaxes(title='', showgrid=False, row=3, col=2)
    fig.update_yaxes(title='Max r', showgrid=False, row=3, col=2)

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
                fig.update_xaxes(showline=True, linewidth=1.5, linecolor='black', 
                                mirror=True, row=row, col=col)
                fig.update_yaxes(showline=True, linewidth=1.5, linecolor='black', 
                                mirror=True, row=row, col=col)

    return fig


app = Dash(__name__)
app.title = 'SDC Heatmap Preview'

app.layout = html.Div(
    [
        html.H3('SDC Result Preview (heatmap + margins)'),
        html.Div([
            html.Span('Alpha: ', style={'marginRight': '8px'}),
            dcc.Input(id='alpha-input', type='number', value=0.05, min=0.0, max=1.0, step=0.0001,
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
    Input('sdc-heatmap', 'hoverData'),
    prevent_initial_call=False
)
def _load(alpha, hover_data):
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
    s = config['fragment_size'].values[0]
    method = config['method'].values[0]


    fig = build_figure(rs_grid, pvals=pv_grid, ts_df=ts_df, alpha=alpha_val,
                      fragment_size=s, method=method)
    return fig, f'Loaded grid {rs_grid.shape[0]}×{rs_grid.shape[1]}, alpha={alpha_val:.3f}', int(s)

@app.callback(
    Output('sdc-heatmap', 'figure', allow_duplicate=True),
    Input('sdc-heatmap', 'hoverData'),
    State('fragment-size-store', 'data'),
    prevent_initial_call=True
)
def update_hover_highlight(hover_data, fragment_size):
    patched_figure = Patch()
    
    # Clear existing shapes
    patched_figure['layout']['shapes'] = []
    
    if hover_data and 'points' in hover_data and fragment_size:
        point = hover_data['points'][0]
        if point.get('curveNumber') == 0:
            hover_start1 = point.get('x')
            hover_start2 = point.get('y')

def update_hover_highlight(hover_data, fragment_size):
    patched_figure = Patch()
    patched_figure['layout']['shapes'] = []
    
    if hover_data and 'points' in hover_data and fragment_size:
        point = hover_data['points'][0]
        
        if point.get('curveNumber') == 0:
            hover_start1 = point.get('x')
            hover_start2 = point.get('y')
            
            # Get axis ranges from current figure
            x_range = current_fig['layout']['xaxis2']['range'] if 'range' in current_fig['layout'].get('xaxis2', {}) else [0, 300]
            y_range = current_fig['layout']['yaxis4']['range'] if 'range' in current_fig['layout'].get('yaxis4', {}) else [0, 300]
            
            # Top plot (xaxis2, yaxis2)
            # Left occlusion
            patched_figure['layout']['shapes'].append({
                'type': 'rect',
                'xref': 'x2', 'yref': 'y2 domain',
                'x0': x_range[0],
                'x1': hover_start1 - fragment_size/2,
                'y0': 0, 'y1': 1,
                'fillcolor': 'rgba(255,255,255,0.7)',
                'line': {'width': 0},
                'layer': 'above'
            })
            # Right occlusion
            patched_figure['layout']['shapes'].append({
                'type': 'rect',
                'xref': 'x2', 'yref': 'y2 domain',
                'x0': hover_start1 + fragment_size/2,
                'x1': x_range[1],
                'y0': 0, 'y1': 1,
                'fillcolor': 'rgba(255,255,255,0.7)',
                'line': {'width': 0},
                'layer': 'above'
            })
            
            # Left plot (xaxis4, yaxis4)
            # Top occlusion
            patched_figure['layout']['shapes'].append({
                'type': 'rect',
                'xref': 'x4 domain', 'yref': 'y4',
                'x0': 0, 'x1': 1,
                'y0': y_range[0],
                'y1': hover_start2 - fragment_size/2,
                'fillcolor': 'rgba(255,255,255,0.7)',
                'line': {'width': 0},
                'layer': 'above'
            })
            # Bottom occlusion
            patched_figure['layout']['shapes'].append({
                'type': 'rect',
                'xref': 'x4 domain', 'yref': 'y4',
                'x0': 0, 'x1': 1,
                'y0': hover_start2 + fragment_size/2,
                'y1': y_range[1],
                'fillcolor': 'rgba(255,255,255,0.7)',
                'line': {'width': 0},
                'layer': 'above'
            })
    
    return patched_figure

if __name__ == '__main__':
    # host 0.0.0.0 so you can hit it from other devices if needed
    app.run_server(debug=True, host='0.0.0.0', port=8051)
