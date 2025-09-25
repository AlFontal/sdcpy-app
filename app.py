import io
import os
import base64
import dash

import pandas as pd
import dash_bootstrap_components as dbc
import redis
from rq import Queue
from rq.job import Job
from rq.exceptions import NoSuchJobError

from dash import html, dcc, no_update
import dash_ag_grid as dag
import tasks
from whitenoise import WhiteNoise
from dash.exceptions import PreventUpdate
from dash.dependencies import Output, Input, State

app = dash.Dash(
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://fonts.googleapis.com/css2?family=Play:wght@400;700&display=swap',
        'https://cdn.jsdelivr.net/npm/ag-grid-community/styles/ag-grid.css',
        'https://cdn.jsdelivr.net/npm/ag-grid-community/styles/ag-theme-alpine.css',
    ],
    suppress_callback_exceptions=True
)

app.title = 'SDCpy app'
server = app.server
server.wsgi_app = WhiteNoise(server.wsgi_app, root='static/')

REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
QUEUE_NAME = os.environ.get('RQ_QUEUE', 'sdcpy')
redis_connection = redis.from_url(REDIS_URL)
job_queue = Queue(QUEUE_NAME, connection=redis_connection)
JOB_TIMEOUT = int(os.environ.get('SDC_JOB_TIMEOUT', 60 * 15))
JOB_RESULT_TTL = int(os.environ.get('SDC_JOB_TTL', 3600))

GITHUB_LOGO = 'github_logo_white.png'
SDCPY_LOGO = 'sdcpy_logo_white.png'
SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '10rem',
    'padding': '2rem 1rem',
    'background': 'linear-gradient(180deg, #1F2A44 0%, #27324D 100%)',
    'color': 'white',
    'boxShadow': '2px 0 12px rgba(15, 23, 42, 0.2)'
}

CONTENT_STYLE = {
    'margin-left': '13rem',
    'margin-right': '2rem',
    'padding': '2.5rem 2rem 3rem',
    'minHeight': '100vh',
}

DEFAULT_PARAMS = {
    'method': 'pearson',
    'labels_fontsize': 6,
    'plot_dpi': 300,
}

sidebar_content = html.Div(
    [
        html.A(
            html.Div(
                html.Img(src=SDCPY_LOGO, className='sidebar-logo'),
                className='sidebar-logo-wrapper'
            ),
            href='https://github.com/AlFontal/sdcpy',
        ),
        html.Hr(className='sidebar-divider'),
        html.A(
            html.Div(
                [
                    html.Span('Source', className='sidebar-link-text'),
                    html.Img(src=GITHUB_LOGO, className='sidebar-icon'),
                ],
                className='sidebar-link'
            ),
            href='https://github.com/AlFontal/sdcpy-app',
            className='sidebar-link-container'
        ),
    ], style=SIDEBAR_STYLE,
)

sidebar = html.Div(sidebar_content, className='desktop-sidebar')

mobile_sidebar = dbc.Offcanvas(
    sidebar_content,
    id='sidebar-offcanvas',
    title='SDCpy',
    placement='start',
    backdrop=True,
    scrollable=True,
    style={'background': SIDEBAR_STYLE['background'], 'color': 'white'}
)

raw_data_store = dcc.Store(id='raw-data-store', data=None)
data_memory_store = dcc.Store(id='data-memory-store')
results_store = dcc.Store(id='results-store', data=None)
job_store = dcc.Store(id='job-store', data=None)
job_interval = dcc.Interval(id='job-interval', interval=.5 * 1000, disabled=True)
theme_store = dcc.Store(id='theme-store', data='light')


def build_progress_content(progress=None):
    description = 'Preparing analysis…'
    if progress:
        description = progress.get('description') or description

    components = [dcc.Markdown(f'**{description}**')]

    if not progress or not progress.get('total'):
        components.append(dbc.Spinner(color='primary', size='sm'))
    else:
        total = progress['total'] or 1
        current = progress.get('current', 0)
        percent = min(100, max(0, int((current / total) * 100)))
        components.append(
            dbc.Progress(value=percent, label=f'{percent}%', striped=True, animated=True, className='my-2')
        )
        components.append(dcc.Markdown(f'{current} / {total} iterations'))

    return components


def build_column_defs(df: pd.DataFrame):
    column_defs = []
    for col in df.columns:
        col_def = {
            'headerName': col,
            'field': col,
        }
        series = df[col]

        if pd.api.types.is_numeric_dtype(series):
            col_def.update({
                'cellClass': 'ag-right-aligned-cell',
            })
        elif pd.api.types.is_datetime64_any_dtype(series):
            col_def.setdefault('valueFormatter', {
                'function': "function(params){return params.value ? new Date(params.value).toLocaleString() : '';}"
            })

        column_defs.append(col_def)

    return column_defs


title_row = html.H2('Scale dependent correlation analysis App', className='page-title')

sidebar_toggle_button = html.Div([
    dbc.Button(
        '☰ Menu',
        id='sidebar-toggle',
        color='light',
        className='sidebar-toggle-button d-lg-none',
        n_clicks=0
    ),
    dbc.Button(
        id='theme-toggle',
        color='light',
        className='theme-toggle-button d-sm-inline-flex',
        n_clicks=0,
        children='🌙 Dark',
    )
], className='top-controls')

instructions_row = html.P(
    'Start by uploading a .csv file containing at least a numerical column and a date column with headers.',
    className='page-subtitle'
)

ts1_dropdown = dcc.Dropdown(options=[{'label': 'Add dataset to select', 'value': 'Empty'}],
                            placeholder='Select time series 1', id='ts1-dropdown')
ts2_dropdown = dcc.Dropdown(options=[{'label': 'Add dataset to select', 'value': 'Empty'}],
                            placeholder='Select time series 2', id='ts2-dropdown')
date_dropdown = dcc.Dropdown(options=[{'label': 'Add dataset to select', 'value': 'Empty'}],
                             placeholder='Select date column', id='date-dropdown')
method_dropdown = dcc.Dropdown(options=[
    {'label': 'Pearson', 'value': 'pearson'},
    {'label': 'Spearman', 'value': 'spearman'},
    ],
    value=DEFAULT_PARAMS['method'],
    id='method-dropdown')

parameter_card = dbc.Card(
    dbc.CardBody([
        html.H4('SDC Parameters', className='section-title'),
        dbc.Row([
            dbc.Col([html.Label('Time Series 1'), ts1_dropdown], className='parameter-col'),
            dbc.Col([html.Label('Time Series 2'), ts2_dropdown], className='parameter-col'),
            dbc.Col([html.Label('Date Column'), date_dropdown], className='parameter-col'),
        ], className='parameter-row'),
        dbc.Row([
            dbc.Col([html.Label('Corr. Method'), method_dropdown], className='parameter-col'),
            dbc.Col([html.Label('Min Lag'),
                     dbc.Input(id='min-lag', placeholder='Default: -inf', type='number')], className='parameter-col'),
            dbc.Col([html.Label('Max Lag'),
                     dbc.Input(id='max-lag', placeholder='Default: inf', type='number')], className='parameter-col'),
            dbc.Col([html.Label('Window Size (s)'),
                     dbc.Input(id='window', placeholder='Select window size', type='number', min=0)], className='parameter-col'),
        ], className='parameter-row'),
    ]),
    className='parameters-card'
)

plot_options_card = dbc.Card(
    dbc.CardBody([
        html.H4('Plot Styling', className='section-title'),
        dbc.Row([
            dbc.Col([html.Label('Plot Title'),
                     dbc.Input(id='plot-title', placeholder='Optional custom title', type='text')],
                    className='parameter-col', width=6),
            dbc.Col([html.Label('Label Font Size'),
                     dbc.Input(id='label-fontsize', type='number', min=5, value=6)],
                    className='parameter-col', width=3),
            dbc.Col([html.Label('Plot DPI'),
                     dbc.Input(id='plot-dpi', type='number', min=72, value=300)],
                    className='parameter-col', width=3),
        ], className='parameter-row'),
        dbc.Row([
            dbc.Col([
                html.Label('Display Options'),
                dbc.Checklist(
                    id='plot-options',
                    options=[
                        {'label': 'Colorbar', 'value': 'colorbar'},
                        {'label': 'Time Series 2', 'value': 'ts2'},
                    ],
                    value=['colorbar', 'ts2'],
                    switch=True,
                    inline=True,
                    className='parameter-switch'
                ),
            ], className='parameter-col', width=12),
        ], className='parameter-row'),
        html.Div(
            [
                dbc.Button('Update Plot', id='update-plot-button', color='secondary', outline=True,
                           className='update-plot-button', n_clicks=0, disabled=True),
                html.Div(id='plot-feedback', className='plot-feedback')
            ],
            className='plot-actions'
        ),
    ]),
    className='parameters-card'
)

file_upload = dbc.Card(
    [
        dbc.CardHeader(
            dbc.Button(
                [
                    html.Span('Upload Dataset', id='upload-card-title', className='card-header-text'),
                    html.Span('▴', id='upload-card-icon', className='card-toggle-icon')
                ],
                id='toggle-upload-card',
                color='link',
                className='card-toggle-button',
                n_clicks=0
            )
        ),
        dbc.Collapse(
            dbc.CardBody([
                html.P('Drag and drop or click to upload a CSV file.', className='card-subtitle'),
                dcc.Upload(
                    id='upload-data',
                    children=html.Div('Upload dataset (.csv)', className='upload-box'),
                    className='upload-container'
                ),
                html.Div(id='output-data-upload')
            ]),
            id='upload-collapse',
            is_open=True
        )
    ],
    className='upload-card collapsible-card'
)

table_preview = dbc.Card(
    [
        dbc.CardHeader(
            dbc.Button(
                [
                    html.Span('Dataset Preview', id='preview-card-title', className='card-header-text'),
                    html.Span('▴', id='preview-card-icon', className='card-toggle-icon')
                ],
                id='toggle-preview-card',
                color='link',
                className='card-toggle-button',
                n_clicks=0
            )
        ),
        dbc.Collapse(
            dbc.CardBody([
                dag.AgGrid(
                    id='data-grid',
                    columnDefs=[],
                    rowData=[],
                    defaultColDef={
                        "sortable": False,
                        "resizable": True,
                        "flex": 1,
                        "minWidth": 140,
                    },
                    dashGridOptions={
                        "pagination": True,
                        "paginationPageSize": 10,
                        "animateRows": False,
                        "rowHeight": 38,
                        "suppressCellSelection": True,
                        "domLayout": "autoHeight",
                    },
                    className='ag-theme-alpine compact-ag-theme',
                )
            ]),
            id='preview-collapse',
            is_open=True
        )
    ],
    className='table-card collapsible-card'
)
progress_div = html.Div(id='progress-div', className='card-section')
results_div = html.Div(id='results-div', className='card-section')

plot_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle('SDC Plot Preview'), close_button=True),
        dbc.ModalBody(html.Img(id='modal-plot-img', style={'width': '100%', 'height': 'auto'})),
        dbc.ModalFooter(dbc.Button('Close', id='close-plot-modal', color='secondary')),
    ],
    id='plot-modal',
    size='xl',
    centered=True,
    is_open=False,
)

run_button = dbc.Button('Run SDC Analysis',
                        disabled=True,
                        color='primary',
                        size='lg',
                        className='run-button',
                        id='run-button')


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if filename and filename.lower().endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            return df.to_dict(orient='list'), None
        return None, dbc.Alert('Unsupported file type. Please upload a .csv file.', color='warning')

    except Exception as exc:
        print(exc)
        return None, dbc.Alert('There was an error processing this file. Please upload a valid .csv file.',
                                color='danger')


@app.callback([Output('raw-data-store', 'data'),
               Output('data-memory-store', 'data'),
               Output('data-grid', 'rowData'),
               Output('data-grid', 'columnDefs'),
               Output('upload-data', 'children'),
               Output('parameters-card-container', 'hidden'),
               Output('plot-options-card-container', 'hidden'),
               Output('run-button-container', 'hidden'),
               Output('results-divider', 'hidden'),
               Output('progress-container', 'hidden'),
               Output('results-container', 'hidden'),
               Output('run-button', 'disabled', allow_duplicate=True),
               Output('update-plot-button', 'disabled', allow_duplicate=True),
               Output('plot-feedback', 'children', allow_duplicate=True),
               Output('upload-card-title', 'children'),
               Output('preview-card-title', 'children'),
               Output('output-data-upload', 'children')],
             Input('upload-data', 'contents'),
             State('upload-data', 'filename'),
             prevent_initial_call=True)
def update_output(content, filename):
    if content is not None:
        data, error = parse_contents(content, filename)
        if error is not None:
            return (no_update, no_update, no_update, no_update,
                    no_update,
                    True,  # parameters card hidden
                    True,  # plot card hidden
                    True,  # run button container hidden
                    True,  # divider hidden
                    True,  # progress container hidden
                    True,  # results container hidden
                    True,  # run button disabled
                    True,  # update plot button disabled
                    None,  # clear feedback
                    'Upload Dataset',
                    'Dataset Preview',
                    error)
        df = pd.DataFrame(data)
        grid_df = df.copy()
        for col in grid_df.columns:
            if pd.api.types.is_datetime64_any_dtype(grid_df[col]):
                grid_df[col] = pd.to_datetime(grid_df[col]).dt.tz_localize(None).dt.strftime('%Y-%m-%d %H:%M:%S')

        row_data = grid_df.fillna('').to_dict('records')
        column_defs = build_column_defs(df)
        serialized = df.to_dict('list')
        return (serialized,
                serialized,
                row_data,
                column_defs,
                html.P(filename, className='upload-filename'),
                False,  # parameters card visible
                False,  # plot card visible
                False,  # run button container visible
                True,   # divider hidden until run starts
                True,   # progress container hidden
                True,   # results container hidden
                False,  # run button enabled
                True,   # keep update plot disabled until results exist
                None,   # clear feedback message
                f'Uploaded dataset: {filename}',
                f'Dataset Preview – {len(df):,} rows × {len(df.columns):,} columns',
                None)
    else:
        raise PreventUpdate


@app.callback([Output('ts1-dropdown', 'options'),
               Output('ts2-dropdown', 'options'),
               Output('date-dropdown', 'options')],
              Input('data-memory-store', 'data'))
def update_series(data):
    if data is not None:
        df = pd.DataFrame(data)
        options = [{'value': col, 'label': col} for col in df.columns]
        return options, options, options
    else:
        raise PreventUpdate


@app.callback(
    Output('run-button', 'disabled', allow_duplicate=True),
    Input('data-memory-store', 'data'),
    prevent_initial_call=True,
)
def enable_run_button(data):
    if data is None:
        raise PreventUpdate
    return False


@app.callback(
    Output('upload-collapse', 'is_open'),
    Output('upload-card-icon', 'children'),
    Input('toggle-upload-card', 'n_clicks'),
    Input('raw-data-store', 'data'),
    State('upload-collapse', 'is_open'),
    prevent_initial_call=True,
)
def toggle_upload_collapse(toggle_clicks, data, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger == 'raw-data-store':
        new_state = False if data is not None else is_open
    else:
        new_state = not is_open

    icon = '▴' if new_state else '▾'
    return new_state, icon


@app.callback(
    Output('preview-collapse', 'is_open'),
    Output('preview-card-icon', 'children'),
    Input('toggle-preview-card', 'n_clicks'),
    State('preview-collapse', 'is_open'),
    prevent_initial_call=True,
)
def toggle_preview_collapse(n_clicks, is_open):
    if not n_clicks:
        raise PreventUpdate
    new_state = not is_open
    icon = '▴' if new_state else '▾'
    return new_state, icon


@app.callback(
    Output('sidebar-offcanvas', 'is_open'),
    Input('sidebar-toggle', 'n_clicks'),
    State('sidebar-offcanvas', 'is_open'),
    prevent_initial_call=True,
)
def toggle_mobile_sidebar(n_clicks, is_open):
    if not n_clicks:
        raise PreventUpdate
    return not is_open


@app.callback(
    Output('theme-store', 'data'),
    Output('theme-toggle', 'children'),
    Input('theme-toggle', 'n_clicks'),
    State('theme-store', 'data'),
    prevent_initial_call=True,
)
def toggle_theme(n_clicks, current):
    if not n_clicks:
        raise PreventUpdate
    new_theme = 'dark' if current == 'light' else 'light'
    button_label = '☀ Light' if new_theme == 'dark' else '🌙 Dark'
    return new_theme, button_label


app.clientside_callback(
    """
    function(theme){
        if(!theme){return ''}
        document.body.classList.remove('light-mode','dark-mode');
        document.body.classList.add(theme + '-mode');
        return '';
    }
    """,
    Output('theme-sync', 'children'),
    Input('theme-store', 'data')
)


@app.callback(
    Output('run-button', 'children', allow_duplicate=True),
    Input('raw-data-store', 'data'),
    prevent_initial_call=True,
)
def set_run_button_label(data):
    if data is None:
        raise PreventUpdate
    return 'Run SDC Analysis'


@app.callback(
    Output('job-store', 'data'),
    Input('run-button', 'n_clicks'),
    State('ts1-dropdown', 'value'),
    State('ts2-dropdown', 'value'),
    State('date-dropdown', 'value'),
    State('method-dropdown', 'value'),
    State('min-lag', 'value'),
    State('max-lag', 'value'),
    State('window', 'value'),
    State('plot-title', 'value'),
    State('label-fontsize', 'value'),
    State('plot-dpi', 'value'),
    State('plot-options', 'value'),
    State('data-memory-store', 'data'),
    prevent_initial_call=True,
)
def enqueue_sdc_job(n_clicks, ts1, ts2, date, method, min_lag, max_lag, window,
                    plot_title, label_fontsize, plot_dpi, plot_options, data):
    if not n_clicks:
        raise PreventUpdate

    if data is None:
        return {'status': 'error', 'message': 'Please upload a dataset before running the analysis.'}

    required = {
        'Time Series 1': ts1,
        'Time Series 2': ts2,
        'Date Column': date,
        'Correlation Method': method,
        'Window Size': window,
    }

    missing = [label for label, value in required.items() if value is None]
    if missing:
        return {
            'status': 'error',
            'message': f"Please provide values for: {', '.join(missing)}."
        }

    plot_title = plot_title.strip() if plot_title else None
    try:
        label_fontsize = int(label_fontsize) if label_fontsize is not None else 12
    except (TypeError, ValueError):
        label_fontsize = 10
    try:
        plot_dpi = int(plot_dpi) if plot_dpi is not None else 300
    except (TypeError, ValueError):
        plot_dpi = 300
    plot_options = plot_options or []
    show_colorbar = 'colorbar' in plot_options
    show_ts2 = 'ts2' in plot_options

    job = job_queue.enqueue(
        'tasks.run_sdc_analysis',
        args=(
            data,
            ts1,
            ts2,
            date,
            method,
            min_lag,
            max_lag,
            window,
            plot_title,
            label_fontsize,
            plot_dpi,
            show_colorbar,
            show_ts2,
        ),
        job_timeout=JOB_TIMEOUT,
        result_ttl=JOB_RESULT_TTL,
    )

    return {'status': 'queued', 'id': job.id, 'message': 'Job queued'}


@app.callback(
    Output('progress-div', 'children'),
    Output('progress-div', 'hidden'),
    Output('job-interval', 'disabled'),
    Output('results-div', 'children'),
    Output('results-store', 'data'),
    Output('run-button', 'disabled', allow_duplicate=True),
    Output('run-button', 'children', allow_duplicate=True),
    Output('update-plot-button', 'disabled'),
    Output('results-divider', 'hidden', allow_duplicate=True),
    Output('progress-container', 'hidden', allow_duplicate=True),
    Output('results-container', 'hidden', allow_duplicate=True),
    Input('job-store', 'data'),
    Input('job-interval', 'n_intervals'),
    prevent_initial_call=True,
)
def manage_job(job_data, _):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    if trigger == 'job-store':
        if not job_data:
            raise PreventUpdate

        if job_data.get('status') == 'error':
            alert = dbc.Alert(job_data['message'], color='danger')
            return [alert], False, True, no_update, no_update, False, dash.no_update, True, True, True, True

        if job_data.get('status') == 'queued':
            progress_children = build_progress_content({'description': job_data.get('message'), 'current': 0, 'total': None})
            return progress_children, False, False, no_update, no_update, True, dash.no_update, True, False, False, True

        raise PreventUpdate

    if not job_data or 'id' not in job_data:
        raise PreventUpdate

    try:
        job = Job.fetch(job_data['id'], connection=redis_connection)
    except NoSuchJobError:
        alert = dbc.Alert('The analysis job could not be found. Please try again.', color='danger')
        return [alert], False, True, no_update, None, False, dash.no_update, True, True, True, True

    if job.is_failed:
        alert = dbc.Alert('The analysis failed. Check the logs and try again.', color='danger')
        return [alert], False, True, no_update, None, False, dash.no_update, True, True, True, True

    progress = job.meta.get('progress', {})

    if job.is_finished:
        result = job.result
        if not result:
            alert = dbc.Alert('No result was returned by the analysis.', color='warning')
            return [alert], False, True, no_update, None, False, dash.no_update, True, True, True, True

        image_div = html.Img(
            src=f"data:image/png;base64,{result['image']}",
            id='sdc-results-img',
            style={'cursor': 'zoom-in'}
        )
        download_button = dbc.Button('Download Results Table', id='download-button', color='secondary')

        results_children = [
            dbc.Row([dbc.Col(dcc.Markdown('### SDC Analysis Results')),
            dbc.Col(html.Div([
                download_button,
                dcc.Download(id='download-results-xlsx')
            ], className='results-actions'))], justify='between'),
            html.Small('Click the plot to open a large preview.', className='plot-hint'),
            image_div,
        ]

        total = progress.get('total') or 1
        success_progress = build_progress_content({'description': 'Completed', 'current': total, 'total': total})

        return success_progress, True, True, results_children, result, False, 'Re-run SDC Analysis', False, False, True, False

    running_progress = build_progress_content(progress)
    return running_progress, False, False, no_update, no_update, True, dash.no_update, True, False, False, True
@app.callback(
    Output("download-results-xlsx", "data"),
    Input("download-button", "n_clicks"),
    Input("results-store", "data"),
    prevent_initial_call=True,
)
def on_download_click(n_clicks, data):
    if n_clicks and data and data.get('excel'):
        file_bytes = base64.b64decode(data['excel'])

        def write_bytes(f):
            f.write(file_bytes)

        return dcc.send_bytes(write_bytes, 'sdc_analysis.xlsx')

    raise PreventUpdate


@app.callback(
    Output('sdc-results-img', 'src'),
    Output('plot-feedback', 'children'),
    Input('update-plot-button', 'n_clicks'),
    State('results-store', 'data'),
    State('plot-title', 'value'),
    State('label-fontsize', 'value'),
    State('plot-dpi', 'value'),
    State('plot-options', 'value'),
    prevent_initial_call=True,
)
def on_update_plot(n_clicks, data, plot_title, label_fontsize, plot_dpi, plot_options):
    if not n_clicks:
        raise PreventUpdate

    if not data or 'analysis' not in data:
        return no_update, no_update

    plot_options = plot_options or []
    show_colorbar = 'colorbar' in plot_options
    show_ts2 = 'ts2' in plot_options

    try:
        label_fontsize = int(label_fontsize) if label_fontsize is not None else DEFAULT_PARAMS['labels_fontsize']
    except (TypeError, ValueError):
        label_fontsize = DEFAULT_PARAMS['labels_fontsize']

    try:
        plot_dpi = int(plot_dpi) if plot_dpi is not None else DEFAULT_PARAMS['plot_dpi']
    except (TypeError, ValueError):
        plot_dpi = DEFAULT_PARAMS['plot_dpi']

    plot_title_clean = plot_title.strip() if plot_title else None

    new_image = tasks.render_sdc_plot_from_payload(
        data['analysis'],
        plot_title=plot_title_clean,
        labels_fontsize=label_fontsize,
        show_colorbar=show_colorbar,
        show_ts2=show_ts2,
        plot_dpi=plot_dpi,
    )

    feedback = html.Span('Plot updated with the latest styling options.', className='plot-feedback-text')

    return f"data:image/png;base64,{new_image}", feedback


@app.callback(
    Output('plot-modal', 'is_open'),
    Output('modal-plot-img', 'src'),
    Input('sdc-results-img', 'n_clicks'),
    Input('close-plot-modal', 'n_clicks'),
    State('plot-modal', 'is_open'),
    State('sdc-results-img', 'src'),
    prevent_initial_call=True,
)
def toggle_plot_modal(open_clicks, close_clicks, is_open, image_src):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger == 'sdc-results-img' and image_src:
        return True, image_src

    if trigger == 'close-plot-modal':
        return False, no_update

    raise PreventUpdate


content_div = html.Div([title_row,
                        html.Div(className='title-underline'),
                        sidebar_toggle_button,
                        instructions_row,
                        file_upload,
                        table_preview,
                        html.Div(parameter_card, hidden=True, id='parameters-card-container'),
                        html.Div(run_button, className='run-button-container', hidden=True, id='run-button-container'),
                        html.Div(progress_div, hidden=True, id='progress-container'),
                        html.Div(plot_options_card, hidden=True, id='plot-options-card-container'),
                        html.Div(className='section-divider', hidden=True, id='results-divider'),
                        html.Div(results_div, hidden=True, id='results-container')],
                       style=CONTENT_STYLE,
                       className='content-wrapper')

app.layout = html.Div(children=[
                                theme_store,
                                sidebar,
                                content_div,
                                raw_data_store,
                                data_memory_store,
                                results_store,
                                job_store,
                                job_interval,
                                plot_modal,
                                mobile_sidebar,
                                html.Div(id='theme-sync', style={'display': 'none'})
                                ])


if __name__ == '__main__':
    # Get port from environment variable, default to 8050
    port = int(os.environ.get('PORT', 8050))
    # Bind to 0.0.0.0 to allow external connections in Docker
    app.run(host='0.0.0.0', port=port, debug=True)
