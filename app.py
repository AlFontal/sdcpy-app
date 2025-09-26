import io
import os
import dash
import tasks
import redis
import base64

import pandas as pd
import dash_ag_grid as dag
import dash_bootstrap_components as dbc

from rq import Queue
from rq.job import Job
from rq.exceptions import NoSuchJobError

from dash import html, dcc, no_update
from whitenoise import WhiteNoise
from dash.exceptions import PreventUpdate
from dash.dependencies import Output, Input, State, ALL

try:
    from pandas.core.tools.datetimes import _guess_datetime_format
except ImportError:  # pragma: no cover - fallback for pandas API changes
    _guess_datetime_format = None  # type: ignore

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
    'n_permutations': 100,
    'plot_width': 7.0,
    'plot_height': 7.0,
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
date_report_store = dcc.Store(id='date-report-store', data=None)


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


def _infer_datetime_format(sample: str | None) -> str | None:
    """Best-effort inference of a strptime-compatible format string for display."""

    if not sample or not isinstance(sample, str):
        return None

    if _guess_datetime_format is None:  # pandas API fallback
        return None

    for dayfirst in (False, True):
        try:
            fmt = _guess_datetime_format(sample, dayfirst=dayfirst, default=None)
        except TypeError:  # pragma: no cover - signature may differ across pandas
            fmt = _guess_datetime_format(sample)
        except Exception:
            fmt = None
        if fmt:
            return fmt
    return None


def sanitize_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {
        'dropped_columns': [],
        'date_columns': []
    }

    # Trim column names and drop duplicates keeping first occurrence
    df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]

    # Drop columns that are entirely empty
    empty_cols = df.columns[df.isna().all()].tolist()
    if empty_cols:
        report['dropped_columns'] = empty_cols
        df = df.drop(columns=empty_cols)

    # Detect datetime-like columns
    for col in df.columns:
        series = df[col]
        info = {
            'column': col,
            'status': 'unchanged',
            'valid_count': 0,
            'total_count': int(len(series)),
            'invalid_count': 0,
            'invalid_examples': [],
            'sample_value': None,
            'parsed_example': None,
            'sample_values': [],
            'parsed_examples': [],
            'suggested_format': None,
        }

        if pd.api.types.is_datetime64_any_dtype(series):
            info['status'] = 'datetime'
            info['valid_count'] = int(series.notna().sum())
            non_null = series.dropna()
            samples = non_null.head(5)
            if not samples.empty:
                info['sample_values'] = [str(val) for val in samples]
                parsed_samples = [val.strftime('%Y-%m-%d') if hasattr(val, 'strftime') else str(val) for val in samples]
                info['parsed_examples'] = parsed_samples
                info['sample_value'] = info['sample_values'][0]
                info['parsed_example'] = info['parsed_examples'][0]
            info['suggested_format'] = '%Y-%m-%d'
            report['date_columns'].append(info)
            continue

        if pd.api.types.is_string_dtype(series) or series.dtype == object:
            parsed = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
            valid_count = int(parsed.notna().sum())
            if valid_count == 0:
                continue

            ratio = valid_count / len(parsed)
            info['valid_count'] = valid_count
            invalid_mask = parsed.isna() & series.notna()
            invalid_rows = invalid_mask[invalid_mask].index.tolist()
            info['invalid_count'] = len(invalid_rows)
            if info['invalid_count']:
                info['invalid_examples'] = [int(i) + 1 for i in invalid_rows[:5]]

            parsed_non_null = parsed.dropna()
            sample_original = series[parsed.notna()].iloc[0] if valid_count else None
            parsed_example = parsed_non_null.iloc[0] if not parsed_non_null.empty else None
            info['sample_value'] = str(sample_original) if sample_original is not None else None
            if parsed_example is not None:
                info['parsed_example'] = parsed_example.strftime('%Y-%m-%d')
                info['suggested_format'] = _infer_datetime_format(info['sample_value']) or ''

            if not parsed_non_null.empty:
                aligned = series.loc[parsed_non_null.index].head(5)
                parsed_samples = parsed_non_null.head(5)
                info['sample_values'] = [str(val) for val in aligned]
                info['parsed_examples'] = [val.strftime('%Y-%m-%d') for val in parsed_samples]
                if info['sample_values']:
                    info['sample_value'] = info['sample_values'][0]
                if info['parsed_examples']:
                    info['parsed_example'] = info['parsed_examples'][0]

            if ratio >= 0.5:
                df[col] = parsed
                info['status'] = 'converted'
            else:
                info['status'] = 'skipped'

            report['date_columns'].append(info)

    return df, report


def build_sanitization_alert(report: dict):
    messages = []

    dropped = report.get('dropped_columns') or []
    if dropped:
        messages.append(html.Li(f"Dropped empty columns: {', '.join(dropped)}"))

    for info in report.get('date_columns', []):
        col = info['column']
        status = info['status']
        valid = info['valid_count']
        total = info['total_count']
        invalid = info['invalid_count']
        examples = info.get('invalid_examples') or []

        if status == 'converted':
            msg = f"Column `{col}` parsed as datetime ({valid}/{total} values)."
            if invalid:
                sample = ', '.join(map(str, examples))
                remaining = invalid - len(examples)
                extra = f" (+{remaining} more)" if remaining > 0 else ''
                msg += f" {invalid} value(s) could not be parsed{extra}. Example rows: {sample}."
        elif status == 'skipped':
            msg = f"Column `{col}` left as text ({valid}/{total} values resembled dates)."
            if invalid:
                sample = ', '.join(map(str, examples))
                msg += f" Example rows treated as non-dates: {sample}."
        elif status == 'datetime':
            msg = f"Column `{col}` detected as datetime."
        else:
            continue

        messages.append(html.Li(msg))

    if not messages:
        return []

    alert_body = [
        html.Strong('Import notes'),
        html.Ul(messages, className='mb-0')
    ]

    return dbc.Alert(alert_body, color='info', className='mt-3', dismissable=True)


def build_date_review_body(report: dict | None,
                           overrides: dict[str, str] | None = None,
                           errors: dict[str, str] | None = None) -> list:
    """Render modal content for reviewing inferred date formats."""

    overrides = overrides or {}
    errors = errors or {}

    if not report:
        return [html.P('No datetime-like columns detected.', className='mb-0')]

    relevant = [info for info in report.get('date_columns', [])
                if info.get('status') in {'converted', 'datetime'}]

    if not relevant:
        return [html.P('No datetime-like columns detected.', className='mb-0')]

    rows: list = [
        html.P(
            'Review the detected date columns. Adjust the format below if the interpretation looks wrong. '
            'Use Python strftime directives (e.g. %d/%m/%Y).',
            className='text-muted'
        )
    ]

    if errors:
        error_messages = [html.Li(f"{col}: {message}") for col, message in errors.items()]
        rows.append(
            dbc.Alert([
                html.Strong('Some overrides could not be applied.'),
                html.Ul(error_messages, className='mb-0')
            ], color='danger')
        )

    for info in relevant:
        col = info['column']
        suggested_format = overrides.get(col) or info.get('suggested_format') or ''
        sample_pairs = list(zip(info.get('sample_values') or [], info.get('parsed_examples') or []))

        if sample_pairs:
            samples_list = html.Ul(
                [
                    html.Li([
                        html.Code(original if original else '—'),
                        html.Span(' → ', className='mx-1'),
                        html.Code(parsed if parsed else '—')
                    ])
                    for original, parsed in sample_pairs
                ],
                className='mb-0 date-sample-list'
            )
        else:
            samples_list = html.P('No parsed examples available yet.', className='text-muted mb-0')

        rows.append(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Strong(col),
                            html.Br(),
                            html.Small('First 5 parsed examples', className='text-muted'),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        samples_list,
                        width=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Input(
                                id={'type': 'date-format-input', 'column': col},
                                placeholder='e.g. %Y-%m-%d',
                                value=suggested_format,
                                debounce=True,
                            ),
                            html.Small('Leave blank to keep the inferred format.', className='text-muted'),
                        ],
                        width=4,
                    ),
                ],
                className='mb-3 align-items-center date-review-row'
            )
        )

    return rows
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
            dbc.Col([html.Label('Date Column'), date_dropdown], className='parameter-col'),
            dbc.Col([html.Label('Time Series 1'), ts1_dropdown], className='parameter-col'),
            dbc.Col([html.Label('Time Series 2'), ts2_dropdown], className='parameter-col'),
            dbc.Col([html.Label('Corr. Method'), method_dropdown], className='parameter-col'),
        ], className='parameter-row'),
        dbc.Row([
            dbc.Col([html.Label('Min Lag'),
                     dbc.Input(id='min-lag', placeholder='Default: -inf', type='number')], className='parameter-col'),
            dbc.Col([html.Label('Max Lag'),
                     dbc.Input(id='max-lag', placeholder='Default: inf', type='number')], className='parameter-col'),
            dbc.Col([html.Label('Window Size (s)'),
                     dbc.Input(id='window', placeholder='Select window size', type='number', min=0)], className='parameter-col'),
            dbc.Col([
                html.Label('Permutations (n)'),
                dbc.Input(
                    id='n-permutations',
                    type='number',
                    min=0,
                    step=1,
                    value=DEFAULT_PARAMS['n_permutations'],
                ),
            ], className='parameter-col', width=3),
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
            dbc.Col([html.Label('Plot Width (in)'),
                     dbc.Input(id='plot-width', type='number', min=1, step=0.5,
                               value=DEFAULT_PARAMS['plot_width'])],
                    className='parameter-col', width=3),
            dbc.Col([html.Label('Plot Height (in)'),
                     dbc.Input(id='plot-height', type='number', min=1, step=0.5,
                               value=DEFAULT_PARAMS['plot_height'])],
                    className='parameter-col', width=3),
            dbc.Col([html.Label('Plot Min Lag'),
                     dbc.Input(id='plot-min-lag', type='number', placeholder='Use analysis min lag')],
                    className='parameter-col', width=3),
            dbc.Col([html.Label('Plot Max Lag'),
                     dbc.Input(id='plot-max-lag', type='number', placeholder='Use analysis max lag')],
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

date_review_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle('Review detected date formats')), 
        dbc.ModalBody(id='date-review-body'),
        dbc.ModalFooter([
            dbc.Button('Cancel', id='dismiss-date-review', color='secondary', className='me-2'),
            dbc.Button('Confirm', id='confirm-date-review', color='primary'),
        ]),
    ],
    id='date-review-modal',
    is_open=False,
    centered=True,
    size='lg',
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
            original_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            sanitized_df, report = sanitize_dataframe(original_df.copy())
            return sanitized_df, None, report, original_df
        return None, dbc.Alert('Unsupported file type. Please upload a .csv file.', color='warning'), None, None

    except Exception as exc:
        print(exc)
        return None, dbc.Alert('There was an error processing this file. Please upload a valid .csv file.',
                                color='danger'), None, None


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
               Output('output-data-upload', 'children'),
               Output('date-report-store', 'data')],
             Input('upload-data', 'contents'),
             State('upload-data', 'filename'),
             prevent_initial_call=True)
def update_output(content, filename):
    if content is None:
        raise PreventUpdate

    sanitized_df, error, report, original_df = parse_contents(content, filename)
    if error is not None:
        return (
            no_update,  # raw data
            no_update,  # memory data
            no_update,  # grid rows
            no_update,  # grid columns
            no_update,  # upload preview text
            True,       # parameters card hidden
            True,       # plot card hidden
            True,       # run button container hidden
            True,       # divider hidden
            True,       # progress container hidden
            True,       # results container hidden
            True,       # run button disabled
            True,       # update plot disabled
            None,       # plot feedback cleared
            'Upload Dataset',
            'Dataset Preview',
            error,
            None,      # reset date report
        )

    grid_df = sanitized_df.copy()
    for col in grid_df.columns:
        if pd.api.types.is_datetime64_any_dtype(grid_df[col]):
            grid_df[col] = pd.to_datetime(grid_df[col]).dt.tz_localize(None).dt.strftime('%Y-%m-%d %H:%M:%S')

    row_data = grid_df.fillna('').to_dict('records')
    column_defs = build_column_defs(sanitized_df)

    serialized_df = sanitized_df.copy()
    for col in serialized_df.columns:
        if pd.api.types.is_datetime64_any_dtype(serialized_df[col]):
            serialized_df[col] = serialized_df[col].dt.tz_localize(None).dt.strftime('%Y-%m-%d %H:%M:%S')
    serialized_df = serialized_df.where(serialized_df.notna(), '')
    serialized = serialized_df.to_dict('list')

    original_serialized = original_df.where(original_df.notna(), None).to_dict('list')

    sanitization_alert = build_sanitization_alert(report)

    return (original_serialized,
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
            True,   # update plot button disabled until results exist
            None,   # clear feedback message
            f'Uploaded dataset: {filename}',
            f'Dataset Preview – {len(sanitized_df):,} rows × {len(sanitized_df.columns):,} columns',
            sanitization_alert,
            report,
            )


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
    Output('date-review-modal', 'is_open'),
    Output('date-review-body', 'children'),
    Output('data-memory-store', 'data', allow_duplicate=True),
    Output('data-grid', 'rowData', allow_duplicate=True),
    Input('date-report-store', 'data'),
    Input('dismiss-date-review', 'n_clicks'),
    Input('confirm-date-review', 'n_clicks'),
    State('date-review-modal', 'is_open'),
    State('raw-data-store', 'data'),
    State('data-memory-store', 'data'),
    State({'type': 'date-format-input', 'column': ALL}, 'value'),
    State({'type': 'date-format-input', 'column': ALL}, 'id'),
    prevent_initial_call=True,
)
def manage_date_review_modal(report_data, dismiss_clicks, confirm_clicks, is_open, raw_data,
                             sanitized_data, override_values, override_ids):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    report_data = report_data or {}

    def as_map(ids, values):
        overrides = {}
        for ctrl_id, value in zip(ids or [], values or []):
            if not isinstance(ctrl_id, dict):
                continue
            column = ctrl_id.get('column')
            if not column:
                continue
            if value is None:
                continue
            value_str = value.strip()
            if value_str:
                overrides[column] = value_str
        return overrides

    overrides_map = as_map(override_ids, override_values)

    if trigger == 'date-report-store':
        date_infos = report_data.get('date_columns') or []
        has_relevant = any(info.get('status') in {'converted', 'datetime'} for info in date_infos)
        body_children = build_date_review_body(report_data)
        return has_relevant, body_children, dash.no_update, dash.no_update

    if trigger == 'dismiss-date-review':
        return False, dash.no_update, dash.no_update, dash.no_update

    if trigger == 'confirm-date-review':
        if raw_data is None or sanitized_data is None:
            return False, dash.no_update, dash.no_update, dash.no_update

        raw_df = pd.DataFrame(raw_data)
        sanitized_df = pd.DataFrame(sanitized_data)

        if raw_df.empty or not overrides_map:
            return False, dash.no_update, dash.no_update, dash.no_update

        errors: dict[str, str] = {}

        for column, fmt in overrides_map.items():
            if column not in raw_df.columns:
                errors[column] = 'Column not found in the uploaded data.'
                continue
            try:
                parsed = pd.to_datetime(raw_df[column], format=fmt, errors='coerce')
            except Exception as exc:  # pragma: no cover - defensive guard
                errors[column] = f'Failed to apply format ({exc}).'
                continue

            non_na = parsed.notna()
            if non_na.sum() == 0 and raw_df[column].notna().sum() > 0:
                errors[column] = 'Format does not match any sample values.'
                continue

            formatted = parsed.dt.strftime('%Y-%m-%d %H:%M:%S')
            sanitized_df[column] = formatted.where(non_na, '')

        if errors:
            body_children = build_date_review_body(report_data, overrides_map, errors)
            return True, body_children, dash.no_update, dash.no_update

        sanitized_serialized = sanitized_df.where(sanitized_df.notna(), '').to_dict('list')

        display_df = sanitized_df.copy()
        for col in display_df.columns:
            if pd.api.types.is_datetime64_any_dtype(display_df[col]):
                display_df[col] = pd.to_datetime(display_df[col]).dt.tz_localize(None).dt.strftime('%Y-%m-%d %H:%M:%S')

        row_data = display_df.fillna('').to_dict('records')

        return False, dash.no_update, sanitized_serialized, row_data

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
    State('n-permutations', 'value'),
    State('plot-title', 'value'),
    State('label-fontsize', 'value'),
    State('plot-dpi', 'value'),
    State('plot-width', 'value'),
    State('plot-height', 'value'),
    State('plot-min-lag', 'value'),
    State('plot-max-lag', 'value'),
    State('plot-options', 'value'),
    State('data-memory-store', 'data'),
    prevent_initial_call=True,
)
def enqueue_sdc_job(n_clicks, ts1, ts2, date, method, min_lag, max_lag, window,
                    n_permutations, plot_title, label_fontsize, plot_dpi,
                    plot_width, plot_height, plot_min_lag, plot_max_lag,
                    plot_options, data):
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

    try:
        n_permutations = int(n_permutations) if n_permutations not in (None, '') else DEFAULT_PARAMS['n_permutations']
    except (TypeError, ValueError):
        n_permutations = DEFAULT_PARAMS['n_permutations']
    if n_permutations < 0:
        n_permutations = DEFAULT_PARAMS['n_permutations']

    def _coerce_positive_float(value, default):
        try:
            if value in (None, ''):
                return default
            value = float(value)
        except (TypeError, ValueError):
            return default
        return value if value > 0 else default

    plot_width = _coerce_positive_float(plot_width, DEFAULT_PARAMS['plot_width'])
    plot_height = _coerce_positive_float(plot_height, DEFAULT_PARAMS['plot_height'])

    def _coerce_optional_int(value):
        if value in (None, ''):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    plot_min_lag = _coerce_optional_int(plot_min_lag)
    plot_max_lag = _coerce_optional_int(plot_max_lag)

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
            n_permutations,
            plot_title,
            label_fontsize,
            plot_dpi,
            show_colorbar,
            show_ts2,
            plot_width,
            plot_height,
            plot_min_lag,
            plot_max_lag,
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
    State('plot-width', 'value'),
    State('plot-height', 'value'),
    State('plot-min-lag', 'value'),
    State('plot-max-lag', 'value'),
    State('plot-options', 'value'),
    prevent_initial_call=True,
)
def on_update_plot(n_clicks, data, plot_title, label_fontsize, plot_dpi,
                   plot_width, plot_height, plot_min_lag, plot_max_lag,
                   plot_options):
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

    plot_settings = (data or {}).get('plot_settings', {})

    def _coerce_positive_float(value, default):
        try:
            if value in (None, ''):
                return default
            value = float(value)
        except (TypeError, ValueError):
            return default
        return value if value > 0 else default

    width_default = plot_settings.get('width', DEFAULT_PARAMS['plot_width'])
    height_default = plot_settings.get('height', DEFAULT_PARAMS['plot_height'])
    plot_width = _coerce_positive_float(plot_width, width_default)
    plot_height = _coerce_positive_float(plot_height, height_default)

    def _coerce_optional_int(value, fallback=None):
        if value in (None, ''):
            return fallback
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    plot_min_lag = _coerce_optional_int(plot_min_lag, plot_settings.get('min_lag'))
    plot_max_lag = _coerce_optional_int(plot_max_lag, plot_settings.get('max_lag'))

    plot_title_clean = plot_title.strip() if plot_title else None

    new_image = tasks.render_sdc_plot_from_payload(
        data['analysis'],
        plot_title=plot_title_clean,
        labels_fontsize=label_fontsize,
        show_colorbar=show_colorbar,
        show_ts2=show_ts2,
        plot_dpi=plot_dpi,
        figsize=(plot_width, plot_height),
        min_lag_override=plot_min_lag,
        max_lag_override=plot_max_lag,
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
                                date_report_store,
                                results_store,
                                job_store,
                                job_interval,
                                plot_modal,
                                date_review_modal,
                                mobile_sidebar,
                                html.Div(id='theme-sync', style={'display': 'none'})
                                ])


if __name__ == '__main__':
    # Get port from environment variable, default to 8050
    port = int(os.environ.get('PORT', 8050))
    # Bind to 0.0.0.0 to allow external connections in Docker
    app.run(host='0.0.0.0', port=port, debug=True)
