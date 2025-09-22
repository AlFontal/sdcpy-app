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
import tasks
from whitenoise import WhiteNoise
from dash.exceptions import PreventUpdate
from dash.dependencies import Output, Input, State

app = dash.Dash(
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://fonts.googleapis.com/css2?family=Play:wght@400;700&display=swap',
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

sidebar = html.Div(
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

data_memory_store = dcc.Store(id='data-memory-store')
results_store = dcc.Store(id='results-store', data=None)
job_store = dcc.Store(id='job-store', data=None)
job_interval = dcc.Interval(id='job-interval', interval=.5 * 1000, disabled=True)


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


title_row = html.H2('Scale dependent correlation analysis App', className='page-title')

instructions_row = html.P(
    'Start by uploading a .csv file containing at the very least a single column with a column header '
    'and numerical values.',
    className='page-subtitle'
)

ts1_dropdown = dcc.Dropdown(options=[{'label': 'Add dataset to select', 'value': 'Empty'}],
                            placeholder='Select time series 1', id='ts1-dropdown')
ts2_dropdown = dcc.Dropdown(options=[{'label': 'Add dataset to select', 'value': 'Empty'}],
                            placeholder='Select time series 2', id='ts2-dropdown')
date_dropdown = dcc.Dropdown(options=[{'label': 'Add dataset to select', 'value': 'Empty'}],
                             placeholder='Select date column', id='date-dropdown')
method_dropdown = dcc.Dropdown(options=[{'label': 'Spearman', 'value': 'spearman'},
                                        {'label': 'Pearson', 'value': 'pearson'}],
                               placeholder='Select correlation method', id='method-dropdown')

parameter_rows = html.Div(
    [
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
        dbc.Row([
            dbc.Col([html.Label('Plot Title'),
                     dbc.Input(id='plot-title', placeholder='Optional custom title', type='text')],
                    className='parameter-col', width=4),
            dbc.Col([html.Label('Label Font Size'),
                     dbc.Input(id='label-fontsize', type='number', min=5, value=10)],
                    className='parameter-col', width=2),
            dbc.Col([html.Label('Plot DPI'),
                     dbc.Input(id='plot-dpi', type='number', min=72, value=300)],
                    className='parameter-col', width=2),
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
            ], className='parameter-col', width=4),
        ], className='parameter-row'),
    ], hidden=True, id='parameters-div', className='parameters-card')

file_upload = dbc.Card(
    [
        dbc.CardBody([
            html.H4('Upload Dataset', className='card-title'),
            html.P('Drag and drop or click to upload a CSV file.', className='card-subtitle'),
            dcc.Upload(
                id='upload-data',
                children=html.Div('Upload dataset (.csv)', className='upload-box'),
                className='upload-container'
            ),
            html.Div(id='output-data-upload')
        ])
    ],
    className='upload-card'
)
progress_div = html.Div(id='progress-div', className='card-section')
results_div = html.Div(id='results-div', className='card-section')

run_button = html.Div(dbc.Button('Run SDC Analysis',
                                 disabled=False,
                                 color='primary',
                                 size='lg',
                                 className='run-button',
                                 id='run-button'),
                      id='run-button-div',
                      hidden=True)


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


@app.callback([Output('data-memory-store', 'data'),
               Output('upload-data', 'children'),
               Output('parameters-div', 'hidden'),
               Output('run-button-div', 'hidden'),
               Output('output-data-upload', 'children')],
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'))
def update_output(content, filename):
    if content is not None:
        data, error = parse_contents(content, filename)
        if error is not None:
            return no_update, no_update, True, True, error
        return data, html.P(filename, className='upload-filename'), False, False, []
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
    Output('run-button', 'disabled'),
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
            return [alert], False, True, no_update, no_update, False

        if job_data.get('status') == 'queued':
            progress_children = build_progress_content({'description': job_data.get('message'), 'current': 0, 'total': None})
            return progress_children, False, False, no_update, no_update, True

        raise PreventUpdate

    if not job_data or 'id' not in job_data:
        raise PreventUpdate

    try:
        job = Job.fetch(job_data['id'], connection=redis_connection)
    except NoSuchJobError:
        alert = dbc.Alert('The analysis job could not be found. Please try again.', color='danger')
        return [alert], False, True, no_update, None, False

    if job.is_failed:
        alert = dbc.Alert('The analysis failed. Check the logs and try again.', color='danger')
        return [alert], False, True, no_update, None, False

    progress = job.meta.get('progress', {})

    if job.is_finished:
        result = job.result
        if not result:
            alert = dbc.Alert('No result was returned by the analysis.', color='warning')
            return [alert], False, True, no_update, None, False

        image_div = html.Img(src=f"data:image/png;base64,{result['image']}", id='sdc-results-img')
        download_button = dbc.Button('Download Results Table', id='download-button', color='secondary')
        update_button = dbc.Button('Update Plot', id='update-plot-button', color='secondary', outline=True,
                                   className='update-plot-button', n_clicks=0)

        results_children = [
            dcc.Markdown('### SDC Analysis Results'),
            html.Div([
                update_button,
                download_button,
                dcc.Download(id='download-results-xlsx')
            ], className='results-actions'),
            html.Div(id='plot-feedback', className='plot-feedback'),
            image_div,
        ]

        total = progress.get('total') or 1
        success_progress = build_progress_content({'description': 'Completed', 'current': total, 'total': total})

        return success_progress, True, True, results_children, result, False

    running_progress = build_progress_content(progress)
    return running_progress, False, False, no_update, no_update, True
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
        alert = dbc.Alert('Please run the SDC analysis before updating the plot.', color='warning', dismissible=True)
        return no_update, alert

    plot_options = plot_options or []
    show_colorbar = 'colorbar' in plot_options
    show_ts2 = 'ts2' in plot_options

    try:
        label_fontsize = int(label_fontsize) if label_fontsize is not None else 12
    except (TypeError, ValueError):
        label_fontsize = 12

    try:
        plot_dpi = int(plot_dpi) if plot_dpi is not None else 300
    except (TypeError, ValueError):
        plot_dpi = 300

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


content_div = html.Div([title_row,
                        html.Div(className='title-underline'),
                        instructions_row,
                        file_upload,
                        parameter_rows,
                        html.Br(),
                        run_button,
                        html.Div(className='section-divider'),
                        progress_div,
                        results_div],
                       style=CONTENT_STYLE,
                       className='content-wrapper')

app.layout = html.Div(children=[sidebar,
                                content_div,
                                data_memory_store,
                                results_store,
                                job_store,
                                job_interval])


if __name__ == '__main__':
    # Get port from environment variable, default to 8050
    port = int(os.environ.get('PORT', 8050))
    # Bind to 0.0.0.0 to allow external connections in Docker
    app.run(host='0.0.0.0', port=port, debug=True)
