# SDCpy App

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Dash](https://img.shields.io/badge/Dash-2.x-119DFF.svg)](https://dash.plotly.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-0db7ed.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-BSD-lightgrey.svg)](./LICENSE)

<img src="https://raw.githubusercontent.com/AlFontal/sdcpy-app/master/static/sdcpy_logo_black.png" width="200" height="250" />

Interactive Dash application for exploring and running [SDCpy](https://github.com/AlFontal/sdcpy)
Scale Dependent Correlation (SDC) analyses through a friendly web interface.

## Highlights
- **Modern UI** : upload data, preview the first rows in an Alpine-themed AgGrid table, and
  configure parameters in dedicated cards.
- **Background execution** : long-running SDC jobs are dispatched to a Redis/RQ worker so the UI stays responsive.
- **Customisable output** : tweak plot titles, DPI, label sizes, and toggle elements before (or after) running;
  regenerate the visual instantly via the `Update Plot` button.
- **Result handling** : view the generated SDC plot in-app, open it in a lightbox, and download the Excel output.

## Getting Started (Docker Compose)
The recommended way to run the app locally is with Docker Compose.

### Prerequisites
- [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/install/) installed

### Launch the stack
```bash
docker compose up --build
```
This starts three services declared in `docker-compose.yml`:

- `redis` : message broker for queued SDC jobs
- `worker` : RQ worker that executes the SDC analysis
- `sdcpy-app` : Dash web application (served on port `8050`)

Once the build finishes, open <http://localhost:8050> in your browser.

### Stopping the stack
```bash
docker compose down
```
Add `-v` if you also want to remove Redis volumes.

## Manual Python Execution (optional)
If you prefer running outside Docker you need Python ≥ 3.11 and Redis available locally. Then:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
redis-server  # in another terminal or via a service manager
python worker.py  # start the RQ worker in a separate shell
python app.py
```
The Dash server will be available at <http://localhost:8050> and the worker must stay running for queued jobs.

## Typical Workflow
1. **Upload CSV** : Drag-and-drop a CSV file. After upload the card collapses to show the filename.
2. **Preview data** : Expand “Dataset Preview” to inspect the first 10 rows (pagination available).
3. **Set parameters** : Choose time series columns, date column, correlations, lags, and window size.
4. **Style output** : Adjust titles, label font size, DPI, and optional plot elements.
5. **Run SDC Analysis** : The job moves to the background worker; progress is shown in the UI.
6. **Inspect & download** : View the resulting plot, open it in the modal for a closer look, download the Excel export.
7. **Update plot styling** : Change plot options and press “Update Plot” to regenerate the visual without rerunning SDC.

## Project Structure
- `app.py` : Dash application with queues, callbacks, and UI components
- `tasks.py` : SDC analysis worker tasks and plotting helpers
- `worker.py` : RQ worker entrypoint
- `docker-compose.yml` : local orchestration for app + worker + Redis
- `Dockerfile` : container image definition (Python 3.11 base)
- `assets/` : static assets, including custom CSS

## License
Distributed under the BSD license (see `pyproject.toml`).

Enjoy exploring SDC analysis through the browser, and feel free to extend the app or open issues/PRs in this repository.

Several upgrades are planned. Stay tuned!
