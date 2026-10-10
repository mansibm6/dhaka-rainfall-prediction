# Dhaka rainfall prediction

Filters monthly Bangladesh weather observations to Dhaka and compares a naive rainfall forecast with an LSTM using 11-step windows.

## Setup and execution

Use Python 3.11 and a separate virtual environment for this project. Run commands from this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python check_inputs.py
python -m jupyterlab
```

On Windows, activate with `.venv\Scripts\activate`. For notebooks, select this environment’s kernel and execute from the top in a fresh kernel. The first code cell exposes `INPUT_PATHS` where old machine-specific paths existed. `PROJECT_DATA_DIR` changes the base for those configured paths; remaining relative paths use the notebook working directory.

## Notebooks

- [Weather Prediction.ipynb](Weather%20Prediction.ipynb)

## Inputs

No fixed external dataset is required for the default entry point. See the notes below for bundled files or command-line input requirements.

## Outputs

dhaka_weather.csv, forecast plots, error metrics, and model_experiments/model_5_lstm.keras.

## Implementation and limitations

The source weather CSV is bundled. Required columns include YEAR, Month, Station Names, Rainfall and the ancillary columns selected in the notebook. Windows retain chronological ordering. MASE is undefined for a constant reference series and now raises an explicit error. Validation selects a checkpoint on the same period subsequently reported as test performance.

Reusable model/analysis functions live in `experiment_weather_prediction.py`. Notebooks retain the experiment sequence, training, and visual narrative.

## Verification

The workspace regression suite checks source/notebook validity and selected behavior with bundled or synthetic inputs. Full experiment training and historical metric reproduction are not implied. See the workspace `REFACTOR_REPORT.md` for exact test coverage and unresolved runtime requirements.
