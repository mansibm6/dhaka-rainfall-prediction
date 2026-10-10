"""Reusable experiment functions extracted from Weather Prediction.ipynb."""

from __future__ import annotations


def plot_time_series(timesteps, values, format=".", start=0, end=None, label=None):
    """
    Plots a timesteps (a series of points in time) against values (a series of values across timesteps).

    Parameters
    ---------
    timesteps : array of timesteps
    values : array of values across time
    format : style of plot, default "."
    start : where to start the plot (setting a value will index from start of timesteps & values)
    end : where to end the plot (setting a value will index from end of timesteps & values)
    label : label to show on plot of values
    """
    import matplotlib.pyplot as plt

    plt.plot(timesteps[start:end], values[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Rainfall (mm)")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)


def mean_absolute_scaled_error(y_true, y_pred):
    """
    Implement MASE (assuming no seasonality of data).
    """
    import tensorflow as tf

    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1]))
    tf.debugging.assert_positive(
        mae_naive_no_season, message="MASE is undefined for a constant series"
    )
    return mae / mae_naive_no_season


def evaluate_preds(y_true, y_pred):
    import tensorflow as tf

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    mae = tf.reduce_mean(tf.abs(y_true - y_pred), axis=-1)
    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    rmse = tf.sqrt(mse)
    mape = 100.0 * tf.reduce_mean(
        tf.abs((y_true - y_pred) / tf.maximum(tf.abs(y_true), tf.keras.backend.epsilon())),
        axis=-1,
    )
    mase = mean_absolute_scaled_error(y_true, y_pred)
    return {
        "mae": mae.numpy(),
        "mse": mse.numpy(),
        "rmse": rmse.numpy(),
        "mape": mape.numpy(),
        "mase": mase.numpy(),
    }


def get_labelled_windows(x, horizon=1):
    """
    Creates labels for windowed dataset.

    E.g. if horizon=1 (default)
    Input: [1, 2, 3, 4, 5, 6] -> Output: ([1, 2, 3, 4, 5], [6])
    """
    return (x[:, :-horizon], x[:, -horizon:])


def make_windows(x, window_size=7, horizon=1):
    """
    Turns a 1D array into a 2D array of sequential windows of window_size.
    """
    import numpy as np

    x = np.asarray(x)
    if x.ndim != 1 or window_size < 1 or horizon < 1 or len(x) < window_size + horizon:
        raise ValueError("Expected a 1D series long enough for a positive window and horizon")
    window_step = np.expand_dims(np.arange(window_size + horizon), axis=0)
    window_indexes = (
        window_step + np.expand_dims(np.arange(len(x) - (window_size + horizon - 1)), axis=0).T
    )
    windowed_array = x[window_indexes]
    windows, labels = get_labelled_windows(windowed_array, horizon=horizon)
    return (windows, labels)


def make_train_test_splits(windows, labels, test_split=0.2):
    """
    Splits matching pairs of windows and labels into train and test splits.
    """
    if not 0 < test_split < 1 or len(windows) != len(labels):
        raise ValueError("Invalid split fraction or mismatched windows/labels")
    split_size = int(len(windows) * (1 - test_split))
    train_windows = windows[:split_size]
    train_labels = labels[:split_size]
    test_windows = windows[split_size:]
    test_labels = labels[split_size:]
    return (train_windows, test_windows, train_labels, test_labels)


def create_model_checkpoint(model_name, save_path="model_experiments"):
    import os
    import tensorflow as tf

    return tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(save_path, model_name + ".keras"),
        verbose=0,
        save_best_only=True,
    )


def make_preds(model, input_data):
    """
    Uses model to make predictions on input_data.

    Parameters
    ----------
    model: trained model
    input_data: windowed input data (same kind of data model was trained on)

    Returns model predictions on input_data.
    """
    import tensorflow as tf

    forecast = model.predict(input_data)
    return tf.squeeze(forecast)
