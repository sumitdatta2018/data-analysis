from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ---------------------------------------------------------
# Rainfall Forecasting Streamlit App
# Artifacts expected from the training notebook:
#   outputs/best_statistical_model.pkl
# or
#   outputs/best_lstm_model.keras
#   outputs/lstm_scaler_bundle.pkl
# Optional source data: cleaned_mumbai_monthly_rains.csv
# ---------------------------------------------------------

st.set_page_config(
    page_title="Rainfall Forecasting",
    page_icon="🌧️",
    layout="wide",
)

st.title("🌧️ Monthly Rainfall Forecasting")
st.caption("Forecast monthly rainfall using the model and preprocessing approach from the training notebook.")

OUTPUT_DIR = Path("outputs")
STAT_BUNDLE_PATH = OUTPUT_DIR / "best_statistical_model.pkl"
LSTM_MODEL_PATH = OUTPUT_DIR / "best_lstm_model.keras"
LSTM_SCALER_PATH = OUTPUT_DIR / "lstm_scaler_bundle.pkl"
DEFAULT_DATA_PATH = Path("cleaned_mumbai_monthly_rains.csv")
SEASON = 12

SARIMAX_OPTIONS = {
    "SARIMAX (1,0,1) × (1,0,1,12)": ((1, 0, 1), (1, 0, 1, 12), False),
    "SARIMAX (1,1,1) × (1,1,1,12)": ((1, 1, 1), (1, 1, 1, 12), False),
    "SARIMAX (0,1,1) × (0,1,1,12)": ((0, 1, 1), (0, 1, 1, 12), False),
    "SARIMAX log1p (1,1,1) × (1,1,1,12)": ((1, 1, 1), (1, 1, 1, 12), True),
}


def load_rainfall_csv(file):
    df = pd.read_csv(file)
    df.columns = df.columns.astype(str).str.strip()
    if not {"Date", "Rainfall"}.issubset(df.columns):
        raise ValueError("The CSV must contain columns named 'Date' and 'Rainfall'.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Rainfall"] = pd.to_numeric(df["Rainfall"], errors="coerce")
    df = df.dropna(subset=["Date", "Rainfall"]).sort_values("Date")
    if df["Date"].duplicated().any():
        raise ValueError("Duplicate dates were found. Please provide one row per month.")
    series = df.set_index("Date")["Rainfall"].astype(float).sort_index()
    series = series.asfreq("MS")
    if series.isna().any():
        raise ValueError(
            f"The monthly series has {int(series.isna().sum())} missing month(s). "
            "Please fill missing months before forecasting."
        )
    if (series < 0).any():
        raise ValueError("Rainfall values cannot be negative.")
    if len(series) < 24:
        raise ValueError("At least 24 months of rainfall history are required.")
    return series


def nonnegative(values):
    return np.maximum(np.asarray(values, dtype=float), 0.0)


def seasonal_naive(history, horizon):
    values = list(history.astype(float).values)
    result = []
    for _ in range(horizon):
        value = max(0.0, float(values[-SEASON]))
        result.append(value)
        values.append(value)
    return np.asarray(result)


def monthly_climatology(history, future_index, years=None):
    hist = history if years is None else history.iloc[-years * SEASON:]
    monthly_means = hist.groupby(hist.index.month).mean()
    return np.asarray([
        max(0.0, float(monthly_means.get(date.month, hist.mean())))
        for date in future_index
    ])


def forecast_ets(history, horizon, transformed=False):
    series = np.log1p(history) if transformed else history
    fitted = ExponentialSmoothing(
        series,
        trend=None,
        seasonal="add",
        seasonal_periods=SEASON,
        initialization_method="estimated",
    ).fit(optimized=True, use_brute=False)
    prediction = np.asarray(fitted.forecast(horizon), dtype=float)
    if transformed:
        prediction = np.expm1(prediction)
    return nonnegative(prediction)


def forecast_sarimax(history, horizon, params):
    order, seasonal_order, transformed = params
    series = np.log1p(history) if transformed else history
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=80)
    prediction = np.asarray(fitted.forecast(horizon), dtype=float)
    if transformed:
        prediction = np.expm1(prediction)
    return nonnegative(prediction)


@st.cache_resource(show_spinner=False)
def load_statistical_bundle(path_string, modified_time):
    return joblib.load(path_string)


@st.cache_resource(show_spinner=False)
def load_lstm_artifacts(model_path, scaler_path, model_mtime, scaler_mtime):
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    bundle = joblib.load(scaler_path)
    return model, bundle["scaler"], int(bundle.get("lookback", 24))


def build_forecast(history, model_name, horizon, stat_bundle=None):
    future_index = pd.date_range(
        history.index[-1] + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )

    if model_name == "Seasonal Naive":
        values = seasonal_naive(history, horizon)
    elif model_name == "Monthly Climatology (all history)":
        values = monthly_climatology(history, future_index)
    elif model_name == "Monthly Climatology (last 30y)":
        values = monthly_climatology(history, future_index, years=30)
    elif model_name == "ETS additive seasonality":
        values = forecast_ets(history, horizon, transformed=False)
    elif model_name == "ETS on log1p":
        values = forecast_ets(history, horizon, transformed=True)
    elif model_name in SARIMAX_OPTIONS:
        values = forecast_sarimax(history, horizon, SARIMAX_OPTIONS[model_name])
    elif stat_bundle and model_name == stat_bundle.get("model_name"):
        params = stat_bundle.get("params")
        if model_name.startswith("SARIMAX"):
            values = forecast_sarimax(history, horizon, params)
        elif model_name == "ETS additive seasonality":
            values = forecast_ets(history, horizon, transformed=False)
        elif model_name == "ETS on log1p":
            values = forecast_ets(history, horizon, transformed=True)
        elif model_name == "Seasonal Naive":
            values = seasonal_naive(history, horizon)
        elif model_name == "Monthly Climatology (all history)":
            values = monthly_climatology(history, future_index)
        elif model_name == "Monthly Climatology (last 30y)":
            values = monthly_climatology(history, future_index, years=30)
        else:
            raise ValueError(f"Unsupported saved model: {model_name}")
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return pd.DataFrame({"Date": future_index, "Predicted_Rainfall": values})


# ---------------------- Sidebar / data ----------------------
with st.sidebar:
    st.header("Forecast settings")
    uploaded_file = st.file_uploader(
        "Upload monthly rainfall CSV",
        type=["csv"],
        help="Required columns: Date and Rainfall. Dates should be monthly.",
    )

    st.divider()
    st.markdown("**Expected CSV format**")
    st.code("Date,Rainfall\n2000-01-01,12.5\n2000-02-01,18.2", language="csv")

# Load trained statistical bundle, if present.
stat_bundle = None
if STAT_BUNDLE_PATH.exists():
    try:
        stat_bundle = load_statistical_bundle(
            str(STAT_BUNDLE_PATH.resolve()), STAT_BUNDLE_PATH.stat().st_mtime
        )
    except Exception as exc:
        st.warning(f"Could not load the saved statistical model bundle: {exc}")

# Determine the history source. An uploaded CSV takes priority.
history = None
data_source = None
try:
    if uploaded_file is not None:
        history = load_rainfall_csv(uploaded_file)
        data_source = "Uploaded CSV"
    elif stat_bundle is not None and isinstance(stat_bundle.get("history_data"), pd.Series):
        history = stat_bundle["history_data"].copy()
        history.index = pd.to_datetime(history.index)
        history = history.astype(float).asfreq("MS")
        data_source = "Training history stored in the saved model bundle"
    elif DEFAULT_DATA_PATH.exists():
        history = load_rainfall_csv(DEFAULT_DATA_PATH)
        data_source = str(DEFAULT_DATA_PATH)
except Exception as exc:
    st.error(str(exc))
    st.stop()

if history is None:
    st.info(
        "No rainfall history or trained model bundle was found. Upload your "
        "`cleaned_mumbai_monthly_rains.csv` file, or place it beside app.py. "
        "For the automatically selected trained model, also include the notebook's "
        "`outputs/` artifacts."
    )
    st.stop()

# ---------------------- Model selection ----------------------
available_models = [
    "Seasonal Naive",
    "Monthly Climatology (all history)",
    "Monthly Climatology (last 30y)",
    "ETS additive seasonality",
    "ETS on log1p",
    *SARIMAX_OPTIONS.keys(),
]

saved_model_name = stat_bundle.get("model_name") if stat_bundle else None
if saved_model_name and saved_model_name not in available_models:
    available_models.append(saved_model_name)

lstm_available = LSTM_MODEL_PATH.exists() and LSTM_SCALER_PATH.exists()
if lstm_available:
    available_models.append("LSTM (saved 12-month model)")

default_model = saved_model_name if saved_model_name in available_models else available_models[0]

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("Historical months", f"{len(history):,}")
with col_b:
    st.metric("History starts", history.index.min().strftime("%b %Y"))
with col_c:
    st.metric("Latest observation", history.index.max().strftime("%b %Y"))

st.caption(f"Data source: {data_source}")

left, right = st.columns([2, 1])
with left:
    selected_model = st.selectbox(
        "Forecasting model",
        available_models,
        index=available_models.index(default_model),
    )
with right:
    if selected_model == "LSTM (saved 12-month model)":
        horizon = 12
        st.number_input("Forecast horizon (months)", min_value=12, max_value=12, value=12, disabled=True)
        st.caption("The saved LSTM was trained to predict 12 months at a time.")
    else:
        horizon = st.number_input(
            "Forecast horizon (months)", min_value=1, max_value=60, value=12, step=1
        )

if selected_model == "LSTM (saved 12-month model)":
    st.caption("LSTM forecast uses the saved Keras model and MinMaxScaler from training.")

run_forecast = st.button("Generate forecast", type="primary", use_container_width=False)

# ---------------------- Historical chart ----------------------
st.subheader("Historical rainfall")
history_frame = history.rename("Rainfall").reset_index()
history_frame.columns = ["Date", "Rainfall"]
history_fig = go.Figure()
history_fig.add_trace(go.Scatter(
    x=history_frame["Date"], y=history_frame["Rainfall"],
    mode="lines", name="Observed rainfall"
))
history_fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Rainfall (dataset units)",
    height=360,
    margin=dict(l=10, r=10, t=25, b=10),
)
st.plotly_chart(history_fig, use_container_width=True)

if run_forecast:
    try:
        with st.spinner("Generating rainfall forecast..."):
            if selected_model == "LSTM (saved 12-month model)":
                model, scaler, lookback = load_lstm_artifacts(
                    str(LSTM_MODEL_PATH.resolve()),
                    str(LSTM_SCALER_PATH.resolve()),
                    LSTM_MODEL_PATH.stat().st_mtime,
                    LSTM_SCALER_PATH.stat().st_mtime,
                )
                if len(history) < lookback:
                    raise ValueError(f"LSTM requires at least {lookback} months of history.")
                scaled = scaler.transform(history.values.reshape(-1, 1))
                last_window = scaled[-lookback:].reshape(1, lookback, 1)
                predicted_scaled = model.predict(last_window, verbose=0).reshape(-1, 1)
                predicted = nonnegative(scaler.inverse_transform(predicted_scaled).ravel())
                future_dates = pd.date_range(
                    history.index[-1] + pd.offsets.MonthBegin(1),
                    periods=12,
                    freq="MS",
                )
                forecast_df = pd.DataFrame({
                    "Date": future_dates,
                    "Predicted_Rainfall": predicted,
                })
            else:
                forecast_df = build_forecast(
                    history, selected_model, int(horizon), stat_bundle
                )

        st.subheader("Forecast results")
        total_forecast = float(forecast_df["Predicted_Rainfall"].sum())
        average_forecast = float(forecast_df["Predicted_Rainfall"].mean())
        peak_row = forecast_df.loc[forecast_df["Predicted_Rainfall"].idxmax()]

        m1, m2, m3 = st.columns(3)
        m1.metric("Forecast total", f"{total_forecast:,.2f}")
        m2.metric("Monthly average", f"{average_forecast:,.2f}")
        m3.metric("Highest forecast month", peak_row["Date"].strftime("%b %Y"))

        combined_fig = go.Figure()
        recent_history = history.iloc[-min(len(history), 60):]
        combined_fig.add_trace(go.Scatter(
            x=recent_history.index, y=recent_history.values,
            mode="lines+markers", name="Historical rainfall"
        ))
        combined_fig.add_trace(go.Scatter(
            x=forecast_df["Date"], y=forecast_df["Predicted_Rainfall"],
            mode="lines+markers", name="Forecast",
            line=dict(dash="dash")
        ))
        combined_fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Rainfall (dataset units)",
            height=430,
            hovermode="x unified",
            margin=dict(l=10, r=10, t=25, b=10),
        )
        st.plotly_chart(combined_fig, use_container_width=True)

        display_df = forecast_df.copy()
        display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m")
        display_df["Predicted_Rainfall"] = display_df["Predicted_Rainfall"].round(3)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download forecast CSV",
            data=csv_bytes,
            file_name="rainfall_forecast.csv",
            mime="text/csv",
        )
    except Exception as exc:
        st.error(f"Forecast failed: {type(exc).__name__}: {exc}")

with st.expander("Deployment notes"):
    st.markdown(
        """
        **Keep these files together:**
        ```text
        your-project/
        ├── app.py
        ├── cleaned_mumbai_monthly_rains.csv   # optional if bundle contains history
        └── outputs/
            ├── best_statistical_model.pkl     # generated by the notebook, OR
            ├── best_lstm_model.keras          # generated if LSTM wins
            └── lstm_scaler_bundle.pkl         # generated with the LSTM model
        ```

        Install dependencies and start the app:
        ```bash
        pip install streamlit pandas numpy joblib plotly statsmodels scikit-learn
        streamlit run app.py
        ```

        If deploying the LSTM artifact, also install TensorFlow:
        ```bash
        pip install tensorflow
        ```

        The notebook's saved statistical bundle contains the selected 12-month model
        name, its parameters, and its training history. The app refits that statistical
        model on the selected history before forecasting. Uploading a CSV replaces the
        stored training history for the forecast.
        """
    )
