"""Streamlit UI for the credit-card fraud detection project.

Run:
    streamlit run fraud_detection_ui.py

Upload the original ``creditcard.csv`` or ``creditcard.csv.zip`` dataset when
the page opens. The application trains the same LightGBM model used by the
project and applies isotonic probability calibration.
"""

from __future__ import annotations

import io
import zipfile

import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split


TARGET = "Class"
FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]


st.set_page_config(
    page_title="FraudGuard | Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {color-scheme: light;}
    .stApp {background: #f5f7fb; color: #172033;}
    .block-container {max-width: 1180px; padding-top: 1.4rem; padding-bottom: 3rem;}
    [data-testid="stHeader"] {background: rgba(245,247,251,.92);}
    [data-testid="stMain"] p,
    [data-testid="stMain"] label,
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3 {color: #172033;}
    .hero {
        padding: 1.55rem 1.8rem; border-radius: 18px; color: white;
        background: linear-gradient(120deg, #102a43, #155eaa 58%, #0ea5a8);
        box-shadow: 0 14px 34px rgba(21, 94, 170, .20); margin-bottom: 1rem;
    }
    .hero h1 {margin: 0; font-size: 2.05rem; color: white !important;}
    .hero p {margin: .4rem 0 0; opacity: .92; color: white !important;}
    .ready-card {
        background: #e8f7ef; color: #087443; border: 1px solid #bce8cf;
        border-left: 5px solid #16a36a; padding: .8rem 1rem; border-radius: 12px;
        font-weight: 650; margin: .2rem 0 1rem;
    }
    .result-safe, .result-risk {
        padding: 1.35rem; border-radius: 16px; color: white; text-align: left;
        box-shadow: 0 10px 28px rgba(15, 23, 42, .13);
    }
    .result-safe {background: linear-gradient(120deg, #087f5b, #16a36a);}
    .result-risk {background: linear-gradient(120deg, #a61b1b, #dc3d4b);}
    .result-safe h2, .result-risk h2 {margin: 0 0 .2rem; color: white !important;}
    .result-safe div, .result-risk div {color: white !important; opacity: .96;}
    div[data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #dfe5ed; padding: .85rem 1rem;
        border-radius: 14px; box-shadow: 0 4px 14px rgba(15, 23, 42, .05);
    }
    div[data-testid="stMetricLabel"] p {color: #5b6679 !important; font-weight: 600;}
    div[data-testid="stMetricValue"] {color: #102a43 !important;}
    div[data-testid="stTabs"] button p {color: #526076 !important; font-weight: 650;}
    div[data-testid="stTabs"] button[aria-selected="true"] p {color: #1d5fcb !important;}
    div[data-testid="stNumberInput"] input,
    div[data-baseweb="input"] input {background: #ffffff !important; color: #172033 !important;}
    div[data-testid="stRadio"] label p {color: #273449 !important;}
    .stButton > button {
        border: 0; border-radius: 11px; color: white; font-weight: 700;
        background: linear-gradient(90deg, #155eaa, #2563eb);
        min-height: 3rem; box-shadow: 0 7px 18px rgba(37,99,235,.20);
    }
    .stButton > button:hover {color: white; border: 0; transform: translateY(-1px);}
    [data-testid="stSidebar"] {background: #111827; border-right: 1px solid #253047;}
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {color: #f8fafc !important;}
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {color: #b9c3d4 !important;}
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: #0b1220; border: 1px dashed #53627a;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small {color: #cbd5e1 !important;}
    .label-pill {
        display: inline-block; padding: .42rem .75rem; border-radius: 999px;
        font-weight: 700; margin: .2rem 0 .8rem;
    }
    .label-fraud {background: #fee2e2; color: #a61b1b; border: 1px solid #fecaca;}
    .label-legit {background: #dcfce7; color: #087443; border: 1px solid #bbf7d0;}
    .helper {color: #64748b; font-size: .9rem; margin-top: -.35rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def read_dataset(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read the Kaggle CSV whether it is uploaded directly or as a ZIP."""
    if filename.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
            csv_files = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not csv_files:
                raise ValueError("The ZIP file does not contain a CSV file.")
            with archive.open(csv_files[0]) as csv_file:
                return pd.read_csv(csv_file)
    return pd.read_csv(io.BytesIO(file_bytes))


def ks_statistic(labels: pd.Series, probabilities: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, probabilities)
    return float(np.max(tpr - fpr))


@st.cache_resource(show_spinner=False)
def train_model(file_bytes: bytes, filename: str):
    """Train and cache the project model so reruns of the UI stay responsive."""
    frame = read_dataset(file_bytes, filename)
    missing = [column for column in [*FEATURES, TARGET] if column not in frame.columns]
    if missing:
        raise ValueError("Dataset columns are missing: " + ", ".join(missing))

    features = frame[FEATURES]
    labels = frame[TARGET].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_samples=50,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    valid_raw = model.predict_proba(x_valid)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(valid_raw, y_valid)

    test_raw = model.predict_proba(x_test)[:, 1]
    test_calibrated = calibrator.predict(test_raw)
    fpr, tpr, _ = roc_curve(y_test, test_raw)
    observed, predicted = calibration_curve(
        y_test, test_calibrated, n_bins=10, strategy="quantile"
    )
    metrics = {
        "auc": roc_auc_score(y_test, test_raw),
        "gini": 2 * roc_auc_score(y_test, test_raw) - 1,
        "ks": ks_statistic(y_test, test_raw),
        "pr_auc": average_precision_score(y_test, test_raw),
        "brier": brier_score_loss(y_test, test_calibrated),
        "fraud_rate": labels.mean(),
    }
    chart_data = {
        "roc": pd.DataFrame({"False positive rate": fpr, "True positive rate": tpr}),
        "calibration": pd.DataFrame(
            {"Predicted probability": predicted, "Observed fraud rate": observed}
        ),
    }
    return model, calibrator, frame, metrics, chart_data


def predict_transaction(model, calibrator, transaction: pd.DataFrame):
    raw_probability = float(model.predict_proba(transaction)[:, 1][0])
    calibrated_probability = float(calibrator.predict([raw_probability])[0])
    contributions = model.predict(transaction, pred_contrib=True)[0][:-1]
    drivers = (
        pd.DataFrame(
            {
                "Feature": FEATURES,
                "Impact": contributions,
                "Direction": np.where(contributions >= 0, "Raises risk", "Lowers risk"),
            }
        )
        .assign(Absolute_impact=lambda data: data["Impact"].abs())
        .sort_values("Absolute_impact", ascending=False)
        .head(8)
    )
    return raw_probability, calibrated_probability, drivers


st.markdown(
    """
    <div class="hero">
      <h1>🛡️ FraudGuard</h1>
      <p>LightGBM credit-card fraud detection with calibrated probabilities and explainable predictions</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Project setup")
    upload = st.file_uploader(
        "Upload the dataset",
        type=["csv", "zip"],
        help="Use creditcard.csv or creditcard.csv.zip.",
    )
    threshold = st.slider(
        "Fraud alert threshold",
        min_value=0.01,
        max_value=0.90,
        value=0.10,
        step=0.01,
        help="A lower threshold catches more fraud but may produce more false alerts.",
    )
    st.caption("Student: S Aditya Anirudh · 245890012")

if upload is None:
    st.info("Upload **creditcard.csv.zip** in the sidebar to start the demonstration.")
    left, right = st.columns(2)
    with left:
        st.subheader("What the application does")
        st.write("1. Trains LightGBM on legitimate and fraudulent transactions.")
        st.write("2. Calibrates the risk score using isotonic regression.")
        st.write("3. Predicts fraud risk for a selected transaction.")
    with right:
        st.subheader("Project results")
        st.metric("ROC-AUC", "0.9609")
        st.metric("Fraud rate", "0.17%")
    st.stop()

try:
    with st.spinner("Training and calibrating the model…"):
        model, calibrator, dataset, metrics, charts = train_model(
            upload.getvalue(), upload.name
        )
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

st.markdown(
    f"<div class='ready-card'>✓ Model ready &nbsp;·&nbsp; {len(dataset):,} transactions loaded</div>",
    unsafe_allow_html=True,
)

metric_columns = st.columns(5)
metric_columns[0].metric("ROC-AUC", f"{metrics['auc']:.4f}")
metric_columns[1].metric("PR-AUC", f"{metrics['pr_auc']:.4f}")
metric_columns[2].metric("Gini", f"{metrics['gini']:.4f}")
metric_columns[3].metric("KS statistic", f"{metrics['ks']:.4f}")
metric_columns[4].metric("Fraud rate", f"{metrics['fraud_rate']:.2%}")

prediction_tab, performance_tab, about_tab = st.tabs(
    ["Transaction check", "Model performance", "About the model"]
)

with prediction_tab:
    mode = st.radio(
        "Input method",
        ["Select a dataset transaction", "Enter values manually"],
        horizontal=True,
    )

    if mode == "Select a dataset transaction":
        row_number = st.number_input(
            "Dataset row number",
            min_value=0,
            max_value=len(dataset) - 1,
            value=min(541, len(dataset) - 1),
            step=1,
        )
        transaction = dataset.loc[[int(row_number)], FEATURES]
        actual_class = int(dataset.loc[int(row_number), TARGET])
        label_class = "label-fraud" if actual_class == 1 else "label-legit"
        label_text = "Known fraud · Class 1" if actual_class == 1 else "Known legitimate · Class 0"
        st.markdown(
            f"<span class='label-pill {label_class}'>{label_text}</span>",
            unsafe_allow_html=True,
        )
    else:
        defaults = dataset[FEATURES].median(numeric_only=True)
        values = {}
        first, second = st.columns(2)
        with first:
            values["Time"] = st.number_input("Time", value=float(defaults["Time"]))
        with second:
            values["Amount"] = st.number_input(
                "Amount", min_value=0.0, value=float(defaults["Amount"])
            )
        with st.expander("Advanced anonymized features V1–V28"):
            columns = st.columns(4)
            for index, feature in enumerate(FEATURES[1:-1]):
                with columns[index % 4]:
                    values[feature] = st.number_input(
                        feature,
                        value=float(defaults[feature]),
                        format="%.5f",
                    )
        transaction = pd.DataFrame([{feature: values[feature] for feature in FEATURES}])

    if st.button("Analyze transaction", type="primary", use_container_width=True):
        raw_probability, probability, drivers = predict_transaction(
            model, calibrator, transaction
        )
        is_alert = probability >= threshold
        css_class = "result-risk" if is_alert else "result-safe"
        title = "High-risk review recommended" if is_alert else "Low-risk transaction"
        icon = "⚠️" if is_alert else "✅"
        action = (
            "The score is above your alert threshold. Send this transaction for review."
            if is_alert
            else "The score is below your alert threshold. No alert is required."
        )
        st.markdown(
            f"<div class='{css_class}'><h2>{icon} {title}</h2>"
            f"<div>Calibrated fraud probability: <b>{probability:.2%}</b> &nbsp;·&nbsp; {action}</div></div>",
            unsafe_allow_html=True,
        )
        probability_columns = st.columns(3)
        probability_columns[0].metric("Calibrated probability", f"{probability:.2%}")
        probability_columns[1].metric("Raw ranking score", f"{raw_probability:.2%}")
        probability_columns[2].metric("Decision threshold", f"{threshold:.0%}")

        st.subheader("Main factors behind this prediction")
        chart = drivers.set_index("Feature")["Impact"].sort_values()
        st.bar_chart(chart, horizontal=True, color="#1261a0")
        st.markdown(
            "<div class='helper'>Positive bars raise fraud risk; negative bars lower it. "
            "The values are local LightGBM/SHAP contributions for this transaction.</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            drivers[["Feature", "Direction", "Impact"]],
            hide_index=True,
            use_container_width=True,
            column_config={"Impact": st.column_config.NumberColumn(format="%.4f")},
        )

        with st.expander("View transaction values"):
            st.dataframe(transaction.T.rename(columns={transaction.index[0]: "Value"}))

with performance_tab:
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.subheader("ROC curve")
        st.line_chart(
            charts["roc"].set_index("False positive rate"),
            color="#1261a0",
        )
        st.caption("A curve closer to the top-left indicates better fraud separation.")
    with chart_right:
        st.subheader("Probability calibration")
        st.line_chart(
            charts["calibration"].set_index("Predicted probability"),
            color="#00a6a6",
        )
        st.caption("Closer agreement means predicted risk better matches observed fraud.")
    st.metric("Calibrated Brier score", f"{metrics['brier']:.4f}")

with about_tab:
    st.subheader("How the system works")
    st.markdown(
        """
        - **LightGBM** learns nonlinear patterns among `Time`, `Amount`, and `V1–V28`.
        - **Class weighting** helps the model learn from the very small fraud class.
        - **Isotonic calibration** converts the raw model score into a more meaningful probability.
        - **Local contributions** show which features raised or lowered one transaction's risk.

        This is an educational demonstration. A production banking system would also require
        cost-based threshold selection, monitoring, human review, and privacy controls.
        """
    )

