import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


st.set_page_config(
    page_title="Premium Wine Screener",
    page_icon=":wine_glass:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #6B4226;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #7A7A7A;
        margin-top: 0;
        margin-bottom: 1.2rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
    .result-card {
        padding: 1.4rem 1.6rem;
        border-radius: 12px;
        margin-top: 0.8rem;
        margin-bottom: 1rem;
    }
    .result-premium {
        background-color: #F4EBD0;
        border: 2px solid #C9A227;
    }
    .result-standard {
        background-color: #F0F0F0;
        border: 2px solid #A0A0A0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-header">Premium White Wine Screener</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Flag likely Premium-grade batches (expert quality score >= 7) from lab '
    'measurements alone, so your QC team can prioritise tasting review instead of treating every '
    'batch equally.</p>',
    unsafe_allow_html=True,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "wine_final_model.pkl")


@st.cache_resource
def load_model_bundle(path):
    bundle = joblib.load(path)
    required_keys = {"model", "threshold", "features"}
    if not required_keys.issubset(bundle.keys()):
        raise ValueError(f"Model file is missing expected keys: {required_keys - set(bundle.keys())}")
    return bundle


try:
    bundle = load_model_bundle(MODEL_PATH)
    model = bundle["model"]
    threshold = bundle["threshold"]
    feature_order = bundle["features"]
    model_load_error = None
except FileNotFoundError:
    model_load_error = (
        f"Could not find 'wine_final_model.pkl' next to this app. "
        f"Run the training notebook first so it saves the model file, then restart the app."
    )
except Exception as e:
    model_load_error = f"Failed to load the model file: {e}"

if model_load_error:
    st.error(model_load_error)
    st.stop()

FEATURE_META = {
    "fixed acidity":         {"min": 3.5,  "max": 15.0, "default": 6.8,   "step": 0.1,  "unit": "g/dm3", "help": "Tartaric acid content."},
    "volatile acidity":      {"min": 0.05, "max": 1.2,  "default": 0.26,  "step": 0.01, "unit": "g/dm3", "help": "Acetic acid; too high causes a vinegar taste."},
    "citric acid":           {"min": 0.0,  "max": 1.7,  "default": 0.32,  "step": 0.01, "unit": "g/dm3", "help": "Adds freshness and flavour."},
    "residual sugar":        {"min": 0.5,  "max": 66.0, "default": 4.7,   "step": 0.1,  "unit": "g/dm3", "help": "Sugar remaining after fermentation."},
    "chlorides":              {"min": 0.005,"max": 0.35, "default": 0.042, "step": 0.001,"unit": "g/dm3", "help": "Salt content."},
    "free sulfur dioxide":   {"min": 1.0,  "max": 290.0,"default": 33.0,  "step": 1.0,  "unit": "mg/dm3","help": "Free SO2; prevents microbial growth and oxidation."},
    "total sulfur dioxide":  {"min": 5.0,  "max": 445.0,"default": 133.0, "step": 1.0,  "unit": "mg/dm3","help": "Free + bound SO2."},
    "density":                {"min": 0.985,"max": 1.04, "default": 0.994, "step": 0.0001,"unit": "g/cm3","help": "Related to sugar and alcohol content."},
    "pH":                     {"min": 2.7,  "max": 3.85, "default": 3.18,  "step": 0.01, "unit": "",      "help": "Acidity on a 0 (very acidic) to 14 scale."},
    "sulphates":              {"min": 0.2,  "max": 1.1,  "default": 0.48,  "step": 0.01, "unit": "g/dm3", "help": "Additive; contributes to SO2 levels."},
    "alcohol":                 {"min": 8.0,  "max": 14.5, "default": 10.4,  "step": 0.1,  "unit": "% vol", "help": "Alcohol content by volume."},
}

PRESETS = {
    "Typical Standard wine": {
        "fixed acidity": 6.9, "volatile acidity": 0.29, "citric acid": 0.33, "residual sugar": 5.2,
        "chlorides": 0.046, "free sulfur dioxide": 35.0, "total sulfur dioxide": 140.0,
        "density": 0.9955, "pH": 3.16, "sulphates": 0.47, "alcohol": 9.8,
    },
    "Typical Premium wine": {
        "fixed acidity": 6.6, "volatile acidity": 0.22, "citric acid": 0.34, "residual sugar": 4.0,
        "chlorides": 0.036, "free sulfur dioxide": 32.0, "total sulfur dioxide": 116.0,
        "density": 0.9915, "pH": 3.21, "sulphates": 0.50, "alcohol": 11.6,
    },
}

for feat, meta in FEATURE_META.items():
    if f"slider_{feat}" not in st.session_state:
        st.session_state[f"slider_{feat}"] = meta["default"]
if "has_predicted" not in st.session_state:
    st.session_state.has_predicted = False


def apply_preset(values: dict):
    for feat, val in values.items():
        st.session_state[f"slider_{feat}"] = val



with st.sidebar:
    st.header("Quick demo presets")
    st.caption("Load a realistic example, then hit Predict.")
    for name, values in PRESETS.items():
        st.button(name, use_container_width=True, on_click=apply_preset, args=(values,))

    st.button(
        "Reset to defaults",
        use_container_width=True,
        on_click=apply_preset,
        args=({feat: meta["default"] for feat, meta in FEATURE_META.items()},),
    )

    st.divider()
    st.header("About this tool")
    st.caption(
        "Trained on 3,961 white wine samples (UCI Wine Quality dataset) using a "
        "hyperparameter-tuned Random Forest. Predicts whether a wine is likely to "
        "score Premium (quality >= 7) from an expert tasting panel, based solely "
        "on lab-measurable chemistry — no tasting required."
    )
    st.caption(f"Decision threshold: {threshold:.3f} (calibrated to maximise F1 on the Premium class)")

st.subheader("1. Enter batch measurements")

groups = {
    "Acidity & Sugar": ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "pH"],
    "Sulfur & Preservatives": ["free sulfur dioxide", "total sulfur dioxide", "sulphates", "chlorides"],
    "Physical & Alcohol": ["density", "alcohol"],
}

validation_errors = []
cols = st.columns(3)
for col, (group_name, feats) in zip(cols, groups.items()):
    with col:
        st.markdown(f"**{group_name}**")
        for feat in feats:
            meta = FEATURE_META[feat]
            label = f"{feat.title()} ({meta['unit']})" if meta["unit"] else feat.title()
            st.slider(
                label,
                min_value=float(meta["min"]),
                max_value=float(meta["max"]),
                step=float(meta["step"]),
                help=meta["help"],
                key=f"slider_{feat}",
            )

current_inputs = {feat: st.session_state[f"slider_{feat}"] for feat in FEATURE_META}

if current_inputs["free sulfur dioxide"] > current_inputs["total sulfur dioxide"]:
    validation_errors.append(
        "Free sulfur dioxide cannot be greater than total sulfur dioxide. Please adjust the sliders above."
    )

if validation_errors:
    for err in validation_errors:
        st.error(err)

st.subheader("2. Run the prediction")
predict_clicked = st.button(
    "Predict Quality",
    type="primary",
    disabled=bool(validation_errors),
    use_container_width=False,
)

if predict_clicked:
    st.session_state.has_predicted = True

if st.session_state.has_predicted and not validation_errors:
    try:
        input_row = pd.DataFrame([current_inputs])[feature_order]
        proba_premium = model.predict_proba(input_row)[0, 1]
        is_premium = proba_premium >= threshold

        card_class = "result-premium" if is_premium else "result-standard"
        label_text = "PREMIUM" if is_premium else "STANDARD"

        st.markdown(f'<div class="result-card {card_class}">', unsafe_allow_html=True)
        result_col1, result_col2, result_col3 = st.columns([1.2, 1, 1])
        with result_col1:
            if is_premium:
                st.success(f"Predicted grade: **{label_text}**")
            else:
                st.info(f"Predicted grade: **{label_text}**")
        with result_col2:
            st.metric("Premium probability", f"{proba_premium*100:.1f}%")
        with result_col3:
            st.metric("Decision threshold", f"{threshold*100:.1f}%")
        st.progress(min(max(proba_premium, 0.0), 1.0))
        st.markdown("</div>", unsafe_allow_html=True)

        st.caption(
            "The model flags a batch as Premium when its predicted probability of scoring quality "
            "≥ 7 meets or exceeds the calibrated threshold above. Use this as a triage signal to "
            "prioritise tasting review, not as a replacement for it."
        )

        if hasattr(model, "feature_importances_"):
            st.subheader("3. What's driving this prediction")
            importance_df = pd.DataFrame({
                "Feature": feature_order,
                "Importance": model.feature_importances_,
            }).sort_values("Importance", ascending=False)
            st.bar_chart(importance_df.set_index("Feature"))
            st.caption(
                "Model-wide feature importance (not specific to this single batch) — alcohol and "
                "density are consistently the strongest predictors of Premium quality across the "
                "training data."
            )

    except Exception as e:
        st.error(f"Something went wrong while generating the prediction: {e}. Please check your inputs and try again.")

elif not st.session_state.has_predicted:
    st.info("Adjust the sliders above (or load a preset from the sidebar), then click **Predict Quality** to see a result.")
