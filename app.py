from pathlib import Path
import pickle

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# =====================================================================
# 1. CONFIG: EDIT THIS SECTION TO MATCH YOUR MODEL / DATASET
# =====================================================================
APP_TITLE = "Influencer Campaign Predictor"
TARGET_LABEL = "Predicted campaign revenue"
TARGET_FORMAT = "{:,.2f}"
TARGET_UNIT = "USD"

MODEL_PATH = Path(__file__).resolve().parent / "models" / "linear_regression_model.pkl"

FEATURES = [
    "creator_followers_k",
    "engagement_rate_pct",
    "paid_boost_spend_usd",
    "content_format",
    "audience_intent_tier",
]
NUMERIC = ["creator_followers_k", "engagement_rate_pct", "paid_boost_spend_usd"]
st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")


@st.cache_resource(show_spinner=False)
def load_model():
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the saved model at {MODEL_PATH}: {error}")
    st.stop()

expected_features = list(model.feature_names_in_)
formats = [
    feature.removeprefix("content_format_")
    for feature in expected_features
    if feature.startswith("content_format_")
]
tiers = [
    feature.removeprefix("audience_intent_tier_")
    for feature in expected_features
    if feature.startswith("audience_intent_tier_")
]


def model_inputs(df: pd.DataFrame) -> pd.DataFrame:
    inputs = df[FEATURES].copy()
    invalid_formats = set(inputs["content_format"].dropna()) - set(formats)
    invalid_tiers = set(inputs["audience_intent_tier"].dropna()) - set(tiers)
    if invalid_formats or invalid_tiers:
        raise ValueError(
            f"Unsupported categories. Content formats: {formats}; audience tiers: {tiers}."
        )
    encoded = pd.DataFrame(index=inputs.index)
    for feature in expected_features:
        if feature.startswith("content_format_"):
            encoded[feature] = inputs["content_format"].eq(
                feature.removeprefix("content_format_")
            )
        elif feature.startswith("audience_intent_tier_"):
            encoded[feature] = inputs["audience_intent_tier"].eq(
                feature.removeprefix("audience_intent_tier_")
            )
        elif feature in inputs.columns:
            encoded[feature] = inputs[feature]
        else:
            raise ValueError(f"Unsupported model feature: {feature}")
    return encoded.reindex(columns=expected_features)


def predict(df: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(model_inputs(df)), dtype=float).ravel()


def fmt(v: float) -> str:
    s = TARGET_FORMAT.format(v)
    return f"{s} {TARGET_UNIT}".strip()


# =====================================================================
# 3. SIDEBAR: MODEL + INPUTS
# =====================================================================
st.title(f"📈 {APP_TITLE}")

with st.sidebar:
    st.header("🎛️ Inputs")
    followers = st.number_input("Creator followers (thousands)", 0.0, 1_000_000.0, 0.0, 0.1,
                                help="Enter 250 for 250,000 followers.")
    st.caption(f"= {followers * 1000:,.0f} followers")
    engagement = st.number_input("Engagement rate (%)", 0.0, 100.0, 0.0, 0.1,
                                 help="Enter 5.0 for 5%.")
    spend = st.number_input("Paid boost spend (USD)", 0.0, 10_000_000.0, 0.0, 1.0)
    content_format = st.selectbox("Content format", formats)
    intent_tier = st.selectbox("Audience intent tier", tiers)

    if engagement > 30:
        st.warning("Engagement rate looks unusually high. Check it's entered as a percentage (5.0 = 5%).")

base = {
    "creator_followers_k": followers,
    "engagement_rate_pct": engagement,
    "paid_boost_spend_usd": spend,
    "content_format": content_format,
    "audience_intent_tier": intent_tier,
}
base_df = pd.DataFrame([base])


def safe_predict(df):
    try:
        return predict(df)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Check that the input values and category choices match the model's training data.")
        st.stop()


# =====================================================================
# 4. TABS
# =====================================================================
tab_pred, tab_grid = st.tabs(
    ["🎯 Prediction", "🧩 Format × Intent"]
)

# ---------- Prediction ----------
with tab_pred:
    pred = float(safe_predict(base_df)[0])
    c1, c2, c3 = st.columns([2, 1, 1])
    c1.metric(TARGET_LABEL, fmt(pred))
    c2.metric("Reach (followers)", f"{followers * 1000:,.0f}")
    c3.metric("Spend per 1k followers", f"${spend / followers:,.2f}" if followers else "n/a")

    st.subheader("Input summary")
    st.dataframe(base_df.T.rename(columns={0: "value"}), use_container_width=True)

    # Spend uplift: what does paid boost add vs. zero spend?
    zero = pd.DataFrame([{**base, "paid_boost_spend_usd": 0.0}])
    p0 = float(safe_predict(zero)[0])
    u1, u2 = st.columns(2)
    u1.metric("Without paid boost", fmt(p0))
    u2.metric("Lift from paid boost", fmt(pred - p0),
              delta=f"{(pred - p0) / p0 * 100:,.1f}%" if p0 else None)

    if st.button("➕ Save as scenario", type="primary"):
        st.session_state.setdefault("scenarios", []).append({**base, "prediction": pred})
        st.toast("Scenario saved")

# ---------- Format x Intent ----------
with tab_grid:
    st.caption("Predicted outcome for every content format and audience tier at the current numeric inputs.")
    combos = pd.DataFrame([{**base, "content_format": f, "audience_intent_tier": t}
                           for f in formats for t in tiers])
    combos["prediction"] = safe_predict(combos)

    heat = alt.Chart(combos).mark_rect().encode(
        x=alt.X("audience_intent_tier:N", sort=tiers, title="Audience intent tier"),
        y=alt.Y("content_format:N", sort=formats, title="Content format"),
        color=alt.Color("prediction:Q", scale=alt.Scale(scheme="viridis"), title=TARGET_LABEL),
        tooltip=["content_format", "audience_intent_tier", alt.Tooltip("prediction:Q", format=",.3f")])
    text = heat.mark_text(baseline="middle").encode(
        text=alt.Text("prediction:Q", format=",.1f"), color=alt.value("white"))
    st.altair_chart(heat + text, use_container_width=True)

    best = combos.loc[combos["prediction"].idxmax()]
    st.success(f"Best combination: **{best['content_format']}** × **{best['audience_intent_tier']}** → {fmt(best['prediction'])}")
    st.dataframe(combos.sort_values("prediction", ascending=False).reset_index(drop=True),
                 use_container_width=True)
    template = pd.DataFrame([base])