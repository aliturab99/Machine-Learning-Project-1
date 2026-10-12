from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "classification_model.pkl"
DATA_PATH = ROOT / "raw_data" / "ecommerce_return_classification.csv"
TARGET = "return_risk_label"
TEST_SIZE = 0.2
RANDOM_STATE = 42
EXCLUDED_FEATURES = {"newsletter_subscriber", "packaging_color"}
FIELD_LABELS = {
    "cart_value_usd": "Cart value (USD)",
    "past_return_rate": "Customer past return rate",
    "size_bracketing_flag": "Size bracketing used",
    "delivery_days_taken": "Delivery time (days)",
    "discount_pct_applied": "Discount applied (%)",
    "product_category": "Product category",
    "fit_review_sentiment": "Fit review sentiment",
    "loyalty_card_color": "Loyalty card color",
}

st.set_page_config(
    page_title="Ecommerce Return Risk Classifier",
    page_icon="📦",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_classification_assets():
    with MODEL_PATH.open("rb") as model_file:
        classification_model = pickle.load(model_file)

    dataset = pd.read_csv(DATA_PATH)
    target = dataset[TARGET]
    inputs = dataset.drop(
        columns=[TARGET, *EXCLUDED_FEATURES],
        errors="ignore",
    )
    expected_features = list(classification_model.feature_names_in_)
    excluded_model_features = [
        feature
        for feature in expected_features
        if any(
            feature == excluded or feature.startswith(f"{excluded}_")
            for excluded in EXCLUDED_FEATURES
        )
    ]
    if excluded_model_features:
        raise ValueError(
            "The saved classifier still uses excluded features "
            f"{excluded_model_features}. Retrain it without newsletter_subscriber "
            "and packaging_color."
        )

    all_categorical_features = inputs.select_dtypes(
        include=["object", "str", "string"]
    ).columns
    categorical_features = [
        feature
        for feature in all_categorical_features
        if any(
            model_feature.startswith(f"{feature}_")
            for model_feature in expected_features
        )
    ]
    all_numeric_features = inputs.select_dtypes(include=np.number).columns
    numeric_features = [
        feature for feature in all_numeric_features if feature in expected_features
    ]
    active_features = [*numeric_features, *categorical_features]
    inputs = inputs[active_features]
    category_options = {
        feature: sorted(inputs[feature].dropna().unique().tolist())
        for feature in categorical_features
    }
    encoded_inputs = pd.get_dummies(
        inputs, columns=categorical_features, drop_first=True
    )
    missing_model_features = [
        feature
        for feature in expected_features
        if feature not in encoded_inputs.columns
    ]
    if missing_model_features:
        raise ValueError(
            "The current dataset cannot provide all saved-model features: "
            f"{missing_model_features}. Retrain the model on the current dataset."
        )
    encoded_inputs = encoded_inputs.reindex(columns=expected_features)

    X_train, X_test, _, y_test = train_test_split(
        encoded_inputs,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    scaler = StandardScaler()
    scaler.fit(X_train[numeric_features])
    X_test = X_test.copy()
    X_test[numeric_features] = X_test[numeric_features].astype(float)
    X_test.loc[:, numeric_features] = scaler.transform(X_test[numeric_features])

    return (
        classification_model,
        scaler,
        categorical_features,
        category_options,
        numeric_features,
        X_test,
        y_test,
        inputs,
    )


def classification_model_input(
    values,
    expected_features,
    categorical_features,
    numeric_features,
    scaler,
):
    encoded = pd.DataFrame(index=[0])
    for feature in expected_features:
        if feature in numeric_features:
            encoded[feature] = [float(values[feature])]
            continue
        for category_feature in categorical_features:
            prefix = f"{category_feature}_"
            if feature.startswith(prefix):
                encoded[feature] = [
                    values[category_feature] == feature.removeprefix(prefix)
                ]
                break
        else:
            raise ValueError(f"Unsupported classification model feature: {feature}")

    encoded.loc[:, numeric_features] = scaler.transform(encoded[numeric_features])
    return encoded.reindex(columns=expected_features)


def field_label(feature):
    return FIELD_LABELS.get(feature, feature.replace("_", " ").title())


st.title("📦 Ecommerce Return Risk Classifier")
st.caption(
    "Estimate an order's return risk and review performance on a reproducible "
    "held-out 20% test split. Numeric scaling is fit on training rows only."
)

try:
    (
        classification_model,
        classification_scaler,
        classification_categories,
        classification_options,
        classification_numeric_features,
        classification_X_test,
        classification_y_test,
        classification_inputs,
    ) = load_classification_assets()
except Exception as error:
    st.error(f"Could not load the classification model or its training data: {error}")
    st.stop()

with st.form("return_risk_prediction_form"):
    st.subheader("Order details")
    input_columns = st.columns(2)
    submitted_classification_values = {}

    for index, feature in enumerate(classification_inputs.columns):
        container = input_columns[index % len(input_columns)]
        label = field_label(feature)
        with container:
            if feature in classification_categories:
                value = st.selectbox(
                    label,
                    classification_options[feature],
                    key=f"classification_{feature}",
                )
            elif set(classification_inputs[feature].dropna().unique()) <= {0, 1}:
                value = st.selectbox(
                    label,
                    [0, 1],
                    format_func=lambda item: "Yes" if item else "No",
                    key=f"classification_{feature}",
                )
            elif pd.api.types.is_integer_dtype(classification_inputs[feature].dtype):
                value = st.number_input(
                    label,
                    value=int(classification_inputs[feature].median()),
                    step=1,
                    key=f"classification_{feature}",
                )
            else:
                value = st.number_input(
                    label,
                    value=float(classification_inputs[feature].median()),
                    step=0.01,
                    key=f"classification_{feature}",
                )
            submitted_classification_values[feature] = value

    classification_submitted = st.form_submit_button(
        "Predict return risk",
        type="primary",
        width="stretch",
    )

if classification_submitted:
    st.session_state["submitted_return_risk"] = submitted_classification_values

classification_input = st.session_state.get("submitted_return_risk")
if classification_input is not None:
    prediction_input = classification_model_input(
        classification_input,
        list(classification_model.feature_names_in_),
        classification_categories,
        classification_numeric_features,
        classification_scaler,
    )
    predicted_class = int(classification_model.predict(prediction_input)[0])
    risk_probability = float(
        classification_model.predict_proba(prediction_input)[0][
            list(classification_model.classes_).index(1)
        ]
    )
    risk_label = "High return risk" if predicted_class == 1 else "Low return risk"
    result_col, probability_col = st.columns(2)
    result_col.metric("Predicted outcome", risk_label)
    probability_col.metric("High-risk probability", f"{risk_probability:.1%}")
    st.progress(risk_probability)
    st.caption("Class 1 = high return risk; class 0 = low return risk.")
    st.subheader("Submitted order")
    st.dataframe(
        pd.DataFrame(
            {"Value": pd.Series(classification_input, dtype="string")}
        ),
        width="stretch",
    )
else:
    st.info("Enter order details and select **Predict return risk**.")

test_predictions = classification_model.predict(classification_X_test)
high_risk_precision = precision_score(
    classification_y_test, test_predictions, pos_label=1, zero_division=0
)
high_risk_recall = recall_score(
    classification_y_test, test_predictions, pos_label=1, zero_division=0
)
high_risk_f1 = f1_score(
    classification_y_test, test_predictions, pos_label=1, zero_division=0
)

st.subheader("Held-out test-set performance")
metric_accuracy, metric_precision, metric_recall, metric_f1 = st.columns(4)
metric_accuracy.metric(
    "Accuracy", f"{accuracy_score(classification_y_test, test_predictions):.1%}"
)
metric_precision.metric("High-risk precision", f"{high_risk_precision:.1%}")
metric_recall.metric("High-risk recall", f"{high_risk_recall:.1%}")
metric_f1.metric("High-risk F1", f"{high_risk_f1:.1%}")

matrix = confusion_matrix(classification_y_test, test_predictions, labels=[0, 1])
st.dataframe(
    pd.DataFrame(
        matrix,
        index=["Actual low risk (0)", "Actual high risk (1)"],
        columns=["Predicted low risk (0)", "Predicted high risk (1)"],
    ),
    width="stretch",
)
