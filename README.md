# Machine Learning Project

This project contains two machine learning dashboards built with Streamlit and saved trained models.

Live app: https://ali-turab-machine-learning-project-1.streamlit.app/

## Overview

The app is designed to show how two different ML problem types work in real life:

1. Regression model: predicts a continuous numeric value
2. Classification model: predicts a category (high risk vs low risk)

Both models were trained on structured business data and then saved as `.pkl` files so the app can load them and make predictions without retraining each time.

## Project structure

- `app.py` — revenue prediction dashboard
- `pages/2_Return_Risk.py` — return-risk dashboard
- `models/` — saved trained models
- `raw_data/` — datasets used for training and evaluation
- `classification_model/` — notebook for the return-risk model
- `regression_model/` — notebook for the influencer revenue model
- `requirements.txt` — project dependencies

## Model 1: Influencer Campaign Revenue Predictor

This is a regression model.

### What it predicts
It predicts the likely campaign revenue for an influencer campaign based on inputs such as:

- number of followers (in thousands)
- engagement rate
- paid boost spend
- content format
- audience intent tier

### Why it is regression
Regression is used when the output is a continuous number, not a class label. Here, the model predicts a dollar value such as `$12,450.30` instead of saying "good" or "bad".

### How it works in the app
- User enters campaign information in the sidebar
- The app prepares the input in the same format used during training
- It encodes categorical values like content format and audience intent
- The saved model predicts revenue
- The dashboard shows:
  - predicted revenue
  - reach
  - spend per 1k followers
  - format × intent comparison grid
  - best-performing content/audience combination

### Example
If a creator has high engagement, moderate spend, and a product-focused audience, the model estimates the likely revenue for that campaign scenario.

## Model 2: Ecommerce Return Risk Classifier

This is a classification model.

### What it predicts
It predicts whether an order is likely to have a high return risk or low return risk based on features such as:

- cart value
- past return rate
- discount percentage
- delivery time
- sentiment of product reviews
- product category
- loyalty card color
- other order-related variables

### Why it is classification
Classification is used when the result is one of a few categories. In this case, the model chooses between:

- `0` = low return risk
- `1` = high return risk

### How it works in the app
- User enters order details in the form
- The app transforms the values into the right model input format
- Numeric values are standardized using the training scaler
- Categorical features are one-hot encoded
- The saved classifier predicts the probability of high return risk
- The dashboard displays:
  - predicted class
  - risk probability
  - test performance metrics
  - confusion matrix

### Example
If a customer orders a product with low satisfaction signals, long delivery time, and high discount usage, the model may flag that order as high return risk.

## Model deployment approach

The app does not retrain the model when a user opens the dashboard. Instead:

- the model is trained offline in the notebook
- the trained model is saved to the `models/` folder
- the Streamlit app loads the saved `.pkl` file at runtime
- the app makes predictions directly from that saved model

This is a common production pattern because it is fast, reliable, and avoids unnecessary retraining in deployment.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run locally

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

## Requirements

- Python
- Streamlit
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
