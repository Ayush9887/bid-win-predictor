import pandas as pd
import numpy as np
import streamlit as st
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_excel("Data for Pricing Engine updated ACtionitem.xlsx")
df = df.drop(columns=['Sr no'])
df = df[df['Price($ / Kg)'] <= df['Price($ / Kg)'].quantile(0.99)]
df['Result_binary'] = df['Result(w/L)'].map({'Won': 1, 'Loss': 0})

# Features and Target
X = df.drop(columns=['Result(w/L)', 'Result_binary'])
y = df['Result_binary']

# Identify columns
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(exclude=['object']).columns.tolist()

# Preprocessing pipelines
numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

# Preprocess X for determining constraints
X_preprocessed = preprocessor.fit_transform(X)
n_features = X_preprocessed.shape[1]

# Monotonic constraints setup
monotonic_constraints = [0] * n_features
if 'Price($ / Kg)' in numerical_features:
    price_index = numerical_features.index('Price($ / Kg)')
    monotonic_constraints[price_index] = -1

# Final model
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    monotone_constraints=tuple(monotonic_constraints[:len(numerical_features)])
)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', xgb_model)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Suggest price adjustment function
def suggest_price_reduction(single_input: pd.DataFrame, model_pipeline, price_col='Price($ / Kg)', step=1.0):
    input_copy = single_input.copy()
    original_price = input_copy[price_col].values[0]
    max_attempts = 20
    for i in range(1, max_attempts + 1):
        input_copy[price_col] = original_price - i * step
        prob = model_pipeline.predict_proba(input_copy)[0][1]
        if prob >= 0.5:
            return input_copy[price_col].values[0], prob
    return None, model_pipeline.predict_proba(single_input)[0][1]

# Streamlit UI
st.title("L&T Bid Win Predictor")
st.write("Upload or enter bid details to predict Win/Loss and get price suggestions.")

sample_input = {}
for col in numerical_features:
    sample_input[col] = st.number_input(f"{col}", value=float(X[col].median()))
for col in categorical_features:
    sample_input[col] = st.selectbox(f"{col}", options=X[col].dropna().unique())

if st.button("Predict Win/Loss"):
    input_df = pd.DataFrame([sample_input])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"Prediction: ✅ Won with probability {probability:.2f}")
    else:
        st.error(f"Prediction: ❌ Loss with probability {1 - probability:.2f}")
        new_price, new_prob = suggest_price_reduction(input_df, model)
        if new_price:
            st.info(f"Suggest reducing price to ${new_price:.2f}/Kg → new win probability: {new_prob:.2f}")
        else:
            st.warning("Even with price reduction, unlikely to win this bid.")
