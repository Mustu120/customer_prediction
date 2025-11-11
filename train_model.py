import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json
import warnings

# Ignore warnings for a cleaner output
warnings.filterwarnings('ignore')

print("--- [1/6] Loading Dataset ---")
# Load the dataset
df = pd.read_csv('telecom_customer_churn.csv')

print("--- [2/6] Cleaning and Imputing Data ---")
# --- 1. Initial Cleaning & Feature Dropping ---

# Drop identifier, high-cardinality location, and data leakage columns
columns_to_drop = ['Customer ID', 'City', 'Zip Code', 'Latitude', 'Longitude', 'Churn Category', 'Churn Reason']
df_cleaned = df.drop(columns=columns_to_drop)

# Correct data anomalies (negative monthly charge)
df_cleaned['Monthly Charge'] = df_cleaned['Monthly Charge'].clip(lower=0)

# --- 2. Handle Missing Values (Imputation) ---

# For 'Internet Type', fill NaN with 'No Internet'
df_cleaned['Internet Type'] = df_cleaned['Internet Type'].fillna('No Internet')

# For 'Avg Monthly GB Download', fill NaN with 0
df_cleaned['Avg Monthly GB Download'] = df_cleaned['Avg Monthly GB Download'].fillna(0)

# For other categorical 'Yes'/'No' internet columns, fill NaN with 'No Internet Service'
for col in ['Online Security', 'Online Backup', 'Device Protection Plan',
            'Premium Tech Support', 'Streaming TV', 'Streaming Movies',
            'Streaming Music', 'Unlimited Data']:
    if col in df_cleaned.columns:
        df_cleaned[col] = df_cleaned[col].fillna('No Internet Service')

# 'Avg Monthly Long Distance Charges': fill NaN with 0
df_cleaned['Avg Monthly Long Distance Charges'] = df_cleaned['Avg Monthly Long Distance Charges'].fillna(0)

# 'Multiple Lines': fill NaN with 'No Phone Service'
df_cleaned['Multiple Lines'] = df_cleaned['Multiple Lines'].fillna('No Phone Service')

# Drop any other stray NaNs if they exist
df_cleaned = df_cleaned.dropna()

print("--- [3/6] Defining Features and Preprocessor ---")
# --- 3. Define Features (X) and Target (y) ---
target = 'Customer Status'
y = df_cleaned[target]
X = df_cleaned.drop(columns=target)

# --- 4. Identify Feature Types ---
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Numerical Features: {numerical_features}")
print(f"Categorical Features: {categorical_features}")

# --- 5. Create Preprocessing Pipeline ---
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# --- 6. Define the Model ---
model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')

# --- 7. Create the Full Prediction Pipeline ---
main_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

print("--- [4/6] Training Model ---")
# --- 8. Split and Train ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train the pipeline
main_pipeline.fit(X_train, y_train)
print("--- Model Training Complete ---")

# --- 9. Evaluate the model ---
print("--- [5/6] Evaluating Model ---")
y_pred = main_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=np.unique(y).tolist())

print(f"\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)

# --- 10. Save the Pipeline and supporting files ---
print("--- [6/6] Saving Files ---")
model_filename = 'customer_status_predictor.joblib'
joblib.dump(main_pipeline, model_filename)
print(f"Model pipeline saved to {model_filename}")

# Save sample data for the app
X_test.to_csv('sample_app_data.csv', index=False)
print(f"Sample data saved to sample_app_data.csv")

# Save UI schema for the app
ui_elements = { 'numerical': {}, 'categorical': {} }

for col in numerical_features:
    min_val = float(X_test[col].min())
    max_val = float(X_test[col].max())
    mean_val = float(X_test[col].mean())
    ui_elements['numerical'][col] = {'min': min_val, 'max': max_val, 'default': mean_val}

for col in categorical_features:
    options = X_test[col].unique().tolist()
    ui_elements['categorical'][col] = {'options': [str(opt) for opt in options]}

with open('app_ui_schema.json', 'w') as f:
    json.dump(ui_elements, f, indent=4)
print(f"App UI schema saved to app_ui_schema.json")

print("\n--- All Done ---")