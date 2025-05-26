import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Set dataset path
DATASET_FOLDER = '../dataset'
MODEL_PATH = '../models/trained_model.pkl'

# Collect CSV files
csv_files = [os.path.join(DATASET_FOLDER, f) for f in os.listdir(DATASET_FOLDER) if f.endswith('.csv')]

# Read and concatenate all data
df_list = [pd.read_csv(f) for f in csv_files]
df = pd.concat(df_list, ignore_index=True)

print("Columns in DataFrame:", list(df.columns))

# Standardize column names
df.columns = df.columns.str.strip().str.lower()

# Check necessary columns
expected_columns = ['max_temp', 'min_temp', 'avg_temp', 'rh_max', 'rh_min', 'rh_avg', 'l', 'dew_max', 'dew_min', 'dew_avg', 'wind_maxms', 'et0']
missing_cols = [col for col in expected_columns if col not in df.columns]
if missing_cols:
    print(f"Missing columns: {missing_cols}")
    exit(1)

# Drop missing data
df = df.dropna(subset=expected_columns)

# Feature selection and target
X = df[['max_temp', 'min_temp', 'avg_temp', 'rh_max', 'rh_min', 'rh_avg', 'l', 'dew_max', 'dew_min', 'dew_avg', 'wind_maxms']]
y = df['et0']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model trained. MSE on test set: {mse:.4f}")

# Save the model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"Model saved at {MODEL_PATH}")
