import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

# Load all datasets
files = [
    "dataset_one_open_env.csv", "dataset_two_open_env.csv",
    "dataset_three_open_env.csv", "dataset_four_open_env.csv",
    "dataset_one_close_env.csv", "dataset_two_close_env.csv",
    "dataset_three_close_env.csv", "dataset_four_close_env.csv"
]

# Combine them
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Drop non-numeric and irrelevant columns
for col in df.columns:
    if 'date' in col.lower() or 'time' in col.lower():
        df.drop(columns=[col], inplace=True)
    if not pd.api.types.is_numeric_dtype(df[col]):
        df.drop(columns=[col], inplace=True)

# Drop extra unnamed/index columns if any
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Separate features and target
X = df.drop(columns=['et0'])
y = df['et0']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "eto_model.pkl")

# Evaluate
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
