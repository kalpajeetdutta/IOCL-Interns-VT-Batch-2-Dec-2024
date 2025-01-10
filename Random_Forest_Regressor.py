import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime

# Loading Dataset
file_path = '/Users/kalpajeetdutta/Downloads/training_dataset_new.xlsx'
df = pd.read_excel(file_path)

# Data Cleaning (Data Preprocessing)
date_columns = ['Notif.date', 'Changed On', 'Req. start']
time_columns = ['Created at', 'Changed at', 'Notif. Time']

for col in date_columns:
    df[col] = pd.to_datetime(df[col].replace('-', pd.NA), errors='coerce')

for col in time_columns:
    df[col] = pd.to_timedelta(df[col].replace('-', pd.NA), errors='coerce')

df.dropna(subset=['Changed On', 'Notif.date'], inplace=True)

# Feature Engineering
df['Maintenance Days'] = (df['Changed On'] - df['Notif.date']).dt.total_seconds() / 86400
df['Maintenance Days'] = df['Maintenance Days'].round().astype(int)
df['Equipment Age (days)'] = (datetime.now() - df['Changed On']).dt.total_seconds() / 86400
df['Equipment Age (days)'] = df['Equipment Age (days)'].round().astype(int)

columns_to_drop = ['Order', 'Changed By', 'Created By', 'Reported by', 
                   'Plt for WorkCtr', 'User status', 'Cost Center',
                   'Created at', 'Changed On', 'Changed at', 'Notif. Time', 'Req. start']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# Separate Features and Target
X = df.drop(columns=['Maintenance Days'])
y = df['Maintenance Days']

categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

for col in categorical_columns:
    X[col] = X[col].astype(str)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Preprocessing and Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, random_state=42))
])

# Train the Model
pipeline.fit(X_train, y_train)

# Model Evaluation
y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Predict Maintenance Days for All Equipment
df['Predicted Maintenance Days'] = pipeline.predict(X)
df['Predicted Maintenance Days'] = df['Predicted Maintenance Days'].round().astype(int)
df['Predicted Maintenance Days'] = df['Predicted Maintenance Days'].clip(lower=0)

# Save to Output
output_path = '/Users/kalpajeetdutta/Downloads/regressive_maintenance_output_new.csv'
df.to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")