import pandas as pd
import xgboost as xgb
import joblib
import os

# Load processed data
data = pd.read_csv('data/processed_data.csv')

# Assume the last column is the target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train XGBoost model
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X, y)

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the model
joblib.dump(model, 'models/champion_model.pkl')

print("✅ Champion model trained and saved as models/champion_model.pkl") 