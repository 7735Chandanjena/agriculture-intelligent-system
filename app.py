# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify 
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import os


app = Flask(__name__)

# --- CONFIGURATION ---
# Ensure these filenames match your local environment
DATA_PATH = 'agri_intelligent_system_data.csv'
MODEL_REG_PATH = 'agri_yield_model.pkl'
MODEL_CLF_PATH = 'agri_crop_model.pkl'
SCALER_PATH = 'agri_full_scaler.pkl'

def load_agricultural_data():
    """
    Reads the existing CSV dataset and prepares it for the KNN engines.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"CRITICAL ERROR: {DATA_PATH} not found. Please ensure the CSV is in the same directory.")
    
    print(f"Loading agricultural data from {DATA_PATH}...")
    return pd.read_csv(DATA_PATH)

def train_and_persist_system():
    """
    Trains the KNN Classifier (Crop) and Regressor (Yield/Resources) 
    using the data loaded from the CSV.
    """
    df = load_agricultural_data()
    
    # Feature columns used for KNN distance calculations
    features = ['soil_moisture', 'avg_temp', 'humidity', 'soil_ph', 'rainfall_mm']
    X = df[features]
    
    # Scale features (Standardization is mandatory for KNN)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Engine 1: Crop Classification (Optimal planting window and crop type)
    clf = KNeighborsClassifier(n_neighbors=7, weights='distance')
    clf.fit(X_scaled, df['optimal_crop'])
    
    # Engine 2: Resource/Yield Prediction (Multi-output Regression)
    reg = KNeighborsRegressor(n_neighbors=5, weights='distance')
    # Predicting: Yield, Irrigation needed, and Fertilizer requirement
    target_cols = ['expected_yield_kg', 'irrigation_needed_liters', 'fertilizer_req_kg']
    reg.fit(X_scaled, df[target_cols])
    
    # Save artifacts for production use
    joblib.dump(clf, MODEL_CLF_PATH)
    joblib.dump(reg, MODEL_REG_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("Intelligent Agriculture Engines Trained Successfully from CSV.")

# Check if models exist; if not, train them using the CSV data
if not os.path.exists(MODEL_REG_PATH) or not os.path.exists(MODEL_CLF_PATH):
    train_and_persist_system()

# Load persistence layers into memory for low-latency inference
clf_engine = joblib.load(MODEL_CLF_PATH)
reg_engine = joblib.load(MODEL_REG_PATH)
scaler_engine = joblib.load(SCALER_PATH)
df_history = load_agricultural_data()

@app.route('/api/v1/recommend', methods=['POST'])
def recommend():
    """
    Inference endpoint: Takes real-time sensor data and returns 
    KNN-based recommendations.
    """
    try:
        req = request.json
        print(req)
        # Validate input parameters
        params = ['moisture', 'temp', 'humidity', 'ph', 'rainfall']
        if not all(p in req for p in params):
            return jsonify({"error": "Incomplete sensor data. Required: moisture, temp, humidity, ph, rainfall"}), 400
            
        # Format input for the model
        input_data = np.array([[req['moisture'], req['temp'], req['humidity'], req['ph'], req['rainfall']]])
        input_scaled = scaler_engine.transform(input_data)
        
        # Dual-Engine Inference
        optimal_crop = clf_engine.predict(input_scaled)[0]
        numeric_preds = reg_engine.predict(input_scaled)[0] # [Yield, Irrigation, Fertilizer]
        
        # Extract Historical Peers (Explainability)
        distances, indices = clf_engine.kneighbors(input_scaled)
        historical_matches = df_history.iloc[indices[0]].to_dict('records')
        
        return jsonify({
            "status": "success",
            "recommendation": {
                "optimal_crop": optimal_crop,
                "expected_yield": f"{round(numeric_preds[0], 2)} kg/acre",
                "irrigation_plan": f"{round(numeric_preds[1], 2)} Liters",
                "fertilization_strategy": f"{round(numeric_preds[2], 2)} kg/acre",
                "planting_window": "Optimal" if 20 <= req['temp'] <= 30 else "Sub-optimal (Temperature Warning)"
            },
            "historical_basis": historical_matches
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    """Serves the intelligent dashboard UI."""
    try:
        with open('index.html', 'r') as f:
            return f.read()
    except FileNotFoundError:
        return "Dashboard UI file (index.html) not found.", 404

if __name__ == '__main__':
    # Start the Flask production-ready server
    app.run(host='0.0.0.0', port=5050, debug=False)

# here we do not to clean the data because this data is clean data it self so here we are not doing any cleaning