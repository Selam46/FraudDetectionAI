from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

file_handler = RotatingFileHandler(
    'logs/fraud_detection.log', 
    maxBytes=10485760,  # 10MB
    backupCount=10
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

# Load the model and scaler
try:
    model = joblib.load('../models/best_model_Gradient Boosting.joblib')
    scaler = joblib.load('../models/scaler.joblib')
    app.logger.info("Model and scaler loaded successfully")
except Exception as e:
    app.logger.error(f"Error loading model or scaler: {str(e)}")
    raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for fraud prediction"""
    try:
        # Get data from request
        data = request.get_json()
        app.logger.info(f"Received prediction request: {data}")

        # Validate input
        required_features = ['user_id', 'purchase_value', 'age', 'ip_address']
        if not all(feature in data for feature in required_features):
            app.logger.error("Missing required features in request")
            return jsonify({
                'error': 'Missing required features',
                'required_features': required_features
            }), 400

        # Prepare features
        features = np.array([
            data['user_id'],
            data['purchase_value'],
            data['age'],
            data['ip_address']
            # Add other features as needed
        ]).reshape(1, -1)

        # Scale features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict_proba(features_scaled)[0]
        fraud_probability = float(prediction[1])

        # Log prediction
        app.logger.info(
            f"Prediction made for user {data['user_id']}: {fraud_probability}"
        )

        # Return prediction
        return jsonify({
            'fraud_probability': fraud_probability,
            'timestamp': datetime.now().isoformat(),
            'user_id': data['user_id']
        })

    except Exception as e:
        app.logger.error(f"Error making prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 