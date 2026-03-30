"""
OncoGuardian Flask API Server
=============================

Production-ready REST API for cancer risk predictions.
Serves predictions from the trained sklearn model to Flutter mobile app.

Environment:
- Local: http://localhost:5000
- Firebase Cloud Functions: https://YOUR_PROJECT.cloudfunctions.net

Endpoints:
- POST /predict - Get cancer risk prediction
- POST /recommendations - Get dietary recommendations  
- GET /health - Health check
- GET /model-info - Model metadata

Author: OncoGuardian Team
Date: 2024
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from predictor import OncoGuardianPredictor
from datetime import datetime
import os
import json
import logging

# ===== SETUP =====
app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter requests

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize predictor (loads models once on startup)
try:
    predictor = OncoGuardianPredictor(verbose=True)
    PREDICTOR_LOADED = True
except Exception as e:
    logger.error(f"Failed to load predictor: {e}")
    PREDICTOR_LOADED = False

# ===== ERROR HANDLERS =====
def error_response(message, status_code=400):
    """Create standardized error response"""
    return jsonify({
        'success': False,
        'error': message,
        'timestamp': datetime.now().isoformat()
    }), status_code


# ===== ENDPOINTS =====

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify API is running.
    
    Response:
    {
        "status": "ok",
        "predictor_loaded": true,
        "timestamp": "2024-03-25T10:30:00.000000"
    }
    """
    return jsonify({
        'status': 'ok' if PREDICTOR_LOADED else 'predictor_error',
        'predictor_loaded': PREDICTOR_LOADED,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/model-info', methods=['GET'])
def get_model_info():
    """
    Get metadata about the underlying ML model.
    
    Response:
    {
        "model_type": "RandomForestClassifier",
        "cancer_types": ["Breast", "Lung", "Colorectal", "Prostate", "Pancreatic"],
        "num_features": 26,
        "feature_names": [...],
        "accuracy": 0.87,
        "timestamp": "2024-03-25T10:30:00.000000"
    }
    """
    if not PREDICTOR_LOADED:
        return error_response("Predictor not loaded", 503)
    
    try:
        return jsonify({
            'model_type': predictor.metadata.get('model_type', 'Unknown'),
            'cancer_types': predictor.cancer_types,
            'num_features': len(predictor.feature_names),
            'feature_names': predictor.feature_names,
            'accuracy': predictor.metadata.get('accuracy', None),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return error_response(f"Error fetching model info: {str(e)}", 500)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make a cancer risk prediction based on patient data.
    
    Request Body (JSON):
    {
        "Age": 45,
        "Gender": 1,
        "Height": 1.7,
        "Weight": 65,
        "Smoking": 0,
        "Alcohol_Use": 5,
        "Physical_Activity": 8,
        "Diet_Red_Meat": 3,
        "Diet_Salted_Processed": 2,
        "Fruit_Veg_Intake": 9,
        "Air_Pollution": 4,
        "Occupational_Hazards": 1,
        "Family_History": 0,
        "BRCA_Mutation": 0,
        "H_Pylori_Infection": 0,
        "Calcium_Intake": 7
    }
    
    Response:
    {
        "success": true,
        "prediction": {
            "predicted_cancer_type": "Breast",
            "risk_level": "HIGH",
            "confidence": 0.92,
            "probabilities": {
                "Breast": 0.92,
                "Lung": 0.05,
                "Colorectal": 0.02,
                "Prostate": 0.01,
                "Pancreatic": 0.00
            }
        },
        "timestamp": "2024-03-25T10:30:00.000000"
    }
    """
    if not PREDICTOR_LOADED:
        return error_response("Predictor not loaded", 503)
    
    try:
        # Get patient data from request
        patient_data = request.get_json()
        
        if not patient_data:
            return error_response("No JSON data provided", 400)
        
        # Validate required fields
        required_fields = [
            'Age', 'Gender', 'Height', 'Weight', 'Smoking', 'Alcohol_Use',
            'Physical_Activity', 'Diet_Red_Meat', 'Diet_Salted_Processed',
            'Fruit_Veg_Intake', 'Air_Pollution', 'Occupational_Hazards',
            'Family_History', 'BRCA_Mutation', 'H_Pylori_Infection', 'Calcium_Intake'
        ]
        
        missing_fields = [f for f in required_fields if f not in patient_data]
        if missing_fields:
            return error_response(
                f"Missing required fields: {', '.join(missing_fields)}", 
                400
            )
        
        # Make prediction
        prediction_dict = predictor.predict(patient_data)
        
        return jsonify({
            'success': True,
            'prediction': prediction_dict,
            'timestamp': datetime.now().isoformat()
        })
        
    except ValueError as e:
        return error_response(f"Invalid input data: {str(e)}", 400)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return error_response(f"Prediction failed: {str(e)}", 500)


@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    """
    Get personalized dietary recommendations based on prediction.
    
    Request Body (JSON):
    {
        "Age": 45,
        "Gender": 1,
        ... (same as predict),
        "cancer_type": "Breast"  // Optional: if not provided, uses top prediction
    }
    
    Response:
    {
        "success": true,
        "recommendations": {
            "risk_level": "HIGH",
            "cancer_type": "Breast",
            "recommended_foods": ["Broccoli", "Green tea", ...],
            "foods_to_avoid": ["Red meat", "Processed foods", ...],
            "supplements": ["Vitamin D", "Omega-3", ...],
            "lifestyle_tips": ["Exercise 30 mins daily", ...]
        },
        "timestamp": "2024-03-25T10:30:00.000000"
    }
    """
    if not PREDICTOR_LOADED:
        return error_response("Predictor not loaded", 503)
    
    try:
        data = request.get_json()
        
        if not data:
            return error_response("No JSON data provided", 400)
        
        # Get cancer type (use provided one or predict it)
        cancer_type = data.get('cancer_type')
        if not cancer_type:
            prediction_dict = predictor.predict(data)
            cancer_type = prediction_dict['predicted_cancer_type']
        
        # Get recommendations
        recommendations = predictor.get_recommendations(data, cancer_type)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Recommendations error: {e}")
        return error_response(f"Failed to generate recommendations: {str(e)}", 500)


@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """
    Make predictions for multiple patients at once.
    Useful for bulk analysis.
    
    Request Body (JSON):
    {
        "patients": [
            { "id": "P001", "Age": 45, ... },
            { "id": "P002", "Age": 52, ... }
        ]
    }
    
    Response:
    {
        "success": true,
        "predictions": [
            { "id": "P001", "prediction": {...} },
            { "id": "P002", "prediction": {...} }
        ],
        "failed": [],
        "timestamp": "2024-03-25T10:30:00.000000"
    }
    """
    if not PREDICTOR_LOADED:
        return error_response("Predictor not loaded", 503)
    
    try:
        data = request.get_json()
        patients = data.get('patients', [])
        
        if not patients:
            return error_response("No patients provided", 400)
        
        results = []
        failed = []
        
        for patient in patients:
            try:
                patient_id = patient.get('id', 'Unknown')
                prediction = predictor.predict(patient)
                results.append({
                    'id': patient_id,
                    'prediction': prediction
                })
            except Exception as e:
                failed.append({
                    'id': patient.get('id', 'Unknown'),
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'predictions': results,
            'failed': failed,
            'total': len(patients),
            'successful': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return error_response(f"Batch prediction failed: {str(e)}", 500)


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return error_response("Endpoint not found", 404)


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return error_response("Internal server error", 500)


# ===== RUNTIME =====
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'
    
    print("\n" + "="*70)
    print("🚀 OncoGuardian Flask API Server")
    print("="*70)
    print(f"port: {port}")
    print(f"debug: {debug}")
    print(f"URL: http://localhost:{port}")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)
