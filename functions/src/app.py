"""
OncoGuardian Flask API Server
=============================

Production-ready REST API for cancer risk predictions.
Serves predictions from the trained sklearn model to Flutter mobile app.

Environment:
- Local: http://localhost:5050
- Firebase Cloud Functions: https://YOUR_PROJECT.cloudfunctions.net

Supported Cancer Types:
- Breast, Colon, Lung, Prostate, Skin

Endpoints:
- GET /health - Health check
- GET /model-info - Model metadata
- POST /predict - Get cancer risk prediction + dietary recommendations (combined)

Author: OncoGuardian Team
Date: 2024
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import os
import logging
import pandas as pd
import numpy as np
import joblib
from typing import Dict

# ===== SETUP =====
app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter requests

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== PREDICTOR CLASS =====
class OncoGuardianPredictor:
    """
    OncoGuardian Cancer Risk Predictor
    Loads trained models from training.py and provides predictions with recommendations
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize predictor by loading trained model artifacts."""
        if verbose:
            print("\n" + "="*70)
            print("🔬 INITIALIZING ONCOGUARDIAN PREDICTOR")
            print("="*70)
        
        try:
            # Load trained artifacts from training.py
            self.model = joblib.load('models/model.pkl')
            self.label_encoders = joblib.load('models/label_encoders.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.feature_names = joblib.load('models/feature_names.pkl')
            self.cancer_types = joblib.load('models/cancer_types.pkl')
            
            try:
                self.best_params = joblib.load('models/best_params.pkl')
            except:
                self.best_params = {}
            
            try:
                metadata_df = pd.read_csv('models/model_metadata.csv')
                self.metadata = metadata_df.iloc[0].to_dict()
            except:
                self.metadata = {'model_type': type(self.model).__name__}
            
            # Initialize food database
            self.food_database = self._initialize_food_database()
            self.last_bmi = None
            self.last_obesity = None
            
            if verbose:
                print(f"✅ Model loaded: {self.metadata.get('model_type', 'Unknown')}")
                print(f"✅ Features: {len(self.feature_names)} total")
                print(f"✅ Cancer types: {', '.join(self.cancer_types)}")
                print("="*70 + "\n")
        
        except FileNotFoundError as e:
            logger.error(f"Failed to load model artifacts: {e}")
            raise
    
    def _initialize_food_database(self) -> Dict:
        """Initialize comprehensive food recommendation database based on cancer type and risk level."""
        return {
            'Lung': {
                'HIGH': {
                    'foods': ['🍎 Apples (rich in flavonoids)', '🥦 Cruciferous vegetables (broccoli, kale, cauliflower)', '🟡 Turmeric and ginger (anti-inflammatory)', '🍵 Green tea (rich in EGCG antioxidants)', '🧄 Garlic and onions (organosulfur compounds)', '🐟 Fatty fish (salmon, sardines - omega-3)', '🥕 Carrots and sweet potatoes (beta-carotene)', '🌰 Brazil nuts (selenium)'],
                    'avoid': ['🚫 Smoked and cured foods', '🚫 Processed meats', '🚫 Fried and deep-fried foods', '🚫 Excessive salt and sodium', '🚫 Alcohol consumption', '🚫 Trans fats and hydrogenated oils'],
                    'diet_plan': 'Anti-inflammatory diet rich in antioxidants. Focus on colorful vegetables, omega-3 fatty acids, and limit exposure to airborne carcinogens. Consider Mediterranean diet pattern with emphasis on plant-based foods.',
                    'supplements': ['Vitamin D', 'Selenium', 'Green tea extract'],
                    'lifestyle': ['Avoid smoking and second-hand smoke', 'Regular exercise', 'Air purification at home']
                },
                'MEDIUM': {
                    'foods': ['🥗 Colorful vegetables (all types)', '🍊 Citrus fruits (vitamin C)', '🥜 Nuts and seeds (vitamin E)', '🐟 Fatty fish (2-3 times/week)', '🌾 Whole grains (fiber)', '🫘 Legumes (beans, lentils)', '🍄 Mushrooms (beta-glucans)'],
                    'avoid': ['🚫 Processed foods', '🚫 Excessive red meat', '🚫 Sugary snacks and beverages', '🚫 Trans fats'],
                    'diet_plan': 'Immune-boosting diet with emphasis on vitamins C, E, and beta-carotene. Maintain 5+ servings of fruits/vegetables daily. Include antioxidant-rich foods and stay hydrated.',
                    'supplements': ['Vitamin C', 'Vitamin E', 'Zinc'],
                    'lifestyle': ['Regular exercise', 'Stress management', 'Adequate sleep']
                },
                'LOW': {
                    'foods': ['🥗 Fresh fruits and vegetables', '🍗 Lean proteins', '🌾 Whole grains', '🥑 Healthy fats (olive oil, avocado)', '🍵 Herbal teas', '💧 Plenty of water'],
                    'avoid': ['🚫 Limit processed foods', '🚫 Avoid smoking', '🚫 Moderate alcohol'],
                    'diet_plan': 'Maintain healthy lifestyle with balanced nutrition. Focus on prevention through diet rich in antioxidants and regular physical activity. Annual check-ups recommended.',
                    'supplements': ['Multivitamin (optional)'],
                    'lifestyle': ['Regular exercise', 'No smoking', 'Limit alcohol', 'Stress management']
                }
            },
            'Breast': {
                'HIGH': {
                    'foods': ['🥦 Cruciferous vegetables (broccoli, cauliflower, Brussels sprouts)', '🍵 Green tea (EGCG compounds for breast health)', '🟡 Turmeric with black pepper (curcumin)', '🌰 Flaxseeds (ground, for lignans)', '🍄 Mushrooms (shiitake, maitake)', '🫐 Berries (blueberries, strawberries - ellagic acid)', '🥕 Orange vegetables (beta-carotene)', '🐟 Fatty fish (omega-3s)'],
                    'avoid': ['🚫 Processed meats', '🚫 Alcohol (even moderate amounts)', '🚫 High-fat dairy products', '🚫 Sugary beverages', '🚫 Soy isoflavone supplements (whole soy foods OK)', '🚫 Trans fats'],
                    'diet_plan': 'Mediterranean diet with emphasis on plant-based foods and omega-3 fatty acids. Focus on fiber-rich foods (25-30g/day). Maintain healthy body weight through diet and exercise.',
                    'supplements': ['Vitamin D', 'Omega-3', 'Calcium (if needed)'],
                    'lifestyle': ['Regular mammograms', 'Breast self-exams', 'Maintain healthy weight', 'Regular exercise']
                },
                'MEDIUM': {
                    'foods': ['🥗 Colorful fruits and vegetables', '🌾 Whole grains (oats, quinoa, brown rice)', '🫘 Legumes (beans, lentils, chickpeas)', '🍗 Lean proteins (chicken, turkey)', '🥜 Nuts and seeds (especially walnuts)', '🥦 Fermented foods (yogurt, kimchi, kefir)', '🍵 Green tea'],
                    'avoid': ['🚫 Processed foods', '🚫 Excessive red meat', '🚫 Fried foods', '🚫 High-glycemic foods and sugars'],
                    'diet_plan': 'Balanced diet with 5+ servings of fruits/vegetables daily. Focus on fiber-rich foods and phytonutrients. Include fermented foods for gut health. Limit saturated fats.',
                    'supplements': ['Vitamin D', 'Calcium'],
                    'lifestyle': ['Monthly breast self-exams', 'Annual check-ups', 'Regular exercise', 'Limit alcohol']
                },
                'LOW': {
                    'foods': ['🥗 Variety of fruits and vegetables', '🌾 Whole grains', '🥑 Healthy fats (olive oil, avocado)', '🍗 Lean proteins', '🍵 Green tea (optional)', '💧 Water'],
                    'avoid': ['🚫 Limit processed foods', '🚫 Maintain healthy weight', '🚫 Moderate alcohol (if any)'],
                    'diet_plan': 'Standard healthy diet for prevention with emphasis on phytonutrients. Maintain healthy lifestyle with regular physical activity and breast awareness.',
                    'supplements': ['Not necessary with balanced diet'],
                    'lifestyle': ['Regular breast awareness', 'Healthy weight', 'Exercise 150 min/week']
                }
            },
            'Colon': {
                'HIGH': {
                    'foods': ['🌾 High-fiber foods (oats, barley, psyllium)', '🫘 Legumes (beans, lentils, chickpeas)', '🥦 Cruciferous vegetables', '🧄 Garlic and onions (prebiotics)', '🥬 Fermented foods (yogurt, kimchi, sauerkraut)', '🥛 Calcium-rich foods (leafy greens, fortified plant milks)', '🍎 Apples (pectin)', '🌰 Brazil nuts (selenium)'],
                    'avoid': ['🚫 Red meat (beef, pork, lamb)', '🚫 Processed meats (bacon, sausage, ham)', '🚫 Fried foods', '🚫 Alcohol', '🚫 Refined carbohydrates', '🚫 Low-fiber foods'],
                    'diet_plan': 'High-fiber (30-35g/day), low-fat diet with emphasis on gut health and probiotics. Include prebiotic foods to feed healthy gut bacteria. Stay well-hydrated with fiber intake.',
                    'supplements': ['Calcium', 'Vitamin D', 'Probiotics'],
                    'lifestyle': ['Regular colonoscopy screening', 'Daily exercise', 'Maintain healthy weight', 'No smoking']
                },
                'MEDIUM': {
                    'foods': ['🌾 Whole grains (at least 3 servings/day)', '🥗 Fresh fruits and vegetables', '🫘 Legumes', '🐟 Fish (especially fatty fish)', '🥜 Nuts and seeds', '🥬 Fermented foods'],
                    'avoid': ['🚫 Processed meats', '🚫 Excessive red meat (limit to 1-2x/week)', '🚫 Refined sugars', '🚫 Trans fats'],
                    'diet_plan': 'Fiber-rich diet with emphasis on whole foods and gut microbiome support. Aim for 25-30g fiber daily. Include variety of plant foods for diverse gut bacteria.',
                    'supplements': ['Calcium', 'Vitamin D'],
                    'lifestyle': ['Regular screening after 45', 'Daily physical activity', 'Hydration']
                },
                'LOW': {
                    'foods': ['🌾 High-fiber foods', '🥗 Fruits and vegetables', '🌾 Whole grains', '🍗 Lean proteins', '🥛 Probiotic foods (yogurt, kefir)', '💧 Plenty of water'],
                    'avoid': ['🚫 Limit red meat', '🚫 Avoid processed foods', '🚫 Stay hydrated'],
                    'diet_plan': 'Maintain high-fiber diet for colon health. Regular physical activity and adequate hydration. Follow recommended screening guidelines based on age.',
                    'supplements': ['None specific'],
                    'lifestyle': ['Regular exercise', 'Fiber-rich diet', 'Hydration', 'Screening per guidelines']
                }
            },
            'Prostate': {
                'HIGH': {
                    'foods': ['🍅 Cooked tomatoes (with olive oil for lycopene absorption)', '🍇 Pomegranate juice (ellagitannins)', '🍵 Green tea (catechins)', '🌱 Soy products (tofu, edamame, tempeh)', '🥦 Broccoli and cauliflower (sulforaphane)', '🌰 Brazil nuts (selenium)', '🐟 Fatty fish (omega-3s)', '🍄 Mushrooms'],
                    'avoid': ['🚫 High-fat dairy products', '🚫 Red meat', '🚫 Processed foods', '🚫 Excessive calcium supplements (>1500mg/day)', '🚫 Saturated fats', '🚫 Grilled/charred meats'],
                    'diet_plan': 'Low-fat, plant-based diet with lycopene-rich foods and omega-3 fatty acids. Include cooked tomatoes with healthy fat for optimal lycopene absorption. Limit dairy and calcium supplements.',
                    'supplements': ['Vitamin D', 'Selenium', 'Green tea extract'],
                    'lifestyle': ['Regular PSA testing as recommended', 'Exercise', 'Healthy weight', 'Limit calcium']
                },
                'MEDIUM': {
                    'foods': ['🍅 Cooked tomatoes', '🥦 Cruciferous vegetables', '🐟 Fish rich in omega-3', '🥜 Nuts and seeds', '🫘 Legumes', '🍵 Green tea', '🌾 Whole grains'],
                    'avoid': ['🚫 High-fat foods', '🚫 Processed meats', '🚫 Excessive dairy', '🚫 Fried foods'],
                    'diet_plan': 'Balanced diet with emphasis on plant proteins and healthy fats. Include lycopene-rich foods several times per week. Maintain healthy weight through diet and exercise.',
                    'supplements': ['Vitamin D', 'Omega-3'],
                    'lifestyle': ['Regular exercise', 'Healthy weight', 'Limit alcohol', 'PSA discussion with doctor']
                },
                'LOW': {
                    'foods': ['🥗 Variety of fruits and vegetables', '🌾 Whole grains', '🥑 Healthy fats', '🍗 Lean proteins', '🍅 Tomatoes', '🍵 Green tea'],
                    'avoid': ['🚫 Limit red meat', '🚫 Maintain healthy weight', '🚫 Regular exercise'],
                    'diet_plan': 'Standard healthy diet for prevention with focus on antioxidants. Regular physical activity and weight management. Discuss screening with healthcare provider.',
                    'supplements': ['None specific'],
                    'lifestyle': ['Regular exercise', 'Healthy diet', 'Screening discussion']
                }
            },
            'Skin': {
                'HIGH': {
                    'foods': ['🥕 Orange/yellow vegetables (carrots, sweet potatoes, pumpkin)', '🥬 Dark leafy greens (spinach, kale, collards)', '🫐 Berries (blueberries, raspberries)', '🍵 Green tea (polyphenols)', '🐟 Fatty fish (omega-3s)', '🥜 Nuts and seeds (vitamin E)', '🍅 Tomatoes (lycopene)', '🍊 Citrus fruits (vitamin C)'],
                    'avoid': ['🚫 Processed foods', '🚫 Excessive sugar', '🚫 Fried foods', '🚫 Excessive alcohol', '🚫 High-fat foods'],
                    'diet_plan': 'Antioxidant-rich diet with vitamins A, C, E and omega-3s for health. Focus on colorful plant foods. Maintain healthy weight and adequate hydration.',
                    'supplements': ['Vitamin D', 'Antioxidants', 'Probiotics'],
                    'lifestyle': ['Maintain healthy weight', 'Regular exercise', 'No smoking', 'Diabetes prevention']
                },
                'MEDIUM': {
                    'foods': ['🥗 Colorful fruits and vegetables', '🌾 Whole grains', '🍗 Lean proteins', '🥑 Healthy fats', '🫘 Legumes'],
                    'avoid': ['🚫 Processed foods', '🚫 Sugary foods', '🚫 Fatty foods', '🚫 Alcohol'],
                    'diet_plan': 'Balanced diet with emphasis on whole foods. Maintain healthy body weight and monitor blood sugar levels.',
                    'supplements': ['Vitamin D'],
                    'lifestyle': ['Regular exercise', 'Healthy weight', 'Monitor blood sugar']
                },
                'LOW': {
                    'foods': ['🥗 Fruits and vegetables', '🌾 Whole grains', '🍗 Lean proteins', '🥑 Healthy fats', '💧 Water'],
                    'avoid': ['🚫 Limit processed foods', '🚫 Maintain portions', '🚫 Moderation in all foods'],
                    'diet_plan': 'Maintain healthy diet and active lifestyle for prevention. Regular physical activity and weight management.',
                    'supplements': ['None specific'],
                    'lifestyle': ['Active lifestyle', 'Healthy eating', 'Annual check-ups']
                }
            }
        }
    
    def preprocess_input(self, patient_data: Dict) -> np.ndarray:
        """Preprocess patient input data with feature engineering."""
        df = pd.DataFrame([patient_data])
        
        # Auto-calculate BMI & Obesity from Height/Weight
        if 'Height' in df.columns and 'Weight' in df.columns:
            try:
                height_m = float(df['Height'].values[0])
                weight_kg = float(df['Weight'].values[0])
                
                bmi = weight_kg / (height_m ** 2)
                
                if bmi < 18.5:
                    obesity_score = 2
                elif bmi < 25:
                    obesity_score = 4
                elif bmi < 30:
                    obesity_score = 6
                else:
                    obesity_score = min(10, 7 + (bmi - 30) / 5)
                
                df['Obesity'] = obesity_score
                self.last_bmi = bmi
                self.last_obesity = obesity_score
            except Exception as e:
                if 'Obesity' not in df.columns:
                    df['Obesity'] = 5
        
        df = df.drop(columns=['Height', 'Weight'], errors='ignore')
        
        # Feature engineering (matches training.py exactly)
        df['Smoking_Years_Risk'] = df['Smoking'] * (df['Age'] / 40)
        df['Alcohol_Years_Risk'] = df['Alcohol_Use'] * (df['Age'] / 40)
        df['Metabolic_Risk'] = (df['Obesity'] * 2 + (10 - df['Physical_Activity'])) / 3
        df['Inflammatory_Score'] = (df['H_Pylori_Infection'] * 2 + df['Diet_Red_Meat'] + df['Obesity']) / 4
        df['Genetic_Vulnerability'] = (df['Family_History'] + df['BRCA_Mutation']) * (40 / (df['Age'] + 1))
        df['Environmental_Exposure'] = (df['Occupational_Hazards'] + df['Air_Pollution']) / 2
        df['Nutrition_Defense'] = (df['Fruit_Veg_Intake'] * 1.5 + df['Calcium_Intake']) - df['Diet_Red_Meat']
        df['Lifestyle_Balance'] = ((df['Physical_Activity'] + df['Fruit_Veg_Intake'] + df['Calcium_Intake']) - (df['Smoking'] + df['Alcohol_Use'] + df['Obesity'])) / 6
        df['Metabolic_Age_Risk'] = (df['Age'] * df['Obesity'] * (10 - df['Physical_Activity'])) / 100
        df['Total_Carcinogen_Burden'] = (df['Smoking'] + df['Alcohol_Use'] + df['Occupational_Hazards'] + df['Air_Pollution'] + df['H_Pylori_Infection'] + df['Obesity']) / 6
        
        # Ensure all required features
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        X = df[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def get_risk_level(self, confidence: float) -> str:
        """Classify risk level based on confidence."""
        if confidence < 0.5:
            return 'LOW'
        elif confidence < 0.75:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def predict(self, patient_data: Dict) -> Dict:
        """Make cancer risk prediction."""
        X_scaled = self.preprocess_input(patient_data)
        
        y_pred = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled)
        
        predicted_cancer_idx = y_pred[0]
        predicted_cancer_type = self.cancer_types[predicted_cancer_idx]
        confidence = float(y_pred_proba[0].max())
        
        result = {
            'predicted_cancer_type': predicted_cancer_type,
            'risk_level': self.get_risk_level(confidence),
            'confidence': confidence,
            'probabilities': {
                cancer: float(prob) 
                for cancer, prob in zip(self.cancer_types, y_pred_proba[0])
            }
        }
        
        if self.last_bmi is not None:
            result['bmi_calculated'] = round(self.last_bmi, 2)
            result['obesity_score'] = round(self.last_obesity, 1)
        
        return result
    
    def get_recommendations(self, patient_data: Dict, cancer_type: str = None) -> Dict:
        """Get personalized dietary recommendations."""
        prediction = self.predict(patient_data)
        
        if cancer_type is None:
            cancer_type = prediction['predicted_cancer_type']
        
        confidence = prediction['probabilities'].get(cancer_type, 0.5)
        risk_level = self.get_risk_level(confidence)
        
        if cancer_type not in self.food_database:
            return {
                'cancer_type': cancer_type,
                'risk_level': risk_level,
                'confidence': float(confidence),
                'message': f'No recommendations available for {cancer_type}'
            }
        
        recommendations = self.food_database[cancer_type].get(risk_level, {})
        
        result = {
            'cancer_type': cancer_type,
            'risk_level': risk_level,
            'confidence': float(confidence),
            'recommended_foods': recommendations.get('foods', []),
            'foods_to_avoid': recommendations.get('avoid', []),
            'diet_plan': recommendations.get('diet_plan', ''),
            'supplements': recommendations.get('supplements', []),
            'lifestyle_tips': recommendations.get('lifestyle', [])
        }
        
        return result


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


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return error_response("Endpoint not found", 404)


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return error_response("Internal server error", 500)


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
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return error_response(f"Error fetching model info: {str(e)}", 500)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make a cancer risk prediction and get personalized recommendations.
    
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
            "bmi_calculated": 22.49,
            "predicted_cancer_type": "Breast",
            "risk_level": "HIGH",
            "confidence": 0.92,
            "obesity_score": 4,
            "probabilities": {
                "Breast": 0.04,
                "Colon": 0.12,
                "Lung": 0.05,
                "Prostate": 0.54,
                "Skin": 0.22
            }
        },
        "recommendations": {
            "cancer_type": "Breast",
            "risk_level": "HIGH",
            "confidence": 0.92,
            "recommended_foods": ["Broccoli", "Green tea", ...],
            "foods_to_avoid": ["Red meat", "Processed foods", ...],
            "diet_plan": "Mediterranean diet with emphasis on...",
            "supplements": ["Vitamin D", "Omega-3", ...],
            "lifestyle_tips": ["Exercise 30 mins daily", ...]
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
        cancer_type = prediction_dict['predicted_cancer_type']
        
        # Get recommendations
        recommendations = predictor.get_recommendations(patient_data, cancer_type)
        
        return jsonify({
            'success': True,
            'prediction': prediction_dict,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        })
        
    except ValueError as e:
        return error_response(f"Invalid input data: {str(e)}", 400)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return error_response(f"Prediction failed: {str(e)}", 500)


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
