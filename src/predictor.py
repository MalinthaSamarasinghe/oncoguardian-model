# -*- coding: utf-8 -*-
"""
OncoGuardian Predictor Module
==============================

Provides prediction functionality and personalized dietary recommendations
based on cancer risk factors.

Classes:
    OncoGuardianPredictor: Main predictor class for risk assessment and recommendations

Author: OncoGuardian Team
Date: 2024
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')


class OncoGuardianPredictor:
    """
    OncoGuardian Cancer Risk Predictor with Food Recommendations
    
    This class loads trained models and provides:
    - Risk predictions for different cancer types
    - Risk level classification (LOW, MEDIUM, HIGH)
    - Personalized dietary recommendations
    - Food suggestions and restrictions
    """

    def __init__(self, model_path='models/model.pkl', verbose=True):
        """
        Initialize the predictor with trained models.

        Args:
            model_path (str): Path to the trained model
            verbose (bool): Print initialization status
        """
        if verbose:
            print("\n" + "="*60)
            print("🔬 INITIALIZING ONCOGUARDIAN PREDICTOR")
            print("="*60)

        try:
            # Load all artifacts
            self.model = joblib.load('models/model.pkl')
            self.label_encoders = joblib.load('models/label_encoders.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.feature_names = joblib.load('models/feature_names.pkl')
            self.cancer_types = joblib.load('models/cancer_types.pkl')

            # Load metadata if available
            try:
                metadata_df = pd.read_csv('models/model_metadata.csv')
                self.metadata = metadata_df.iloc[0].to_dict()
            except:
                self.metadata = {'model_type': type(self.model).__name__}

            # Initialize food database
            self.food_database = self._initialize_food_database()

            # Define categorical feature mappings
            self.categorical_mappings = {
                'Gender': {'Male': 1, 'Female': 2, 'Other': 3},
                'Smoking': {'Never': 1, 'Former': 2, 'Current': 3},
                'Alcohol_Use': {'None': 1, 'Moderate': 2, 'Heavy': 3},
                'Obesity': {'No': 1, 'Yes': 2},
                'Family_History': {'No': 1, 'Yes': 2},
                'Diet_Red_Meat': {'Low': 1, 'Medium': 2, 'High': 3},
                'Diet_Salted_Processed': {'Low': 1, 'Medium': 2, 'High': 3},
                'Fruit_Veg_Intake': {'Low': 1, 'Medium': 2, 'High': 3},
                'Physical_Activity': {'Sedentary': 1, 'Moderate': 2, 'High': 3},
                'Air_Pollution': {'Low': 1, 'Medium': 2, 'High': 3},
                'Occupational_Hazards': {'Low': 1, 'Medium': 2, 'High': 3},
                'BRCA_Mutation': {'No': 1, 'Yes': 2, 'Unknown': 1},
                'H_Pylori_Infection': {'No': 1, 'Yes': 2, 'Unknown': 1},
                'Calcium_Intake': {'Low': 1, 'Medium': 2, 'High': 3},
            }

            if verbose:
                print(f"✅ Model loaded: {self.metadata.get('model_type', 'Unknown')}")
                print(f"✅ Features: {len(self.feature_names)}")
                print(f"✅ Cancer types: {', '.join(self.cancer_types)}")
                print(f"✅ Food database: {len(self.food_database)} cancer types")
                print("="*60)

        except FileNotFoundError as e:
            print(f"❌ Error loading models: {e}")
            print("   Please run src/training.py first to train models")
            raise

    def _initialize_food_database(self) -> Dict:
        """
        Initialize comprehensive food recommendation database.
        Based on scientific research and dietary guidelines.
        
        Returns:
            dict: Food recommendations by cancer type and risk level
        """
        return {
            'Lung': {
                'HIGH': {
                    'foods': [
                        '🍎 Apples (rich in flavonoids)',
                        '🥦 Cruciferous vegetables (broccoli, kale, cauliflower)',
                        '🟡 Turmeric and ginger (anti-inflammatory)',
                        '🍵 Green tea (rich in EGCG antioxidants)',
                        '🧄 Garlic and onions (organosulfur compounds)',
                        '🐟 Fatty fish (salmon, sardines - omega-3)',
                        '🥕 Carrots and sweet potatoes (beta-carotene)',
                        '🌰 Brazil nuts (selenium)'
                    ],
                    'avoid': [
                        '🚫 Smoked and cured foods',
                        '🚫 Processed meats',
                        '🚫 Fried foods',
                        '🚫 Excessive salt',
                        '🚫 Alcohol',
                        '🚫 Trans fats'
                    ],
                    'supplements': ['Vitamin D', 'Selenium', 'Green tea extract'],
                    'lifestyle': ['Avoid smoking', 'Regular exercise', 'Air purification']
                },
                'MEDIUM': {
                    'foods': [
                        '🥗 Colorful vegetables',
                        '🍊 Citrus fruits (vitamin C)',
                        '🥜 Nuts and seeds',
                        '🐟 Fatty fish (2-3 times/week)',
                        '🌾 Whole grains',
                        '🫘 Legumes (beans, lentils)',
                        '🍄 Mushrooms'
                    ],
                    'avoid': [
                        '🚫 Processed foods',
                        '🚫 Excess red meat',
                        '🚫 Sugary snacks',
                        '🚫 Trans fats'
                    ],
                    'supplements': ['Vitamin C', 'Vitamin E', 'Zinc'],
                    'lifestyle': ['Regular exercise', 'Stress management', 'Adequate sleep']
                },
                'LOW': {
                    'foods': [
                        '🥗 Fresh fruits and vegetables',
                        '🍗 Lean proteins',
                        '🌾 Whole grains',
                        '🥑 Healthy fats',
                        '🍵 Herbal teas',
                        '💧 Water'
                    ],
                    'avoid': [
                        '🚫 Limit processed foods',
                        '🚫 Avoid smoking',
                        '🚫 Moderate alcohol'
                    ],
                    'supplements': ['Multivitamin (optional)'],
                    'lifestyle': ['Regular exercise', 'No smoking', 'Limit alcohol']
                }
            },
            'Breast': {
                'HIGH': {
                    'foods': [
                        '🥦 Cruciferous vegetables',
                        '🍵 Green tea',
                        '🟡 Turmeric with black pepper',
                        '🌰 Ground flaxseeds',
                        '🍄 Mushrooms (shiitake, maitake)',
                        '🫐 Berries',
                        '🥕 Orange vegetables',
                        '🐟 Fatty fish'
                    ],
                    'avoid': [
                        '🚫 Processed meats',
                        '🚫 Alcohol',
                        '🚫 High-fat dairy',
                        '🚫 Sugary drinks',
                        '🚫 Trans fats'
                    ],
                    'supplements': ['Vitamin D', 'Omega-3', 'Calcium'],
                    'lifestyle': ['Regular mammograms', 'Maintain healthy weight', 'Exercise']
                },
                'MEDIUM': {
                    'foods': [
                        '🥗 Leafy greens',
                        '🍎 Apples and berries',
                        '🥕 Carrots',
                        '🐟 Fish (2-3 times/week)',
                        '🌾 Whole grains',
                        '🫘 Legumes'
                    ],
                    'avoid': [
                        '🚫 Excess red meat',
                        '🚫 High-fat foods',
                        '🚫 Alcohol'
                    ],
                    'supplements': ['Vitamin D', 'Calcium'],
                    'lifestyle': ['Regular exercise', 'Maintain weight', 'Stress reduction']
                },
                'LOW': {
                    'foods': [
                        '🥗 Balanced diet',
                        '🍗 Lean proteins',
                        '🌾 Whole grains',
                        '🥑 Healthy fats',
                        '💧 Plenty of water'
                    ],
                    'avoid': [
                        '🚫 Maintain moderation'
                    ],
                    'supplements': ['Multivitamin'],
                    'lifestyle': ['Regular check-ups', 'Exercise', 'Healthy lifestyle']
                }
            }
        }

    def preprocess_input(self, patient_data: Dict) -> np.ndarray:
        """
        Preprocess patient input data for prediction.

        Args:
            patient_data (dict): Patient information with feature values

        Returns:
            np.ndarray: Preprocessed and scaled feature vector
        """
        # Create DataFrame with feature names
        df = pd.DataFrame([patient_data])

        # Encode categorical features
        for col in df.columns:
            if col in self.label_encoders and col != 'Cancer_Type':
                if col in self.categorical_mappings:
                    # Use dictionary mapping
                    df[col] = df[col].map(self.categorical_mappings[col])
                else:
                    # Use label encoder
                    le = self.label_encoders[col]
                    df[col] = le.transform(df[col].astype(str))

        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0

        # Select only required features
        X = df[self.feature_names]

        # Scale features
        X_scaled = self.scaler.transform(X)

        return X_scaled

    def predict(self, patient_data: Dict) -> Dict:
        """
        Predict cancer risk for a patient.

        Args:
            patient_data (dict): Patient information

        Returns:
            dict: Prediction results with probabilities for each cancer type
        """
        # Preprocess input
        X_scaled = self.preprocess_input(patient_data)

        # Get predictions
        y_pred = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled)

        # Map to cancer types
        predicted_cancer_type = self.cancer_types[y_pred[0]]

        # Create results dictionary
        result = {
            'predicted_cancer_type': predicted_cancer_type,
            'confidence': float(y_pred_proba[0].max()),
            'probabilities': {
                cancer: float(prob) 
                for cancer, prob in zip(self.cancer_types, y_pred_proba[0])
            }
        }

        return result

    def get_risk_level(self, confidence: float) -> str:
        """
        Classify risk level based on prediction confidence.

        Args:
            confidence (float): Model prediction confidence (0-1)

        Returns:
            str: Risk level ('LOW', 'MEDIUM', 'HIGH')
        """
        if confidence < 0.5:
            return 'LOW'
        elif confidence < 0.75:
            return 'MEDIUM'
        else:
            return 'HIGH'

    def get_recommendations(
        self, 
        patient_data: Dict, 
        cancer_type: str = None
    ) -> Dict:
        """
        Get personalized dietary recommendations.

        Args:
            patient_data (dict): Patient information
            cancer_type (str): Specific cancer type (if None, use prediction)

        Returns:
            dict: Personalized recommendations
        """
        # Get prediction if cancer type not specified
        if cancer_type is None:
            prediction = self.predict(patient_data)
            cancer_type = prediction['predicted_cancer_type']
            confidence = prediction['confidence']
        else:
            prediction = self.predict(patient_data)
            confidence = prediction['probabilities'].get(cancer_type, 0.5)

        # Determine risk level
        risk_level = self.get_risk_level(confidence)

        # Get food database for cancer type
        if cancer_type not in self.food_database:
            return {
                'cancer_type': cancer_type,
                'risk_level': risk_level,
                'confidence': float(confidence),
                'message': f'Limited recommendations available for {cancer_type}'
            }

        recommendations = self.food_database[cancer_type].get(risk_level, {})

        result = {
            'cancer_type': cancer_type,
            'risk_level': risk_level,
            'confidence': float(confidence),
            'recommended_foods': recommendations.get('foods', []),
            'foods_to_avoid': recommendations.get('avoid', []),
            'supplements': recommendations.get('supplements', []),
            'lifestyle_tips': recommendations.get('lifestyle', [])
        }

        return result

    def get_full_assessment(self, patient_data: Dict) -> Dict:
        """
        Get complete assessment including prediction and recommendations.

        Args:
            patient_data (dict): Patient information

        Returns:
            dict: Complete assessment report
        """
        prediction = self.predict(patient_data)
        cancer_type = prediction['predicted_cancer_type']
        
        recommendations = self.get_recommendations(
            patient_data, 
            cancer_type
        )

        assessment = {
            'prediction': prediction,
            'recommendations': recommendations,
            'all_probabilities': prediction['probabilities']
        }

        return assessment

    def batch_predict(self, patients_data: List[Dict]) -> List[Dict]:
        """
        Make predictions for multiple patients.

        Args:
            patients_data (list): List of patient data dictionaries

        Returns:
            list: List of prediction results
        """
        results = []
        for patient_data in patients_data:
            result = self.predict(patient_data)
            results.append(result)
        return results


def example_usage():
    """Example usage of the OncoGuardianPredictor."""
    
    # Initialize predictor
    predictor = OncoGuardianPredictor()

    # Example patient data
    patient_data = {
        'Age': 45,
        'Gender': 'Female',
        'Smoking': 'Never',
        'Alcohol_Use': 'Moderate',
        'Obesity': 'No',
        'Family_History': 'Yes',
        'Diet_Red_Meat': 'Low',
        'Diet_Salted_Processed': 'Low',
        'Fruit_Veg_Intake': 'High',
        'Physical_Activity': 'High',
        'Air_Pollution': 'Low',
        'Occupational_Hazards': 'Low',
        'BRCA_Mutation': 'No',
        'H_Pylori_Infection': 'No',
        'Calcium_Intake': 'High',
    }

    print("\n" + "="*60)
    print("EXAMPLE PREDICTION")
    print("="*60)

    # Get full assessment
    assessment = predictor.get_full_assessment(patient_data)

    print(f"\n🔮 Prediction Results:")
    print(f"   Predicted Cancer Type: {assessment['prediction']['predicted_cancer_type']}")
    print(f"   Confidence: {assessment['prediction']['confidence']:.2%}")

    print(f"\n📊 All Cancer Type Probabilities:")
    for cancer, prob in assessment['all_probabilities'].items():
        print(f"   {cancer}: {prob:.2%}")

    print(f"\n🍽️ Recommendations:")
    rec = assessment['recommendations']
    print(f"   Risk Level: {rec['risk_level']}")
    print(f"   Recommended Foods:")
    for food in rec['recommended_foods'][:5]:
        print(f"      {food}")
    print(f"\n   Foods to Avoid:")
    for food in rec['foods_to_avoid'][:3]:
        print(f"      {food}")
    print(f"\n   Supplements:")
    for supp in rec['supplements']:
        print(f"      • {supp}")
    print(f"\n   Lifestyle Tips:")
    for tip in rec['lifestyle_tips']:
        print(f"      • {tip}")


if __name__ == '__main__':
    example_usage()
