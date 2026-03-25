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
from typing import Dict, List

warnings.filterwarnings('ignore')


class OncoGuardianPredictor:
    """
    OncoGuardian Cancer Risk Predictor with Food Recommendations
    
    This class loads trained models and provides:
    - Risk predictions for different cancer types
    - Risk level classification (LOW, MEDIUM, HIGH)
    - Personalized dietary recommendations
    - Food suggestions and restrictions
    - Feature engineering (10 derived features for accurate predictions)
    
    Note: The preprocess_input method automatically generates engineered features
    that match the training pipeline (Feature Engineering)
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
                        '🚫 Fried and deep-fried foods',
                        '🚫 Excessive salt and sodium',
                        '🚫 Alcohol consumption',
                        '🚫 Trans fats and hydrogenated oils'
                    ],
                    'diet_plan': 'Anti-inflammatory diet rich in antioxidants. Focus on colorful vegetables, omega-3 fatty acids, and limit exposure to airborne carcinogens. Consider Mediterranean diet pattern with emphasis on plant-based foods.',
                    'supplements': ['Vitamin D', 'Selenium', 'Green tea extract'],
                    'lifestyle': ['Avoid smoking and second-hand smoke', 'Regular exercise', 'Air purification at home']
                },
                'MEDIUM': {
                    'foods': [
                        '🥗 Colorful vegetables (all types)',
                        '🍊 Citrus fruits (vitamin C)',
                        '🥜 Nuts and seeds (vitamin E)',
                        '🐟 Fatty fish (2-3 times/week)',
                        '🌾 Whole grains (fiber)',
                        '🫘 Legumes (beans, lentils)',
                        '🍄 Mushrooms (beta-glucans)'
                    ],
                    'avoid': [
                        '🚫 Processed foods',
                        '🚫 Excessive red meat',
                        '🚫 Sugary snacks and beverages',
                        '🚫 Trans fats'
                    ],
                    'diet_plan': 'Immune-boosting diet with emphasis on vitamins C, E, and beta-carotene. Maintain 5+ servings of fruits/vegetables daily. Include antioxidant-rich foods and stay hydrated.',
                    'supplements': ['Vitamin C', 'Vitamin E', 'Zinc'],
                    'lifestyle': ['Regular exercise', 'Stress management', 'Adequate sleep']
                },
                'LOW': {
                    'foods': [
                        '🥗 Fresh fruits and vegetables',
                        '🍗 Lean proteins',
                        '🌾 Whole grains',
                        '🥑 Healthy fats (olive oil, avocado)',
                        '🍵 Herbal teas',
                        '💧 Plenty of water'
                    ],
                    'avoid': [
                        '🚫 Limit processed foods',
                        '🚫 Avoid smoking',
                        '🚫 Moderate alcohol'
                    ],
                    'diet_plan': 'Maintain healthy lifestyle with balanced nutrition. Focus on prevention through diet rich in antioxidants and regular physical activity. Annual check-ups recommended.',
                    'supplements': ['Multivitamin (optional)'],
                    'lifestyle': ['Regular exercise', 'No smoking', 'Limit alcohol', 'Stress management']
                }
            },
            'Breast': {
                'HIGH': {
                    'foods': [
                        '🥦 Cruciferous vegetables (broccoli, cauliflower, Brussels sprouts)',
                        '🍵 Green tea (EGCG compounds for breast health)',
                        '🟡 Turmeric with black pepper (curcumin)',
                        '🌰 Flaxseeds (ground, for lignans)',
                        '🍄 Mushrooms (shiitake, maitake)',
                        '🫐 Berries (blueberries, strawberries - ellagic acid)',
                        '🥕 Orange vegetables (beta-carotene)',
                        '🐟 Fatty fish (omega-3s)'
                    ],
                    'avoid': [
                        '🚫 Processed meats',
                        '🚫 Alcohol (even moderate amounts)',
                        '🚫 High-fat dairy products',
                        '🚫 Sugary beverages',
                        '🚫 Soy isoflavone supplements (whole soy foods OK)',
                        '🚫 Trans fats'
                    ],
                    'diet_plan': 'Mediterranean diet with emphasis on plant-based foods and omega-3 fatty acids. Focus on fiber-rich foods (25-30g/day). Maintain healthy body weight through diet and exercise.',
                    'supplements': ['Vitamin D', 'Omega-3', 'Calcium (if needed)'],
                    'lifestyle': ['Regular mammograms', 'Breast self-exams', 'Maintain healthy weight', 'Regular exercise']
                },
                'MEDIUM': {
                    'foods': [
                        '🥗 Colorful fruits and vegetables',
                        '🌾 Whole grains (oats, quinoa, brown rice)',
                        '🫘 Legumes (beans, lentils, chickpeas)',
                        '🍗 Lean proteins (chicken, turkey)',
                        '🥜 Nuts and seeds (especially walnuts)',
                        '🥦 Fermented foods (yogurt, kimchi, kefir)',
                        '🍵 Green tea'
                    ],
                    'avoid': [
                        '🚫 Processed foods',
                        '🚫 Excessive red meat',
                        '🚫 Fried foods',
                        '🚫 High-glycemic foods and sugars'
                    ],
                    'diet_plan': 'Balanced diet with 5+ servings of fruits/vegetables daily. Focus on fiber-rich foods and phytonutrients. Include fermented foods for gut health. Limit saturated fats.',
                    'supplements': ['Vitamin D', 'Calcium'],
                    'lifestyle': ['Monthly breast self-exams', 'Annual check-ups', 'Regular exercise', 'Limit alcohol']
                },
                'LOW': {
                    'foods': [
                        '🥗 Variety of fruits and vegetables',
                        '🌾 Whole grains',
                        '🥑 Healthy fats (olive oil, avocado)',
                        '🍗 Lean proteins',
                        '🍵 Green tea (optional)',
                        '💧 Water'
                    ],
                    'avoid': [
                        '🚫 Limit processed foods',
                        '🚫 Maintain healthy weight',
                        '🚫 Moderate alcohol (if any)'
                    ],
                    'diet_plan': 'Standard healthy diet for prevention with emphasis on phytonutrients. Maintain healthy lifestyle with regular physical activity and breast awareness.',
                    'supplements': ['Not necessary with balanced diet'],
                    'lifestyle': ['Regular breast awareness', 'Healthy weight', 'Exercise 150 min/week']
                }
            },
            'Colon': {
                'HIGH': {
                    'foods': [
                        '🌾 High-fiber foods (oats, barley, psyllium)',
                        '🫘 Legumes (beans, lentils, chickpeas)',
                        '🥦 Cruciferous vegetables',
                        '🧄 Garlic and onions (prebiotics)',
                        '🥬 Fermented foods (yogurt, kimchi, sauerkraut)',
                        '🥛 Calcium-rich foods (leafy greens, fortified plant milks)',
                        '🍎 Apples (pectin)',
                        '🌰 Brazil nuts (selenium)'
                    ],
                    'avoid': [
                        '🚫 Red meat (beef, pork, lamb)',
                        '🚫 Processed meats (bacon, sausage, ham)',
                        '🚫 Fried foods',
                        '🚫 Alcohol',
                        '🚫 Refined carbohydrates',
                        '🚫 Low-fiber foods'
                    ],
                    'diet_plan': 'High-fiber (30-35g/day), low-fat diet with emphasis on gut health and probiotics. Include prebiotic foods to feed healthy gut bacteria. Stay well-hydrated with fiber intake.',
                    'supplements': ['Calcium', 'Vitamin D', 'Probiotics'],
                    'lifestyle': ['Regular colonoscopy screening', 'Daily exercise', 'Maintain healthy weight', 'No smoking']
                },
                'MEDIUM': {
                    'foods': [
                        '🌾 Whole grains (at least 3 servings/day)',
                        '🥗 Fresh fruits and vegetables',
                        '🫘 Legumes',
                        '🐟 Fish (especially fatty fish)',
                        '🥜 Nuts and seeds',
                        '🥬 Fermented foods'
                    ],
                    'avoid': [
                        '🚫 Processed meats',
                        '🚫 Excessive red meat (limit to 1-2x/week)',
                        '🚫 Refined sugars',
                        '🚫 Trans fats'
                    ],
                    'diet_plan': 'Fiber-rich diet with emphasis on whole foods and gut microbiome support. Aim for 25-30g fiber daily. Include variety of plant foods for diverse gut bacteria.',
                    'supplements': ['Calcium', 'Vitamin D'],
                    'lifestyle': ['Regular screening after 45', 'Daily physical activity', 'Hydration']
                },
                'LOW': {
                    'foods': [
                        '🌾 High-fiber foods',
                        '🥗 Fruits and vegetables',
                        '🌾 Whole grains',
                        '🍗 Lean proteins',
                        '🥛 Probiotic foods (yogurt, kefir)',
                        '💧 Plenty of water'
                    ],
                    'avoid': [
                        '🚫 Limit red meat',
                        '🚫 Avoid processed foods',
                        '🚫 Stay hydrated'
                    ],
                    'diet_plan': 'Maintain high-fiber diet for colon health. Regular physical activity and adequate hydration. Follow recommended screening guidelines based on age.',
                    'supplements': ['None specific'],
                    'lifestyle': ['Regular exercise', 'Fiber-rich diet', 'Hydration', 'Screening per guidelines']
                }
            },
            'Prostate': {
                'HIGH': {
                    'foods': [
                        '🍅 Cooked tomatoes (with olive oil for lycopene absorption)',
                        '🍇 Pomegranate juice (ellagitannins)',
                        '🍵 Green tea (catechins)',
                        '🌱 Soy products (tofu, edamame, tempeh)',
                        '🥦 Broccoli and cauliflower (sulforaphane)',
                        '🌰 Brazil nuts (selenium)',
                        '🐟 Fatty fish (omega-3s)',
                        '🍄 Mushrooms'
                    ],
                    'avoid': [
                        '🚫 High-fat dairy products',
                        '🚫 Red meat',
                        '🚫 Processed foods',
                        '🚫 Excessive calcium supplements (>1500mg/day)',
                        '🚫 Saturated fats',
                        '🚫 Grilled/charred meats'
                    ],
                    'diet_plan': 'Low-fat, plant-based diet with lycopene-rich foods and omega-3 fatty acids. Include cooked tomatoes with healthy fat for optimal lycopene absorption. Limit dairy and calcium supplements.',
                    'supplements': ['Vitamin D', 'Selenium', 'Green tea extract'],
                    'lifestyle': ['Regular PSA testing as recommended', 'Exercise', 'Healthy weight', 'Limit calcium']
                },
                'MEDIUM': {
                    'foods': [
                        '🍅 Cooked tomatoes',
                        '🥦 Cruciferous vegetables',
                        '🐟 Fish rich in omega-3',
                        '🥜 Nuts and seeds',
                        '🫘 Legumes',
                        '🍵 Green tea',
                        '🌾 Whole grains'
                    ],
                    'avoid': [
                        '🚫 High-fat foods',
                        '🚫 Processed meats',
                        '🚫 Excessive dairy',
                        '🚫 Fried foods'
                    ],
                    'diet_plan': 'Balanced diet with emphasis on plant proteins and healthy fats. Include lycopene-rich foods several times per week. Maintain healthy weight through diet and exercise.',
                    'supplements': ['Vitamin D', 'Omega-3'],
                    'lifestyle': ['Regular exercise', 'Healthy weight', 'Limit alcohol', 'PSA discussion with doctor']
                },
                'LOW': {
                    'foods': [
                        '🥗 Variety of fruits and vegetables',
                        '🌾 Whole grains',
                        '🥑 Healthy fats',
                        '🍗 Lean proteins',
                        '🍅 Tomatoes',
                        '🍵 Green tea'
                    ],
                    'avoid': [
                        '🚫 Limit red meat',
                        '🚫 Maintain healthy weight',
                        '🚫 Regular exercise'
                    ],
                    'diet_plan': 'Standard healthy diet for prevention with focus on antioxidants. Regular physical activity and weight management. Discuss screening with healthcare provider.',
                    'supplements': ['None specific'],
                    'lifestyle': ['Regular exercise', 'Healthy diet', 'Screening discussion']
                }
            },
            'Skin': {
                'HIGH': {
                    'foods': [
                        '🥕 Orange/yellow vegetables (carrots, sweet potatoes, pumpkin)',
                        '🥬 Dark leafy greens (spinach, kale, collards)',
                        '🫐 Berries (blueberries, raspberries - ellagic acid)',
                        '🍵 Green tea (polyphenols)',
                        '🐟 Fatty fish (omega-3 for inflammation)',
                        '🥜 Nuts and seeds (vitamin E)',
                        '🍅 Tomatoes (lycopene)',
                        '🍊 Citrus fruits (vitamin C)'
                    ],
                    'avoid': [
                        '🚫 Processed foods',
                        '🚫 Excessive sugar',
                        '🚫 Fried foods',
                        '🚫 Excessive alcohol',
                        '🚫 Sun-sensitizing foods (for some - citrus before sun exposure)'
                    ],
                    'diet_plan': 'Antioxidant-rich diet with vitamins A, C, E and omega-3s for skin health. Focus on colorful plant foods for photoprotection. Always combine with sun protection measures.',
                    'supplements': ['Vitamin D', 'Vitamin C', 'Vitamin E', 'Omega-3'],
                    'lifestyle': ['Daily sunscreen SPF 30+', 'Avoid peak sun hours', 'Regular skin checks', 'No tanning beds']
                },
                'MEDIUM': {
                    'foods': [
                        '🥗 Colorful fruits and vegetables',
                        '🍊 Citrus fruits (vitamin C for collagen)',
                        '🥜 Nuts and seeds',
                        '🍵 Green tea',
                        '🌾 Whole grains',
                        '🐟 Fish'
                    ],
                    'avoid': [
                        '🚫 Processed foods',
                        '🚫 Excessive sugar',
                        '🚫 Trans fats'
                    ],
                    'diet_plan': 'Diet rich in antioxidants and vitamins for skin protection. Include vitamin C for collagen production and vitamin E for cell membrane protection.',
                    'supplements': ['Vitamin D'],
                    'lifestyle': ['Daily sun protection', 'Regular skin exams', 'Stay hydrated']
                },
                'LOW': {
                    'foods': [
                        '🥗 Fruits and vegetables',
                        '🥑 Healthy fats',
                        '🍗 Lean proteins',
                        '🌾 Whole grains',
                        '🍵 Green tea',
                        '💧 Water for hydration'
                    ],
                    'avoid': [
                        '🚫 Limit processed foods',
                        '🚫 Sun protection',
                        '🚫 Stay hydrated'
                    ],
                    'diet_plan': 'Maintain healthy diet with sun-protective nutrients. Continue sun-safe behaviors and regular skin self-exams.',
                    'supplements': ['None specific'],
                    'lifestyle': ['Sun protection', 'Skin self-exams', 'Hydration', 'Avoid tanning']
                }
            }
        }

    def preprocess_input(self, patient_data: Dict) -> np.ndarray:
        """
        Preprocess patient input data for prediction.
        Includes feature engineering to match training pipeline.

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

        # ===== FEATURE ENGINEERING =====
        # Create engineered features to match training pipeline
        try:
            # 1. Lifestyle Risk Score
            df['Lifestyle_Risk'] = (df['Smoking'] + df['Alcohol_Use'] + df['Obesity']) / 3
            
            # 2. Diet Quality Score
            df['Diet_Quality'] = (df['Fruit_Veg_Intake'] * 2 - df['Diet_Red_Meat'] - df['Diet_Salted_Processed']) / 4
            
            # 3. Environmental Risk Score
            df['Environmental_Risk'] = (df['Air_Pollution'] + df['Occupational_Hazards']) / 2
            
            # 4. Genetic Risk Score
            df['Genetic_Risk'] = df['Family_History'] + df['BRCA_Mutation']
            
            # 5. Activity-Obesity Ratio
            df['Activity_Obesity_Ratio'] = df['Physical_Activity'] / (df['Obesity'] + 1)
            
            # 6. Infection-Age Risk
            df['Infection_Age_Risk'] = df['H_Pylori_Infection'] * (df['Age'] / 50)
            
            # 7. Calcium-Diet Protection
            df['Calcium_Diet_Protection'] = df['Calcium_Intake'] * df['Diet_Quality']
            
            # 8. Age-Smoking Risk
            df['Age_Smoking_Risk'] = df['Age'] * df['Smoking'] / 10
            
            # 9. Gender-Genetic Risk
            df['Gender_Genetic_Risk'] = df['Gender'] * df['BRCA_Mutation']
            
            # 10. Protective Factors Score
            df['Protective_Factors'] = (df['Physical_Activity'] + df['Diet_Quality'] + df['Calcium_Intake']) / 3
            
        except KeyError as e:
            print(f"⚠️ Warning: Could not create engineered feature {e}. This may cause prediction errors.")

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

    # Example patient data (using numeric values - same format as training data)
    patient_data = {
        'Age': 45,                    # Age in years (25-90)
        'Gender': 0,                  # 0=Female, 1=Male
        'Smoking': 3,                 # 0-10 scale (3=Light smoker)
        'Alcohol_Use': 5,             # 0-10 scale (5=Moderate drinker)
        'Obesity': 4,                 # 0-10 scale
        'Family_History': 1,          # 0=No, 1=Yes
        'Diet_Red_Meat': 6,           # 0-10 scale (higher = more red meat)
        'Diet_Salted_Processed': 4,   # 0-10 scale
        'Fruit_Veg_Intake': 8,        # 0-10 scale (higher = more fruits/veggies)
        'Physical_Activity': 7,       # 0-10 scale (higher = more active)
        'Air_Pollution': 5,           # 0-10 scale
        'Occupational_Hazards': 3,    # 0-10 scale
        'BRCA_Mutation': 0,           # 0=No, 1=Yes
        'H_Pylori_Infection': 0,      # 0=No, 1=Yes
        'Calcium_Intake': 7,          # 0-10 scale (higher = more calcium)
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
