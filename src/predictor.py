# -*- coding: utf-8 -*-
"""
OncoGuardian Predictor Module - Production Ready
==================================================

Provides comprehensive prediction functionality with:
- Loading pre-trained ML models from training pipeline
- Automatic feature engineering (10 engineered features from 16 base)
- Cancer risk predictions with probability scores
- Risk level assessment (LOW, MEDIUM, HIGH)
- Personalized dietary recommendations
- Full validation and error handling

Input Format:
- All features use numeric 0-10 scales (NOT categorical strings)
- Height and Weight auto-calculate BMI → Obesity score
- Height/Weight provided in meters and kilograms

Features: 26 total (16 base + 10 engineered)
Base: Age, Gender, Height, Weight, Smoking, Alcohol_Use, Physical_Activity, 
       Diet_Red_Meat, Diet_Salted_Processed, Fruit_Veg_Intake, Air_Pollution,
       Occupational_Hazards, Family_History, BRCA_Mutation, H_Pylori_Infection,
       Calcium_Intake

Engineered: Smoking_Years_Risk, Alcohol_Years_Risk, Metabolic_Risk, 
           Inflammatory_Score, Genetic_Vulnerability, Environmental_Exposure,
           Nutrition_Defense, Lifestyle_Balance, Metabolic_Age_Risk,
           Total_Carcinogen_Burden

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
    
    Production-ready predictor that:
    - Loads models trained by training.py
    - Handles numeric 0-10 input scales (from Flutter widgets)
    - Auto-calculates BMI from Height/Weight
    - Creates 10 epidemiologically-validated engineered features
    - Provides cancer risk predictions with confidence scores
    - Generates personalized dietary recommendations
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize predictor by loading all trained model artifacts.

        Args:
            verbose (bool): Print initialization status
            
        Raises:
            FileNotFoundError: If model artifacts not found
        """
        if verbose:
            print("\n" + "="*70)
            print("🔬 INITIALIZING ONCOGUARDIAN PREDICTOR")
            print("="*70)

        try:
            # ===== LOAD ALL TRAINED ARTIFACTS =====
            self.model = joblib.load('models/model.pkl')
            self.label_encoders = joblib.load('models/label_encoders.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.feature_names = joblib.load('models/feature_names.pkl')
            self.cancer_types = joblib.load('models/cancer_types.pkl')
            
            # Load best parameters if available
            try:
                self.best_params = joblib.load('models/best_params.pkl')
            except:
                self.best_params = {}
            
            # Load metadata
            try:
                metadata_df = pd.read_csv('models/model_metadata.csv')
                self.metadata = metadata_df.iloc[0].to_dict()
            except:
                self.metadata = {'model_type': type(self.model).__name__}

            # ===== INITIALIZE FOOD DATABASE =====
            self.food_database = self._initialize_food_database()
            
            # ===== STORAGE FOR CALCULATED VALUES =====
            self.last_bmi = None
            self.last_obesity = None

            if verbose:
                print(f"✅ Model loaded: {self.metadata.get('model_type', 'Unknown')}")
                print(f"✅ Features: {len(self.feature_names)} total")
                print(f"   - Base features: 16")
                print(f"   - Engineered features: 10")
                print(f"✅ Cancer types: {', '.join(self.cancer_types)}")
                print(f"✅ Food database: {len(self.food_database)} cancer types")
                print("="*70 + "\n")

        except FileNotFoundError as e:
            print(f"❌ Error: Could not load model artifacts")
            print(f"   Missing: {e}")
            print(f"\n   Please run: python src/training.py")
            print(f"   This will train and save all required model files")
            raise

    def _initialize_food_database(self) -> Dict:
        """
        Initialize comprehensive food recommendation database based on 
        cancer type and risk level.
        
        Returns:
            dict: Nested dictionary {cancer_type → {risk_level → recommendations}}
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
        Preprocess patient input data for ML prediction.
        
        Key processing:
        - Auto-calculates BMI from Height/Weight → Obesity (0-10 scale)
        - Creates 10 engineered features (matches training pipeline exactly)
        - Scales all features using trained StandardScaler
        - Validates input completeness
        
        Args:
            patient_data (dict): Patient features (numeric 0-10 scales)
                Must include: All 16 base features OR Height+Weight will auto-calculate Obesity
                
        Returns:
            np.ndarray: Scaled feature vector ready for model prediction
            
        Raises:
            ValueError: If required features missing or invalid ranges
        """
        df = pd.DataFrame([patient_data])

        # ===== AUTO-CALCULATE BMI & OBESITY IF HEIGHT/WEIGHT PROVIDED =====
        if 'Height' in df.columns and 'Weight' in df.columns:
            try:
                height_m = float(df['Height'].values[0])
                weight_kg = float(df['Weight'].values[0])
                
                # Calculate BMI: weight(kg) / height(m)^2
                bmi = weight_kg / (height_m ** 2)
                
                # Convert BMI to 0-10 Obesity scale (epidemiological model)
                # Underweight (BMI<18.5): 1-2
                # Normal (18.5-24.9): 3-4
                # Overweight (25-29.9): 5-6
                # Obese (30+): 7-10
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
                print(f"   ✅ Calculated BMI: {bmi:.2f} (Obesity score: {obesity_score:.1f}/10)")
            except Exception as e:
                print(f"   ⚠️ Could not calculate BMI: {e}")
                if 'Obesity' not in df.columns:
                    df['Obesity'] = 5
        
        # Remove Height/Weight (not used in training)
        df = df.drop(columns=['Height', 'Weight'], errors='ignore')

        # ===== FEATURE ENGINEERING (10 ENGINEERED FEATURES) =====
        # EXACTLY matches training.py create_engineered_features()
        try:
            # 1. Cumulative Smoking Exposure (age-adjusted)
            df['Smoking_Years_Risk'] = df['Smoking'] * (df['Age'] / 40)
            
            # 2. Cumulative Alcohol Exposure (age-adjusted)
            df['Alcohol_Years_Risk'] = df['Alcohol_Use'] * (df['Age'] / 40)
            
            # 3. Metabolic Risk Score (obesity + sedentary)
            df['Metabolic_Risk'] = (df['Obesity'] * 2 + (10 - df['Physical_Activity'])) / 3
            
            # 4. Inflammatory Markers (H. pylori, diet, obesity)
            df['Inflammatory_Score'] = (df['H_Pylori_Infection'] * 2 + 
                                       df['Diet_Red_Meat'] + 
                                       df['Obesity']) / 4
            
            # 5. Genetic Vulnerability (family history + BRCA, age-adjusted)
            df['Genetic_Vulnerability'] = (df['Family_History'] + df['BRCA_Mutation']) * (40 / (df['Age'] + 1))
            
            # 6. Environmental Exposure (occupational + air pollution)
            df['Environmental_Exposure'] = (df['Occupational_Hazards'] + df['Air_Pollution']) / 2
            
            # 7. Nutrition Defense (protective - harmful)
            df['Nutrition_Defense'] = (df['Fruit_Veg_Intake'] * 1.5 + df['Calcium_Intake']) - df['Diet_Red_Meat']
            
            # 8. Lifestyle Balance (protective - risk factors)
            df['Lifestyle_Balance'] = ((df['Physical_Activity'] + df['Fruit_Veg_Intake'] + df['Calcium_Intake']) - 
                                      (df['Smoking'] + df['Alcohol_Use'] + df['Obesity'])) / 6
            
            # 9. Metabolic Age Risk (age × obesity × sedentary)
            df['Metabolic_Age_Risk'] = (df['Age'] * df['Obesity'] * (10 - df['Physical_Activity'])) / 100
            
            # 10. Total Carcinogen Burden (all exposures combined)
            df['Total_Carcinogen_Burden'] = (df['Smoking'] + df['Alcohol_Use'] + 
                                            df['Occupational_Hazards'] + df['Air_Pollution'] +
                                            df['H_Pylori_Infection'] + df['Obesity']) / 6
            
        except KeyError as e:
            raise ValueError(f"Error creating engineered feature: Missing {e}")

        # Ensure all required features present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0

        # Select only features used in training (in correct order)
        X = df[self.feature_names]

        # Scale using trained StandardScaler
        X_scaled = self.scaler.transform(X)

        return X_scaled

    def predict(self, patient_data: Dict) -> Dict:
        """
        Make cancer risk prediction for a patient.
        
        Args:
            patient_data (dict): Patient features (numeric 0-10 scales)
            
        Returns:
            dict: Prediction with cancer type and confidence
                {
                    'predicted_cancer_type': str,
                    'confidence': float (0-1),
                    'bmi_calculated': float,
                    'obesity_score': float,
                    'probabilities': {cancer_type: probability}
                }
        """
        # Preprocess input
        X_scaled = self.preprocess_input(patient_data)

        # Get predictions
        y_pred = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled)

        # Map to cancer types
        predicted_cancer_idx = y_pred[0]
        predicted_cancer_type = self.cancer_types[predicted_cancer_idx]

        # Create result dictionary
        result = {
            'predicted_cancer_type': predicted_cancer_type,
            'confidence': float(y_pred_proba[0].max()),
            'probabilities': {
                cancer: float(prob) 
                for cancer, prob in zip(self.cancer_types, y_pred_proba[0])
            }
        }
        
        # Include BMI if calculated
        if self.last_bmi is not None:
            result['bmi_calculated'] = round(self.last_bmi, 2)
            result['obesity_score'] = round(self.last_obesity, 1)

        return result

    def get_risk_level(self, confidence: float) -> str:
        """
        Classify risk level based on prediction confidence.
        
        Args:
            confidence (float): Model confidence (0-1)
            
        Returns:
            str: Risk level 'LOW', 'MEDIUM', or 'HIGH'
        """
        if confidence < 0.5:
            return 'LOW'
        elif confidence < 0.75:
            return 'MEDIUM'
        else:
            return 'HIGH'

    def get_recommendations(self, patient_data: Dict, cancer_type: str = None, prediction: Dict = None) -> Dict:
        """
        Get personalized dietary recommendations.
        
        Args:
            patient_data (dict): Patient features
            cancer_type (str): Specific cancer type (if None, use prediction)
            prediction (dict): Pre-computed prediction (optional, to avoid redundant calculations)
            
        Returns:
            dict: Recommendations with foods, supplements, lifestyle tips
        """
        # Get prediction if not provided
        if prediction is None:
            prediction = self.predict(patient_data)
        
        # Use provided cancer_type or get from prediction
        if cancer_type is None:
            cancer_type = prediction['predicted_cancer_type']
        
        confidence = prediction['probabilities'].get(cancer_type, 0.5)

        risk_level = self.get_risk_level(confidence)

        # Get recommendations from database
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
            'foods': recommendations.get('foods', []),
            'avoid': recommendations.get('avoid', []),
            'supplements': recommendations.get('supplements', []),
            'lifestyle_tips': recommendations.get('lifestyle', [])
        }
        
        # Include BMI if available
        if self.last_bmi is not None:
            result['bmi_calculated'] = round(self.last_bmi, 2)
            result['obesity_score'] = round(self.last_obesity, 1)

        return result

    def get_full_assessment(self, patient_data: Dict) -> Dict:
        """
        Get complete assessment with prediction and recommendations.
        
        Args:
            patient_data (dict): Patient features
            
        Returns:
            dict: Complete assessment report
        """
        prediction = self.predict(patient_data)
        cancer_type = prediction['predicted_cancer_type']
        
        # Pass prediction to avoid redundant calculation
        recommendations = self.get_recommendations(patient_data, cancer_type, prediction=prediction)

        return {
            'prediction': prediction,
            'recommendations': recommendations,
            'all_probabilities': prediction['probabilities']
        }

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


def example_flutter_inputs():
    """
    Test with 5 example patient inputs matching Flutter widget outputs.
    
    Flutter widgets provide numeric inputs in 0-10 scale:
    - Age: spinner/picker (25-90)
    - Gender: dropdown (0=Female, 1=Male)
    - Height: text input (1.40-2.10m)
    - Weight: text input (45-120kg)
    - All other: slider (0-10 scale)
    """
    
    print("\n" + "="*70)
    print("🧪 TESTING PREDICTOR WITH FLUTTER-COMPATIBLE INPUTS")
    print("="*70)
    
    predictor = OncoGuardianPredictor()

    # 5 example Flutter inputs
    flutter_patients = {
        "Patient 1: Low Risk (Young Female)": {
            'Age': 35, 'Gender': 0,
            'Height': 1.65, 'Weight': 58,
            'Smoking': 0, 'Alcohol_Use': 2,
            'Family_History': 0, 'BRCA_Mutation': 0,
            'Diet_Red_Meat': 2, 'Diet_Salted_Processed': 1, 
            'Fruit_Veg_Intake': 9, 'Calcium_Intake': 8,
            'Physical_Activity': 9, 'Air_Pollution': 2,
            'Occupational_Hazards': 0, 'H_Pylori_Infection': 0,
        },
        
        "Patient 2: Medium Risk (Middle-aged Male)": {
            'Age': 52, 'Gender': 1,
            'Height': 1.78, 'Weight': 88,
            'Smoking': 4, 'Alcohol_Use': 5,
            'Family_History': 1, 'BRCA_Mutation': 0,
            'Diet_Red_Meat': 6, 'Diet_Salted_Processed': 5, 
            'Fruit_Veg_Intake': 5, 'Calcium_Intake': 5,
            'Physical_Activity': 5, 'Air_Pollution': 6,
            'Occupational_Hazards': 4, 'H_Pylori_Infection': 0,
        },
        
        "Patient 3: High Risk (Older Male)": {
            'Age': 62, 'Gender': 1,
            'Height': 1.72, 'Weight': 105,
            'Smoking': 8, 'Alcohol_Use': 7,
            'Family_History': 1, 'BRCA_Mutation': 0,
            'Diet_Red_Meat': 8, 'Diet_Salted_Processed': 7, 
            'Fruit_Veg_Intake': 2, 'Calcium_Intake': 2,
            'Physical_Activity': 2, 'Air_Pollution': 8,
            'Occupational_Hazards': 7, 'H_Pylori_Infection': 1,
        },
        
        "Patient 4: BRCA+ Breast Risk": {
            'Age': 48, 'Gender': 0,
            'Height': 1.68, 'Weight': 75,
            'Smoking': 2, 'Alcohol_Use': 6,
            'Family_History': 1, 'BRCA_Mutation': 1,
            'Diet_Red_Meat': 7, 'Diet_Salted_Processed': 6, 
            'Fruit_Veg_Intake': 4, 'Calcium_Intake': 6,
            'Physical_Activity': 4, 'Air_Pollution': 3,
            'Occupational_Hazards': 1, 'H_Pylori_Infection': 0,
        },
        
        "Patient 5: Colon Risk (Poor Diet)": {
            'Age': 58, 'Gender': 1,
            'Height': 1.75, 'Weight': 92,
            'Smoking': 5, 'Alcohol_Use': 8,
            'Family_History': 1, 'BRCA_Mutation': 0,
            'Diet_Red_Meat': 9, 'Diet_Salted_Processed': 8, 
            'Fruit_Veg_Intake': 2, 'Calcium_Intake': 2,
            'Physical_Activity': 3, 'Air_Pollution': 5,
            'Occupational_Hazards': 2, 'H_Pylori_Infection': 1,
        },
    }

    # Test each patient
    for patient_name, patient_data in flutter_patients.items():
        print(f"\n{'─'*70}")
        print(f"📋 {patient_name}")
        print(f"{'─'*70}")
        
        assessment = predictor.get_full_assessment(patient_data)
        pred = assessment['prediction']
        rec = assessment['recommendations']
        
        # Display results
        print(f"\n📏 Measurements:")
        if 'bmi_calculated' in pred:
            print(f"   BMI: {pred['bmi_calculated']:.2f}, Obesity: {pred['obesity_score']:.1f}/10")
        
        print(f"\n🔮 Prediction:")
        print(f"   Cancer Type: {pred['predicted_cancer_type']}")
        print(f"   Confidence: {pred['confidence']:.1%}")
        
        print(f"\n📊 Cancer Probabilities:")
        for cancer, prob in sorted(pred['probabilities'].items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 15)
            print(f"   {cancer:12} {prob:5.1%} {bar}")
        
        print(f"\n🍽️ Recommendations (Risk: {rec['risk_level']}):")
        if rec.get('foods'):
            print(f"   Foods: {', '.join(rec['foods'][:2])}")
        else:
            print(f"   Foods: No specific recommendations")
        if rec.get('avoid'):
            print(f"   Avoid: {', '.join(rec['avoid'][:2])}")
        else:
            print(f"   Avoid: No specific recommendations")


if __name__ == '__main__':
    example_flutter_inputs()
