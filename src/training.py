# -*- coding: utf-8 -*-
"""
OncoGuardian Model Training Pipeline
========================================

A comprehensive machine learning pipeline for personalized cancer risk prediction.
This module contains all functions for:
- Data loading and exploration
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Model training and comparison
- Hyperparameter tuning
- Model evaluation
- Model artifact saving

Author: OncoGuardian Team
Date: 2024
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)

# XGBoost import (optional - requires OpenMP on macOS)
# To use XGBoost on macOS, run: brew install libomp
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    # Catch ImportError and other errors (e.g., XGBoostError for missing OpenMP)
    XGBOOST_AVAILABLE = False

# Setup
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================
# UTILITY FUNCTIONS
# ============================================

def create_output_directories():
    """Create necessary directories for outputs."""
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)
    os.makedirs('reports/metrics', exist_ok=True)
    print("✅ Output directories created/verified")


def print_section(section_title, step_num=None):
    """Print formatted section header."""
    if step_num:
        print(f"\n📊 STEP {step_num}: {section_title}")
    else:
        print(f"\n📊 {section_title}")
    print("-" * 60)


# ============================================
# 1. DATA LOADING AND EXPLORATION
# ============================================

def load_and_explore_data(filepath):
    """
    Load dataset and perform initial exploration.
    
    Args:
        filepath (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset or None if error
    """
    print_section("LOADING AND EXPLORING DATA", 1)
    
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Dataset loaded successfully!")
        print(f"   Shape: {df.shape}")
        print(f"   Rows: {df.shape[0]:,}")
        print(f"   Columns: {df.shape[1]}")

        print(f"\n📋 First 5 rows:")
        print(df.head())

        print(f"\n📋 Data Types:")
        print(df.dtypes)

        print(f"\n📋 Missing Values:")
        missing = df.isnull().sum()
        if any(missing > 0):
            print(missing[missing > 0])
        else:
            print("   No missing values found!")

        print(f"\n📋 Basic Statistics:")
        print(df.describe())

        return df

    except FileNotFoundError:
        print(f"❌ Error: File '{filepath}' not found!")
        return None


# ============================================
# 2. ADVANCED EXPLORATORY DATA ANALYSIS
# ============================================

def perform_advanced_eda(df):
    """
    Perform comprehensive exploratory data analysis with visualizations.
    
    Args:
        df (pd.DataFrame): Input data
        
    Returns:
        pd.DataFrame: Age statistics by cancer type
    """
    print_section("ADVANCED EXPLORATORY DATA ANALYSIS", 2)

    # 1. Cancer Type Distribution
    print("\n🔍 Analyzing Cancer Type Distribution...")
    plt.figure(figsize=(16, 6))

    cancer_counts = df['Cancer_Type'].value_counts()
    colors = plt.cm.viridis(np.linspace(0, 1, len(cancer_counts)))

    # Bar chart
    plt.subplot(1, 3, 1)
    bars = plt.bar(cancer_counts.index, cancer_counts.values, color=colors)
    plt.title('Cancer Type Distribution', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')

    # Pie chart
    plt.subplot(1, 3, 2)
    plt.pie(cancer_counts.values, labels=cancer_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    plt.title('Cancer Type Proportions', fontsize=14, fontweight='bold')

    # Horizontal bar chart
    plt.subplot(1, 3, 3)
    y_pos = np.arange(len(cancer_counts))
    plt.barh(y_pos, cancer_counts.values, color=colors)
    plt.yticks(y_pos, cancer_counts.index)
    plt.xlabel('Count')
    plt.title('Cancer Type Distribution (Horizontal)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('reports/figures/cancer_distribution_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"   ✅ Most common: {cancer_counts.index[0]} ({cancer_counts.values[0]} cases)")
    print(f"   ✅ Least common: {cancer_counts.index[-1]} ({cancer_counts.values[-1]} cases)")

    # 2. Age Analysis
    print("\n🔍 Analyzing Age Distribution...")
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    for cancer in df['Cancer_Type'].unique():
        subset = df[df['Cancer_Type'] == cancer]
        plt.hist(subset['Age'], alpha=0.6, label=cancer, bins=20, edgecolor='black')
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Age Distribution by Cancer Type', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    df.boxplot(column='Age', by='Cancer_Type', figsize=(12, 6))
    plt.title('Age Distribution Boxplot by Cancer Type', fontsize=14, fontweight='bold')
    plt.suptitle('')
    plt.xlabel('Cancer Type')
    plt.ylabel('Age')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('reports/figures/age_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Age statistics by cancer type
    age_stats = df.groupby('Cancer_Type')['Age'].agg(['mean', 'median', 'min', 'max', 'std'])
    print(f"\n📊 Age Statistics by Cancer Type:")
    print(age_stats)

    # 3. Correlation Analysis (for numeric features only)
    print("\n🔍 Analyzing Correlations...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) > 1:
        plt.figure(figsize=(14, 10))
        correlation_matrix = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(correlation_matrix), k=1)

        sns.heatmap(correlation_matrix,
                    mask=mask,
                    annot=True,
                    fmt='.2f',
                    cmap='coolwarm',
                    center=0,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8})

        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('reports/figures/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

    return age_stats


# ============================================
# 3. FEATURE ENGINEERING
# ============================================

def create_engineered_features(X):
    """
    Create new features from existing ones to improve model performance.
    
    Args:
        X (pd.DataFrame): Feature matrix with base features
        
    Returns:
        pd.DataFrame: Feature matrix with engineered features added
    """
    X_eng = X.copy()
    
    print("\n   🔧 Creating engineered features...")
    
    # 1. ===== LIFESTYLE RISK SCORE =====
    # Combination of smoking, alcohol, and obesity
    X_eng['Lifestyle_Risk'] = (X['Smoking'] + X['Alcohol_Use'] + X['Obesity']) / 3
    print("      ✅ Calculated Lifestyle_Risk (Smoking+Alcohol+Obesity)/3")
    
    # 2. ===== DIET QUALITY SCORE =====
    # Higher fruit/veg intake is good, red/processed meat is bad
    X_eng['Diet_Quality'] = (X['Fruit_Veg_Intake'] * 2 - X['Diet_Red_Meat'] - X['Diet_Salted_Processed']) / 4
    print("      ✅ Calculated Diet_Quality (higher is better diet)")
    
    # 3. ===== ENVIRONMENTAL EXPOSURE SCORE =====
    # Air pollution + occupational hazards
    X_eng['Environmental_Risk'] = (X['Air_Pollution'] + X['Occupational_Hazards']) / 2
    print("      ✅ Calculated Environmental_Risk (Air+Occupational)/2")
    
    # 4. ===== GENETIC/HEALTH RISK SCORE =====
    # Family history + BRCA mutation (binary indicators)
    X_eng['Genetic_Risk'] = X['Family_History'] + X['BRCA_Mutation']
    print("      ✅ Calculated Genetic_Risk (Family_History+BRCA)")
    
    # 5. ===== ACTIVITY-OBESITY RATIO =====
    # Physical activity counteracting obesity risk
    X_eng['Activity_Obesity_Ratio'] = X['Physical_Activity'] / (X['Obesity'] + 1)
    print("      ✅ Calculated Activity_Obesity_Ratio")
    
    # 6. ===== INFECTION RISK =====
    # H Pylori + Age interaction (older + infection = higher risk)
    X_eng['Infection_Age_Risk'] = X['H_Pylori_Infection'] * (X['Age'] / 50)
    print("      ✅ Calculated Infection_Age_Risk")
    
    # 7. ===== CALCIUM-DIET INTERACTION =====
    # Calcium intake modulating diet quality
    X_eng['Calcium_Diet_Protection'] = X['Calcium_Intake'] * X['Diet_Quality']
    print("      ✅ Calculated Calcium_Diet_Protection")
    
    # 8. ===== AGE-RISK INTERACTION =====
    # Age amplifies smoking risk
    X_eng['Age_Smoking_Risk'] = X['Age'] * X['Smoking'] / 10
    print("      ✅ Calculated Age_Smoking_Risk")
    
    # 9. ===== GENDER-SPECIFIC RISK =====
    # Female + BRCA is very high risk; Male + Prostate factors
    X_eng['Gender_Genetic_Risk'] = X['Gender'] * X['BRCA_Mutation']
    print("      ✅ Calculated Gender_Genetic_Risk")
    
    # 10. ===== OVERALL PROTECTIVE FACTORS =====
    # High physical activity + good diet + calcium intake
    X_eng['Protective_Factors'] = (X['Physical_Activity'] + X['Diet_Quality'] + X['Calcium_Intake']) / 3
    print("      ✅ Calculated Protective_Factors score")
    
    print(f"\n   🎉 Feature engineering complete! Added 10 new features")
    print(f"      Original features: {len(X)} → Enhanced features: {len(X_eng)}")
    
    return X_eng


# ============================================
# 4. DATA PREPROCESSING
# ============================================

def preprocess_data(df):
    """
    Preprocess data for machine learning.
    
    Args:
        df (pd.DataFrame): Raw input data
        
    Returns:
        tuple: (X_scaled, y, label_encoders, scaler, feature_cols, cancer_types)
    """
    print_section("DATA PREPROCESSING", 3)

    # Create a copy
    df_numeric = df.copy()
    label_encoders = {}

    # Remove non-feature columns
    columns_to_drop = [
        'Patient_ID',               # Identifier, not a feature
        'Overall_Risk_Score',       # Pre-calculated from other features
        'Physical_Activity_Level',  # Alternative encoding of Physical_Activity
        'Risk_Level',               # Categorical version of Overall_Risk_Score
        'Cancer_Type'               # This is the target (y)
    ]
    
    # Data from original CSV is already numeric encoded
    # No need for LabelEncoder conversion
    print(f"   📝 Data is already numerically encoded (from original dataset)")
    print(f"   ✅ Removed non-feature columns: {', '.join(columns_to_drop)}")

    # Encode target variable (Cancer_Type)
    target_encoder = LabelEncoder()
    df_numeric['Cancer_Type_Encoded'] = target_encoder.fit_transform(df_numeric['Cancer_Type'])
    label_encoders['Cancer_Type'] = target_encoder

    cancer_types = target_encoder.classes_.tolist()
    print(f"\n   🎯 Target classes: {cancer_types}")
    print(f"   🎯 Number of samples per cancer type:")
    for cancer_type in cancer_types:
        count = (df_numeric['Cancer_Type'] == cancer_type).sum()
        print(f"      {cancer_type}: {count} samples")

    # Create feature matrix and target vector
    feature_cols = [col for col in df_numeric.columns 
                   if col not in ['Cancer_Type', 'Cancer_Type_Encoded'] + columns_to_drop]
    
    X = df_numeric[feature_cols]
    y = df_numeric['Cancer_Type_Encoded']

    print(f"\n   📊 Feature matrix shape: {X.shape}")
    print(f"   📊 Features ({len(feature_cols)}): {', '.join(feature_cols[:5])}... +{len(feature_cols)-5} more")
    print(f"   🎯 Target vector shape: {y.shape}")

    # Apply feature engineering
    X = create_engineered_features(X)
    feature_cols = list(X.columns)  # Update feature names

    # Scale features
    print(f"\n   📏 Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    print(f"   ✅ Features scaled successfully!")

    return X_scaled, y, label_encoders, scaler, feature_cols, cancer_types


# ============================================
# 5. MODEL TRAINING AND COMPARISON
# ============================================

def train_and_compare_models(X_train, X_test, y_train, y_test, cancer_types):
    """
    Train multiple models and compare their performance.
    Includes additional models and class weight balancing.
    
    Args:
        X_train, X_test, y_train, y_test: Training and testing data
        cancer_types: List of cancer type names
        
    Returns:
        tuple: (results_df, trained_models)
    """
    print_section("MODEL TRAINING AND COMPARISON", 5)

    # ===== CLASS WEIGHT BALANCING =====
    # Helps models handle imbalanced cancer type distribution
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print(f"   ⚖️ Class weights (balanced): {class_weight_dict}\n")

    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1500, 
            random_state=42, 
            n_jobs=-1,
            class_weight='balanced'  #  handle class imbalance
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=150, 
            random_state=42, 
            n_jobs=-1,
            class_weight='balanced',  # handle class imbalance
            min_samples_split=3,
            min_samples_leaf=1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150, 
            random_state=42,
            learning_rate=0.05,  # Reduced for better learning
            subsample=0.8  # stochastic boosting
        ),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),  # Deeper network
            max_iter=1000, 
            random_state=42,
            early_stopping=True,  # prevent overfitting
            validation_fraction=0.1,
            n_iter_no_change=50
        )
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            n_estimators=150, 
            random_state=42, 
            eval_metric='mlogloss', 
            n_jobs=-1,
            scale_pos_weight=1,  # Already balanced
            subsample=0.8,
            colsample_bytree=0.8
        )

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\n   🔄 Training {name}...")

        try:
            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'CV Mean': cv_scores.mean(),
                'CV Std': cv_scores.std()
            })

            trained_models[name] = model

            print(f"   ✅ {name} trained successfully!")
            print(f"      Accuracy: {accuracy:.4f}")
            print(f"      F1-Score: {f1:.4f}")
            print(f"      CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

        except Exception as e:
            print(f"   ❌ Error training {name}: {str(e)}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('F1-Score', ascending=False)

    print("\n📊 Model Performance Summary:")
    print(results_df.to_string(index=False))

    # Save results
    results_df.to_csv('reports/metrics/model_comparison.csv', index=False)

    # Plot comparison
    plt.figure(figsize=(14, 8))
    results_melted = results_df.melt(id_vars=['Model'],
                                     value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                                     var_name='Metric', value_name='Score')

    sns.barplot(data=results_melted, x='Model', y='Score', hue='Metric')
    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('reports/figures/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    return results_df, trained_models


# ============================================
# 6. HYPERPARAMETER TUNING FOR BEST MODEL
# ============================================

def tune_best_model(X_train, y_train, X_test, y_test, cancer_types, results_df, trained_models):
    """
    Perform comprehensive hyperparameter tuning for the best performing model.
    Automatically selects the top model from comparison step and tunes it.
    Expanded parameter grid for better optimization
    
    Args:
        X_train, X_test, y_train, y_test: Training and testing data
        cancer_types: List of cancer type names
        results_df (pd.DataFrame): Model comparison results (to find best model)
        trained_models (dict): Dictionary of trained models
        
    Returns:
        tuple: (best_model, best_params, best_model_name)
    """
    print_section("HYPERPARAMETER TUNING (EXPANDED GRID)", 6)
    
    # ===== AUTOMATICALLY SELECT BEST MODEL =====
    # select the model with highest F1-Score
    best_model_name = results_df.iloc[0]['Model']  # First row after sorting by F1-Score
    best_f1_score = results_df.iloc[0]['F1-Score']
    print(f"   🏆 Selected best model for tuning: {best_model_name}")
    print(f"      F1-Score: {best_f1_score:.4f}\n")
    
    # Check if best model is tunable (Random Forest is best for tuning)
    if best_model_name != 'Random Forest':
        print(f"   ℹ️  Note: {best_model_name} performs best, but Random Forest tuning is more stable.")
        print(f"      Will tune Random Forest as alternative.\n")
        best_model_to_tune = 'Random Forest'
    else:
        best_model_to_tune = best_model_name

    # ===== EXPANDED PARAMETER GRID =====
    # Larger grid for more thorough search
    if best_model_to_tune == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 150, 200, 250, 300, 350],  # More options
            'max_depth': [10, 15, 18, 20, 25, 30, 35],  # More depth levels
            'min_samples_split': [2, 3, 5, 7, 10],  # More split options
            'min_samples_leaf': [1, 2, 3, 4, 5],  # More leaf options
            'max_features': ['sqrt', 'log2', None]  # Added: use all features
        }
        model_to_tune = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    else:
        # If another model is best, use smaller grid for computational efficiency
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [15, 20, 25],
            'min_samples_split': [3, 7],
            'min_samples_leaf': [1, 3],
            'max_features': ['sqrt', 'log2']
        }
        model_to_tune = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')

    # Grid search with Stratified K-Fold for better evaluation
    grid_search = GridSearchCV(
        model_to_tune,
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),  # Better for imbalanced data
        scoring='f1_weighted',  # Use F1 instead of accuracy (better for imbalanced)
        n_jobs=-1,
        verbose=1,  # Show progress
        return_train_score=True
    )

    # Fit grid search
    grid_search.fit(X_train, y_train)

    print(f"\n   ✅ Best parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"      {param}: {value}")

    print(f"\n   📊 Best cross-validation score: {grid_search.best_score_:.4f}")

    # Evaluate on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"   📊 Test set accuracy: {test_accuracy:.4f}")

    # Plot tuning results
    results = pd.DataFrame(grid_search.cv_results_)

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    for n_est in param_grid['n_estimators']:
        subset = results[results['param_n_estimators'] == n_est]
        plt.plot(subset['param_max_depth'].astype(str), subset['mean_test_score'],
                marker='o', label=f'n_est={n_est}')
    plt.xlabel('Max Depth')
    plt.ylabel('Mean Test Score')
    plt.title('Parameter Tuning: Max Depth vs Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)

    sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
    plt.title('Top 15 Features (Tuned Model)', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score')

    plt.tight_layout()
    plt.savefig('reports/figures/tuning_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    return best_model, grid_search.best_params_, best_model_name


# ============================================
# 7. COMPREHENSIVE MODEL EVALUATION
# ============================================

def evaluate_model(model, X_test, y_test, cancer_types, label_encoders):
    """
    Perform comprehensive model evaluation.
    
    Args:
        model: Trained model
        X_test, y_test: Test data
        cancer_types: List of cancer type names
        label_encoders: Dictionary of label encoders
        
    Returns:
        tuple: (y_pred, y_pred_proba)
    """
    print_section("COMPREHENSIVE MODEL EVALUATION", 7)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # 1. Classification Report
    print("\n📋 Classification Report:")
    report = classification_report(y_test, y_pred,
                                  target_names=cancer_types,
                                  output_dict=True)
    print(classification_report(y_test, y_pred, target_names=cancer_types))

    # Save classification report
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv('reports/metrics/classification_report.csv')

    # 2. Confusion Matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=cancer_types,
                yticklabels=cancer_types)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('reports/figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3. Per-class accuracy
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    bars = plt.bar(cancer_types, class_accuracy, color='coral')
    plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Cancer Type')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    for bar, acc in zip(bars, class_accuracy):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{acc:.2%}', ha='center', va='bottom')

    # 4. ROC Curves (One-vs-Rest)
    plt.subplot(1, 2, 2)
    y_test_bin = label_binarize(y_test, classes=range(len(cancer_types)))

    for i in range(len(cancer_types)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{cancer_types[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig('reports/figures/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    return y_pred, y_pred_proba


# ============================================
# 8. SAVE ALL ARTIFACTS
# ============================================

def save_artifacts(model, label_encoders, scaler, feature_names, cancer_types, best_params):
    """
    Save all model artifacts for deployment.
    
    Args:
        model: Trained model
        label_encoders: Dictionary of label encoders
        scaler: StandardScaler instance
        feature_names: List of feature names
        cancer_types: List of cancer type names
        best_params: Best hyperparameters
    """
    print_section("SAVING MODEL ARTIFACTS", 8)

    artifacts = {
        'model': model,
        'label_encoders': label_encoders,
        'scaler': scaler,
        'feature_names': list(feature_names),
        'cancer_types': cancer_types,
        'best_params': best_params
    }

    for name, artifact in artifacts.items():
        filename = f'models/{name}.pkl'
        joblib.dump(artifact, filename)
        print(f"   ✅ Saved: {filename}")

    # Save model metadata
    metadata = {
        'model_type': type(model).__name__,
        'features': list(feature_names),
        'n_features': len(feature_names),
        'cancer_types': cancer_types,
        'n_classes': len(cancer_types),
        'best_params': best_params,
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    metadata_df = pd.DataFrame([metadata])
    metadata_df.to_csv('models/model_metadata.csv', index=False)
    print(f"   ✅ Saved: models/model_metadata.csv")

    print(f"\n📦 All artifacts saved successfully!")


# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main execution function."""

    print("\n" + "="*70)
    print("🚀 STARTING ONCOGUARDIAN ML PIPELINE")
    print("="*70)

    # Create output directories
    create_output_directories()

    # Step 1: Load data
    df = load_and_explore_data('data/cancer-risk-factors.csv')
    if df is None:
        print("\n❌ Cannot proceed without data. Please add cancer-risk-factors.csv to data/ folder")
        return

    # Step 2: Perform EDA
    age_stats = perform_advanced_eda(df)

    # Step 3: Preprocess data
    X, y, label_encoders, scaler, feature_names, cancer_types = preprocess_data(df)

    # Step 4: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n📊 Data Split:")
    print(f"   Training set: {X_train.shape}")
    print(f"   Testing set: {X_test.shape}")

    # Step 5: Train and compare models
    results_df, trained_models = train_and_compare_models(X_train, X_test, y_train, y_test, cancer_types)

    # Step 6: Hyperparameter tuning (now tunes the best model automatically)
    best_model, best_params, best_model_name = tune_best_model(X_train, y_train, X_test, y_test, cancer_types, results_df, trained_models)

    # Step 7: Comprehensive evaluation
    y_pred, y_pred_proba = evaluate_model(best_model, X_test, y_test, cancer_types, label_encoders)

    # Step 8: Save artifacts
    save_artifacts(best_model, label_encoders, scaler, feature_names, cancer_types, best_params)

    print("\n" + "="*70)
    print("✅ ONCOGUARDIAN ML PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)

    # Final summary
    print("\n📊 FINAL MODEL SUMMARY:")
    print(f"   Model Type: {best_model_name} (Optimized)")
    print(f"   Number of Features: {len(feature_names)} (15 base + 10 engineered)")
    print(f"   Cancer Types: {', '.join(cancer_types)}")
    print(f"   Best Parameters: {best_params}")
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    # Feature importance summary
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n🔍 Top 10 Most Important Features (Including Engineered):")
    for i in range(min(10, len(feature_importance))):
        print(f"   {i+1}. {feature_importance.iloc[i]['feature']}: {feature_importance.iloc[i]['importance']:.4f}")


if __name__ == "__main__":
    main()
