# OncoGuardian Model - Code Explanation Guide

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Code Structure](#code-structure)
3. [Detailed Explanation](#detailed-explanation)
4. [How to Run](#how-to-run)
5. [Understanding Each Step](#understanding-each-step)

---

## 🎯 Project Overview

**OncoGuardian** is a machine learning system that predicts personalized cancer risk based on various health and lifestyle factors. It uses multiple classification algorithms trained on cancer risk factor data.

### Key Components:
- **Data Preparation**: Loading, exploring, and preprocessing data
- **Model Training**: Training 5 different ML algorithms
- **Model Selection**: Hyperparameter tuning and comparison
- **Evaluation**: Comprehensive metrics and visualizations
- **Prediction**: Making risk predictions and providing dietary recommendations

---

## 📁 Code Structure

```
oncoguardian-model/
├── src/
│   ├── training.py           # Main training pipeline (7 steps)
│   ├── predictor.py          # Prediction and recommendation system
│   ├── generate_data.py      # Sample data generation
│   └── utils.py              # (Optional) Utility functions
├── data/
│   └── cancer-risk-factors.csv    # Input dataset
├── models/                        # Trained model artifacts
│   ├── model.pkl                  # Trained Random Forest
│   ├── label_encoders.pkl         # Feature encoders
│   ├── scaler.pkl                 # Feature scaler
│   ├── feature_names.pkl          # Feature list
│   ├── cancer_types.pkl           # Cancer type names
│   └── model_metadata.csv         # Model information
├── reports/
│   ├── figures/                   # Visualizations
│   │   ├── cancer_distribution_comprehensive.png
│   │   ├── age_analysis_comprehensive.png
│   │   ├── correlation_matrix.png
│   │   ├── model_comparison.png
│   │   ├── tuning_results.png
│   │   ├── confusion_matrix.png
│   │   └── roc_curves.png
│   └── metrics/                   # CSV metrics files
│       ├── model_comparison.csv
│       └── classification_report.csv
├── requirements.txt
└── README.md
```

---

## 🔍 Detailed Explanation

### 1. **training.py** - The Main Pipeline

This file contains ALL the functions needed to train the model. Let's break it down:

#### **Step 1: Data Loading and Exploration**
```python
def load_and_explore_data(filepath):
    """Load dataset and perform initial exploration."""
```
**What it does:**
- Reads the CSV file containing cancer risk factors
- Displays:
  - Dataset shape (rows, columns)
  - First 5 rows of data
  - Data types (numeric, text, etc.)
  - Missing values
  - Statistical summary (mean, std, min, max)

**Why it's important:**
- Understand data quality
- Identify missing or incorrect values
- Get familiar with the dataset structure

**Example Output:**
```
✅ Dataset loaded successfully!
   Shape: (500, 15)
   Rows: 500
   Columns: 15
```

---

#### **Step 2: Advanced Exploratory Data Analysis (EDA)**
```python
def perform_advanced_eda(df):
    """Perform comprehensive exploratory data analysis with visualizations."""
```

**What it does:**
1. **Cancer Type Distribution**: Shows which cancer types are most common
   - Bar chart: How many samples for each cancer type
   - Pie chart: Percentage distribution
   - Horizontal bar: For better readability

2. **Age Distribution**: Analyzes age patterns
   - Histogram: Shows age distribution for each cancer type
   - Boxplot: Visualizes age ranges (min, max, median, quartiles)

3. **Correlation Analysis**: Shows relationships between features
   - Heatmap: Displays correlation matrix
   - Values range from -1 to 1
     - 1: Perfect positive correlation
     - 0: No correlation
     - -1: Perfect negative correlation

**Why it's important:**
- Understand data patterns and distributions
- Identify which features are correlated
- Spot any data anomalies
- Inform feature engineering decisions

**Generated Files:**
```
reports/figures/
├── cancer_distribution_comprehensive.png
├── age_analysis_comprehensive.png
└── correlation_matrix.png
```

---

#### **Step 3: Data Preprocessing**
```python
def preprocess_data(df):
    """Preprocess data for machine learning."""
```

**What it does:**

1. **Label Encoding**: Converts text to numbers
   ```
   Original:      Gender: 'Male', 'Female', 'Other'
   Encoded:       Gender: 1, 2, 3
   
   Original:      Smoking: 'Never', 'Former', 'Current'
   Encoded:       Smoking: 1, 2, 3
   ```
   **Why?** ML algorithms only understand numbers

2. **Feature Scaling (StandardScaler)**
   ```
   Original Age values: 20, 45, 80
   Scaled values: -1.5, 0.2, 1.3 (normalized to mean=0, std=1)
   ```
   **Why?** Puts all features on same scale (very important for algorithms like SVM, Neural Networks)

3. **Separating Features (X) and Target (y)**
   ```
   X = All features (age, gender, smoking, etc.) - Input to model
   y = Cancer_Type - What we want to predict (Output)
   ```

**Output:**
- `X_scaled`: Preprocessed features (scaled)
- `y`: Target variable (encoded)
- `label_encoders`: Dictionary to convert predictions back to original values
- `scaler`: Used for new predictions

---

#### **Step 4: Model Training and Comparison**
```python
def train_and_compare_models(X_train, X_test, y_train, y_test, cancer_types):
    """Train multiple models and compare their performance."""
```

**What it does:**
Trains 5 different machine learning algorithms:

1. **Logistic Regression**: Fast, simple, interpretable
   - Good for binary/multi-class classification
   - Provides probability scores

2. **Random Forest**: Ensemble method, powerful
   - Combination of many decision trees
   - Robust to outliers

3. **Gradient Boosting**: Sequential ensemble
   - Builds trees one by one, each fixing errors of previous
   - Usually very accurate

4. **XGBoost**: Advanced gradient boosting
   - Optimized implementation
   - Often wins machine learning competitions

5. **Neural Network (MLP)**: Deep learning approach
   - Multiple layers of neurons
   - Can capture complex patterns

**For each model:**
```python
# 1. Train on training data
model.fit(X_train, y_train)

# 2. Predict on test data
y_pred = model.predict(X_test)

# 3. Calculate metrics
accuracy = accuracy_score(y_test, y_pred)  # % of correct predictions
precision = precision_score(y_test, y_pred, average='weighted')  # True positives / all positives
recall = recall_score(y_test, y_pred, average='weighted')  # True positives / all actual positives
f1 = f1_score(y_test, y_pred, average='weighted')  # Harmonic mean of precision & recall

# 4. Cross-validation (extra check)
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
# Splits data into 5 parts, trains/tests on each - ensures model generalizes
```

**Metrics Explained:**
```
Accuracy:  90% = 90 out of 100 predictions correct
Precision: How many predicted positives are actually positive
Recall:    Of all actual positives, how many did we find
F1-Score:  Balance between precision and recall (0-1, higher is better)
CV Score:  Average accuracy on 5 different data splits
```

**Output:**
```
Model Performance Summary:
Model                  Accuracy  Precision  Recall  F1-Score  CV Mean
Random Forest          0.9200    0.9180     0.9200  0.9190    0.9100
XGBoost                0.9050    0.9020     0.9050  0.9035    0.8950
Gradient Boosting      0.8850    0.8820     0.8850  0.8835    0.8750
Neural Network         0.8650    0.8620     0.8650  0.8635    0.8550
Logistic Regression    0.8200    0.8180     0.8200  0.8190    0.8100

reports/metrics/model_comparison.csv (saved)
reports/figures/model_comparison.png (saved)
```

---

#### **Step 5: Hyperparameter Tuning**
```python
def tune_best_model(X_train, y_train, X_test, y_test, cancer_types):
    """Perform hyperparameter tuning for Random Forest."""
```

**What it does:**

Hyperparameters are settings we choose (not learned from data). For Random Forest:

```python
param_grid = {
    'n_estimators': [100, 200, 300],      # Number of trees
    'max_depth': [10, 20, 30, None],      # How deep each tree grows
    'min_samples_split': [2, 5, 10],      # Min samples to split a node
    'min_samples_leaf': [1, 2, 4],        # Min samples in leaf node
    'max_features': ['sqrt', 'log2']      # Features per split
}
```

**Example:**
```
Configuration 1: n_estimators=100, max_depth=10, ...  → Accuracy: 0.91
Configuration 2: n_estimators=200, max_depth=20, ...  → Accuracy: 0.92 ✅ BEST
Configuration 3: n_estimators=300, max_depth=30, ...  → Accuracy: 0.88
...
(Tests 3000+ combinations)
```

**GridSearchCV:**
- Tests all combinations of parameters
- Uses 5-fold cross-validation (more reliable)
- Returns best parameters

**Output:**
```
Best parameters found:
   n_estimators: 200
   max_depth: 20
   min_samples_split: 5
   min_samples_leaf: 2
   max_features: sqrt

Best cross-validation score: 0.9250
Test set accuracy: 0.9200

Feature importances (what contributes most to predictions):
1. Age: 0.25
2. Smoking: 0.20
3. Family_History: 0.18
...
```

---

#### **Step 6: Comprehensive Model Evaluation**
```python
def evaluate_model(model, X_test, y_test, cancer_types, label_encoders):
    """Perform comprehensive model evaluation."""
```

**What it does:**

1. **Classification Report**: Detailed metrics per cancer type
```
           precision  recall  f1-score  support
Lung          0.92     0.93     0.93      50
Breast        0.91     0.90     0.91      48
Colon         0.89     0.88     0.89      40
Prostate      0.95     0.96     0.95      52
Skin          0.88     0.87     0.88      40
```

2. **Confusion Matrix**: Shows what model predicted vs actual
```
           Predicted
Actual    Lung Breast Colon Prostate Skin
Lung       46    2      1      0       1
Breast      1   43      2      1       1
Colon       1    2     35      1       1
...
```
Diagonal = correct predictions, Off-diagonal = errors

3. **Per-Class Accuracy**
```
Lung       92% accuracy
Breast     90% accuracy
Colon      89% accuracy
...
Shows which cancer types the model predicts best
```

4. **ROC Curves (One-vs-Rest)**
```
AUC (Area Under Curve) = 0.95 means:
- If you randomly pick one positive and one negative sample
- Model would rank the positive higher 95% of the time
- Range: 0.5 (random) to 1.0 (perfect)
```

**Generated Files:**
```
reports/figures/
├── confusion_matrix.png
├── roc_curves.png

reports/metrics/
└── classification_report.csv
```

---

#### **Step 7: Save Artifacts**
```python
def save_artifacts(model, label_encoders, scaler, feature_names, cancer_types, best_params):
    """Save all model artifacts for deployment."""
```

**What it does:**
Saves everything needed to make predictions later:

```
models/
├── model.pkl              ← The trained Random Forest model
├── label_encoders.pkl    ← How to convert text → numbers
├── scaler.pkl            ← How to scale features
├── feature_names.pkl     ← Which features the model expects
├── cancer_types.pkl      ← List of cancer types
└── model_metadata.csv    ← Model information (type, date, params)
```

**Why?** 
- Don't need to retrain every time
- Can load and use for predictions immediately
- Can share model with others

---

### 2. **predictor.py** - Making Predictions

This file provides the `OncoGuardianPredictor` class for using the trained model.

```python
from src.predictor import OncoGuardianPredictor

# Initialize
predictor = OncoGuardianPredictor()

# Prepare patient data
patient = {
    'Age': 45,
    'Gender': 'Female',
    'Smoking': 'Never',
    'Alcohol_Use': 'Moderate',
    'Obesity': 'No',
    ...
}

# Get prediction
prediction = predictor.predict(patient)
print(prediction)
# Output:
# {
#     'predicted_cancer_type': 'Breast',
#     'confidence': 0.92,
#     'probabilities': {
#         'Lung': 0.05,
#         'Breast': 0.92,
#         'Colon': 0.02,
#         'Prostate': 0.01,
#         'Skin': 0.00
#     }
# }

# Get dietary recommendations
recommendations = predictor.get_recommendations(patient)
print(recommendations)
# Output:
# {
#     'cancer_type': 'Breast',
#     'risk_level': 'HIGH',
#     'recommended_foods': [
#         '🥦 Cruciferous vegetables',
#         '🍵 Green tea',
#         ...
#     ],
#     'foods_to_avoid': [...],
#     'supplements': [...],
#     'lifestyle_tips': [...]
# }
```

---

### 3. **generate_data.py** - Creating Sample Data

Generates realistic sample data for training:

```python
from src.generate_data import generate_sample_dataset, save_dataset

# Generate 500 samples
df = generate_sample_dataset(n_samples=500)

# Save to CSV
save_dataset(df, 'data/cancer-risk-factors.csv')
```

**Generated Features:**
- Age: 20-85 years
- Gender: Male, Female, Other
- Smoking: Never, Former, Current
- Alcohol_Use: None, Moderate, Heavy
- Obesity: No, Yes
- Family_History: No, Yes
- And 9 more risk factors...
- **Target:** Cancer_Type (Lung, Breast, Colon, Prostate, Skin)

---

## 🚀 How to Run

### **Option 1: Quick Start (Recommended for First Time)**

```bash
# 1. Generate sample data
python src/generate_data.py

# 2. Run training pipeline
python src/training.py

# 3. Make predictions
python src/predictor.py
```

### **Option 2: With Your Own Data**

```bash
# 1. Place your CSV in data/cancer-risk-factors.csv
# 2. Make sure it has these columns:
#    - Cancer_Type (target)
#    - Age, Gender, Smoking, Alcohol_Use, Obesity, Family_History
#    - Diet_Red_Meat, Diet_Salted_Processed, Fruit_Veg_Intake
#    - Physical_Activity, Air_Pollution, Occupational_Hazards
#    - BRCA_Mutation, H_Pylori_Infection, Calcium_Intake

# 3. Run training
python src/training.py

# 4. Make predictions
python src/predictor.py
```

---

## 📊 Understanding Each Step

### **Data Split**
```
Original Data (500 samples)
├── Training Set (80%, 400 samples) → Used to train model
└── Test Set (20%, 100 samples)     → Used to evaluate model

Why?
- Training on test data would overfit (memorize answers)
- Need independent test data to verify the model generalizes
```

### **Cross-Validation**
```
K-Fold Cross-Validation (K=5):

Fold 1: Train[2-5], Test[1]
Fold 2: Train[1,3-5], Test[2]
Fold 3: Train[1-2,4-5], Test[3]
Fold 4: Train[1-3,5], Test[4]
Fold 5: Train[1-4], Test[5]

Average CV Score = (Fold1 + Fold2 + ... + Fold5) / 5

Why?
- Single test might be lucky/unlucky
- CV gives more reliable estimate of model performance
- Ensures model generalizes across different data splits
```

### **Feature Importance**
```
Why certain features matter more:

Feature                Importance
Age                    0.25  ████████████████████████
Smoking                0.20  ████████████████████
Family_History         0.18  ██████████████████
Obesity                0.12  ████████████
Physical_Activity      0.10  ██████████
Diet_Red_Meat          0.08  ████████
Other features...      0.07

Age is 2.5x more important than Physical Activity!
This makes sense for cancer prediction.
```

---

## ⚙️ Key Concepts Explained

### **Machine Learning Workflow**
```
1. DATA COLLECTION → Raw data
2. DATA EXPLORATION → Understand patterns
3. PREPROCESSING → Clean & normalize
4. TRAIN/TEST SPLIT → Divide into training/testing
5. MODEL TRAINING → Fit algorithm to training data
6. HYPERPARAMETER TUNING → Optimize model settings
7. EVALUATION → Test on independent data
8. DEPLOYMENT → Use for predictions
9. MONITORING → Track performance over time
```

### **Why Multiple Models?**
```
Different algorithms have different strengths:

Random Forest:
✓ Handles non-linear relationships
✓ Feature importance built-in
✓ Robust to outliers
✗ Can overfit if not tuned

Gradient Boosting:
✓ Highly accurate
✓ Handles mixed data types
✗ Slow to train
✗ Hard to interpret

XGBoost:
✓ Very accurate
✓ Fast
✗ Black box (hard to interpret)

Logistic Regression:
✓ Fast
✓ Interpretable
✓ Provides probabilities
✗ Assumes linear relationships

Neural Network:
✓ Can learn complex patterns
✗ Need lots of data
✗ Slow to train
```

We train all of them and pick the best!

---

## 📈 Metrics Explained

### **Classification Metrics**

**Accuracy:**
```
Correct Predictions / Total Predictions
= (TP + TN) / (TP + TN + FP + FN)
Range: 0-1 (higher is better)
```

**Precision:**
```
True Positives / (True Positives + False Positives)
"Of predictions positive, how many were correct?"
Use when: Cost of false positives is high
Example: In cancer screening, false positive means unnecessary testing
```

**Recall:**
```
True Positives / (True Positives + False Negatives)
"Of all actual positives, how many did model find?"
Use when: Cost of false negatives is high
Example: In cancer screening, false negative means missing disease
```

**F1-Score:**
```
2 × (Precision × Recall) / (Precision + Recall)
Harmonic mean - balanced metric
Use when: Want balance between precision and recall
```

### **Confusion Matrix for 5 Classes**

```
                 Predicted
              Lung Breast Colon Prostate Skin
           ┌─────────────────────────────────┐
Actual Lung│ 46     2      1       0        1  │  ← Actual Lung samples
      Breas│  1    43      2       1        1  │  ← Actual Breast samples
      Colon│  1     2     35       1        1  │  ← Actual Colon samples
      Prost│  0     1      1      50        0  │  ← Actual Prostate samples
      Skin │  1     1      1       0       37  │  ← Actual Skin samples
           └─────────────────────────────────┘

Diagonal values = Correct predictions
Off-diagonal = Misclassifications
```

---

## 🔧 Troubleshooting

### **Error: File not found**
```
Solution: Generate sample data first
python src/generate_data.py
```

### **Error: Module not found**
```
Solution: Install dependencies
pip install -r requirements.txt
```

### **Error: Shape mismatch in prediction**
```
Solution: Ensure features match training data
Check that all required column names are present in patient data
```

---

## 📚 Additional Resources

### **ML Concepts:**
- Scikit-learn documentation: https://scikit-learn.org/
- Pandas guide: https://pandas.pydata.org/
- Understanding different algorithms

### **Model Export for Flutter:**
- Converting sklearn models to TFLite
- ONNX format for cross-platform compatibility

---

## 📝 Summary

This code provides a **complete ML pipeline** for cancer risk prediction:

1. **Load & Explore** data to understand it
2. **Preprocess** features (encode, scale)
3. **Train multiple** algorithms
4. **Compare** performance to find best
5. **Tune hyperparameters** for optimization
6. **Evaluate comprehensively** on test data
7. **Save artifacts** for deployment
8. **Make predictions** with the trained model
9. **Provide recommendations** based on risk

The modular structure makes it easy to understand each step and modify for your specific needs!
