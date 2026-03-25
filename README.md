# OncoGuardian Model - Python Training Pipeline

A comprehensive machine learning pipeline for personalized cancer risk prediction using multiple classification algorithms.

## Project Structure

```
oncoguardian-model/
├── data/                          # Input datasets
│   └── cancer-risk-factors.csv
├── models/                        # Trained models and artifacts
│   ├── best_params.pkl
│   ├── cancer_types.pkl
│   ├── feature_names.pkl
│   ├── label_encoders.pkl
│   ├── model_metadata.csv
│   ├── model.pkl
│   └── scaler.pkl
├── reports/                       # Generated reports and visualizations
│   ├── figures/                   # Plots and charts
│   └── metrics/                   # Performance metrics
├── src/                           # Source code
│   ├── training.py                # Main training pipeline
│   └── predictor.py               # Prediction and recommendation system
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Setup Instructions

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Dataset is ready!**
   - ✅ `data/cancer-risk-factors.csv` - Contains 2000 real cancer risk factor samples
   - Ready to use immediately!

3. **Run training pipeline:**
   ```bash
   python src/training.py
   ```
   This trains the model with the real data (2000 samples, 15 features, 5 cancer types).

## Features

- **7-Step ML Pipeline:**
  1. Data loading and exploration
  2. Advanced EDA with visualizations
  3. Data preprocessing and encoding
  4. Model training and comparison
  5. Hyperparameter tuning
  6. Comprehensive evaluation
  7. Model artifacts saving

- **Multiple Algorithms:**
  - Logistic Regression
  - Random Forest (tuned)
  - Gradient Boosting
  - XGBoost
  - Neural Network (MLP)

- **Cancer Type Predictions:**
  - Lung, Breast, Colon, Prostate, and more

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip or conda

### Step 1: Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using conda
conda create -n oncoguardian python=3.9
conda activate oncoguardian
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Prepare Data
- Download or create `cancer-risk-factors.csv`
- Place it in the `data/` directory

## Usage

### Training Models
```bash
python src/training.py
```

### Making Predictions
```python
from src.predictor import OncoGuardianPredictor

predictor = OncoGuardianPredictor()
risk_scores = predictor.predict(patient_data)
recommendations = predictor.get_recommendations(patient_data, risk_level)
```

## Model Export for Flutter/TFLITE

After training completes:
1. The best model is saved as `models/model.pkl`
2. To convert for TFLite, use TensorFlow's conversion tools:
   ```bash
   pip install tensorflow
   python convert_to_tflite.py  # (script to be created)
   ```

## Dataset Features

Expected columns in `cancer-risk-factors.csv`:
- Cancer_Type (target variable)
- Age, Gender, Smoking, Alcohol_Use
- Obesity, Family_History
- Diet_Red_Meat, Diet_Salted_Processed, Fruit_Veg_Intake
- Physical_Activity, Air_Pollution, Occupational_Hazards
- BRCA_Mutation, H_Pylori_Infection
- And other risk factors

## Output Files

After training:
- **models/**: Serialized models and encoders
- **reports/figures/**: Visualizations (PNG files)
- **reports/metrics/**: Performance metrics (CSV files)

## Notes for Flutter Integration

- Models are saved as `.pkl` files (Python serialization)
- For Flutter, need to:
  1. Convert model to TFLite format
  2. Or use Python backend API
  3. Or export as ONNX and convert to TFLite

See `FLUTTER_INTEGRATION.md` for detailed steps.
