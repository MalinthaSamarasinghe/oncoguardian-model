# 🚀 Quick Start Guide - OncoGuardian Model

## First Time Setup (5 minutes)

### Step 1: Generate Sample Data
```bash
python src/generate_data.py
```
This creates `data/cancer-risk-factors.csv` with 500 sample records.

**Output:**
```
✅ Dataset saved to data/cancer-risk-factors.csv
   Shape: (500, 15)

📊 Cancer Type Distribution:
   Lung        98
   Breast      105
   Colon       99
   Prostate    104
   Skin        94
```

---

### Step 2: Run Training Pipeline
```bash
python src/training.py
```

This runs all 7 training steps and takes ~2-3 minutes.

**Output:**
```
===============================================
🚀 STARTING ONCOGUARDIAN ML PIPELINE
===============================================

📊 STEP 1: LOADING AND EXPLORING DATA
---
✅ Dataset loaded successfully!
   Shape: (500, 15)
   Rows: 500
   Columns: 15
   ...

📊 STEP 2: ADVANCED EXPLORATORY DATA ANALYSIS
---
🔍 Analyzing Cancer Type Distribution...
✅ Most common: Lung (98 cases)
✅ Least common: Skin (94 cases)
   ...

📊 STEP 4: MODEL TRAINING AND COMPARISON
---
   🔄 Training Logistic Regression...
   ✅ Logistic Regression trained successfully!
      Accuracy: 0.8200
      F1-Score: 0.8190
   ...

📊 STEP 5: HYPERPARAMETER TUNING
---
   🔍 Searching for best hyperparameters...
      Grid size: 3600 combinations
   
   ✅ Best parameters found:
      n_estimators: 200
      max_depth: 20
      ...

📊 FINAL MODEL SUMMARY:
   Model Type: Random Forest (Tuned)
   Number of Features: 14
   Cancer Types: ['Lung', 'Breast', 'Colon', 'Prostate', 'Skin']
   Test Accuracy: 0.9200
   
   Top 5 Most Important Features:
   1. Age: 0.2523
   2. Gender: 0.1834
   3. Smoking: 0.1567
   ...

✅ ONCOGUARDIAN ML PIPELINE COMPLETED SUCCESSFULLY!
================================================
```

---

### Step 3: Make Predictions
```bash
python src/predictor.py
```

**Output:**
```
============================================================
EXAMPLE PREDICTION
============================================================

🔮 Prediction Results:
   Predicted Cancer Type: Breast
   Confidence: 92.45%

📊 All Cancer Type Probabilities:
   Lung: 5.23%
   Breast: 92.45%
   Colon: 1.82%
   Prostate: 0.30%
   Skin: 0.20%

🍽️ Recommendations:
   Risk Level: HIGH
   Recommended Foods:
      🥦 Cruciferous vegetables (broccoli, kale, cauliflower)
      🍵 Green tea (rich in EGCG antioxidants)
      🟡 Turmeric and ginger (anti-inflammatory)
      ...
   
   Foods to Avoid:
      🚫 Processed meats
      🚫 Alcohol
      ...
   
   Supplements:
      • Vitamin D
      • Omega-3
      ...
   
   Lifestyle Tips:
      • Regular mammograms
      • Maintain healthy weight
      • Exercise
```

---

## 📁 What Gets Created

### During Step 1 (Data Generation):
```
data/
└── cancer-risk-factors.csv  (500 samples, 15 columns)
```

### During Step 2 (Training):
```
models/
├── model.pkl                     (Trained Random Forest)
├── label_encoders.pkl            (Feature encoders)
├── scaler.pkl                    (Feature scaler)
├── feature_names.pkl             (Feature list)
├── cancer_types.pkl              (Cancer type names)
└── model_metadata.csv            (Model info)

reports/
├── figures/
│   ├── cancer_distribution_comprehensive.png
│   ├── age_analysis_comprehensive.png
│   ├── correlation_matrix.png
│   ├── model_comparison.png
│   ├── tuning_results.png
│   ├── confusion_matrix.png
│   └── roc_curves.png
└── metrics/
    ├── model_comparison.csv
    └── classification_report.csv
```

---

## 📊 Understanding the Results

### Model Comparison
Shows performance of all 5 algorithms trained. Random Forest usually wins!

### Confusion Matrix
Visualizes what the model predicted vs. what it actually was.
```
Diagonal (correct) = Dark colors
Off-diagonal (errors) = Light colors
```

### Feature Importance
Shows which factors matter most for cancer prediction.

### ROC Curves
Measures how well the model ranks positive samples higher than negative samples.

---

## 🎯 Next Steps

### For University Assignment:
1. ✅ Understand the code (see CODE_EXPLANATION.md)
2. ✅ Run the pipeline with sample data
3. ✅ Get real cancer risk factor data (from university/open datasets)
4. ✅ Retrain with real data
5. ✅ Document results in your report
6. ✅ Prepare for Flutter integration

### For Flutter Integration:
See [FLUTTER_INTEGRATION.md](./FLUTTER_INTEGRATION.md) for:
- Converting model to TFLite format
- Integration with Firebase
- Building the mobile app

---

## 🐛 Common Issues

### Issue: "File not found" error
**Solution:**
```bash
# Make sure you have the data file
python src/generate_data.py
```

### Issue: "Module not installed" error
**Solution:**
```bash
# Install dependencies
pip install -r requirements.txt
```

### Issue: Model accuracy is low (< 70%)
**Solution:**
- Check that data has enough samples (>300)
- Verify all feature columns are present
- Try different hyperparameters
- Get more/better quality data

### Issue: Prediction fails
**Solution:**
- Make sure all required features are in patient data
- Check feature names match exactly
- Verify categorical values are valid (e.g., 'Male' not 'male')

---

## 💡 Tips for Your Assignment

1. **Document Everything:**
   - Add comments explaining your modifications
   - Document any changes to the dataset
   - Explain hyperparameter choices

2. **Show Your Work:**
   - Include visualization screenshots
   - Report metrics (accuracy, precision, recall, F1)
   - Show confusion matrix
   - Display top features

3. **Evaluate Results:**
   - Which algorithm performed best?
   - Why do you think?
   - Are there cancer types that are confused?
   - What factors matter most?

4. **For Mobile Integration:**
   - Save metrics for the report
   - Plan how to export model for Flutter
   - Design recommendation system for app

5. **Prepare for Questions:**
   - Understand each step of the pipeline
   - Know what each metric means
   - Explain your feature choices
   - Discuss limitations and future improvements

---

## 📖 Additional Help

See these files for more information:
- **CODE_EXPLANATION.md** - Detailed line-by-line explanation
- **README.md** - Project overview and setup
- **Code comments** - Each function has docstrings

---

Now you're ready! Start with: `python src/generate_data.py` 🎉
