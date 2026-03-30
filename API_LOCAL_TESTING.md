# OncoGuardian API - Local Testing Guide

## ⚡ Quick Start

### 1. Install Dependencies

```bash
# Activate virtual environment
source .venv/bin/activate

# Install Flask and CORS
pip install Flask==2.3.3 Flask-CORS==4.0.0 -q

# Or install from requirements file
pip install -r requirements-api.txt -q
```

### 2. Run API Server Locally

```bash
# From project root
cd /Users/outerspace/oncoguardian-model

# Activate venv and run Flask
source .venv/bin/activate && python -m flask --app src.app run

# Or run directly
source .venv/bin/activate && python src/app.py
```

**Expected Output:**
```
🚀 OncoGuardian Flask API Server
======================================================================
port: 5000
debug: True
URL: http://localhost:5000
======================================================================
```

### 3. Test Endpoints

#### A. Health Check (Browser)
```
http://localhost:5000/health
```

**Response:**
```json
{
  "status": "ok",
  "predictor_loaded": true,
  "timestamp": "2024-03-25T10:30:00.000000"
}
```

---

## 🧪 API Testing Methods

### Method 1: Using cURL (Terminal)

```bash
# Health check
curl http://localhost:5000/health

# Get model info
curl http://localhost:5000/model-info

# Make prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'

# Get recommendations
curl -X POST http://localhost:5000/recommendations \
  -H "Content-Type: application/json" \
  -d '{
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
    "Calcium_Intake": 7,
    "cancer_type": "Breast"
  }'
```

---

### Method 2: Using Python

```python
import requests
import json

# Base URL
BASE_URL = "http://localhost:5000"

# Patient data
patient_data = {
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

# Test prediction
def test_prediction():
    response = requests.post(f"{BASE_URL}/predict", json=patient_data)
    print("Prediction Response:")
    print(json.dumps(response.json(), indent=2))

# Test recommendations
def test_recommendations():
    data = {**patient_data, "cancer_type": "Breast"}
    response = requests.post(f"{BASE_URL}/recommendations", json=data)
    print("Recommendations Response:")
    print(json.dumps(response.json(), indent=2))

# Test batch prediction
def test_batch_predict():
    patients = [
        {"id": "P001", **patient_data},
        {"id": "P002", **{**patient_data, "Age": 52}},
        {"id": "P003", **{**patient_data, "Smoking": 5}},
    ]
    
    response = requests.post(
        f"{BASE_URL}/batch-predict",
        json={"patients": patients}
    )
    print("Batch Prediction Response:")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    print("Testing OncoGuardian API...\n")
    test_prediction()
    print("\n" + "="*70 + "\n")
    test_recommendations()
    print("\n" + "="*70 + "\n")
    test_batch_predict()
```

**Run the test:**
```bash
source .venv/bin/activate && python test_api.py
```

---

### Method 3: Using Postman / REST Client

**VS Code REST Client:**

Create file `test_api.http`:

```http
### Health Check
GET http://localhost:5000/health

### Get Model Info
GET http://localhost:5000/model-info

### Make Prediction
POST http://localhost:5000/predict
Content-Type: application/json

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

### Get Recommendations
POST http://localhost:5000/recommendations
Content-Type: application/json

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
  "Calcium_Intake": 7,
  "cancer_type": "Breast"
}

### Batch Predict
POST http://localhost:5000/batch-predict
Content-Type: application/json

{
  "patients": [
    {
      "id": "P001",
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
    },
    {
      "id": "P002",
      "Age": 52,
      "Gender": 0,
      "Height": 1.75,
      "Weight": 78,
      "Smoking": 5,
      "Alcohol_Use": 3,
      "Physical_Activity": 5,
      "Diet_Red_Meat": 6,
      "Diet_Salted_Processed": 5,
      "Fruit_Veg_Intake": 4,
      "Air_Pollution": 6,
      "Occupational_Hazards": 2,
      "Family_History": 1,
      "BRCA_Mutation": 0,
      "H_Pylori_Infection": 0,
      "Calcium_Intake": 5
    }
  ]
}
```

Install VS Code extension: `REST Client` by Huachao Mao

Then click "Send Request" above each request.

---

## 🔍 Expected Responses

### Prediction Response
```json
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
```

### Recommendations Response
```json
{
  "success": true,
  "recommendations": {
    "risk_level": "HIGH",
    "cancer_type": "Breast",
    "recommended_foods": [
      "Broccoli",
      "Green tea",
      "Fatty fish (Salmon, Mackerel)",
      "Blueberries",
      "Turmeric"
    ],
    "foods_to_avoid": [
      "Red meat (Beef, Lamb)",
      "Processed meats",
      "Sugar-sweetened beverages",
      "Alcohol (Limited)"
    ],
    "supplements": [
      "Vitamin D3",
      "Omega-3 (Fish Oil)",
      "Curcumin (Turmeric extract)"
    ],
    "lifestyle_tips": [
      "Exercise 30 minutes daily",
      "Maintain healthy weight",
      "Reduce stress through meditation"
    ]
  },
  "timestamp": "2024-03-25T10:30:00.000000"
}
```

---

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError: No module named 'predictor'

**Solution:**
```bash
# Make sure you're in the project root
cd /Users/outerspace/oncoguardian-model

# Run app from console with correct path
PYTHONPATH=/Users/outerspace/oncoguardian-model source .venv/bin/activate && python -m flask --app src.app run
```

### Issue: Port 5000 already in use

**Solution:**
```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9

# Or run on different port
PORT=8000 python src/app.py
```

### Issue: Predictor not loaded / Models not found

**Solution:**
```bash
# Make sure models exist
ls models/

# Run training script first
source .venv/bin/activate && python src/training.py

# Then run API
python src/app.py
```

### Issue: CORS errors in Flutter

**Solution:** The API already has CORS enabled. If issues persist:

```python
# Clear browser cache
# Or test with curl first (no CORS issues)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '...'
```

---

## 📊 API Performance Notes

- **Latency:** ~50-100ms per prediction (on typical machine)
- **Throughput:** ~10-15 predictions/second
- **Memory:** ~200-300MB (model + predictor)
- **Optimal for:** Single predictions, real-time app usage
- **Batch predictions:** Up to 100 patients/request

---

## Next Steps

1. ✅ Test locally with cURL/Python/Postman
2. 📱 Integrate with Flutter app (see `FLUTTER_INTEGRATION.md`)
3. ☁️ Deploy to Firebase Cloud Functions (see `FIREBASE_DEPLOYMENT.md`)
4. 🔐 Add authentication and rate limiting
5. 📈 Monitor predictions and model performance

---

## References

- Flask Documentation: https://flask.palletsprojects.com/
- CORS: https://flask-cors.readthedocs.io/
- Testing REST APIs: https://www.postman.com/
- VS Code REST Client: https://marketplace.visualstudio.com/items?itemName=humao.rest-client
