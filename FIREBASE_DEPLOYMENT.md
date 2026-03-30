# OncoGuardian API - Firebase Cloud Functions Deployment Guide

## 🚀 Deploy Flask API to Firebase Cloud Functions

This guide covers deploying the OncoGuardian Flask API to Firebase so it can be accessed from your Flutter app globally.

---

## Prerequisites

1. **Firebase Account**: https://firebase.google.com/
2. **Firebase CLI**: Install globally
3. **Python 3.9+**: For Cloud Functions runtime
4. **Project Structure**: Must have `src/app.py` and `models/` directory

---

## Step 1: Install Firebase CLI

```bash
# macOS with Homebrew
brew install firebase-tools

# Or npm (any OS)
npm install -g firebase-tools

# Verify installation
firebase --version
```

---

## Step 2: Create Firebase Project

### Option A: Use Existing Google Cloud Project

```bash
# Login to Firebase
firebase login

# List existing projects
firebase projects:list

# Use existing project
firebase init --project=YOUR_PROJECT_ID
```

### Option B: Create New Firebase Project

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Click "Create a project"
3. Enter project name: `oncoguardian-model` (or your choice)
4. Skip Google Analytics (optional)
5. Click "Create project"

---

## Step 3: Initialize Firebase in Your Project

```bash
# From project root
cd /Users/outerspace/oncoguardian-model

# Initialize Firebase
firebase init functions

# Select options:
# ? Which language would you like to use to write Cloud Functions? → TypeScript (or JavaScript)
# ? Do you want to use ESLint to catch probable bugs? → Y
# ? Do you want to install dependencies now? → Y
```

This creates:
```
functions/
├── .eslintrc.json
├── package.json
├── tsconfig.json
└── src/
    └── index.ts
```

---

## Step 4: Create Python Wrapper Function

Firebase Cloud Functions supports Python runtimes. Create a Python-based function.

### A. Update `functions/package.json`

```bash
# In functions directory
cd functions
npm install express cors
```

### B. Create Python Entry Point

Create `functions/src/main.py`:

```python
"""
Firebase Cloud Function wrapper for OncoGuardian Flask API
Deployed to: https://us-central1-YOUR_PROJECT.cloudfunctions.net
"""

import functions_framework
from flask import Flask, request, jsonify
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.predictor import OncoGuardianPredictor
from datetime import datetime

# Initialize predictor
try:
    predictor = OncoGuardianPredictor(verbose=False)
    PREDICTOR_LOADED = True
except Exception as e:
    print(f"Error loading predictor: {e}")
    PREDICTOR_LOADED = False

# Create Flask app
app = Flask(__name__)

# Enable CORS
from flask_cors import CORS
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/predict', methods=['POST'])
def predict():
    if not PREDICTOR_LOADED:
        return jsonify({'error': 'Predictor not loaded'}), 503
    
    try:
        patient_data = request.get_json()
        prediction = predictor.predict(patient_data)
        return jsonify({
            'success': True,
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/recommendations', methods=['POST'])
def recommendations():
    if not PREDICTOR_LOADED:
        return jsonify({'error': 'Predictor not loaded'}), 503
    
    try:
        data = request.get_json()
        cancer_type = data.get('cancer_type')
        
        if not cancer_type:
            prediction = predictor.predict(data)
            cancer_type = prediction['predicted_cancer_type']
        
        recommendations = predictor.get_recommendations(data, cancer_type)
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok' if PREDICTOR_LOADED else 'error',
        'predictor_loaded': PREDICTOR_LOADED
    })

@functions_framework.http
def api(request: functions_framework.Request):
    """HTTP Cloud Function entry point"""
    with app.app_context():
        return app.full_dispatch_request()
```

---

## Step 5: Alternative - Use Node.js Wrapper (Simpler)

Instead of Python, use Node.js to call your local API:

Create `functions/src/index.ts`:

```typescript
import * as functions from "firebase-functions";
import * as admin from "firebase-admin";
import axios from "axios";

admin.initializeApp();

const PREDICTION_API_URL = "https://your-backend-server.com/predict";
// Or for local testing:
// const PREDICTION_API_URL = "http://localhost:5000/predict";

export const predict = functions.https.onRequest(async (req, res) => {
  functions.logger.info("Predict request received", { structuredData: true });

  // Enable CORS
  res.set("Access-Control-Allow-Origin", "*");
  res.set("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS, POST");
  res.set("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    res.status(204).send("");
    return;
  }

  try {
    // Forward request to Python backend
    const response = await axios.post(PREDICTION_API_URL, req.body);
    res.json(response.data);
  } catch (error: any) {
    functions.logger.error("Prediction error:", error);
    res.status(500).json({
      success: false,
      error: error.message,
    });
  }
});

export const recommendations = functions.https.onRequest(async (req, res) => {
  functions.logger.info("Recommendations request received");

  res.set("Access-Control-Allow-Origin", "*");
  res.set("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS, POST");
  res.set("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    res.status(204).send("");
    return;
  }

  try {
    const response = await axios.post(
      `${PREDICTION_API_URL.replace("/predict", "")}/recommendations`,
      req.body
    );
    res.json(response.data);
  } catch (error: any) {
    functions.logger.error("Recommendations error:", error);
    res.status(500).json({
      success: false,
      error: error.message,
    });
  }
});
```

Update `functions/package.json`:

```json
{
  "name": "functions",
  "description": "OncoGuardian Cloud Functions",
  "type": "module",
  "engines": {
    "node": "18"
  },
  "main": "lib/index.js",
  "scripts": {
    "build": "tsc",
    "serve": "npm run build && firebase emulators:start --only functions",
    "shell": "npm run build && firebase functions:shell",
    "start": "npm run shell",
    "deploy": "firebase deploy --only functions",
    "logs": "firebase functions:log"
  },
  "dependencies": {
    "firebase-admin": "^11.8.0",
    "firebase-functions": "^4.4.1",
    "axios": "^1.4.0"
  },
  "devDependencies": {
    "@typescript-eslint/eslint-plugin": "^5.62.0",
    "@typescript-eslint/parser": "^5.62.0",
    "eslint": "^8.45.0",
    "typescript": "^5.1.6"
  }
}
```

---

## Step 6: Deploy to Firebase

### Option A: Deploy with Python Runtime (Recommended for Your Use Case)

```bash
# From project root
firebase deploy --only functions

# Watch logs
firebase functions:log
```

### Option B: Deploy with Node.js Runtime (Simpler)

```bash
# From functions directory
cd functions

# Install dependencies
npm install

# Build TypeScript
npm run build

# Deploy
firebase deploy --only functions

# View logs
firebase functions:log
```

---

## Step 7: Get Your Firebase API URL

After deployment, Firebase gives you a URL like:

```
https://us-central1-your-project-id.cloudfunctions.net/predict
https://us-central1-your-project-id.cloudfunctions.net/recommendations
```

Save this URL - you'll need it in your Flutter app!

---

## Step 8: Test Deployed Function

### Via cURL

```bash
curl -X POST https://us-central1-YOUR_PROJECT_ID.cloudfunctions.net/predict \
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
```

### Via Python

```python
import requests

url = "https://us-central1-YOUR_PROJECT_ID.cloudfunctions.net/predict"

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

response = requests.post(url, json=patient_data)
print(response.json())
```

---

## Step 9: Configure for Flutter App

In your Flutter app (`lib/services/prediction_service.dart`):

```dart
class PredictionService {
  // Your Firebase Cloud Function URL
  final String apiUrl = 'https://us-central1-YOUR_PROJECT_ID.cloudfunctions.net';

  Future<Map> getPrediction(Map<String, dynamic> patientData) async {
    try {
      final response = await http.post(
        Uri.parse('$apiUrl/predict'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(patientData),
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception('Error: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Failed to get prediction: $e');
    }
  }
}
```

---

## 🔐 Security Considerations

### 1. Enable Authentication

Add to `functions/src/index.ts`:

```typescript
export const predict = functions.https.onRequest(async (req, res) => {
  // Check authentication token
  const authHeader = req.headers.authorization;
  
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    res.status(401).json({ error: "Unauthorized" });
    return;
  }

  const token = authHeader.substring(7);
  
  try {
    await admin.auth().verifyIdToken(token);
    // Proceed with prediction
  } catch (error) {
    res.status(401).json({ error: "Invalid token" });
    return;
  }
  // ... rest of function
});
```

### 2. Rate Limiting

Use Firebase Functions with scheduled deletion of old predictions:

```typescript
export const cleanupOldPredictions = functions.pubsub
  .schedule("every 24 hours")
  .onRun(async (context) => {
    const db = admin.firestore();
    const cutoff = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000); // 30 days ago
    
    const batch = db.batch();
    const snapshot = await db.collection("predictions")
      .where("timestamp", "<", cutoff)
      .limit(100)
      .get();
    
    snapshot.docs.forEach((doc) => batch.delete(doc.ref));
    await batch.commit();
  });
```

### 3. Enable IAM Authentication

```bash
# In Firebase Console → Cloud Functions:
# For each function, set access to "Require authentication"
```

---

## 📊 Monitoring & Logs

```bash
# View all function logs
firebase functions:log

# View specific function logs
firebase functions:log --function=predict

# Setup performance alerts
# Firebase Console → Functions → Monitor
```

---

## 🐛 Troubleshooting

### Issue: "Permission denied" error

**Solution:**
```bash
# Enable required APIs
gcloud services enable cloudfunctions.googleapis.com
gcloud services enable firebaseextensions.googleapis.com
gcloud services enable storage-api.googleapis.com
gcloud services enable logging.googleapis.com
```

### Issue: Function timeout

**Solution:** Increase timeout in `firebase.json`:

```json
{
  "functions": {
    "source": "functions",
    "codebase": "default",
    "ignoreCopyBuildPhase": false,
    "runtime": "python39"
  }
}
```

Then update function:

```python
@functions_framework.http
def api(request, timeout=540):  # 9 minutes
    ...
```

### Issue: Models not found in Cloud Function

**Solution:**

Upload models to Google Cloud Storage:

```bash
# Create bucket
gsutil mb gs://oncoguardian-models

# Upload models
gsutil -m cp -r models/* gs://oncoguardian-models/

# Download in function
from google.cloud import storage

def download_models():
    client = storage.Client()
    bucket = client.bucket('oncoguardian-models')
    for blob in bucket.list_blobs():
        blob.download_to_filename(f'models/{blob.name}')
```

Then in function code:

```python
if not os.path.exists('models/model.pkl'):
    download_models()

predictor = OncoGuardianPredictor()
```

---

## 💾 Deploying Models to Firebase

### Option 1: Firebase Storage

```bash
# Create storage bucket
firebase init storage

# Upload models
gsutil cp models/* gs://YOUR_PROJECT.appspot.com/models/
```

Access in code:

```python
from google.cloud import storage
import os

def download_models_from_storage():
    client = storage.Client()
    bucket = client.bucket(f'{PROJECT_ID}.appspot.com')
    
    for model_file in ['model.pkl', 'scaler.pkl', 'label_encoders.pkl']:
        blob = bucket.blob(f'models/{model_file}')
        blob.download_to_filename(f'/tmp/{model_file}')
```

### Option 2: Package Models with Function

Keep models in `functions/models/` directory:

```
functions/
├── src/
│   └── main.py
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── label_encoders.pkl
└── package.json
```

---

## ✅ Deployment Checklist

- [ ] Firebase account created
- [ ] Firebase CLI installed
- [ ] Firebase project initialized
- [ ] Models packaged/uploaded
- [ ] Cloud Function deployed
- [ ] API URL saved
- [ ] CORS enabled
- [ ] Authentication configured (optional)
- [ ] Rate limiting configured (optional)
- [ ] Flutter app updated with API URL
- [ ] End-to-end testing completed

---

## 📱 Next Steps

1. ✅ Deploy API to Firebase Cloud Functions
2. 📱 Integrate URL with Flutter app
3. 🧪 Test predictions from mobile device
4. 🔐 Setup user authentication
5. 📊 Monitor API performance and usage
6. 🚀 Publish Flutter app to stores

---

## References

- [Firebase Cloud Functions Documentation](https://firebase.google.com/docs/functions)
- [Deploy Python Functions](https://firebase.google.com/docs/functions/runtime-support)
- [Firebase Security Rules](https://firebase.google.com/docs/database/security)
- [Google Cloud Platform Console](https://console.cloud.google.com/)
