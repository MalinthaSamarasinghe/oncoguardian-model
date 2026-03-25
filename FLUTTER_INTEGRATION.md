# 🚀 Flutter Integration Guide - OncoGuardian Model

## Converting Python Model to Flutter/TFLite

For your mobile app, you need to export the model in a format Flutter can use. Here are the approaches:

---

## Option 1: REST API with Firebase Cloud Functions (Recommended for Now)

### Why?
- Keep Python model on backend
- Flask/FastAPI server with Firebase Cloud Functions
- Easy to update models without redeploying app
- Secure predictions on backend

### Steps:

#### 1. Create Flask API
```python
# src/app.py
from flask import Flask, request, jsonify
from predictor import OncoGuardianPredictor
import os

app = Flask(__name__)
predictor = OncoGuardianPredictor(verbose=False)

@app.route('/predict', methods=['POST'])
def predict():
    """Make cancer risk prediction"""
    try:
        data = request.json
        prediction = predictor.predict(data)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/recommendations', methods=['POST'])
def recommendations():
    """Get dietary recommendations"""
    try:
        data = request.json
        cancer_type = data.get('cancer_type')
        recs = predictor.get_recommendations(data, cancer_type)
        return jsonify(recs)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=False)
```

#### 2. Deploy to Firebase Cloud Functions

```bash
# Install Firebase CLI
npm install -g firebase-tools
firebase login

# Initialize Firebase project
firebase init

# Deploy
firebase deploy --only functions
```

#### 3. Flutter Code to Call API

```dart
// pubspec.yaml dependencies
dependencies:
  http: ^1.1.0

// lib/services/prediction_service.dart
import 'dart:convert';
import 'package:http/http.dart' as http;

class PredictionService {
  final String apiUrl = 'https://YOUR_FIREBASE_PROJECT.cloudfunctions.net';

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

  Future<Map> getRecommendations(
    Map<String, dynamic> patientData,
    String cancerType,
  ) async {
    try {
      final data = {...patientData, 'cancer_type': cancerType};
      final response = await http.post(
        Uri.parse('$apiUrl/recommendations'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(data),
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception('Error: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Failed to get recommendations: $e');
    }
  }
}

// Usage in Flutter widget
class PredictionScreen extends StatefulWidget {
  @override
  _PredictionScreenState createState() => _PredictionScreenState();
}

class _PredictionScreenState extends State<PredictionScreen> {
  final predictionService = PredictionService();
  bool isLoading = false;

  void _getPrediction() async {
    setState(() => isLoading = true);

    try {
      final patientData = {
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
      };

      final prediction = await predictionService.getPrediction(patientData);
      final recommendations = await predictionService.getRecommendations(
        patientData,
        prediction['predicted_cancer_type'],
      );

      // Display results in UI
      showPredictionResults(prediction, recommendations);
    } catch (e) {
      showError('Error: $e');
    } finally {
      setState(() => isLoading = false);
    }
  }

  void showPredictionResults(Map prediction, Map recommendations) {
    // Update UI with results
    // prediction: {predicted_cancer_type, confidence, probabilities}
    // recommendations: {recommended_foods, foods_to_avoid, supplements, lifestyle_tips}
  }

  void showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message)),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Cancer Risk Prediction')),
      body: Center(
        child: isLoading
            ? CircularProgressIndicator()
            : ElevatedButton(
                onPressed: _getPrediction,
                child: Text('Get Prediction'),
              ),
      ),
    );
  }
}
```

---

## Option 2: TFLite Model (On-Device Prediction)

This allows model to run on device without internet.

### Requirements:
- TensorFlow Lite model (.tflite)
- Flutter TFLite plugin

### Step 1: Convert Model to TFLite

The sklearn Random Forest cannot be directly converted to TFLite. You need to:

#### Method A: Train with TensorFlow (Recommended for production)
```python
import tensorflow as tf
from tensorflow import keras

# Build neural network instead
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(14,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(5, activation='softmax')  # 5 cancer types
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train_onehot, epochs=50, validation_split=0.2)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

#### Method B: Use ONNX converter (More complex)
```bash
# Convert sklearn to ONNX
pip install skl2onnx onnx

# Then convert ONNX to TFLite
pip install onnx-tf
python -m onnx_tf.backend prepare model.onnx model_tf
```

### Step 2: Integrate with Flutter

#### pubspec.yaml
```yaml
dependencies:
  flutter:
    sdk: flutter
  tflite_flutter: ^0.9.0
  tflite_flutter_helper: ^0.0.1
```

#### Dart Code
```dart
import 'package:tflite_flutter/tflite_flutter.dart';

class OncoGuardianPredictor {
  late Interpreter interpreter;
  
  Future<void> loadModel() async {
    interpreter = await Interpreter.fromAsset('assets/model.tflite');
  }

  List<List<double>> predict(List<double> input) {
    var output = List.generate(5, (i) => [0.0]);
    interpreter.run(input, output);
    return output;
  }

  void dispose() {
    interpreter.close();
  }
}
```

---

## Option 3: Python Backend with Firebase Realtime Database

### Setup:

#### 1. Python Backend
```python
# src/firebase_service.py
import firebase_admin
from firebase_admin import db
from predictor import OncoGuardianPredictor

firebase_admin.initialize_app(options={
    'databaseURL': 'https://YOUR_PROJECT.firebaseio.com'
})

predictor = OncoGuardianPredictor(verbose=False)

def handle_prediction_request(patient_id, patient_data):
    """Handle prediction request from Firebase"""
    prediction = predictor.predict(patient_data)
    recommendations = predictor.get_recommendations(patient_data)
    
    # Store in Firebase
    db.reference(f'predictions/{patient_id}').set({
        'prediction': prediction,
        'recommendations': recommendations,
        'timestamp': str(datetime.now())
    })
```

#### 2. Flutter Integration
```dart
import 'package:firebase_database/firebase_database.dart';

class FirebasePredictionService {
  final database = FirebaseDatabase.instance.ref();

  Future<Map> getPrediction(String patientId, Map patientData) async {
    // Send prediction request
    await database
        .child('prediction_requests')
        .child(patientId)
        .set(patientData);

    // Listen for response
    return database
        .child('predictions')
        .child(patientId)
        .onValue
        .first
        .then((event) => Map.from(event.snapshot.value as Map))
        .timeout(Duration(seconds: 30));
  }
}
```

---

## Recommended Approach for Your Project

1. **For Assignment:** Use **Option 1 (REST API)**
   - Simple to understand
   - Can run locally for testing
   - Scales well with Firebase

2. **For Production:** Combine **Option 1 + Option 2**
   - Critical predictions on server (Python model)
   - Quick predictions on device (TFLite for common cases)
   - Best of both worlds

3. **Data Flow:**
```
Flutter App
    ↓
Form Input (age, gender, smoking, etc.)
    ↓
Validation (check all fields)
    ↓
API Call (REST to Firebase Cloud Function)
    ↓
Python Backend
    ↓
Load Model
    ↓
Preprocess Input
    ↓
Make Prediction
    ↓
Get Recommendations
    ↓
Send JSON Response
    ↓
Flutter App displays results
```

---

## Firebase Firestore Data Structure

```
/patients/{userId}
├── /profile
│   ├── age: 45
│   ├── gender: "Female"
│   └── ...
├── /predictions
│   ├── {predictionId}
│   │   ├── predicted_cancer: "Breast"
│   │   ├── confidence: 0.92
│   │   ├── timestamp: "2024-03-25T10:30:00"
│   │   └── probabilities
│   │       ├── Lung: 0.05
│   │       ├── Breast: 0.92
│   │       └── ...
└── /recommendations
    ├── {predictionId}
    │   ├── risk_level: "HIGH"
    │   ├── foods: [...]
    │   └── lifestyle_tips: [...]
```

---

## For Your Assignment Report

Document:
1. ✅ Model training metrics (accuracy, F1, etc.)
2. ✅ Feature importance analysis
3. ✅ Confusion matrix and ROC curves
4. ✅ Cancer type predictions (which are easiest/hardest)
5. ✅ How model will integrate with Flutter
6. ✅ Firebase architecture diagram
7. ✅ API endpoint documentation
8. ✅ Data privacy considerations

---

## Next Steps

1. **Immediate:**
   ```bash
   python src/training.py
   ```

2. **Short term:**
   - Create Flask API wrapper
   - Set up local testing

3. **Medium term:**
   - Deploy to Firebase Cloud Functions
   - Create Flutter screens
   - Integrate with Firebase

4. **Long term:**
   - Add user authentication
   - Store prediction history
   - Implement in-app model updates
   - Add more cancer types

---

## Troubleshooting

### API not responding?
- Check internet connection
- Verify Firebase project is active
- Check Cloud Function logs

### Model predictions inconsistent?
- Ensure data preprocessing matches training
- Check feature names and order
- Verify categorical encoding

### TFLite conversion failing?
- Use TensorFlow model instead of sklearn
- Check input/output shapes
- Use recommended conversion tools

---

For more help, see:
- [Firebase Documentation](https://firebase.google.com/docs)
- [Flutter Firebase Guide](https://firebase.google.com/docs/flutter/setup)
- [TFLite Documentation](https://www.tensorflow.org/lite)
- [CODE_EXPLANATION.md](./CODE_EXPLANATION.md)
