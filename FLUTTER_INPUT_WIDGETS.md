# OncoGuardian Flutter Input Widgets Documentation

## Input Widget Specifications for User Data Collection

This document provides detailed specifications for each input field in the OncoGuardian Flutter mobile app, including widget types, validation rules, and value ranges.

---

## Overview: All 15 Required Input Features

| # | Feature Name | Widget Type | Unit/Range | Input Format | Example |
|---|---|---|---|---|---|
| 1 | Age | TextField | 25-90 years | Numeric only | 45 |
| 2 | Gender | RadioListTile | Binary | 0 or 1 | Female (0) |
| 3 | Smoking | Slider | 0-10 scale | Integer | 5 |
| 4 | Alcohol_Use | Slider | 0-10 scale | Integer | 3 |
| 5 | Obesity | Slider | 0-10 scale | Integer | 4 |
| 6 | Family_History | RadioListTile | Binary | 0 or 1 | Yes (1) |
| 7 | Diet_Red_Meat | Slider | 0-10 scale | Integer | 6 |
| 8 | Diet_Salted_Processed | Slider | 0-10 scale | Integer | 4 |
| 9 | Fruit_Veg_Intake | Slider | 0-10 scale | Integer | 8 |
| 10 | Physical_Activity | Slider | 0-10 scale | Integer | 7 |
| 11 | Air_Pollution | Slider | 0-10 scale | Integer | 5 |
| 12 | Occupational_Hazards | Slider | 0-10 scale | Integer | 3 |
| 13 | BRCA_Mutation | RadioListTile | Binary | 0 or 1 | No (0) |
| 14 | H_Pylori_Infection | RadioListTile | Binary | 0 or 1 | No (0) |
| 15 | Calcium_Intake | Slider | 0-10 scale | Integer | 7 |

---

## Detailed Widget Specifications

### 1. **Age** - TextField (Numbers Only)

**Widget Type:** `TextField` with numeric input only

**Specifications:**
```dart
TextField(
  keyboardType: TextInputType.number,
  inputFormatters: [
    FilteringTextInputFormatter.digitsOnly,
    LengthLimitingTextInputFormatter(2),
  ],
  decoration: InputDecoration(
    labelText: 'Age (years)',
    hintText: 'Enter age between 25-90',
    prefixIcon: Icon(Icons.person),
    suffixText: 'years',
    border: OutlineInputBorder(),
  ),
  validator: (value) {
    if (value == null || value.isEmpty) {
      return 'Age is required';
    }
    final age = int.tryParse(value);
    if (age == null || age < 25 || age > 90) {
      return 'Age must be between 25 and 90';
    }
    return null;
  },
)
```

**Python Translation:**
```python
age = int(user_input['Age'])  # Must be 25-90
```

---

### 2. **Gender** - RadioListTile

**Widget Type:** `Radio` or `RadioListTile` (2 options)

**Specifications:**
```dart
// Option 1: Using RadioListTile
RadioListTile<int>(
  title: Text('Female'),
  value: 0,
  groupValue: selectedGender,
  onChanged: (int? value) {
    setState(() => selectedGender = value);
  },
)

RadioListTile<int>(
  title: Text('Male'),
  value: 1,
  groupValue: selectedGender,
  onChanged: (int? value) {
    setState(() => selectedGender = value);
  },
)

// Option 2: Using SegmentedButton (Modern)
SegmentedButton<int>(
  segments: [
    ButtonSegment(
      value: 0,
      label: Text('Female'),
      icon: Icon(Icons.female),
    ),
    ButtonSegment(
      value: 1,
      label: Text('Male'),
      icon: Icon(Icons.male),
    ),
  ],
  selected: {selectedGender},
  onSelectionChanged: (Set<int> newSelection) {
    setState(() => selectedGender = newSelection.first);
  },
)
```

**Encoding:**
- `Female` = 0
- `Male` = 1

**Python Translation:**
```python
gender = int(user_input['Gender'])  # 0 or 1
```

---

### 3-5. **Smoking, Alcohol_Use, Obesity** - Sliders (0-10 Scale)

**Widget Type:** `Slider` with numeric display

**Common Specification Template:**
```dart
// Smoking Level Slider
Column(
  children: [
    Text('Smoking Level: ${smokingLevel.toInt()}'),
    Slider(
      value: smokingLevel.toDouble(),
      min: 0,
      max: 10,
      divisions: 10,
      label: smokingLevel.toInt().toString(),
      onChanged: (double value) {
        setState(() => smokingLevel = value);
      },
    ),
    Padding(
      padding: EdgeInsets.symmetric(vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text('Never'),
          Text('Occasionally'),
          Text('Regularly'),
          Text('Heavily'),
        ],
      ),
    ),
  ],
)

// Similar for Alcohol_Use and Obesity
```

**Scale Interpretation:**

**Smoking (0-10):**
- 0 = Never smoked
- 1-3 = Former smoker (quit)
- 4-6 = Light smoker
- 7-8 = Moderate smoker
- 9-10 = Heavy smoker

**Alcohol_Use (0-10):**
- 0 = No consumption
- 1-3 = Rare/occasional
- 4-6 = Moderate (1-2 drinks/day)
- 7-8 = Heavy (3-4 drinks/day)
- 9-10 = Very heavy (5+ drinks/day)

**Obesity (0-10):**
- 0 = Underweight (BMI < 18.5)
- 1-3 = Normal weight (BMI 18.5-24.9)
- 4-6 = Overweight (BMI 25-29.9)
- 7-8 = Obese Level 1 (BMI 30-34.9)
- 9-10 = Obese Level 2+ (BMI 35+)

**Python Translation:**
```python
smoking = int(slider_value)  # 0-10
alcohol_use = int(slider_value)  # 0-10
obesity = int(slider_value)  # 0-10
```

---

### 6. **Family_History** - RadioListTile (Binary)

**Widget Type:** `Radio` or `SegmentedButton` (2 options)

**Specifications:**
```dart
// Using RadioListTile
Column(
  children: [
    Text('Family History of Cancer:', style: TextStyle(fontWeight: FontWeight.bold)),
    RadioListTile<int>(
      title: Text('No'),
      value: 0,
      groupValue: familyHistory,
      onChanged: (int? value) {
        setState(() => familyHistory = value);
      },
    ),
    RadioListTile<int>(
      title: Text('Yes'),
      value: 1,
      groupValue: familyHistory,
      onChanged: (int? value) {
        setState(() => familyHistory = value);
      },
    ),
  ],
)
```

**Encoding:**
- `No` = 0
- `Yes` = 1

**Python Translation:**
```python
family_history = int(user_input['Family_History'])  # 0 or 1
```

---

### 7-9. **Diet Features** - Sliders (0-10 Scale)

**Widgets:** `Slider` (same as smoking/alcohol)

**Scale Interpretations:**

**Diet_Red_Meat (0-10) - Consumption Frequency:**
- 0 = Never
- 1-3 = Rarely (< 1x/month)
- 4-6 = Moderate (1-2x/week)
- 7-8 = Frequent (3-4x/week)
- 9-10 = Very frequent (Daily+)

**Diet_Salted_Processed (0-10) - Consumption Frequency:**
- 0 = Never eat processed foods
- 1-3 = Rarely (< 1x/week)
- 4-6 = Moderate (2-3x/week)
- 7-8 = Frequent (4-5x/week)
- 9-10 = Very frequent (Daily)

**Fruit_Veg_Intake (0-10) - Daily Servings:**
- 0 = Never eat fruits/vegetables
- 1-3 = <1 serving/day
- 4-6 = 2-3 servings/day
- 7-8 = 4-5 servings/day
- 9-10 = 5+ servings/day (recommended)

**Python Translation:**
```python
diet_red_meat = int(slider_value)  # 0-10
diet_salted_processed = int(slider_value)  # 0-10
fruit_veg_intake = int(slider_value)  # 0-10
```

---

### 10. **Physical_Activity** - Slider (0-10 Scale)

**Widget Type:** `Slider`

**Scale Interpretation (Hours/Week):**
- 0 = Sedentary (0 hours)
- 1-3 = Very light (1-2 hours/week)
- 4-6 = Light (3-4 hours/week)
- 7-8 = Moderate (5-6 hours/week)
- 9-10 = Very active (7+ hours/week)

**Specifications:**
```dart
Column(
  children: [
    Text('Physical Activity Level: ${activityLevel.toInt()}'),
    Slider(
      value: activityLevel.toDouble(),
      min: 0,
      max: 10,
      divisions: 10,
      label: activityLevel.toInt().toString(),
      onChanged: (double value) {
        setState(() => activityLevel = value);
      },
    ),
    Text('0 = Sedentary, 10 = Very Active (7+ hrs/week)'),
  ],
)
```

**Python Translation:**
```python
physical_activity = int(slider_value)  # 0-10
```

---

### 11. **Air_Pollution** - Slider (0-10 Scale)

**Widget Type:** `Slider`

**Scale Interpretation (AQI/Exposure Level):**
- 0 = Clean air (AQI 0-50)
- 1-3 = Moderate (AQI 51-100)
- 4-6 = Unhealthy for sensitive groups (AQI 101-150)
- 7-8 = Unhealthy (AQI 151-200)
- 9-10 = Very unhealthy/Hazardous (AQI 201+)

**Python Translation:**
```python
air_pollution = int(slider_value)  # 0-10
```

---

### 12. **Occupational_Hazards** - Slider (0-10 Scale)

**Widget Type:** `Slider`

**Scale Interpretation (Exposure Risk):**
- 0 = No occupational hazards
- 1-3 = Low exposure (safe environment)
- 4-6 = Moderate exposure (some hazards present)
- 7-8 = High exposure (significant hazards)
- 9-10 = Very high exposure (dangerous workplace)

**Examples:**
- Office work = 1-2
- Factory work = 5-7
- Chemical plant = 8-10
- Construction = 6-8
- Healthcare = 4-6

**Python Translation:**
```python
occupational_hazards = int(slider_value)  # 0-10
```

---

### 13-14. **BRCA_Mutation & H_Pylori_Infection** - RadioListTile (Binary)

**Widget Type:** `Radio` or `SegmentedButton`

**BRCA_Mutation Specifications:**
```dart
Column(
  children: [
    Text('BRCA Mutation Status:', style: TextStyle(fontWeight: FontWeight.bold)),
    RadioListTile<int>(
      title: Text('No BRCA Mutation'),
      subtitle: Text('Low genetic risk'),
      value: 0,
      groupValue: brcaMutation,
      onChanged: (int? value) {
        setState(() => brcaMutation = value);
      },
    ),
    RadioListTile<int>(
      title: Text('Has BRCA Mutation'),
      subtitle: Text('High genetic risk (BRCA1 or BRCA2)'),
      value: 1,
      groupValue: brcaMutation,
      onChanged: (int? value) {
        setState(() => brcaMutation = value);
      },
    ),
  ],
)

// H_Pylori_Infection follows same pattern
```

**Encoding:**
- `No` = 0
- `Yes` = 1

**Python Translation:**
```python
brca_mutation = int(user_input['BRCA_Mutation'])  # 0 or 1
h_pylori_infection = int(user_input['H_Pylori_Infection'])  # 0 or 1
```

---

### 15. **Calcium_Intake** - Slider (0-10 Scale)

**Widget Type:** `Slider`

**Scale Interpretation (Daily Intake in mg):**
- 0 = None (0 mg)
- 1-3 = Low (<400 mg/day)
- 4-6 = Moderate (400-800 mg/day)
- 7-8 = Good (800-1000 mg/day)
- 9-10 = Excellent (1000+ mg/day)

**Python Translation:**
```python
calcium_intake = int(slider_value)  # 0-10
```

---

## Complete Flutter Form Example

```dart
import 'package:flutter/material.dart';

class CancerRiskForm extends StatefulWidget {
  @override
  _CancerRiskFormState createState() => _CancerRiskFormState();
}

class _CancerRiskFormState extends State<CancerRiskForm> {
  final _formKey = GlobalKey<FormState>();
  
  // Store field values
  late int age;
  late int gender = 0;
  late double smoking = 5;
  late double alcoholUse = 5;
  late double obesity = 5;
  late int familyHistory = 0;
  late double dietRedMeat = 5;
  late double dietSaltedProcessed = 5;
  late double fruitVegIntake = 5;
  late double physicalActivity = 5;
  late double airPollution = 5;
  late double occupationalHazards = 3;
  late int brcaMutation = 0;
  late int hPyloriInfection = 0;
  late double calciumIntake = 7;

  @override
  Widget build(BuildContext context) {
    return Form(
      key: _formKey,
      child: SingleChildScrollView(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Age TextField
            TextFormField(
              keyboardType: TextInputType.number,
              decoration: InputDecoration(
                labelText: 'Age (years)',
                hintText: '25-90',
              ),
              validator: (value) {
                if (value == null || value.isEmpty) return 'Required';
                final a = int.tryParse(value);
                if (a == null || a < 25 || a > 90) {
                  return 'Age must be 25-90';
                }
                return null;
              },
              onSaved: (value) => age = int.parse(value!),
            ),
            
            // Gender RadioListTile
            Text('Gender:', style: TextStyle(fontWeight: FontWeight.bold)),
            RadioListTile<int>(
              title: Text('Female'),
              value: 0,
              groupValue: gender,
              onChanged: (int? value) {
                setState(() => gender = value!);
              },
            ),
            RadioListTile<int>(
              title: Text('Male'),
              value: 1,
              groupValue: gender,
              onChanged: (int? value) {
                setState(() => gender = value!);
              },
            ),
            
            // Smoking Slider
            Text('Smoking Level: ${smoking.toInt()}'),
            Slider(
              value: smoking,
              min: 0,
              max: 10,
              divisions: 10,
              onChanged: (value) {
                setState(() => smoking = value);
              },
            ),
            
            // Add similar widgets for other fields...
            
            // Submit Button
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
                if (_formKey.currentState!.validate()) {
                  _formKey.currentState!.save();
                  // Send data to backend/model
                  _submitForm();
                }
              },
              child: Text('Get Cancer Risk Prediction'),
            ),
          ],
        ),
      ),
    );
  }

  void _submitForm() {
    // Create data dictionary matching Python model input
    Map<String, dynamic> patientData = {
      'Age': age,
      'Gender': gender,
      'Smoking': smoking.toInt(),
      'Alcohol_Use': alcoholUse.toInt(),
      'Obesity': obesity.toInt(),
      'Family_History': familyHistory,
      'Diet_Red_Meat': dietRedMeat.toInt(),
      'Diet_Salted_Processed': dietSaltedProcessed.toInt(),
      'Fruit_Veg_Intake': fruitVegIntake.toInt(),
      'Physical_Activity': physicalActivity.toInt(),
      'Air_Pollution': airPollution.toInt(),
      'Occupational_Hazards': occupationalHazards.toInt(),
      'BRCA_Mutation': brcaMutation,
      'H_Pylori_Infection': hPyloriInfection,
      'Calcium_Intake': calciumIntake.toInt(),
    };
    
    // Send to your backend API
    // final result = await apiClient.predictCancerRisk(patientData);
  }
}
```

---

## Python Backend Integration

```python
# In predictor.py or api.py
def get_prediction_from_flutter(patient_data: dict):
    """
    Accept patient data from Flutter app and return prediction.
    
    Expected format:
    {
        'Age': 45,                          # int: 25-90
        'Gender': 0,                        # int: 0 or 1
        'Smoking': 5,                       # int: 0-10
        'Alcohol_Use': 3,                   # int: 0-10
        'Obesity': 4,                       # int: 0-10
        'Family_History': 1,                # int: 0 or 1
        'Diet_Red_Meat': 6,                 # int: 0-10
        'Diet_Salted_Processed': 4,         # int: 0-10
        'Fruit_Veg_Intake': 8,              # int: 0-10
        'Physical_Activity': 7,             # int: 0-10
        'Air_Pollution': 5,                 # int: 0-10
        'Occupational_Hazards': 3,          # int: 0-10
        'BRCA_Mutation': 0,                 # int: 0 or 1
        'H_Pylori_Infection': 0,            # int: 0 or 1
        'Calcium_Intake': 7,                # int: 0-10
    }
    """
    predictor = OncoGuardianPredictor()
    assessment = predictor.get_full_assessment(patient_data)
    return assessment
```

---

## Summary Table: Widget Implementation Checklist

| Feature | Widget | Value Type | Constraint | Status |
|---|---|---|---|---|
| Age | TextField | int | 25-90, digits only | ✅ |
| Gender | RadioListTile | int | 0 or 1 | ✅ |
| Smoking | Slider | int | 0-10 | ✅ |
| Alcohol_Use | Slider | int | 0-10 | ✅ |
| Obesity | Slider | int | 0-10 | ✅ |
| Family_History | RadioListTile | int | 0 or 1 | ✅ |
| Diet_Red_Meat | Slider | int | 0-10 | ✅ |
| Diet_Salted_Processed | Slider | int | 0-10 | ✅ |
| Fruit_Veg_Intake | Slider | int | 0-10 | ✅ |
| Physical_Activity | Slider | int | 0-10 | ✅ |
| Air_Pollution | Slider | int | 0-10 | ✅ |
| Occupational_Hazards | Slider | int | 0-10 | ✅ |
| BRCA_Mutation | RadioListTile | int | 0 or 1 | ✅ |
| H_Pylori_Infection | RadioListTile | int | 0 or 1 | ✅ |
| Calcium_Intake | Slider | int | 0-10 | ✅ |

---

## Quick Reference: Input Validation Rules

```python
# Validation rules for backend
VALIDATION_RULES = {
    'Age': {'type': int, 'min': 25, 'max': 90},
    'Gender': {'type': int, 'values': [0, 1]},
    'Smoking': {'type': int, 'min': 0, 'max': 10},
    'Alcohol_Use': {'type': int, 'min': 0, 'max': 10},
    'Obesity': {'type': int, 'min': 0, 'max': 10},
    'Family_History': {'type': int, 'values': [0, 1]},
    'Diet_Red_Meat': {'type': int, 'min': 0, 'max': 10},
    'Diet_Salted_Processed': {'type': int, 'min': 0, 'max': 10},
    'Fruit_Veg_Intake': {'type': int, 'min': 0, 'max': 10},
    'Physical_Activity': {'type': int, 'min': 0, 'max': 10},
    'Air_Pollution': {'type': int, 'min': 0, 'max': 10},
    'Occupational_Hazards': {'type': int, 'min': 0, 'max': 10},
    'BRCA_Mutation': {'type': int, 'values': [0, 1]},
    'H_Pylori_Infection': {'type': int, 'values': [0, 1]},
    'Calcium_Intake': {'type': int, 'min': 0, 'max': 10},
}
```

---

## Notes for Implementation

- **Binary fields** (Gender, Family_History, etc.): Use RadioListTile or SegmentedButton for clarity
- **Scale fields** (0-10): Use Slider with divisions=10 for step-by-step selection
- **Age field**: TextField with numeric validation only
- **All values** must be converted to `int` before sending to model
- **Display labels** with descriptions to help users understand what each scale means
- **Save all values** in order to match training data feature order
