# OncoGuardian Flutter Integration - Complete Implementation Guide

## 📱 Flutter + REST API Integration

This guide provides complete Flutter code to integrate with the OncoGuardian REST API.

---

## 📋 Table of Contents

1. [Project Setup](#project-setup)
2. [Dependencies](#dependencies)
3. [API Service Layer](#api-service-layer)
4. [Main App Structure](#main-app-structure)
5. [Prediction Screen](#prediction-screen)
6. [Results Display](#results-display)
7. [State Management](#state-management)
8. [Error Handling](#error-handling)
9. [Testing](#testing)

---

## Project Setup

### 1. Create Flutter Project

```bash
# Create new Flutter project
flutter create oncoguardian_mobile

cd oncoguardian_mobile

# Get dependencies
flutter pub get
```

---

## Dependencies

### pubspec.yaml

```yaml
name: oncoguardian_mobile
description: OncoGuardian Cancer Risk Prediction Mobile App
publish_to: 'none'

version: 1.0.0+1

environment:
  sdk: '>=2.19.0 <4.0.0'

dependencies:
  flutter:
    sdk: flutter

  # HTTP & API
  http: ^1.1.0
  dio: ^5.3.1

  # State Management
  provider: ^6.0.0
  bloc: ^8.1.2
  flutter_bloc: ^8.1.3

  # UI & Styling
  cupertino_icons: ^1.0.2
  google_fonts: ^5.1.0
  flutter_spinkit: ^5.2.0

  # Data
  json_serializable: ^6.7.1
  json_annotation: ^4.8.1

  # Storage
  shared_preferences: ^2.2.2
  sqflite: ^2.3.0

  # Firebase (Optional)
  firebase_core: ^2.24.0
  firebase_analytics: ^10.7.0
  firebase_auth: ^4.14.0

  # Utilities
  intl: ^0.19.0
  logger: ^2.0.1
  connectivity_plus: ^5.0.2

dev_dependencies:
  flutter_test:
    sdk: flutter
  flutter_lints: ^3.0.0
  build_runner: ^2.4.6
  json_serializable: ^6.7.1

flutter:
  uses-material-design: true
  assets:
    - assets/images/
    - assets/icons/
  fonts:
    - family: CustomFont
      fonts:
        - asset: assets/fonts/CustomFont-Regular.ttf
        - asset: assets/fonts/CustomFont-Bold.ttf
          weight: 700
```

---

## API Service Layer

### 1. Models

Create `lib/models/prediction.dart`:

```dart
import 'package:json_annotation/json_annotation.dart';

part 'prediction.g.dart';

@JsonSerializable()
class Prediction {
  final bool success;
  final PredictionData? prediction;
  final String? error;
  final String timestamp;

  Prediction({
    required this.success,
    this.prediction,
    this.error,
    required this.timestamp,
  });

  factory Prediction.fromJson(Map<String, dynamic> json) =>
      _$PredictionFromJson(json);
  Map<String, dynamic> toJson() => _$PredictionToJson(this);
}

@JsonSerializable()
class PredictionData {
  @JsonKey(name: 'predicted_cancer_type')
  final String predictedCancerType;
  
  @JsonKey(name: 'risk_level')
  final String riskLevel;
  
  final double confidence;
  final Map<String, double> probabilities;

  PredictionData({
    required this.predictedCancerType,
    required this.riskLevel,
    required this.confidence,
    required this.probabilities,
  });

  factory PredictionData.fromJson(Map<String, dynamic> json) =>
      _$PredictionDataFromJson(json);
  Map<String, dynamic> toJson() => _$PredictionDataToJson(this);
}

@JsonSerializable()
class Recommendations {
  final bool success;
  final RecommendationData? recommendations;
  final String? error;
  final String timestamp;

  Recommendations({
    required this.success,
    this.recommendations,
    this.error,
    required this.timestamp,
  });

  factory Recommendations.fromJson(Map<String, dynamic> json) =>
      _$RecommendationsFromJson(json);
  Map<String, dynamic> toJson() => _$RecommendationsToJson(this);
}

@JsonSerializable()
class RecommendationData {
  @JsonKey(name: 'risk_level')
  final String riskLevel;
  
  @JsonKey(name: 'cancer_type')
  final String cancerType;
  
  @JsonKey(name: 'recommended_foods')
  final List<String> recommendedFoods;
  
  @JsonKey(name: 'foods_to_avoid')
  final List<String> foodsToAvoid;
  
  final List<String> supplements;
  
  @JsonKey(name: 'lifestyle_tips')
  final List<String> lifestyleTips;

  RecommendationData({
    required this.riskLevel,
    required this.cancerType,
    required this.recommendedFoods,
    required this.foodsToAvoid,
    required this.supplements,
    required this.lifestyleTips,
  });

  factory RecommendationData.fromJson(Map<String, dynamic> json) =>
      _$RecommendationDataFromJson(json);
  Map<String, dynamic> toJson() => _$RecommendationDataToJson(this);
}

@JsonSerializable()
class PatientData {
  final int Age;
  final int Gender;
  final double Height;
  final double Weight;
  final int Smoking;
  final int Alcohol_Use;
  final int Physical_Activity;
  final int Diet_Red_Meat;
  final int Diet_Salted_Processed;
  final int Fruit_Veg_Intake;
  final int Air_Pollution;
  final int Occupational_Hazards;
  final int Family_History;
  final int BRCA_Mutation;
  final int H_Pylori_Infection;
  final int Calcium_Intake;

  PatientData({
    required this.Age,
    required this.Gender,
    required this.Height,
    required this.Weight,
    required this.Smoking,
    required this.Alcohol_Use,
    required this.Physical_Activity,
    required this.Diet_Red_Meat,
    required this.Diet_Salted_Processed,
    required this.Fruit_Veg_Intake,
    required this.Air_Pollution,
    required this.Occupational_Hazards,
    required this.Family_History,
    required this.BRCA_Mutation,
    required this.H_Pylori_Infection,
    required this.Calcium_Intake,
  });

  factory PatientData.fromJson(Map<String, dynamic> json) =>
      _$PatientDataFromJson(json);
  Map<String, dynamic> toJson() => _$PatientDataToJson(this);
}
```

### 2. API Service

Create `lib/services/api_service.dart`:

```dart
import 'package:http/http.dart' as http;
import 'dart:convert';
import '../models/prediction.dart';
import 'package:logger/logger.dart';

class ApiService {
  // Configure this for your deployment
  static const String _baseUrl = 
    'https://us-central1-YOUR_PROJECT.cloudfunctions.net';
  
  // For local testing:
  // static const String _baseUrl = 'http://localhost:5000';

  final http.Client client;
  final Logger logger = Logger();

  ApiService({http.Client? client}) 
    : client = client ?? http.Client();

  /// Get prediction from API
  Future<Prediction> getPrediction(PatientData patientData) async {
    try {
      logger.i('Sending prediction request for patient');
      
      final response = await client.post(
        Uri.parse('$_baseUrl/predict'),
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: jsonEncode(patientData.toJson()),
      ).timeout(
        const Duration(seconds: 30),
        onTimeout: () => throw Exception('Request timeout'),
      );

      logger.d('Response status: ${response.statusCode}');
      
      if (response.statusCode == 200) {
        final jsonData = jsonDecode(response.body);
        logger.i('Prediction received successfully');
        return Prediction.fromJson(jsonData);
      } else {
        logger.e('API Error: ${response.statusCode}');
        throw Exception(
          'Failed to get prediction: ${response.statusCode}'
        );
      }
    } on http.ClientException catch (e) {
      logger.e('Network error: $e');
      throw Exception('Network error: Check your connection');
    } catch (e) {
      logger.e('Error: $e');
      rethrow;
    }
  }

  /// Get recommendations from API
  Future<Recommendations> getRecommendations(
    PatientData patientData,
    String cancerType,
  ) async {
    try {
      logger.i('Requesting recommendations for $cancerType');
      
      final data = {
        ...patientData.toJson(),
        'cancer_type': cancerType,
      };

      final response = await client.post(
        Uri.parse('$_baseUrl/recommendations'),
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: jsonEncode(data),
      ).timeout(
        const Duration(seconds: 30),
        onTimeout: () => throw Exception('Request timeout'),
      );

      logger.d('Response status: ${response.statusCode}');
      
      if (response.statusCode == 200) {
        final jsonData = jsonDecode(response.body);
        logger.i('Recommendations received successfully');
        return Recommendations.fromJson(jsonData);
      } else {
        logger.e('API Error: ${response.statusCode}');
        throw Exception(
          'Failed to get recommendations: ${response.statusCode}'
        );
      }
    } on http.ClientException catch (e) {
      logger.e('Network error: $e');
      throw Exception('Network error: Check your connection');
    } catch (e) {
      logger.e('Error: $e');
      rethrow;
    }
  }

  /// Check API health
  Future<bool> checkHealth() async {
    try {
      final response = await client.get(
        Uri.parse('$_baseUrl/health'),
      ).timeout(
        const Duration(seconds: 10),
        onTimeout: () => throw Exception('Health check timeout'),
      );
      return response.statusCode == 200;
    } catch (e) {
      logger.w('Health check failed: $e');
      return false;
    }
  }

  void dispose() {
    client.close();
  }
}
```

---

## Main App Structure

### lib/main.dart

```dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'services/api_service.dart';
import 'screens/home_screen.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        Provider<ApiService>(create: (_) => ApiService()),
      ],
      child: MaterialApp(
        title: 'OncoGuardian',
        theme: ThemeData(
          primarySwatch: Colors.blue,
          useMaterial3: true,
          appBarTheme: const AppBarTheme(
            backgroundColor: Color(0xFF007AFF),
            elevation: 0,
          ),
        ),
        home: const HomeScreen(),
        debugShowCheckedModeBanner: false,
      ),
    );
  }
}
```

---

## Prediction Screen

### lib/screens/prediction_screen.dart

```dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../models/prediction.dart';
import '../services/api_service.dart';
import 'results_screen.dart';

class PredictionScreen extends StatefulWidget {
  const PredictionScreen({Key? key}) : super(key: key);

  @override
  State<PredictionScreen> createState() => _PredictionScreenState();
}

class _PredictionScreenState extends State<PredictionScreen> {
  final _formKey = GlobalKey<FormState>();
  bool _isLoading = false;

  // Patient data form fields
  int age = 45;
  int gender = 1; // 1 = Female, 0 = Male
  double height = 1.7;
  double weight = 65;
  int smoking = 0;
  int alcoholUse = 5;
  int physicalActivity = 8;
  int dietRedMeat = 3;
  int dietSaltedProcessed = 2;
  int fruitVegIntake = 9;
  int airPollution = 4;
  int occupationalHazards = 1;
  int familyHistory = 0;
  int brcaMutation = 0;
  int hPyloriInfection = 0;
  int calciumIntake = 7;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Cancer Risk Assessment'),
        centerTitle: true,
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Form(
                key: _formKey,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _buildSectionHeader('Demographics'),
                    _buildSlider('Age', age, 18, 100, (v) => age = v.toInt()),
                    _buildGenderDropdown(),
                    _buildSlider(
                      'Height (m)',
                      height,
                      1.4,
                      2.2,
                      (v) => height = v,
                    ),
                    _buildSlider('Weight (kg)', weight, 40, 150, (v) => weight = v),
                    
                    const SizedBox(height: 20),
                    _buildSectionHeader('Lifestyle Factors'),
                    _buildSlider('Smoking (0-10)', smoking.toDouble(), 0, 10,
                        (v) => smoking = v.toInt()),
                    _buildSlider('Alcohol Use (0-10)', alcoholUse.toDouble(), 0, 10,
                        (v) => alcoholUse = v.toInt()),
                    _buildSlider('Physical Activity (0-10)', 
                        physicalActivity.toDouble(), 0, 10,
                        (v) => physicalActivity = v.toInt()),
                    
                    const SizedBox(height: 20),
                    _buildSectionHeader('Diet & Nutrition'),
                    _buildSlider('Red Meat Diet (0-10)', 
                        dietRedMeat.toDouble(), 0, 10,
                        (v) => dietRedMeat = v.toInt()),
                    _buildSlider('Salted/Processed (0-10)', 
                        dietSaltedProcessed.toDouble(), 0, 10,
                        (v) => dietSaltedProcessed = v.toInt()),
                    _buildSlider('Fruit/Veg Intake (0-10)', 
                        fruitVegIntake.toDouble(), 0, 10,
                        (v) => fruitVegIntake = v.toInt()),
                    _buildSlider('Calcium Intake (0-10)', 
                        calciumIntake.toDouble(), 0, 10,
                        (v) => calciumIntake = v.toInt()),
                    
                    const SizedBox(height: 20),
                    _buildSectionHeader('Environmental & Medical'),
                    _buildSlider('Air Pollution (0-10)', 
                        airPollution.toDouble(), 0, 10,
                        (v) => airPollution = v.toInt()),
                    _buildSlider('Occupational Hazards (0-10)', 
                        occupationalHazards.toDouble(), 0, 10,
                        (v) => occupationalHazards = v.toInt()),
                    _buildCheckbox('Family History', familyHistory == 1, 
                        (v) => familyHistory = v! ? 1 : 0),
                    _buildCheckbox('BRCA Mutation', brcaMutation == 1, 
                        (v) => brcaMutation = v! ? 1 : 0),
                    _buildCheckbox('H. Pylori Infection', hPyloriInfection == 1, 
                        (v) => hPyloriInfection = v! ? 1 : 0),
                    
                    const SizedBox(height: 30),
                    SizedBox(
                      width: double.infinity,
                      height: 56,
                      child: ElevatedButton(
                        onPressed: _submitForm,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFF007AFF),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                        ),
                        child: const Text(
                          'Get Prediction',
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                            color: Colors.white,
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(height: 20),
                  ],
                ),
              ),
            ),
    );
  }

  Widget _buildSectionHeader(String title) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 16),
      child: Text(
        title,
        style: const TextStyle(
          fontSize: 18,
          fontWeight: FontWeight.bold,
          color: Color(0xFF007AFF),
        ),
      ),
    );
  }

  Widget _buildSlider(
    String label,
    double value,
    double min,
    double max,
    Function(double) onChanged,
  ) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(label, style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w500)),
            Text(value.toStringAsFixed(1), 
              style: const TextStyle(fontSize: 14, fontWeight: FontWeight.bold, color: Color(0xFF007AFF)),
            ),
          ],
        ),
        Slider(
          value: value,
          min: min,
          max: max,
          divisions: ((max - min) * 2).toInt(),
          label: value.toStringAsFixed(1),
          onChanged: onChanged,
        ),
        const SizedBox(height: 8),
      ],
    );
  }

  Widget _buildGenderDropdown() {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: DropdownButtonFormField<int>(
        value: gender,
        decoration: InputDecoration(
          labelText: 'Gender',
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(8),
          ),
        ),
        items: const [
          DropdownMenuItem(value: 0, child: Text('Male')),
          DropdownMenuItem(value: 1, child: Text('Female')),
        ],
        onChanged: (value) {
          if (value != null) gender = value;
        },
      ),
    );
  }

  Widget _buildCheckbox(String label, bool value, Function(bool?) onChanged) {
    return CheckboxListTile(
      title: Text(label),
      value: value,
      onChanged: onChanged,
      contentPadding: EdgeInsets.zero,
    );
  }

  void _submitForm() async {
    if (_formKey.currentState == null) return;
    
    _formKey.currentState!.save();
    
    setState(() => _isLoading = true);

    try {
      final patientData = PatientData(
        Age: age,
        Gender: gender,
        Height: height,
        Weight: weight,
        Smoking: smoking,
        Alcohol_Use: alcoholUse,
        Physical_Activity: physicalActivity,
        Diet_Red_Meat: dietRedMeat,
        Diet_Salted_Processed: dietSaltedProcessed,
        Fruit_Veg_Intake: fruitVegIntake,
        Air_Pollution: airPollution,
        Occupational_Hazards: occupationalHazards,
        Family_History: familyHistory,
        BRCA_Mutation: brcaMutation,
        H_Pylori_Infection: hPyloriInfection,
        Calcium_Intake: calciumIntake,
      );

      final apiService = context.read<ApiService>();
      final prediction = await apiService.getPrediction(patientData);

      if (!mounted) return;

      if (prediction.success && prediction.prediction != null) {
        // Get recommendations
        final recommendations = await apiService.getRecommendations(
          patientData,
          prediction.prediction!.predictedCancerType,
        );

        if (!mounted) return;

        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => ResultsScreen(
              prediction: prediction.prediction!,
              recommendations: recommendations.recommendations,
            ),
          ),
        );
      } else {
        _showErrorDialog(prediction.error ?? 'Unknown error');
      }
    } catch (e) {
      _showErrorDialog('Error: $e');
    } finally {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  void _showErrorDialog(String message) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Error'),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }
}
```

---

## Results Display

### lib/screens/results_screen.dart

```dart
import 'package:flutter/material.dart';
import '../models/prediction.dart';

class ResultsScreen extends StatelessWidget {
  final PredictionData prediction;
  final RecommendationData? recommendations;

  const ResultsScreen({
    Key? key,
    required this.prediction,
    this.recommendations,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Prediction Results'),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Risk Level Card
            _buildRiskLevelCard(),
            const SizedBox(height: 20),

            // Cancer Type & Confidence
            _buildPredictionCard(),
            const SizedBox(height: 20),

            // Probability Distribution
            _buildProbabilityChart(),
            const SizedBox(height: 20),

            // Recommendations
            if (recommendations != null) ...[
              _buildRecommendationsSection(),
            ],

            const SizedBox(height: 30),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: () => Navigator.popUntil(context, (route) => route.isFirst),
                child: const Text('New Assessment'),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildRiskLevelCard() {
    final riskColor = _getRiskColor(prediction.riskLevel);
    
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      color: riskColor.withOpacity(0.1),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Row(
          children: [
            Icon(
              _getRiskIcon(prediction.riskLevel),
              color: riskColor,
              size: 48,
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Overall Risk Level',
                    style: TextStyle(fontSize: 14, color: Colors.grey),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    prediction.riskLevel.toUpperCase(),
                    style: TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                      color: riskColor,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPredictionCard() {
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Primary Cancer Type',
              style: TextStyle(fontSize: 14, color: Colors.grey),
            ),
            const SizedBox(height: 8),
            Text(
              prediction.predictedCancerType,
              style: const TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.bold,
                color: Color(0xFF007AFF),
              ),
            ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text('Confidence Score:'),
                Text(
                  '${(prediction.confidence * 100).toStringAsFixed(1)}%',
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value: prediction.confidence,
                minHeight: 8,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildProbabilityChart() {
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Probability Distribution',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            ...prediction.probabilities.entries.map((entry) {
              return Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(entry.key),
                        Text(
                          '${(entry.value * 100).toStringAsFixed(1)}%',
                          style: const TextStyle(fontWeight: FontWeight.bold),
                        ),
                      ],
                    ),
                    const SizedBox(height: 4),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(4),
                      child: LinearProgressIndicator(
                        value: entry.value,
                        minHeight: 6,
                      ),
                    ),
                  ],
                ),
              );
            }).toList(),
          ],
        ),
      ),
    );
  }

  Widget _buildRecommendationsSection() {
    final recs = recommendations!;
    
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Personalized Recommendations',
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 12),
        
        // Recommended Foods
        _buildRecommendationCard(
          'Recommended Foods',
          recs.recommendedFoods,
          Colors.green,
        ),
        const SizedBox(height: 12),
        
        // Foods to Avoid
        _buildRecommendationCard(
          'Foods to Avoid',
          recs.foodsToAvoid,
          Colors.red,
        ),
        const SizedBox(height: 12),
        
        // Supplements
        _buildRecommendationCard(
          'Recommended Supplements',
          recs.supplements,
          Colors.orange,
        ),
        const SizedBox(height: 12),
        
        // Lifestyle Tips
        _buildRecommendationCard(
          'Lifestyle Tips',
          recs.lifestyleTips,
          Colors.blue,
        ),
      ],
    );
  }

  Widget _buildRecommendationCard(
    String title,
    List<String> items,
    Color color,
  ) {
    return Card(
      elevation: 1,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                CircleAvatar(
                  radius: 4,
                  backgroundColor: color,
                ),
                const SizedBox(width: 8),
                Text(
                  title,
                  style: const TextStyle(fontWeight: FontWeight.bold),
                ),
              ],
            ),
            const SizedBox(height: 12),
            ...items.map((item) {
              return Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Row(
                  children: [
                    Icon(Icons.check_circle, size: 16, color: color),
                    const SizedBox(width: 8),
                    Expanded(child: Text(item)),
                  ],
                ),
              );
            }).toList(),
          ],
        ),
      ),
    );
  }

  Color _getRiskColor(String riskLevel) {
    if (riskLevel == 'HIGH') return Colors.red;
    if (riskLevel == 'MEDIUM') return Colors.orange;
    return Colors.green;
  }

  IconData _getRiskIcon(String riskLevel) {
    if (riskLevel == 'HIGH') return Icons.warning;
    if (riskLevel == 'MEDIUM') return Icons.info;
    return Icons.check_circle;
  }
}
```

---

## State Management

### lib/providers/prediction_provider.dart

```dart
import 'package:flutter/material.dart';
import '../models/prediction.dart';
import '../services/api_service.dart';

class PredictionProvider extends ChangeNotifier {
  final ApiService apiService;

  PredictionProvider({required this.apiService});

  Prediction? _lastPrediction;
  Recommendations? _lastRecommendations;
  bool _isLoading = false;
  String? _error;

  Prediction? get lastPrediction => _lastPrediction;
  Recommendations? get lastRecommendations => _lastRecommendations;
  bool get isLoading => _isLoading;
  String? get error => _error;

  Future<void> predict(PatientData patientData) async {
    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      _lastPrediction = await apiService.getPrediction(patientData);
      
      if (_lastPrediction != null) {
        _lastRecommendations = await apiService.getRecommendations(
          patientData,
          _lastPrediction!.prediction!.predictedCancerType,
        );
      }
    } catch (e) {
      _error = e.toString();
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  void clearPrediction() {
    _lastPrediction = null;
    _lastRecommendations = null;
    _error = null;
    notifyListeners();
  }
}
```

---

## Testing

### test/api_service_test.dart

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:mockito/mockito.dart';
import 'package:oncoguardian_mobile/models/prediction.dart';
import 'package:oncoguardian_mobile/services/api_service.dart';

class MockClient extends Mock implements http.Client {}

void main() {
  group('ApiService', () {
    late ApiService apiService;
    late MockClient mockClient;

    setUp(() {
      mockClient = MockClient();
      apiService = ApiService(client: mockClient);
    });

    test('getPrediction returns Prediction on successful response', () async {
      final patientData = PatientData(
        Age: 45,
        Gender: 1,
        Height: 1.7,
        Weight: 65,
        Smoking: 0,
        Alcohol_Use: 5,
        Physical_Activity: 8,
        Diet_Red_Meat: 3,
        Diet_Salted_Processed: 2,
        Fruit_Veg_Intake: 9,
        Air_Pollution: 4,
        Occupational_Hazards: 1,
        Family_History: 0,
        BRCA_Mutation: 0,
        H_Pylori_Infection: 0,
        Calcium_Intake: 7,
      );

      when(mockClient.post(
        any,
        headers: anyNamed('headers'),
        body: anyNamed('body'),
      )).thenAnswer((_) async => http.Response(
        '''{
          "success": true,
          "prediction": {
            "predicted_cancer_type": "Breast",
            "risk_level": "HIGH",
            "confidence": 0.92,
            "probabilities": {"Breast": 0.92}
          },
          "timestamp": "2024-03-25T10:30:00"
        }''',
        200,
      ));

      final prediction = await apiService.getPrediction(patientData);

      expect(prediction.success, true);
      expect(prediction.prediction!.predictedCancerType, 'Breast');
      expect(prediction.prediction!.confidence, 0.92);
    });
  });
}
```

---

## Configuration

Create `.env` file:

```env
API_URL=https://us-central1-YOUR_PROJECT.cloudfunctions.net
FIREBASE_WEB_API_KEY=YOUR_WEB_API_KEY
FIREBASE_PROJECT_ID=YOUR_PROJECT_ID
```

---

## Summary

✅ **Complete Flutter Integration includes:**
- REST API service layer
- Model serialization (JSON)
- Prediction form with 16 input fields
- Real-time results display
- Dietary recommendations UI
- Error handling and loading states
- State management with Provider
- Unit tests
- Environment configuration

🚀 **Next Steps:**
1. Replace `API_URL` with your Firebase endpoint
2. Run `flutter pub get`
3. Run `flutter run` on device/emulator
4. Test predictions end-to-end

---

## References

- [Flutter HTTP Guide](https://flutter.dev/docs/development/data-and-backend/json)
- [Provider Package](https://pub.dev/packages/provider)
- [JSON Serialization](https://flutter.dev/docs/development/data-and-backend/json/json-serialization)
