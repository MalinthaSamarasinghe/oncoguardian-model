# -*- coding: utf-8 -*-
"""
Data Generation for OncoGuardian Model Training
================================================

This script generates sample cancer risk factor data for model training and testing.
The data is based on realistic cancer risk factors and epidemiological data.

Author: OncoGuardian Team
Date: 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_sample_dataset(n_samples=500, random_state=42) -> pd.DataFrame:
    """
    Generate a realistic sample dataset for cancer risk prediction.

    Args:
        n_samples (int): Number of samples to generate
        random_state (int): Random seed for reproducibility

    Returns:
        pd.DataFrame: Generated dataset with cancer risk factors
    """
    np.random.seed(random_state)

    cancer_types = ['Lung', 'Breast', 'Colon', 'Prostate', 'Skin']

    # Generate features
    data = {
        'Age': np.random.randint(20, 85, n_samples),
        'Gender': np.random.choice(['Male', 'Female', 'Other'], n_samples),
        'Smoking': np.random.choice(['Never', 'Former', 'Current'], n_samples),
        'Alcohol_Use': np.random.choice(['None', 'Moderate', 'Heavy'], n_samples),
        'Obesity': np.random.choice(['No', 'Yes'], n_samples,p=[0.7, 0.3]),
        'Family_History': np.random.choice(['No', 'Yes'], n_samples, p=[0.8, 0.2]),
        'Diet_Red_Meat': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'Diet_Salted_Processed': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'Fruit_Veg_Intake': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'Physical_Activity': np.random.choice(['Sedentary', 'Moderate', 'High'], n_samples),
        'Air_Pollution': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'Occupational_Hazards': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.6, 0.3, 0.1]),
        'BRCA_Mutation': np.random.choice(['No', 'Yes', 'Unknown'], n_samples, p=[0.8, 0.1, 0.1]),
        'H_Pylori_Infection': np.random.choice(['No', 'Yes', 'Unknown'], n_samples, p=[0.85, 0.1, 0.05]),
        'Calcium_Intake': np.random.choice(['Low', 'Medium', 'High'], n_samples),
    }

    # Cancer Type (target) - with some correlation to features
    cancer_types_list = []
    for i in range(n_samples):
        # Add some correlation to make prediction meaningful
        if data['Smoking'][i] == 'Current':
            weight = [0.5, 0.1, 0.15, 0.1, 0.15]  # Higher lung cancer
        elif data['Gender'][i] == 'Female' and data['BRCA_Mutation'][i] == 'Yes':
            weight = [0.1, 0.6, 0.1, 0.05, 0.15]  # Higher breast cancer
        elif data['Age'][i] > 65:
            weight = [0.2, 0.2, 0.25, 0.2, 0.15]  # More colon/prostate
        else:
            weight = [0.2, 0.2, 0.2, 0.2, 0.2]  # Balanced

        cancer_types_list.append(np.random.choice(cancer_types, p=weight))

    data['Cancer_Type'] = cancer_types_list

    df = pd.DataFrame(data)

    # Shuffle rows
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df


def save_dataset(df: pd.DataFrame, filepath: str = 'data/cancer-risk-factors.csv'):
    """
    Save generated dataset to CSV file.

    Args:
        df (pd.DataFrame): Dataset to save
        filepath (str): Path to save the CSV file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"✅ Dataset saved to {filepath}")
    print(f"   Shape: {df.shape}")
    print(f"\n📊 Dataset Summary:")
    print(df.head())
    print(f"\n📊 Cancer Type Distribution:")
    print(df['Cancer_Type'].value_counts())


def main():
    """Main execution function."""
    print("="*60)
    print("🔬 GENERATING SAMPLE DATASET")
    print("="*60)

    # Generate dataset
    print("\n📝 Generating 500 sample records...")
    df = generate_sample_dataset(n_samples=500, random_state=42)

    # Save dataset
    print("\n💾 Saving dataset...")
    save_dataset(df)

    print("\n✅ Done! You can now run the training pipeline:")
    print("   python src/training.py")


if __name__ == '__main__':
    main()
