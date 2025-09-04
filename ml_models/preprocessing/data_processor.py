import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import joblib
import os
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta

class HealthDataProcessor:
    """Data preprocessing utility for health data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_names = []
        self.categorical_features = []
        self.numerical_features = []
        
    def load_health_data(self, filepath: str) -> pd.DataFrame:
        """Load health data from various formats"""
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            df = pd.read_json(filepath)
        elif filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format. Use CSV, JSON, or Excel files.")
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate health data"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Validate data ranges
        df = self._validate_data_ranges(df)
        
        # Remove outliers
        df = self._remove_outliers(df)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in health data"""
        # For numerical columns, use median imputation
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
        
        # For categorical columns, use mode imputation
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
        
        return df
    
    def _validate_data_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate health data ranges"""
        # Heart rate validation (30-200 bpm)
        if 'heart_rate' in df.columns:
            df.loc[df['heart_rate'] < 30, 'heart_rate'] = 30
            df.loc[df['heart_rate'] > 200, 'heart_rate'] = 200
        
        # Blood pressure validation
        if 'blood_pressure_systolic' in df.columns:
            df.loc[df['blood_pressure_systolic'] < 70, 'blood_pressure_systolic'] = 70
            df.loc[df['blood_pressure_systolic'] > 200, 'blood_pressure_systolic'] = 200
        
        if 'blood_pressure_diastolic' in df.columns:
            df.loc[df['blood_pressure_diastolic'] < 40, 'blood_pressure_diastolic'] = 40
            df.loc[df['blood_pressure_diastolic'] > 130, 'blood_pressure_diastolic'] = 130
        
        # Temperature validation (35-42°C)
        if 'temperature' in df.columns:
            df.loc[df['temperature'] < 35, 'temperature'] = 35
            df.loc[df['temperature'] > 42, 'temperature'] = 42
        
        # Oxygen saturation validation (70-100%)
        if 'oxygen_saturation' in df.columns:
            df.loc[df['oxygen_saturation'] < 70, 'oxygen_saturation'] = 70
            df.loc[df['oxygen_saturation'] > 100, 'oxygen_saturation'] = 100
        
        # Respiratory rate validation (8-40 breaths/min)
        if 'respiratory_rate' in df.columns:
            df.loc[df['respiratory_rate'] < 8, 'respiratory_rate'] = 8
            df.loc[df['respiratory_rate'] > 40, 'respiratory_rate'] = 40
        
        # Glucose level validation (50-500 mg/dL)
        if 'glucose_level' in df.columns:
            df.loc[df['glucose_level'] < 50, 'glucose_level'] = 50
            df.loc[df['glucose_level'] > 500, 'glucose_level'] = 500
        
        # BMI validation (10-50)
        if 'bmi' in df.columns:
            df.loc[df['bmi'] < 10, 'bmi'] = 10
            df.loc[df['bmi'] > 50, 'bmi'] = 50
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame, method='iqr') -> pd.DataFrame:
        """Remove outliers using IQR method"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Replace outliers with bounds
            df.loc[df[col] < lower_bound, col] = lower_bound
            df.loc[df[col] > upper_bound, col] = upper_bound
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer new features from existing health data"""
        # Age-related features
        if 'date_of_birth' in df.columns:
            df['age'] = pd.to_datetime(df['date_of_birth']).apply(
                lambda x: (datetime.now() - x).days // 365
            )
        
        # Blood pressure features
        if 'blood_pressure_systolic' in df.columns and 'blood_pressure_diastolic' in df.columns:
            df['pulse_pressure'] = df['blood_pressure_systolic'] - df['blood_pressure_diastolic']
            df['mean_arterial_pressure'] = df['blood_pressure_diastolic'] + (df['pulse_pressure'] / 3)
        
        # BMI calculation
        if 'weight_kg' in df.columns and 'height_cm' in df.columns:
            df['bmi_calculated'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)
        
        # Health risk scores
        df = self._calculate_risk_scores(df)
        
        return df
    
    def _calculate_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various health risk scores"""
        # Cardiovascular risk score
        if all(col in df.columns for col in ['age', 'heart_rate', 'blood_pressure_systolic']):
            df['cardiovascular_risk'] = (
                (df['age'] / 100) * 0.3 +
                (df['heart_rate'] / 200) * 0.2 +
                (df['blood_pressure_systolic'] / 200) * 0.5
            )
        
        # Metabolic risk score
        if all(col in df.columns for col in ['bmi', 'glucose_level']):
            df['metabolic_risk'] = (
                (df['bmi'] / 50) * 0.5 +
                (df['glucose_level'] / 500) * 0.5
            )
        
        # Respiratory risk score
        if all(col in df.columns for col in ['oxygen_saturation', 'respiratory_rate']):
            df['respiratory_risk'] = (
                ((100 - df['oxygen_saturation']) / 30) * 0.6 +
                (df['respiratory_rate'] / 40) * 0.4
            )
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for machine learning"""
        # Identify feature columns
        if target_column:
            feature_cols = [col for col in df.columns if col != target_column]
        else:
            feature_cols = df.columns
        
        # Separate numerical and categorical features
        self.numerical_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
        
        # Create feature dataframe
        features = df[feature_cols].copy()
        
        # Encode categorical variables
        for col in self.categorical_features:
            if col in features.columns:
                le = LabelEncoder()
                features[col] = le.fit_transform(features[col].astype(str))
                self.label_encoders[col] = le
        
        # Scale numerical features
        if self.numerical_features:
            features[self.numerical_features] = self.scaler.fit_transform(features[self.numerical_features])
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        # Prepare target variable
        target = None
        if target_column and target_column in df.columns:
            target = df[target_column]
        
        return features, target
    
    def save_preprocessor(self, filepath: str):
        """Save the preprocessor for later use"""
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'imputer': self.imputer
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(preprocessor_data, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: str):
        """Load a saved preprocessor"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor file not found: {filepath}")
        
        preprocessor_data = joblib.load(filepath)
        
        self.scaler = preprocessor_data['scaler']
        self.label_encoders = preprocessor_data['label_encoders']
        self.feature_names = preprocessor_data['feature_names']
        self.categorical_features = preprocessor_data['categorical_features']
        self.numerical_features = preprocessor_data['numerical_features']
        self.imputer = preprocessor_data['imputer']
        
        print(f"Preprocessor loaded from {filepath}")
        return self
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using the fitted preprocessor"""
        if not self.feature_names:
            raise ValueError("Preprocessor not fitted. Please fit the preprocessor first.")
        
        # Select only the features used during training
        features = df[self.feature_names].copy()
        
        # Encode categorical variables
        for col in self.categorical_features:
            if col in features.columns and col in self.label_encoders:
                features[col] = self.label_encoders[col].transform(features[col].astype(str))
        
        # Scale numerical features
        if self.numerical_features:
            features[self.numerical_features] = self.scaler.transform(features[self.numerical_features])
        
        return features

def main():
    """Example usage of the HealthDataProcessor"""
    print("🏥 Health Data Processor Example")
    print("=" * 40)
    
    # Create processor
    processor = HealthDataProcessor()
    
    # Generate sample data
    np.random.seed(42)
    sample_data = {
        'age': np.random.normal(45, 15, 100).astype(int),
        'heart_rate': np.random.normal(75, 15, 100).astype(int),
        'blood_pressure_systolic': np.random.normal(120, 20, 100).astype(int),
        'blood_pressure_diastolic': np.random.normal(80, 10, 100).astype(int),
        'temperature': np.random.normal(37.0, 0.5, 100),
        'oxygen_saturation': np.random.normal(98.0, 2.0, 100),
        'gender': np.random.choice(['male', 'female'], 100),
        'smoking_status': np.random.choice(['never', 'former', 'current'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    
    print(f"Original data shape: {df.shape}")
    print(f"Sample data:\n{df.head()}")
    
    # Clean data
    df_clean = processor.clean_data(df)
    print(f"\nCleaned data shape: {df_clean.shape}")
    
    # Engineer features
    df_engineered = processor.engineer_features(df_clean)
    print(f"\nEngineered features: {[col for col in df_engineered.columns if col not in df_clean.columns]}")
    
    # Prepare features
    features, target = processor.prepare_features(df_engineered)
    print(f"\nPrepared features shape: {features.shape}")
    print(f"Feature names: {processor.feature_names}")
    
    # Save preprocessor
    processor.save_preprocessor("ml_models/preprocessing/health_preprocessor.pkl")
    
    print("\n✅ Data processing completed successfully!")

if __name__ == "__main__":
    main()
