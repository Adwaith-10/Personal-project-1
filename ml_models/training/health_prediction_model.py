import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class HealthPredictionModel:
    """Health prediction model using multiple algorithms"""
    
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        
    def generate_sample_data(self, n_samples=1000):
        """Generate sample health data for training"""
        np.random.seed(42)
        
        # Generate realistic health data
        data = {
            'age': np.random.normal(45, 15, n_samples).astype(int),
            'heart_rate': np.random.normal(75, 15, n_samples).astype(int),
            'blood_pressure_systolic': np.random.normal(120, 20, n_samples).astype(int),
            'blood_pressure_diastolic': np.random.normal(80, 10, n_samples).astype(int),
            'temperature': np.random.normal(37.0, 0.5, n_samples),
            'oxygen_saturation': np.random.normal(98.0, 2.0, n_samples),
            'respiratory_rate': np.random.normal(16, 4, n_samples).astype(int),
            'glucose_level': np.random.normal(100, 20, n_samples),
            'bmi': np.random.normal(25, 5, n_samples),
            'gender': np.random.choice(['male', 'female'], n_samples),
            'smoking_status': np.random.choice(['never', 'former', 'current'], n_samples),
            'diabetes': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'hypertension': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        }
        
        df = pd.DataFrame(data)
        
        # Create health risk labels based on the data
        df['health_risk'] = self._calculate_health_risk(df)
        
        return df
    
    def _calculate_health_risk(self, df):
        """Calculate health risk based on various factors"""
        risk_scores = []
        
        for _, row in df.iterrows():
            score = 0
            
            # Age risk
            if row['age'] > 65:
                score += 2
            elif row['age'] > 50:
                score += 1
            
            # Heart rate risk
            if row['heart_rate'] > 100 or row['heart_rate'] < 60:
                score += 1
            
            # Blood pressure risk
            if row['blood_pressure_systolic'] > 140 or row['blood_pressure_diastolic'] > 90:
                score += 2
            
            # Temperature risk
            if row['temperature'] > 38.0 or row['temperature'] < 36.0:
                score += 1
            
            # Oxygen saturation risk
            if row['oxygen_saturation'] < 95:
                score += 2
            
            # Glucose risk
            if row['glucose_level'] > 126:
                score += 2
            elif row['glucose_level'] > 100:
                score += 1
            
            # BMI risk
            if row['bmi'] > 30:
                score += 1
            elif row['bmi'] < 18.5:
                score += 1
            
            # Smoking risk
            if row['smoking_status'] == 'current':
                score += 2
            elif row['smoking_status'] == 'former':
                score += 1
            
            # Medical conditions
            if row['diabetes'] == 1:
                score += 2
            if row['hypertension'] == 1:
                score += 2
            
            # Categorize risk
            if score >= 6:
                risk_scores.append('high')
            elif score >= 3:
                risk_scores.append('medium')
            else:
                risk_scores.append('low')
        
        return risk_scores
    
    def preprocess_data(self, df):
        """Preprocess the data for training"""
        # Create features
        features = df[['age', 'heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic',
                      'temperature', 'oxygen_saturation', 'respiratory_rate', 'glucose_level', 'bmi']].copy()
        
        # Add engineered features
        features['pulse_pressure'] = features['blood_pressure_systolic'] - features['blood_pressure_diastolic']
        features['mean_arterial_pressure'] = features['blood_pressure_diastolic'] + (features['pulse_pressure'] / 3)
        
        # Encode categorical variables
        features['gender_encoded'] = self.label_encoder.fit_transform(df['gender'])
        features['smoking_encoded'] = LabelEncoder().fit_transform(df['smoking_status'])
        
        # Add medical conditions
        features['diabetes'] = df['diabetes']
        features['hypertension'] = df['hypertension']
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        # Scale numerical features
        numerical_features = ['age', 'heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic',
                            'temperature', 'oxygen_saturation', 'respiratory_rate', 'glucose_level', 'bmi',
                            'pulse_pressure', 'mean_arterial_pressure']
        
        features[numerical_features] = self.scaler.fit_transform(features[numerical_features])
        
        return features
    
    def train_model(self, X, y):
        """Train the selected model"""
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train the model
        self.model.fit(X, y)
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model"""
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': y_pred
        }
    
    def predict_health_risk(self, features):
        """Predict health risk for new data"""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Preprocess features
        features_scaled = self.scaler.transform([features])
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        return {
            'prediction': prediction,
            'probability': probability,
            'confidence': max(probability)
        }
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save. Please train the model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model and preprocessing objects
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'training_date': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        
        print(f"Model loaded from {filepath}")
        return self

def main():
    """Main training function"""
    print("🏥 Health Prediction Model Training")
    print("=" * 50)
    
    # Initialize model
    model = HealthPredictionModel(model_type='xgboost')
    
    # Generate sample data
    print("📊 Generating sample data...")
    df = model.generate_sample_data(n_samples=2000)
    print(f"Generated {len(df)} samples")
    
    # Preprocess data
    print("🔧 Preprocessing data...")
    X = model.preprocess_data(df)
    y = df['health_risk']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    print(f"🤖 Training {model.model_type} model...")
    model.train_model(X_train, y_train)
    
    # Evaluate model
    print("📈 Evaluating model...")
    results = model.evaluate_model(X_test, y_test)
    
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, results['predictions']))
    
    # Save model
    model_path = "ml_models/models/health_prediction_model.pkl"
    model.save_model(model_path)
    
    # Test prediction
    print("\n🧪 Testing prediction...")
    sample_features = [45, 75, 120, 80, 37.0, 98.0, 16, 100, 25, 15, 40, 0, 0, 0]  # Example features
    prediction = model.predict_health_risk(sample_features)
    print(f"Sample prediction: {prediction['prediction']} (confidence: {prediction['confidence']:.3f})")
    
    print("\n✅ Training completed successfully!")

if __name__ == "__main__":
    main()
