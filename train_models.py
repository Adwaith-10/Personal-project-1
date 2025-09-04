#!/usr/bin/env python3
"""
Script to train machine learning models for Health AI Twin
"""

import sys
import os
from pathlib import Path

# Add the ml_models directory to the path
sys.path.append(str(Path(__file__).parent / "ml_models"))

from training.health_prediction_model import main as train_health_model

def main():
    """Train all ML models"""
    print("🤖 Training Health AI Twin ML Models")
    print("=" * 50)
    
    try:
        # Train health prediction model
        print("\n🏥 Training Health Prediction Model...")
        train_health_model()
        
        print("\n✅ All models trained successfully!")
        print("\n📁 Models saved to:")
        print("   - ml_models/models/health_prediction_model.pkl")
        print("   - ml_models/preprocessing/health_preprocessor.pkl")
        
    except Exception as e:
        print(f"❌ Error training models: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
