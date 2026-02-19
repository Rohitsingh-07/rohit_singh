import pandas as pd
import joblib
import os
from .load_data import load_and_clean_data

def score_all():
    """
    This script uses our trained model to give every person a 'Risk Score'
    from 0 to 100.
    """
    # 1. Load the data and the trained model
    df = load_and_clean_data()
    
    model_path = 'outputs/model.joblib'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run train_model first.")
        return
        
    model = joblib.load(model_path)
    
    # 2. Separate features for prediction
    X = df.drop(columns=['charges', 'high_cost'])
    
    # 3. Generate risk scores (probabilities * 100)
    # predict_proba returns [prob_low_cost, prob_high_cost]
    probabilities = model.predict_proba(X)[:, 1]
    df['risk_score'] = (probabilities * 100).round(2)
    
    # 4. Save the results
    os.makedirs('outputs', exist_ok=True)
    output_path = 'outputs/scored.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\nScoring complete! Results saved to {output_path}")
    
    # 5. Show Top 10 highest risk people
    print("\nTop 10 High-Risk Profiles:")
    top_10 = df.sort_values(by='risk_score', ascending=False).head(10)
    print(top_10[['age', 'smoker', 'bmi', 'charges', 'risk_score']])

if __name__ == "__main__":
    score_all()