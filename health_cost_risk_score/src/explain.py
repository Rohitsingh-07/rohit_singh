import joblib
import pandas as pd
import os

def explain_model():
    """
    This script explains 'why' the model makes certain predictions.
    It looks at the weight (coefficients) the model gives to each factor.
    """
    model_path = 'outputs/model.joblib'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run train_model first.")
        return
        
    model = joblib.load(model_path)
    
    # Extract the steps from our pipeline
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']
    
    # Get the names of the features after they were transformed
    # 1. Numeric features
    num_features = ['age', 'bmi', 'children']
    # 2. Categorical features (getting the names after OneHotEncoding)
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(['sex', 'smoker', 'region'])
    
    all_feature_names = list(num_features) + list(cat_features)
    
    # Get the weights (coefficients) from the model
    weights = classifier.coef_[0]
    
    # Create a simple table of factors and their impact
    impact_df = pd.DataFrame({
        'Factor': all_feature_names,
        'Impact_Score': weights
    }).sort_values(by='Impact_Score', ascending=False)
    
    # Generate the insights text
    insights = []
    insights.append("=== Health Cost Risk Drivers ===\n")
    insights.append("These factors are the strongest predictors of being in the top 10% of health costs:\n")
    
    for _, row in impact_df.head(5).iterrows():
        factor = row['Factor']
        score = row['Impact_Score']
        insights.append(f"- {factor}: Strongly increases risk (score: {score:.2f})")
    
    insights.append("\n=== Simple Interpretation ===")
    
    # Basic logic to provide human-friendly summary
    top_factor = impact_df.iloc[0]['Factor']
    if 'smoker' in top_factor:
        insights.append("Interpretation: Smoking status is the primary driver of high healthcare costs in this data.")
    elif 'bmi' in top_factor:
        insights.append("Interpretation: BMI (Body Mass Index) is a major contributor to high healthcare costs.")
    elif 'age' in top_factor:
        insights.append("Interpretation: Age is the leading predictor for increased healthcare costs.")
    else:
        insights.append(f"Interpretation: {top_factor} is the most significant factor in predicting high costs.")

    # Save to file
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/insights.txt', 'w') as f:
        f.write("\n".join(insights))
        
    print(f"\nInsights saved to outputs/insights.txt")
    print("\nSummary of Risk Drivers:")
    print("\n".join(insights[:6]))

if __name__ == "__main__":
    explain_model()