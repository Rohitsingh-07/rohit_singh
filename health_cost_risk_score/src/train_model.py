import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from .load_data import load_and_clean_data

def train():
    """
    This function tracks how our AI learns to predict health costs.
    It prepares the data, builds a 'Pipeline' (a sequence of steps), 
    and saves the finished model.
    """
    # 1. Load the data
    df = load_and_clean_data()
    
    # 2. Define Features (X) and Target (y)
    # We want to predict 'high_cost' using everything except 'charges' and 'high_cost'
    X = df.drop(columns=['charges', 'high_cost'])
    y = df['high_cost']
    
    # 3. Identify different types of columns
    categorical_features = ['sex', 'smoker', 'region']
    numeric_features = ['age', 'bmi', 'children']
    
    # 4. Create Preprocessing Steps
    # We turn text (categories) into numbers and scale numbers to a similar range
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ]
    )
    
    # 5. Build the simple Model Pipeline
    # Logistic Regression is great for 'yes/no' (0/1) predictions
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    # 6. Split data into Training (80%) and Testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 7. Train the model
    print("\nTraining the model... please wait.")
    model_pipeline.fit(X_train, y_train)
    
    # 8. Evaluate how well it did
    y_pred = model_pipeline.predict(X_test)
    y_prob = model_pipeline.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\nModel Evaluation Results:")
    print(f"- Accuracy (exact matches): {acc:.2%}")
    print(f"- ROC-AUC (prediction quality): {auc:.2f}")
    print("- Confusion Matrix:")
    print(cm)
    
    # 9. Save the model to the 'outputs' folder
    os.makedirs('outputs', exist_ok=True)
    model_path = 'outputs/model.joblib'
    joblib.dump(model_pipeline, model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    train()