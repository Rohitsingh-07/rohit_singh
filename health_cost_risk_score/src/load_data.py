import pandas as pd
import os

# The URL to the raw CSV data on GitHub Gist
DATA_URL = "https://gist.githubusercontent.com/meperezcuello/82a9f1c1c473d6585e750ad2e3c05a41/raw/Medical_Cost.csv"

def load_and_clean_data():
    """
    This function downloads the medical cost dataset, cleans it, 
    and defines what 'high cost' means for our health risk score.
    """
    print(f"Fetching data from: {DATA_URL}")
    
    # Load the dataset directly from the URL using pandas
    df = pd.read_csv(DATA_URL)
    
    # Clean up column names (remove any extra spaces)
    df.columns = df.columns.str.strip()
    
    # Handle missing values: For this simple project, we drop any rows with missing data
    df = df.dropna()
    
    # Define 'High Cost' as the top 10% of charges
    # We find the 90th percentile value
    threshold = df['charges'].quantile(0.90)
    
    # Create a new column 'high_cost': 1 if charges are high, 0 otherwise
    df['high_cost'] = (df['charges'] >= threshold).astype(int)
    
    # Calculate percentage for reporting
    pct_high_cost = df['high_cost'].mean() * 100
    
    print(f"Data loaded successfully. Total rows: {len(df)}")
    print(f"High-cost threshold (90th percentile): ${threshold:.2f}")
    print(f"Percentage of high-cost cases: {pct_high_cost:.1f}%")
    
    return df

if __name__ == "__main__":
    # If we run this script directly, just show the first few rows
    data = load_and_clean_data()
    print("\nFirst 5 rows of cleaned data:")
    print(data.head())