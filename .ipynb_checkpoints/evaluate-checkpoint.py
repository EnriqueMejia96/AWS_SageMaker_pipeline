# This script will use your evaluation logic

import argparse
import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--test-data", type=str)
    args = parser.parse_args()
    
    # Load model
    model = joblib.load(args.model)
    
    # Load test data
    test_df = pd.read_csv(args.test_data)
    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target']
    
    # Evaluate model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Print or save the evaluation result
    print(f"Test Accuracy: {accuracy}")
