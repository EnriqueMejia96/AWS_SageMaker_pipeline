import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # Import joblib for model serialization

def preprocess_data(data):
    X = data.drop(columns=['target'])
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardizing the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Converting to DataFrames
    train_df = pd.DataFrame(X_train, columns=X.columns)
    train_df['target'] = y_train.reset_index(drop=True)
    test_df = pd.DataFrame(X_test, columns=X.columns)
    test_df['target'] = y_test.reset_index(drop=True)

    return train_df, test_df, scaler

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, help="Path to input data")
    parser.add_argument("--output-train", type=str, help="Path to save the training dataset")
    parser.add_argument("--output-test", type=str, help="Path to save the testing dataset")
    parser.add_argument("--output-scaler", type=str, help="Path to save the scaler model")
    args = parser.parse_args()
    
    # Load dataset
    data = pd.read_csv(args.input_data)
    
    train_df, test_df, scaler = preprocess_data(data)
    
    # Save processed data
    train_df.to_csv(args.output_train, index=False)
    test_df.to_csv(args.output_test, index=False)
    
    # Save the scaler model
    scaler_path = os.path.join(args.output_scaler, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
