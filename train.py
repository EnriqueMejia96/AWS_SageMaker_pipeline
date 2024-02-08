# This script will include your model training logic

import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=str)
    args = parser.parse_args()
    
    # Load data
    train_df = pd.read_csv(args.train_data)
    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, os.path.join("/opt/ml/model", "model.joblib"))
