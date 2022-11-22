import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import mlflow

from sklearn.pipeline import Pipeline,make_pipeline

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import numpy as np


def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.25)
    parser.add_argument('--random-split', type=int, default=0)
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.data)

    df = pd.read_csv(args.data)

    #drop cabin feature due to too many missing values
    df.drop("Cabin",axis=1,inplace=True)

    #fill missing values with mean of cabins age
    df["Age"] = df[["Age","Pclass"]].apply(AgeImputation,axis=1)
    # Embarked Column has Just 2 Null values, we will drop that
    df.dropna(inplace=True)
    # Drop Redundant Columns like PassengerId, Name, and Ticket 
    # That's Not Important in Classification Problem
    df.drop(columns=["PassengerId","Name","Ticket"],inplace=True)
    # Extract X and y
    X = df.drop("Survived",axis=1)
    y = df["Survived"]
    
    random_state=args.random_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=args.test_train_ratio, random_state=random_state, stratify=y)
    
    # creating a pipeline for OneHotEncoding of Categorical Columns
    categorical_processor = ColumnTransformer(transformers=[
    ("OHE",OneHotEncoder(drop='first'),["Sex","Embarked"]),
    ],remainder="passthrough")

    pipe = Pipeline(steps=[
    ("Categorical_Processor",categorical_processor),
    ("Standard Scaling",StandardScaler())
    ])
    
    print('Running preprocessing and feature engineering transformations')
    train_features = pd.DataFrame(pipe.fit_transform(X_train))
    test_features = pd.DataFrame(pipe.transform(X_test))
    
    # concat to ensure Label column is the first column in dataframe
    train_full = pd.concat([pd.DataFrame(y_train.values, columns=['Label']), train_features], axis=1)
    test_full = pd.concat([pd.DataFrame(y_test.values, columns=['Label']), test_features], axis=1)
    
    print('Train data shape after preprocessing: {}'.format(train_features.shape))
    mlflow.log_metric("num_samples_train", train_features.shape[0])
    mlflow.log_metric("num_features_train", train_features.shape[1] - 1)
    print('Test data shape after preprocessing: {}'.format(test_features.shape))
    mlflow.log_metric("num_samples_test", test_features.shape[0])
    mlflow.log_metric("num_features_test", test_features.shape[1] - 1)
    
    

    # output paths are mounted as folder, therefore, we are adding a filename to the path
    train_full.to_csv(os.path.join(args.train_data, "data.csv"), index=False)

    test_full.to_csv(os.path.join(args.test_data, "data.csv"), index=False)

    # Stop Logging
    mlflow.end_run()

# Age Column Imputation
def AgeImputation(value):
    age = value[0]
    pclass = value[1]
    
    if np.isnan(age):
        if pclass == 1:
            return 38.10
        elif pclass == 2:
            return 29.87
        else:
            return 25.14
        
    else:
        return age
        

if __name__ == "__main__":
    main()