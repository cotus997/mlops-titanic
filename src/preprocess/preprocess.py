import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import mlflow

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.exceptions import DataConversionWarning
from sklearn.compose import make_column_transformer


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
    df.sample(frac=1)
    
    COLS = df.columns
    newcolorder = ['PAY_AMT1','BILL_AMT1'] + list(COLS[1:])[:11] + list(COLS[1:])[12:17] + list(COLS[1:])[18:]
    
    random_state=args.random_split
    
    X_train, X_test, y_train, y_test = train_test_split(df.drop('Label', axis=1), df['Label'], 
                                                        test_size=args.test_train_ratio, random_state=random_state)
    
    preprocess = make_column_transformer(
        (StandardScaler(),['PAY_AMT1']),
        (MinMaxScaler(),['BILL_AMT1']),
    remainder='passthrough')
    
    print('Running preprocessing and feature engineering transformations')
    train_features = pd.DataFrame(preprocess.fit_transform(X_train), columns = newcolorder)
    test_features = pd.DataFrame(preprocess.transform(X_test), columns = newcolorder)
    
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


if __name__ == "__main__":
    main()