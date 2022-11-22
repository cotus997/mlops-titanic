import argparse
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import os
import pandas as pd
import mlflow
import xgboost as xgb
from azureml.core import Workspace


def select_first_file(path):
    """Selects first file in folder, use under assumption there is only one file in folder
    Args:
        path (str): path to directory or file to choose
    Returns:
        str: full path of selected file
    """
    files = os.listdir(path)
    return os.path.join(path, files[0])

def get_model_metrics(model, X,y):
    preds = model.predict(X)
    mse = mean_squared_error(preds, y)
    metrics = {"mse": mse}
    return metrics


# Start Logging
mlflow.start_run()

# enable autologging
mlflow.xgboost.autolog()
ws = Workspace.from_config()
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
os.makedirs("./outputs", exist_ok=True)


def main():
    """Main function of the script."""
    
    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    parser.add_argument("--max_depth", required=False, default=6, type=int)
    parser.add_argument("--learning_rate", required=False, default=0.3, type=float)
    parser.add_argument("--registered_model_name", type=str, help="model name")
    parser.add_argument("--model", type=str, help="path to model file")
    args = parser.parse_args()

    # paths are mounted as folder, therefore, we are selecting the file from folder
    train_df = pd.read_csv(select_first_file(args.train_data))

    # Extracting the label column
    y_train = train_df.pop("Label")

    # convert the dataframe values to array
    X_train = train_df.values

    # paths are mounted as folder, therefore, we are selecting the file from folder
    test_df = pd.read_csv(select_first_file(args.test_data))

    # Extracting the label column
    y_test = test_df.pop("Label")

    # convert the dataframe values to array
    X_test = test_df.values

    print(f"Training with data of shape {X_train.shape}")
    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42, eta=args.learning_rate,max_depth=args.max_depth)
    
    xgb_model.fit(X_train, y_train)

    #y_pred = xgb_model.predict(X_test)

    #print(classification_report(y_test, y_pred))
    metrics= get_model_metrics(xgb_model,X_test,y_test)
    for (k, v) in metrics.items():
        mlflow.log_metric(f'val_{k}',v)
    # Registering the model to the workspace
    print("Registering the model via MLFlow")
    mlflow.xgboost.log_model(
        xgb_model=xgb_model,
        registered_model_name=args.registered_model_name,
        artifact_path=args.registered_model_name,
    )

    # Saving the model to a file
    mlflow.xgboost.save_model(
        xgb_model=xgb_model,
        path=os.path.join(args.model, "trained_model"),
    )

    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    main()