
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline

import argparse
import os
import pandas as pd
import xgboost as xgb
import shutil

import mlflow
from azureml.core import Workspace, Dataset
from mlflow.tracking import MlflowClient


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





def main():
    """Main function of the script."""
    
    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=False, default="titanic")
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.25)
    parser.add_argument('--random_split', type=int, default=0)
    parser.add_argument("--max_depth", required=False, default=6, type=int)
    parser.add_argument("--learning_rate", required=False, default=0.3, type=float)
    parser.add_argument("--registered_model_name", type=str, required=False, help="model name")
    parser.add_argument("--model", type=str, help="path to model file")
    args = parser.parse_args()

    # Start Logging
    
    ws = Workspace.from_config()
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    #mlflow.set_experiment("train")
    mlflow.start_run(nested=True)
    # enable autologging
    mlflow.xgboost.autolog()
    os.makedirs("./outputs", exist_ok=True)

    df=retrieve_registered_dataset(ws)
    X = df.drop("Survived",axis=1)
    y = df["Survived"]
    
    random_state=args.random_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=args.test_train_ratio, random_state=random_state, stratify=y)
    
    xgb_model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            eta=args.learning_rate,
            max_depth=args.max_depth,
            use_label_encoder =False
        )

    # creating a pipeline for OneHotEncoding of Categorical Columns
    categorical_processor = ColumnTransformer(transformers=[
    ("OHE",OneHotEncoder(drop='first'),["Sex","Embarked"]),
    ],remainder="passthrough")

    pipe = Pipeline(steps=[
    ("Categorical_Processor",categorical_processor),
    ("Standard Scaling",StandardScaler()),
    ("Classifier",xgb_model)
    ])



   

    
    
    
    pipe.fit(X_train, y_train)

    #y_pred = xgb_model.predict(X_test)

    #print(classification_report(y_test, y_pred))
    metrics= get_model_metrics(pipe,X_test,y_test)
    for (k, v) in metrics.items():
        mlflow.log_metric(f'mse',v)
    # Registering the model to the workspace
    print("Saved model via MLFlow")
    #mlflow.xgboost.log_model(
    #    xgb_model=xgb_model,
    #    registered_model_name=args.registered_model_name,
    #    artifact_path=args.registered_model_name,
    #)
    if os.path.exists(os.path.join(args.model, "trained_model")):
        shutil.rmtree(os.path.join(args.model, "trained_model"), ignore_errors=True)
    # Saving the model to a file
    mlflow.sklearn.save_model(
        sk_model=pipe,
        path=os.path.join(args.model, "trained_model")
    )
    os.rename(os.path.join(args.model, "trained_model","model.pkl"), os.path.join(args.model, "trained_model","titanic-xgb.pkl"))
    #print(f'saved model: {saved_model}\n run info: {mlflow_run.info}')

    # Stop Logging
    mlflow.end_run()


def retrieve_registered_dataset(ws:Workspace)->pd.DataFrame:
    exp=mlflow.get_experiment_by_name("titanic-pipeline")
    runs=mlflow.search_runs(exp.experiment_id,output_format="list")
    last_run=runs[-2]

    run_id = last_run.info.run_id
    print(run_id)
    finished_mlflow_run = MlflowClient().get_run(run_id)
    params = finished_mlflow_run.data.params

    dataset_id=params['dataset_ID']
    dataset=Dataset.get_by_id(ws,id=dataset_id)
    mlflow.set_tag("dataset_id",dataset_id)
    return dataset.to_pandas_dataframe()

if __name__ == "__main__":
    main()