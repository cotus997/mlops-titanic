
import argparse
import pandas as pd
import mlflow
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


import numpy as np
from azureml.core import Workspace, Dataset
from azureml.core.run import Run

def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--experiment_name", type=str, required=False, default="titanic")
    #parser.add_argument('--random-split', type=int, default=0)
    parser.add_argument("--output_data", type=str, required=False, help="path to train data")
    #parser.add_argument("--test_data", type=str, help="path to test data")
    args = parser.parse_args()
    #
    ##Connecting with the workspace
    #credential=DefaultAzureCredential()
#
    #try:
    #    ml_client = MLClient.from_config(credential=credential, path='.')
    #except Exception as ex:
    #    # NOTE: Update following workspace information to contain
    #    #       your subscription ID, resource group name, and workspace name
    #    client_config = {
    #        "subscription_id": "f90533aa-280d-40b9-9949-a7ba0ee9511f",
    #        "resource_group": "mlops-RG",
    #        "workspace_name": "mlops-AML-WS",
    #    }
#
    #    # write and reload from config file
    #    import json, os
#
    #    config_path = "../.azureml/config.json"
    #    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    #    with open(config_path, "w") as fo:
    #        fo.write(json.dumps(client_config))
    #    ml_client = MLClient.from_config(credential=credential, path=config_path)
#
#
    #azureml_mlflow_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
    aml_workspace = Workspace.from_config()
    '''Workspace.get(
        name="mlops-AML-WS",
        subscription_id="f90533aa-280d-40b9-9949-a7ba0ee9511f",
        resource_group="mlops-RG",
    )'''
    azureml_mlflow_uri= aml_workspace.get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(azureml_mlflow_uri)
    run = Run.get_context()
    #setting experiment name
    experiment_name = args.experiment_name
    #mlflow.set_experiment("preproc")

    # Start Logging
    mlflow.start_run(nested=True)

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
    df.rename(columns={"Survived":"Label"})


    print('Dataset shape after preprocessing: {}'.format(df.shape))
    mlflow.log_metric("num_samples_dataset", df.shape[0])
    mlflow.log_metric("num_features_dataset", df.shape[1] - 1)
    
    #run.parent.log("num_samples_dataset", df.shape[0])
    #run.parent.log("num_features_dataset", df.shape[1] - 1)

    dataset_info=register_dataset(df)
    for (k,v) in dataset_info.items():
        print(f'{k} - {v}')
        mlflow.log_param(k,v)
        #run.parent.log(k,v)
    df.to_csv("../../data/train/preprocessed.csv")
    

    # Stop Logging'''
    mlflow.end_run()

def register_dataset(df:pd.DataFrame):
    # Connect to the Workspace
    ws = Workspace.from_config()

    # The default datastore is a blob storage container where datasets are stored
    datastore = ws.get_default_datastore()
    # Register the dataset
    ds = Dataset.Tabular.register_pandas_dataframe(
            dataframe=df, 
            name='Titanic_dataset', 
            description='dataset containing the survivors from the Titanic disaster',
            target=datastore
        )

    # Display information about the dataset
    return {"dataset_name":ds.name,"dataset_version":ds.version,"dataset_ID":ds.id}
    #print(ds.name + " v" + str(ds.version) + ' (ID: ' + ds.id + ")")

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