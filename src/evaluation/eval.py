"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation (“Microsoft”) grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any claim of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import argparse
import traceback

import os
import mlflow
from azureml.core import Workspace,Dataset
from azureml.core.run import Run
from azureml.core.model import Model as AMLModel
from mlflow.tracking import MlflowClient


def main():
    """Main function of the script."""
    mlflow.start_run(nested=True)
    run = Run.get_context()
    ws = Workspace.from_config()
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    
    parser = argparse.ArgumentParser("evaluate")
    parser.add_argument("--run_id", type=str, help="Training run ID", required=False)
    parser.add_argument( "--model_name", type=str, help="Name of the Model", default="titanic-xgb.pkl" )
    parser.add_argument( "--model_path", type=str, help="path to the model not yet registered", default="../train/models/trained_model/")
    args = parser.parse_args()
    
    
    model_name = args.model_name
    model_path = args.model_path
    metric_eval = "mse"
    metrics,tags,params=get_train_exp()
    print(metrics)
    allow_run_cancel="true"
    
    
    
    # Parameterize the matrices on which the models should be compared
    # Add golden data set on which all the model performance can be evaluated
    try:
        
    
        model = get_model(
                    model_name=model_name,
                    aml_workspace=ws)
    
        if (model is not None):
            production_model_mse = 10000
            if (metric_eval in model.tags):
                production_model_mse = float(model.tags[metric_eval])
            try:
                new_model_mse = float(metrics.get(metric_eval))
            except TypeError:
                new_model_mse = None
            if (production_model_mse is None or new_model_mse is None):
                print("Unable to find ", metric_eval, " metrics, "
                      "exiting evaluation")
                if((allow_run_cancel).lower() == 'true'):
                    #run.parent.cancel()
                    pass
            else:
                print(
                    "Current Production model {}: {}, ".format(
                        metric_eval, production_model_mse) +
                    "New trained model {}: {}".format(
                        metric_eval, new_model_mse
                    )
                )

            if (new_model_mse < production_model_mse or True):
                print("New trained model performs better, "
                      "Registering model")
                      
                tags[metric_eval]=metrics[metric_eval]
                register_aml_model(
                model_path=model_path,
                model_name=model_name, 
                model_tags=tags, 
                name="evaluation", 
                ws=ws, 
                run=run, 
                dataset_id=tags["dataset_id"]
                )
            else:
                print("New trained model metric is worse than or equal to "
                      "production model so skipping model registration.")
                if((allow_run_cancel).lower() == 'true'):
                    #run.parent.cancel()
                    pass
        else:
            # model_path,
            # model_name,
            # model_tags,
            # name,
            # ws,
            # run_id,
            # dataset_id
            tags[metric_eval]=metrics[metric_eval]
            register_aml_model(
                model_path=model_path,
                model_name=model_name, 
                model_tags=tags, 
                name="evaluation", 
                ws=ws, 
                run=run, 
                dataset_id=tags["dataset_id"]
                )
    
    except Exception:
        traceback.print_exc(limit=None, file=None, chain=True)
        print("Something went wrong trying to evaluate. Exiting.")
        raise



def get_train_exp():
    exp=mlflow.get_experiment_by_name("titanic-pipeline")
    runs=mlflow.search_runs(exp.experiment_id,output_format="list")
    last_run=runs[-2]
    # Use MlFlow to retrieve the job that was just completed
    client = MlflowClient()
    run_id = last_run.info.run_id
    print(run_id)
    finished_mlflow_run = MlflowClient().get_run(run_id)

    metrics = finished_mlflow_run.data.metrics
    tags = finished_mlflow_run.data.tags
    params = finished_mlflow_run.data.params

    #print(f'{metrics} - {tags} - {params}')
    return metrics,tags,params


def get_model(
    model_name: str,
    model_version: int = None,  # If none, return latest model
    tag_name: str = None,
    tag_value: str = None,
    aml_workspace: Workspace = None
) -> AMLModel:
    """
    Retrieves and returns a model from the workspace by its name
    and (optional) tag.

    Parameters:
    aml_workspace (Workspace): aml.core Workspace that the model lives.
    model_name (str): name of the model we are looking for
    (optional) model_version (str): model version. Latest if not provided.
    (optional) tag (str): the tag value & name the model was registered under.

    Return:
    A single aml model from the workspace that matches the name and tag, or
    None.
    """
    if aml_workspace is None:
        print("No workspace defined - using current experiment workspace.")
        aml_workspace = get_current_workspace()

    tags = None
    if tag_name is not None or tag_value is not None:
        # Both a name and value must be specified to use tags.
        if tag_name is None or tag_value is None:
            raise ValueError(
                "model_tag_name and model_tag_value should both be supplied"
                + "or excluded"  # NOQA: E501
            )
        tags = [[tag_name, tag_value]]

    model = None
    if model_version is not None:
        # TODO(tcare): Finding a specific version currently expects exceptions
        # to propagate in the case we can't find the model. This call may
        # result in a WebserviceException that may or may not be due to the
        # model not existing.
        model = AMLModel(
            aml_workspace,
            name=model_name,
            version=model_version,
            tags=tags)
    else:
        models = AMLModel.list(
            aml_workspace, name=model_name, tags=tags, latest=True)
        if len(models) == 1:
            model = models[0]
        elif len(models) > 1:
            raise Exception("Expected only one model")

    return model


def register_aml_model(
    model_path,
    model_name,
    model_tags,
    name,
    ws,
    run,
    dataset_id
):
    try:
        tagsValue = {"area": "Titanic_classification",
                     "run_id": run.parent.get_details()['runId'],
                     "experiment_name": name}
        tagsValue.update(model_tags)
       

        model = AMLModel.register(
            workspace=ws,
            model_name=model_name,
            model_path=model_path,
            tags=tagsValue,
            datasets=[('training data',
                       Dataset.get_by_id(ws, dataset_id))])
        os.chdir("..")
        run.parent.log("model_name",model.name)
        run.parent.log("model_description",model.description)
        run.parent.log("model_version",model.version)
        print(
            "Model registered: {} \nModel Description: {} "
            "\nModel Version: {}".format(
                model.name, model.description, model.version
            )
        )
    except Exception:
        traceback.print_exc(limit=None, file=None, chain=True)
        print("Model registration failed")
        raise

if __name__ == "__main__":
    main()