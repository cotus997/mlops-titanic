import azureml.core
from azureml.core import Workspace, Experiment, Datastore, Environment, ScriptRunConfig
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.widgets import RunDetails


from azureml.data.data_reference import DataReference
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.compute_target import ComputeTargetException
import os

from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import DEFAULT_CPU_IMAGE
from azureml.core.runconfig import DockerConfiguration

def getOrCreateCompute(ws:Workspace)-> AmlCompute:
    

    aml_compute_target = "testcot"
    try:
        aml_compute = AmlCompute(ws, aml_compute_target)
        print("found existing compute target.")
    except ComputeTargetException:
        print("creating new compute target")

        provisioning_config = AmlCompute.provisioning_configuration(vm_size = "STANDARD_D2_V2",
                                                                    min_nodes = 0, 
                                                                    max_nodes = 4)    
        aml_compute = ComputeTarget.create(ws, aml_compute_target, provisioning_config)
        aml_compute.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
    finally:
        return aml_compute

def createRunConfig()-> RunConfiguration:
    # create a new runconfig object
    run_config = RunConfiguration()
    myenv = Environment(name="myenv")
    conda_dep = CondaDependencies(conda_dependencies_file_path="ci_dependencies.yml")
    myenv.python.conda_dependencies=conda_dep
    # enable Docker 
    run_config.environment=myenv

    ## set Docker base image to the default CPU-based image
    #run_config.environment.docker.base_image = "mcr.microsoft.com/azureml/lightgbm-3.2-ubuntu18.04-py37-cpu-inference:latest"
    ## use conda_dependencies.yml to create a conda environment in the Docker image for execution
    #run_config.environment.python.user_managed_dependencies = False
    ## specify CondaDependencies obj
    #run_config.environment.python.conda_dependencies = CondaDependencies(conda_dependencies_file_path='ci_dependencies.yml')

    return run_config
    
def create_run_config_docker()-> DockerConfiguration:
    # Specify VM and Python environment:
    vm_env = Environment.from_conda_specification(name='my-test-env', file_path='ci_dependencies.yml')
    vm_env.docker.base_image = 'mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.2-cudnn7-ubuntu18.04'
    run_config = RunConfiguration()
    run_config.docker = docker_config
    docker_config = DockerConfiguration(use_docker=True)
    return docker_config

def main():

    ws = Workspace.from_config()
    print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')

    run_preproc=True
    # Default datastore (Azure blob storage)
    def_blob_store = ws.get_default_datastore()
    #def_blob_store = Datastore(ws, "workspaceblobstore")
    print("Blobstore's name: {}".format(def_blob_store.name))


    #Upload file to datastore

    # Use a CSV to read in the data set.
    file_name = "../../data/rawdata/train.csv"

    if not os.path.exists(file_name):
        raise Exception(
            'Could not find CSV dataset at "%s". If you have bootstrapped your project, you will need to provide a CSV.'  # NOQA: E501
            % file_name
        )  # NOQA: E50
    # Upload file to default datastore in workspace

    target_path = "training-data/"
    def_blob_store.upload_files(
        files=[file_name],
        target_path=target_path,
        overwrite=True,
        show_progress=False,
    )

    blob_input_data = DataReference(
        datastore=def_blob_store,
        data_reference_name="test_data",
        path_on_datastore="training-data/train.csv")
    
    aml_compute = getOrCreateCompute(ws)
    run_config = createRunConfig()
    #processed_data1 = PipelineData("processed_data1",datastore=def_blob_store)
    train_data = PipelineData("train_data1",datastore=def_blob_store)

    source_directory="../../src/preprocess/"
    preprocess_step = PythonScriptStep(
        script_name="preprocess.py", 
        arguments=["--data", blob_input_data],
        inputs=[blob_input_data],
        compute_target=aml_compute, 
        source_directory=source_directory,
        runconfig=run_config
    )
    print("preprocess step created")
    source_directory="../../src/train/"
    train_step = PythonScriptStep(
        name="Train Model",
        script_name="train.py",
        compute_target=aml_compute,
        source_directory=source_directory,
        outputs=[train_data],
        arguments=[
            "--model",
            train_data
        ],
        runconfig=run_config,
        allow_reuse=True,
    )
    print("Step Train created")
    
    source_directory="../../src/evaluation/"
    evaluate_step = PythonScriptStep(
        name="Evaluate Model ",
        script_name="eval.py",
        compute_target=aml_compute,
        source_directory=source_directory,
        inputs=[train_data],
        arguments=[
            "--model_path",
            train_data,
        ],
        runconfig=run_config,
        allow_reuse=False,
    )
    print("Step Evaluate created")

    
    print("Step Register created")
    # Check run_evaluation flag to include or exclude evaluation step.
    train_step.run_after(preprocess_step)
    evaluate_step.run_after(train_step)
    steps = [preprocess_step,train_step, evaluate_step]
    

    train_pipeline = Pipeline(workspace=ws, steps=steps)
    #train_pipeline._set_experiment_name
    train_pipeline.validate()
    published_pipeline = train_pipeline.publish(
        name="preproc-train-register pipeline",
        description="Model training/retraining pipeline"
    )
    print(f"Published pipeline: {published_pipeline.name}")
    #pipeline_run1 = Experiment(ws, 'pipeline-exp').submit(train_pipeline)
    #print("Pipeline is submitted for execution")
    #pipeline_run1.wait_for_completion(show_output=True)

if __name__ == "__main__":
    main()