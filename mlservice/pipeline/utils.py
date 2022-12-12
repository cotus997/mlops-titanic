
from azureml.core import Workspace,Environment
from azureml.core.conda_dependencies import CondaDependencies

from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.core.compute_target import ComputeTargetException

from azureml.core.runconfig import RunConfiguration

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