

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from azureml.core.runconfig import RunConfiguration

from azureml.core import Workspace
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.core.compute_target import ComputeTargetException


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
    mlclient= MLClient.from_config(DefaultAzureCredential())
    env = mlclient.environments.get(name="titanic-env", version="1")
    
    run_config.environment=env

    

    return run_config