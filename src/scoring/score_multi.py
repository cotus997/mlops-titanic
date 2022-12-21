import numpy as np
import joblib
import os
#from azureml.core.model import Model
from inference_schema.schema_decorators \
    import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type \
    import NumpyParameterType
import json

from azureml.core.model import Model

def init():
    global model_1, model_2
    # Here "my_first_model" is the name of the model registered under the workspace.
    # This call will return the path to the .pkl file on the local disk.
    
    model_filename1 = 'titanic-xgb.pkl'
    model_filename2 = 'sklearn_regression_model'
    print(os.listdir(os.getenv('AZUREML_MODEL_DIR')))
    model_1_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'titanic-xgb.pkl', '7', 'titanic-xgb.pkl')
    model_2_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'my-sklearn-model', '6', 'titanic-xgb.pkl')
    #model_1_path = Model.get_model_path(model_name=model_filename1)
    #model_2_path = Model.get_model_path(model_name=model_filename2)
    
    
    # Deserialize the model files back into scikit-learn models.
    model_1 = joblib.load(model_1_path)
    model_2 = joblib.load(model_2_path)
    
#@input_schema('data', NumpyParameterType(np.array([[1.0,0.0,3.0,22.0,1.0,0.0,7.25]])))
#@output_schema(NumpyParameterType(np.array([0.0])))
# Note you can pass in multiple rows for scoring.
def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        data = np.array(data)
        
        # Call predict() on each model
        result_1 = model_1.predict(data)
        result_2 = model_2.predict(data)

        # You can return any JSON-serializable value.
        return {"prediction1": result_1.tolist(), "prediction2": result_2.tolist()}
    except Exception as e:
        result = str(e)
        return result