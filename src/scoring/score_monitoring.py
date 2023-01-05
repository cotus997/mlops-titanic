import joblib
import numpy as np
import pandas as pd
import os
from azureml.monitoring import ModelDataCollector
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
import json

# The init() method is called once, when the web service starts up.
#
# Typically you would deserialize the model file, as shown here using joblib,
# and store it in a global variable so your run() method can access it later.
def init():
    global model
    global inputs_dc, prediction_dc
    #Survived,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked
    inputs_dc = ModelDataCollector("best_model", designation="inputs", feature_names=["Pclass", "Age", "Sex", "SibSp", "Parch", "Fare", "Embarked"])
    prediction_dc = ModelDataCollector("best_model", designation="predictions", feature_names=["Survived"])
    # The AZUREML_MODEL_DIR environment variable indicates
    # a directory containing the model file you registered.
    model_filename = 'titanic-xgb.pkl'
    model_path = os.path.join(os.environ['AZUREML_MODEL_DIR'], model_filename)

    model = joblib.load(model_path)


# The run() method is called each time a request is made to the scoring API.
#
# Shown here are the optional input_schema and output_schema decorators
# from the inference-schema pip package. Using these decorators on your
# run() method parses and validates the incoming payload against
# the example input you provide here. This will also generate a Swagger
# API document for your web service.
@input_schema('data', NumpyParameterType(np.array([[1.0,0.0,3.0,22.0,1.0,0.0,7.25]])))
@output_schema(NumpyParameterType(np.array([0.0])))
def run(data):
    # Use the model object loaded by init().
    #data = json.loads(raw_data)['data']
    #data = np.array(data)
    result = model.predict(data)
    print(f'data: {data} result: {type(result)}')
    inputs_dc.collect(data[0]) #this call is saving our input data into Azure Blob
    res=prediction_dc.collect(result) #this call is saving our prediction data into Azure Blob
    print(res)
    print(f"mdc:{inputs_dc}")
    # You can return any JSON-serializable object.
    return result.tolist()
