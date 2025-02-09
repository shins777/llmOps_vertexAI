
import numpy as np
import pickle

from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.utils import prediction_utils

from sklearn.datasets import load_iris

class CprPredictor(Predictor):
    
    def __init__(self):
        return
    
    def load(self, artifacts_uri: str):

        print("Load start!!!")
        prediction_utils.download_model_artifacts(artifacts_uri)

        with open(f"model.pkl", "rb") as model_f:
            self._model = pickle.load(model_f)

        self._class_names = load_iris().target_names

        print(f"Model : {self._model}")
    
    def predict(self, instances):
        """Performs prediction."""
        inputs = np.asarray(instances)

        print(f"Inputs: {inputs}")
        
        # preprocessed_inputs = self._scaler.preprocess(inputs)
        # print(f"preprocessed_inputs: {preprocessed_inputs}")
        
        outputs = self._model.predict(inputs)
        
        return {"predictions": [self._class_names[class_num] for class_num in outputs]}
