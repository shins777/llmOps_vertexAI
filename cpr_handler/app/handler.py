
import csv
from io import StringIO
import json
from fastapi import Response
from google.cloud.aiplatform.prediction.handler import PredictionHandler

class CprHandler(PredictionHandler):
    """Default prediction handler for the prediction requests sent to the application."""

    async def handle(self, request):
        """Handles a prediction request."""
        request_body = await request.body()
        
        print(f"request_body : {request_body}")
        
        prediction_instances = self._convert_csv_to_list(request_body)
        prediction_results = self._predictor.postprocess(
            self._predictor.predict(self._predictor.preprocess(prediction_instances))
        )
        
        print(f"prediction_results : {prediction_results}")
        
        return Response(content=json.dumps(prediction_results))
    
    def _convert_csv_to_list(self, data):
        """Converts list of string in csv format to list of float.
        
        Example input:
          b"1.1,2.2,3.3,4.4\n2.3,3.4,4.5,5.6\n"
          
        Example output:
            [
                [1.1, 2.2, 3.3, 4.4],
                [2.3, 3.4, 4.5, 5.6],
            ]
        """
        res = []
        for r in csv.reader(StringIO(data.decode("utf-8")), quoting=csv.QUOTE_NONNUMERIC):
            res.append(r)

        print(f"res : {res}")

            
        return res
