import logging
import json
import os
import pickle
import numpy as np
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
# Set path to root directory
import argparse
import sys
import time
sys.path.append(".")
from fastai.vision.all import *
from fastai.vision import *
import torch
import PIL


OUTPUT_CONTENT_TYPE = 'application/json'
INPUT_CONTENT_TYPE = 'application/x-image'
JSON_CONTENT_TYPE = 'application/json'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def model_fn(model_dir):
    
    logger.info(f'Start model load from {model_dir}')
    
    model = None
    inp_mod = None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info('Loading the model.')
    
    with open(os.path.join(model_dir, "export.pkl"), "rb") as inp:
        inp_mod = BytesIO(inp.read())
    
    logger.info('Loading model')    
    
    model = load_learner(inp_mod)
    
    logger.info('Done loading model')
    
    return model

def predict_fn(input_data, model):
    logger.info('Making prediction.')
    tmp_inp = input_data.getvalue()

    output = model.predict(tmp_inp)

    logger.info(f'Prediction output {output}')
    return output

# Deserialize the Invoke request body into an object we can perform prediction on
def input_fn(request_body, content_type=INPUT_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    # process an image uploaded to the endpoint
    if content_type == INPUT_CONTENT_TYPE: return io.BytesIO(request_body)
    # process a URL submitted to the endpoint
    if content_type == JSON_CONTENT_TYPE:
        img_request = requests.get(request_body['url'], stream=True)
        return open_image(io.BytesIO(img_request.content))
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))
    
# Serialize the prediction result into the desired response content type
def output_fn(prediction, accept=JSON_CONTENT_TYPE):        
    logger.info('Serializing the generated output.')
    pred = {}
    pred['Prediction']=prediction[0]
    pred['Tensor']= str(prediction[1])
    pred['Probabilities']= str(prediction[2])
    if accept == JSON_CONTENT_TYPE: return json.dumps(pred), accept
    raise Exception('Requested unsupported ContentType in Accept: {}'.format(accept))  