import json
import logging
import os
import torch
import requests
from PIL import Image
from torchvision import transforms
from torchvision import models
import torch.nn as nn
import io

logger = logging.getLogger(__name__)


def model_fn(model_dir):

    n_classes=3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Loading the model.')
    model = models.resnet34(pretrained=True)

    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, n_classes)

    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f,map_location=torch.device('cpu')))

    model.to(device).eval()
    logger.info('Done loading model')
    return model

def input_fn(image_data, content_type='application/x-image'):
    logger.info('Deserializing the input data.')
    try:
        if content_type == 'application/x-image':
            image_data = Image.open(io.BytesIO(image_data))
            image_data=  image_data.convert('RGB')
    #         image_data = Image.open(requests.get(url, stream=True).raw)
    # def input_fn(image_data):
    #     logger.info('Deserializing the input data.')
    #     image_data=image_data

            image_transform = transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            return image_transform(image_data)

        raise Exception(f'Requested unsupported ContentType in content_type: {content_type}')
    except Exception as e:
        logger.info("something is wrong")

def predict_fn(input_data, model):
    logger.info('Generating prediction based on input parameters.')
    if torch.cuda.is_available():
        input_data = input_data.view(1, 3, 224, 224).cuda()
    else:
        input_data = input_data.view(1, 3, 224, 224)
    with torch.no_grad():
        model.eval()
        out = model(input_data)
        ps = torch.exp(out)
    return ps

def output_fn(prediction_output, accept='application/json'):
    logger.info('Serializing the generated output.')
    classes = {0: 'Priority', 1: 'Roundabout', 2: 'Signal'}

    topk, topclass = prediction_output.topk(1, dim=1)
    result = []

    for i in range(1):
        pred = {'prediction': classes[topclass.cpu().numpy()[0][i]], 'score': f'{topk.cpu().numpy()[0][i] * 100}%'}
        logger.info(f'Adding prediction: {pred}')
        result.append(pred)

    if accept == 'application/json':
        return json.dumps(result), accept
    raise Exception(f'Requested unsupported ContentType in Accept: {accept}')

