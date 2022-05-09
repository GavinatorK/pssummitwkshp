import logging, requests, os, io, glob, time
from fastai.vision.all import *
from PIL import Image
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

JSON_CONTENT_TYPE = 'application/json'
PNG_CONTENT_TYPE = 'application/x-image'

# loads the model into memory from disk and returns it
def model_fn(model_dir):
    logger.info('model_fn')
    learn = load_learner(os.path.join(model_dir, 'model.pth'))
    return learn

# Deserialize the Invoke request body into an object we can perform prediction on
def input_fn(request_body, content_type=PNG_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    # process an image uploaded to the endpoint
    # if content_type == PNG_CONTENT_TYPE: return open_image(io.BytesIO(request_body))
    if content_type == PNG_CONTENT_TYPE:
        
        # image_data = Image.open(io.BytesIO(request_body))
        image_data=bytes(request_body)
        return(image_data)
    # process a URL submitted to the endpoint
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

# Perform prediction on the deserialized object, with the loaded model
def predict_fn(input_object, model):
    logger.info("Calling model")
    start_time = time.time()
    predict_class,predict_idx,predict_values = model.predict(input_object)
    print("--- Inference time: %s seconds ---" % (time.time() - start_time))
    print(f'Predicted class is {str(predict_class)}')
    print(f'Predict confidence score is {predict_values[predict_idx.item()].item()}')
    return dict(class_name = str(predict_class),
        confidence = predict_values[predict_idx.item()].item())

# Serialize the prediction result into the desired response content type
def output_fn(prediction, accept=JSON_CONTENT_TYPE):        
    logger.info('Serializing the generated output.')
    if accept == JSON_CONTENT_TYPE: return json.dumps(prediction), accept
    raise Exception('Requested unsupported ContentType in Accept: {}'.format(accept))  