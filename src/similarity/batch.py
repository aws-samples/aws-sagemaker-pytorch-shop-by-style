import logging
import json
import os
import torch
import pickle
from cnn import CNN
import numpy as np
import gzip
from io import BytesIO, StringIO

OUTPUT_CONTENT_TYPE = 'text/csv'
INPUT_CONTENT_TYPE = 'application/x-npy'
logger = logging.getLogger(__name__)

image_names = []

def model_fn(model_dir):
    
    model_info = {}
    
    with open(os.path.join(model_dir, 'model_info.pth'), 'rb') as f:
        model_info = torch.load(f)
    
    print('model_info: {}'.format(model_info))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info('Current device: {}'.format(device))
    model = CNN(similarity_dims=model_info['simililarity-dims'])

    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    
    model.eval()
    logger.info(model)
    return model


def input_fn(request_body, accept=INPUT_CONTENT_TYPE):
    logger.info('Deserializing the generated input.')
    if accept == INPUT_CONTENT_TYPE:

      #  logger.info(request_body)
        logger.info(len(request_body))
        logger.info(pickle.format_version)
        logger.info(np.version.version)

        request_body = gzip.decompress(request_body)
        (names, tensors) = pickle.load(BytesIO(request_body), fix_imports=True)
        global image_names
        image_names = names
        
        logger.info(tensors.shape)
        logger.info(image_names)
        return torch.from_numpy(tensors)
    raise Exception('Requested unsupported ContentType: ' + accept)


def output_fn(prediction_output, accept=OUTPUT_CONTENT_TYPE):
    logger.info('Serializing the generated output for '+accept)
    if accept == OUTPUT_CONTENT_TYPE:
        stream = StringIO()
        for i in range(len(prediction_output)):
            stream.write(image_names[i]+','+str(prediction_output[i])+'\n')
        return stream.getvalue()
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)


def predict_fn(input_data, model):
    logger.info('Making prediction.')
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(DEVICE)
    input_data = input_data.to(DEVICE)
    
    logger.info(input_data.shape)
    
    img1 = input_data.narrow(0,0,1)
    img2 = input_data.narrow(0,1,input_data.shape[0]-1)
    
    print(img1.shape)
    print(img2.shape)
        
    logger.info(img1.shape)
    logger.info(img2.shape)
    distances = model.forward(img1,img2).tolist()

    logger.info(distances)

    return distances
