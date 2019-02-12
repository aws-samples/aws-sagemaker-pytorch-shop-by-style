import logging
import json
import os
import torch
import pickle
from cnn import CNN
import numpy as np
from io import BytesIO

OUTPUT_CONTENT_TYPE = 'application/json'
INPUT_CONTENT_TYPE = 'application/x-npy'
logger = logging.getLogger(__name__)

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
    return model

def output_fn(prediction_output, accept=OUTPUT_CONTENT_TYPE):
    logger.info('Serializing the generated output.')
    if accept == OUTPUT_CONTENT_TYPE:
        return json.dumps(prediction_output)
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)


def predict_fn(input_data, model):
    logger.info('Making prediction.')
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(DEVICE)
    input_data = input_data.to(DEVICE)
    
    logger.info(input_data.shape)
    images = torch.split(input_data, int(input_data.shape[0]/2))
    
    img1 = images[0].unsqueeze_(0)
    img2 = images[1].unsqueeze_(0)
    logger.info(img1.shape)
    logger.info(img2.shape)
    logger.info(print(model))
    
    distance = model.forward(img1,img2)[0].item()
    logger.info(distance)
    return {"similarity": distance}