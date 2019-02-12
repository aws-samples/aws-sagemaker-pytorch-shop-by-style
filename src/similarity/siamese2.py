import argparse
import os
import sys
import time
import copy
import math
import warnings
import csv
import logging
import tarfile

from PIL import Image
import numpy as np
import pandas as pd

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim

from cnn import CNN as cnn
from index import Zappos50kDynamicTuplesDataset, Zappos50kIndex

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
    
parser = argparse.ArgumentParser(description='PyTorch CNN Siamese Network')

# Hyperparameters sent by the client are passed as command-line arguments to the script.
parser.add_argument('--batch-size', type=int, default=64, help='mini batch size')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--learning-rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--similarity-dims', type=int, default=64, 
                    help='the number of dimensions of the image vectors used for calculating similarity')
parser.add_argument('--similarity-margin', type=float, default=0.03, help='margin of error within labels to be considered correct')
parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer to use for training')
parser.add_argument('--best-model-metric', type=str, default= 'train-loss', help='metric used to checkpoint best model') 

# Data and model checkpoints/otput directories from the container environment
parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

args = parser.parse_args()
    
MODEL_PATH = os.path.join(args.model_dir, 'model.pth')
MODEL_INFO_PATH = os.path.join(args.model_dir, 'model_info.pth')
CHECKPOINT_PATH = os.path.join(args.output_data_dir, 'model.pth')
CHECKPOINT_STATE_PATH = os.path.join(args.output_data_dir, 'model_info.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###############################################################################
# Load data
###############################################################################

TRANSFORMATIONS = \
transforms.Compose([
    transforms.Resize(224), \
    transforms.ToTensor(), \
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) \
])

ZAPPOS50K_INDEX = 'zappos50k.idx'

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))
    
trainDS = Zappos50kDynamicTuplesDataset(ZAPPOS50K_INDEX,args.data_dir)
trainDL = torch.utils.data.DataLoader(trainDS, args.batch_size, shuffle=False)

logger.info('Load data')

###############################################################################
# Build the model
###############################################################################
OPTIMIZER = {'Adam': optim.Adam, 'SGD': optim.SGD}

logger.info('Build the model')
if os.path.isfile(os.path.join(args.data_dir,'model.tar.gz')) :
    
    logger.info('Model exists. Reload model for continued training')
    tar = tarfile.open(os.path.join(args.data_dir,'model.tar.gz'))
    tar.extractall(args.data_dir)
    tar.close()
    model = cnn(args.similarity_dims, 152)
    model.load_state_dict(torch.load(os.path.join(args.data_dir,'model.pth')))
    
else :
    
    logger.info('No model found. Loading a new ResNet model with default pre-trained weights')
    model = cnn(args.similarity_dims, 152)

optimizer = OPTIMIZER[args.optimizer](model.parameters(), lr= args.learning_rate)

# Save arguments used to create model for restoring the model later
with open(MODEL_INFO_PATH, 'wb') as f:
    model_info = {
        'simililarity-dims': args.similarity_dims
    }
    torch.save(model_info, f)
        
###############################################################################
# Training code
###############################################################################

def contrastive_loss(distance, labels):
    
    is_diff = (labels > 0.0).float()
    loss = torch.mean(((1-is_diff) * torch.pow(distance, 2)) +
                        ((is_diff) * torch.pow(torch.abs(labels - distance), 2)))
    return loss

BEST_MODEL_METRIC = {
    'train-loss': 1000.0,
    'train-acc': 0.0,
}

def train_sim_model(model, train_dl, optimizer, num_epochs= args.epochs):
    
    try :
        since = time.time()
        best_loss = 1000.0
        model = model.to(DEVICE)

        ntuples = train_dl.dataset.amplified_size()
        
        for epoch in range(num_epochs):

            logger.info('\n Epoch {}/{}'.format(epoch, num_epochs - 1))
            logger.info('-' * 10)

            # Each epoch has a training and validation phase
            model.train()  # Set model to training mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in train_dl:

                img1 = data['img1'].to(DEVICE)
                img1 = img1.view(-1,img1.shape[-3],img1.shape[-2],img1.shape[-1])
                
                img2 = data['img2'].to(DEVICE)
                img2 = img2.view(-1,img2.shape[-3],img2.shape[-2],img2.shape[-1])
                
                labels = data['labels'].to(DEVICE).float()
                labels = labels.view(-1)

                # zero the parameter gradients
                optimizer.zero_grad()

                distance = model.forward(img1,img2)

                loss = contrastive_loss(distance, labels)
                loss.backward()
                optimizer.step()

                # statistics
                predictions = (torch.abs(distance - labels) < args.similarity_margin).int()
                running_loss += loss.item()
                running_corrects += torch.sum(predictions)

            print()

            BEST_MODEL_METRIC['train-loss'] = running_loss / ntuples
            BEST_MODEL_METRIC['train-acc'] = running_corrects.double() / ntuples

            logger.info('Training set: Average loss: {:.8f}, Average acc: {:.8f} \n'
                        .format(BEST_MODEL_METRIC['train-loss'], BEST_MODEL_METRIC['train-acc']))

            # checkpoint the best model
            if  BEST_MODEL_METRIC[args.best_model_metric] < best_loss:
                best_loss = BEST_MODEL_METRIC[args.best_model_metric]

                logger.info('Saving the best model: {}'.format(best_loss))
                with open(CHECKPOINT_PATH, 'wb') as f:
                    torch.save(model.state_dict(), f)
                with open(CHECKPOINT_STATE_PATH, 'w') as f:
                    f.write('epoch {:3d} | lr: {:5.2f} | loss {:.8f}'
                            .format(epoch, args.learning_rate, best_loss))

        time_elapsed = time.time() - since
        logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        logger.info('Best Loss: {:8f}'.format(best_loss))

        # Load the best saved model.
        with open(CHECKPOINT_PATH, 'rb') as f:
            model.load_state_dict(torch.load(f))

    except: 
        
        # Load the best saved model.
        with open(CHECKPOINT_PATH, 'rb') as f:
            model.load_state_dict(torch.load(f))
        
        if model != None :
            # Move the best model to cpu and resave it
            with open(MODEL_PATH, 'wb') as f:
                torch.save(model.cpu().state_dict(), f)
                   
    return model

model = train_sim_model(model, trainDL, optimizer, num_epochs = args.epochs)

# Move the best model to cpu and resave it
with open(MODEL_PATH, 'wb') as f:
    torch.save(model.cpu().state_dict(), f)