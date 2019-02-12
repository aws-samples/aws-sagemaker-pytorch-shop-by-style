import argparse
import os
import sys
import time
import copy
import math
import warnings
import csv
import logging

from PIL import Image
import numpy as np
import pandas as pd

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim

from cnn import CNN as cnn

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
parser.add_argument('--best-model-metric', type=str, default= 'test-loss', help='metric used to checkpoint best model') 

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

#GROUND_TRUTH_FILENAME = 'ground_truth.csv'

TEST_DS_INDEX = 'zappos50k-tuples-index-test.csv'
TRAIN_DS_INDEX = 'zappos50k-tuples-index-train.csv'

class Zappos50kTuplesDataset(Dataset):
   
    def __init__(self, csv_file, root_dir, transform=None):

        self.index = pd.read_csv(os.path.join(root_dir,csv_file), header=None)
        self.root_dir = root_dir
        self.transform = transform
  
    def __len__(self):
        return self.index.shape[0]

    def __getitem__(self, idx):
        
        img1_name = os.path.join(self.root_dir, self.index.iloc[idx, 0])
        img1 = Image.open(img1_name)
        
        img2_name = os.path.join(self.root_dir, self.index.iloc[idx, 1])
        img2 = Image.open(img2_name)
        
        image1_tensor = self.transform(img1)
        image2_tensor = self.transform(img2)

        label = self.index.iloc[idx, 2]
        
        return {"img1": image1_tensor, "img2": image2_tensor, "label": label}

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))
    
trainDS = Zappos50kTuplesDataset(TRAIN_DS_INDEX, args.data_dir, TRANSFORMATIONS)
testDS = Zappos50kTuplesDataset(TEST_DS_INDEX, args.data_dir, TRANSFORMATIONS)

trainDL = torch.utils.data.DataLoader(trainDS, args.batch_size, shuffle=True)
testDL = torch.utils.data.DataLoader(testDS, args.batch_size, shuffle=False)

logger.info('Load data')

###############################################################################
# Build the model
###############################################################################
OPTIMIZER = {'Adam': optim.Adam, 'SGD': optim.SGD}

logger.info('Build the model')

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
 
def test_model(model, test_dl):
    
    model.eval()
    running_loss = 0.0
    running_corrects = 0
        
    for data in test_dl:
        
        img1 = data['img1'].to(DEVICE)
        img2 = data['img2'].to(DEVICE)
        labels = data['label'].to(DEVICE).float()
            
        distance = model.forward(img1,img2)         
        loss = contrastive_loss(distance, labels)
        predictions = (torch.abs(distance - labels) < args.similarity_margin).int()
        
        running_loss += loss.item()
        running_corrects += torch.sum(predictions)
    
    test_loss = running_loss / len(test_dl.dataset)
    test_acc = running_corrects.double() / len(test_dl.dataset)
    
    logger.info('Test set: Average loss: {:.8f}\n'.format(test_loss, test_acc))
    
    return test_loss, test_acc

BEST_MODEL_METRIC = {
    'train-loss': 1000.0,
    'train-acc': 0.0,
    'test-loss': 1000.0,
    'test-acc': 0.0
}

def train_sim_model(model, train_dl, test_dl, optimizer, num_epochs= args.epochs):
    
    try :
        since = time.time()
        best_loss = 1000.0
        model = model.to(DEVICE)

        for epoch in range(num_epochs):

            logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
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
                
                labels = data['label'].to(DEVICE).float()
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

            BEST_MODEL_METRIC['train-loss'] = running_loss / len(train_dl.dataset)
            BEST_MODEL_METRIC['train-acc'] = running_corrects.double() / len(train_dl.dataset)

            logger.info('Training set: Average loss: {:.8f}, Average acc: {:.8f} \n'
                        .format(BEST_MODEL_METRIC['train-loss'], BEST_MODEL_METRIC['train-acc']))

            BEST_MODEL_METRIC['test-loss'], BEST_MODEL_METRIC['test-acc'] = test_model(model,test_dl)

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

model = train_sim_model(model, trainDL, testDL, optimizer, num_epochs = args.epochs)

# Move the best model to cpu and resave it
with open(MODEL_PATH, 'wb') as f:
    torch.save(model.cpu().state_dict(), f)