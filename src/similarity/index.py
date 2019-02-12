import os
import gzip
import pickle
import hashlib
import random as rand

from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms, utils

WEIGHT_SAME_IMG = 0.0
WEIGHT_DIFF_IMG = 1.0
PARAM_SAME_CATEGORY_WEIGHTING = 0.08
PARAM_SAME_SUBCATEGORY_WEIGHTING = 0.02   
    
    
class Zappos50kIndex :
    
    IDX_STRUCTURE = {
                    "Shoes":{
                        "i":-1,
                        "r":[-1,-1],
                        "Sneakers and Athletic Shoes":{
                            "i":-1,
                            "r":[-1,-1]
                        },
                        "Loafers":{
                            "i":-1,
                            "r":[-1,-1]
                        },
                        "Crib Shoes":{
                            "i":-1,
                            "r":[-1,-1] 
                        },
                        "Prewalker":{
                            "i":-1,
                            "r":[-1,-1]
                        },
                        "Flats":{
                            "i":-1,
                            "r":[-1,-1]
                        },
                        "Clogs and Mules":{
                            "i":-1,
                            "r":[-1,-1]
                        },
                        "Oxfords":{
                            "i":-1,
                            "r":[-1,-1]
                        },
                        "Firstwalker":{
                            "i":-1,
                            "r":[-1,-1]
                        },
                        "Heels":{
                            "i":-1,
                            "r":[-1,-1]
                        },
                        "Boat Shoes":{
                            "i":-1,
                            "r":[-1,-1]
                        }
                    },
                    "Boots":{
                        "i":-1,
                        "r":[-1,-1],
                        "Prewalker Boots":{
                            "i":-1,
                            "r":[-1,-1]
                        },
                        "Ankle":{
                            "i":-1,
                            "r":[-1,-1]
                        },
                        "Over the Knee":{
                            "i":-1,
                            "r":[-1,-1]
                        },
                        "Knee High":{
                            "i":-1,
                            "r":[-1,-1]
                        },
                        "Mid-Calf":{
                            "i":-1,
                            "r":[-1,-1]
                        }
                    },
                    "Slippers":{
                        "i":-1,
                        "r":[-1,-1],
                        "Boot":{
                            "i":-1,
                            "r":[-1,-1]
                        },
                        "Slipper Heels":{
                            "i":-1,
                            "r":[-1,-1]
                        },
                        "Slipper Flats":{
                            "i":-1,
                            "r":[-1,-1]
                        }
                    },
                    "Sandals":{
                        "i":-1,
                        "r":[-1,-1],
                        "Athletic":{
                            "i":-1,
                            "r":[-1,-1]
                        },
                        "Heel":{
                            "i":-1,
                            "r":[-1,-1]
                        },
                        "Flat":{
                            "i":-1,
                            "r":[-1,-1]
                        }
                    }
               }

    IMG_BLACK_LIST = ['Boots/Mid-Calf/Primigi Kids/8022041.89.jpg',
                      'Boots/Mid-Calf/Roper Kids/7675771.248592.jpg',
                      'Shoes/Sneakers and Athletic Shoes/Puma Kids/7587775.215216.jpg',
                      'Shoes/Sneakers and Athletic Shoes/Puma Kids/7649123.238814.jpg',
                      'Shoes/Heels/Aravon/8003190.2783.jpg',
                      'Shoes/Sneakers and Athletic Shoes/Puma Kids/7649125.238816.jpg']
    
    TRANSFORMATIONS = \
        transforms.Compose([
            transforms.Resize(224), \
            transforms.ToTensor(), \
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) \
        ])
    
     
    CATEGORIES_LABELS = [('Shoes', 'Sneakers and Athletic Shoes'),
                          ('Shoes', 'Loafers'),
                          ('Shoes', 'Crib Shoes'),
                          ('Shoes', 'Prewalker'),
                          ('Shoes', 'Flats'),
                          ('Shoes', 'Clogs and Mules'),
                          ('Shoes', 'Oxfords'),
                          ('Shoes', 'Firstwalker'),
                          ('Shoes', 'Heels'),
                          ('Shoes', 'Boat Shoes'),
                          ('Shoes', 'Oxfords'),
                          ('Boots', 'Prewalker Boots'),
                          ('Boots', 'Ankle'),
                          ('Boots', 'Over the Knee'),
                          ('Boots', 'Knee High'),
                          ('Boots', 'Mid-Calf'),
                          ('Slippers', 'Boot'),
                          ('Slippers', 'Slipper Heels'),
                          ('Slippers', 'Slipper Flats'),
                          ('Sandals', 'Athletic'),
                          ('Sandals', 'Heel'),
                          ('Sandals', 'Flat')]
    
    WEIGHT_SAME_IMG = 0.0
    WEIGHT_DIFF_IMG = 1.0
    PARAM_SAME_CATEGORY_WEIGHTING = 0.08
    PARAM_SAME_SUBCATEGORY_WEIGHTING = 0.02  
    
    @classmethod
    def fromfile(self, data_dir, filename) :
                    
        self = self.load(self, os.path.join(data_dir, filename))
        self.data_dir = data_dir+"/"
        return self
        
    def __init__(self, data_dir):
        
        self.idx = Zappos50kIndex.IDX_STRUCTURE        
        self.pdata = []
        self.data_dir = data_dir+"/"
            
        i= 0
        for category in os.listdir(data_dir):

            cid = int(hashlib.sha256(category.encode('utf-8')).hexdigest(), 16) % 10**9
            print(category+": id: "+str(cid)+" i: "+str(i))
            
            self.idx[category]["i"] = cid
            self.idx[category]["r"][0] = i

            for subcat in os.listdir(data_dir+"/"+category):
            
                scid = int(hashlib.sha256(subcat.encode('utf-8')).hexdigest(), 16) % 10**9
                print("  "+subcat+": id: "+str(scid)+" i: "+str(i))
        
                
                self.idx[category][subcat]["i"] = scid
                self.idx[category][subcat]["r"][0] = i

                for (root,dirs,files) in os.walk(data_dir+"/"+category+'/'+subcat):          
                    for f in files:

                        img_path = os.path.join(root.replace(data_dir+"/",''),f)
                        if img_path not in Zappos50kIndex.IMG_BLACK_LIST :
                            self.pdata.append(img_path) 
                            i= i+1

                            self.idx[category][subcat]["r"][1] = i-1                    
                        self.idx[category]["r"][1] = i-1
        
    def get(self, i) :
        return self.pdata[i]
        
    def count(self) :
        return len(self.pdata)
        
    def get_rand_id(self, category, subcategory):
        
        r = self.idx[category][subcategory]['r']
        return rand.randint(r[0],r[1])
    
    def get_training_tuple_tensors(self, i) :
    
        n = len(Zappos50kIndex.CATEGORIES_LABELS)+1
        img1_tensors = torch.empty(n, 3, 224, 224, dtype=torch.float)
        img2_tensors = torch.empty(n, 3, 224, 224, dtype=torch.float)
        labels_tensors = torch.empty(n, dtype=torch.float)
        
        img1 = self.get(i)
        (c1, sc1) = Zappos50kIndex.get_categorization(img1)
        
        img1_tensor = Zappos50kIndex.getImageTensor(self.data_dir+img1, Zappos50kIndex.TRANSFORMATIONS)
        
        img1_tensors[0,:,:,:] = img1_tensor
        img2_tensors[0,:,:,:] = Zappos50kIndex.getImageTensor(self.data_dir+img1, Zappos50kIndex.TRANSFORMATIONS)
        labels_tensors[0] = Zappos50kIndex.WEIGHT_SAME_IMG
        
        k = 1
        for (c,sc) in Zappos50kIndex.CATEGORIES_LABELS :
            
            r = self.get_rand_id(c,sc) 
            
            lw = Zappos50kIndex.WEIGHT_DIFF_IMG
            if (c1 == c) :
                lw = lw - Zappos50kIndex.PARAM_SAME_CATEGORY_WEIGHTING
                
                if (sc1 == sc):
                    while(r == i):
                        r = self.get_rand_id(c,sc)
                
            if (sc1 == sc) :
                lw = lw - Zappos50kIndex.PARAM_SAME_SUBCATEGORY_WEIGHTING
                
            img2 = self.get(r)
            
            img1_tensors[k,:,:,:] = img1_tensor
            img2_tensors[k,:,:,:] = Zappos50kIndex.getImageTensor(self.data_dir+img2, Zappos50kIndex.TRANSFORMATIONS)
            labels_tensors[k] = lw
            k+=1
            
        return {'img1':img1_tensors, 'img2':img2_tensors, 'labels':labels_tensors}
    
    def get_categoryid(self, path) :
        c,sc = Zappos50kIndex.get_categorization(path)
        return (self.idx[c]["i"], self.idx[c][sc]["i"])
        
    @staticmethod
    def get_categorization(path) :
        s= path.split('/')
        return (s[0],s[1])
    
    #the dynamic index generates tuples for training dynamically for each image. This method returns how
    #many tuples are generated per image. If 3 tuples are generated then the amplification is 3x
    @staticmethod
    def get_tuplesize_amplification() :
        return len(Zappos50kIndex.CATEGORIES_LABELS) + 1    
    
    @staticmethod
    def getImageTensor(img_path, transform):
        
        image = Image.open(img_path)
        image_tensor = transform(image)
        return image_tensor

    def save(self, filename) :
        
        try :
            with gzip.open(filename, 'wb') as idxfile:
                pickle.dump(self, idxfile)
        finally :
            idxfile.close()

    def load(self, filename) :
            
        try :
            with gzip.open(filename,'rb') as idxfile:
                return pickle.load(idxfile)
              
        finally :
            idxfile.close()

class Zappos50kDynamicTuplesDataset(Dataset):
   
    def __init__(self, idx_filename, root_dir, transform=None):

        self.index = Zappos50kIndex.fromfile(root_dir,idx_filename)
        self.transform = transform
  
    def __len__(self):
        return self.index.count()
    
    def amplified_size(self):
        return self.index.count() * Zappos50kIndex.get_tuplesize_amplification()

    def __getitem__(self, i):
        return self.index.get_training_tuple_tensors(i)