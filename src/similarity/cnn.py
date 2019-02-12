import torch
import torch.nn as nn
import torchvision.models as models

class CNN(nn.Module):

    COSINE_SIMILARITY = 0
    L1_SIMILARITY = 1
    L2_SIMILARITY = 2
    PARAM_DISTANCE_DECAY_RATE = 0.5
    
    @staticmethod
    def exp_manhattan_distance(v1,v2) :

        d = torch.pow(nn.functional.pairwise_distance(v1,v2,1),CNN.PARAM_DISTANCE_DECAY_RATE)
        return 1.0 - torch.exp(-d)

    @staticmethod
    def exp_euclidian_distance(v1,v2) :
        d = torch.pow(nn.functional.pairwise_distance(v1,v2,2),CNN.PARAM_DISTANCE_DECAY_RATE)
        return 1.0 - torch.exp(-d)

    @staticmethod
    def inv_cosine_similarity(v1,v2) :
        return torch.abs(torch.abs(nn.functional.cosine_similarity(v1,v2)) - 1.0)

    sim_fns = [inv_cosine_similarity,  exp_manhattan_distance, exp_euclidian_distance]
    sim_fn = sim_fns[L1_SIMILARITY]
    
    def build_model(self, dims, v):
        
        if v == 18:
            cnn = models.resnet18(pretrained=True)
        elif v == 34:
            cnn = models.resnet34(pretrained=True)
        elif v == 50:
            cnn = models.resnet50(pretrained=True)
        elif v == 101:
            cnn = models.resnet101(pretrained=True)
        elif v == 152:
            cnn = models.resnet152(pretrained=True)
        else:
            cnn = models.resnet152(pretrained=True)
            
        lastlayer_in = cnn.fc.in_features
        cnn.fc = nn.Linear(lastlayer_in, dims)
        
        active_layers = {"fc.weight":1,"fc.bias":1}
        for name, param in cnn.named_parameters():
            if name not in active_layers:
                param.requires_grad = False
         
        self.sim_model = cnn
       
    def __init__(self, similarity_dims, version=152):
        
        super(CNN, self).__init__()
        self.build_model(similarity_dims, version)
                                    
    def parameters(self):
        return self.sim_model.fc.parameters()
    
    def forward(self, img1, img2):
            
        cnn_out1 = self.sim_model(img1)
        cnn_out2 = self.sim_model(img2)
        
        return CNN.sim_fn(cnn_out1, cnn_out2)