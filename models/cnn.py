import torch
import torch.nn as nn
from config import CHANNELS, FEATURE_MAP_HEIGHT , FEATURE_MAP_WIDTH , NUM_NEURONS , NUM_CLASSES, FEATURE_MAP_DEPTH

class cnn(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.mapfeatures = nn.Sequential (
            nn.Conv3d( 3, CHANNELS, 3 , 1 ),
            nn.ReLU(),
            nn.MaxPool3d(2,2) #kernel and stride here are two so no overlapping happens with window and itself when it slides on data set
        )
        self.classifier = nn.Sequential (
           nn.Linear ( CHANNELS * FEATURE_MAP_DEPTH * FEATURE_MAP_HEIGHT * FEATURE_MAP_WIDTH, NUM_NEURONS),
           nn.ReLU(),
           nn.Linear( NUM_NEURONS , NUM_CLASSES)
        )
    def forward(self , x):
        x = self.mapfeatures( x )
        x = x.view( x.size( 0 ), -1 ) #flattening step cuz linear layers need 1d vector momken ne write torch.flatten(x, 1) instead
        x = self.classifier(x)
        return x