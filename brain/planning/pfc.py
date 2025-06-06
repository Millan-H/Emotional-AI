import cv2
import numpy as np
import sys
import os
import math

# Add the project root directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Project root
sys.path.append(parent_dir)

# Now import using the directory structure
from organism.annCNNimplementation import Network

class PFC:
    def __init__(self):
        pass
    def plan(self): #planning transformer, optimal situation transformer, future planning limiting needed for adequate reduction
        pass
    def getGoalUpdates(self):
        pass
    def train(self, actualState, predictedState):
        pass
    
