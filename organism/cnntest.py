import cv2
import numpy as np
import sys
import os
import math
import pickle

# Add the project root directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Project root
sys.path.append(parent_dir)

# Now import using the directory structure
from organism.annCNNimplementation import Network

network=Network([3072,256,256,256,10], 4)

dataset=pickle.load(open("C:/Users/milla/Downloads/cifar-10-python (1)/cifar-10-batches-py/data_batch_1", "rb"), encoding='bytes')
datasetReformatted=[]
for i in range(len(dataset[b'data'])):
    datasetReformatted.append([dataset[b'labels'][i],[]])
    spliceCount=-1
    for j in range(int(len(dataset[b'data'][i])/3)):
        if j%32==0:
            datasetReformatted[i][1].append([])
            spliceCount+=1
        datasetReformatted[i][1][spliceCount].append([dataset[b'data'][i][j],dataset[b'data'][i][j+1024],dataset[b'data'][i][j+2048]])

network.train(datasetReformatted, 0.01, epochs=4, rl=False)