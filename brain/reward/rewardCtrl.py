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
from planning.pfc import PFC

class RewardController:
    def __init__(self, inVectorLength, discountFactor, baseEmotionCount):
        self.inVectorLength=inVectorLength
        self.emotions=[Network([self.inVectorLength,128,128,1],name=0)]
        self.weightList=[1]
        self.discountFactor=-1*discountFactor
        self.prevReaction=0
        self.reaction=0
        self.reactionStorage=[[]]
        self.baseEmotionCount=baseEmotionCount

    def addEmotionalState(self,trainData):
        newEmotion=Network([self.inVectorLength,128,128,1],name=self.emotions)
        newEmotion.train(trainData)
        self.emotions.append(newEmotion)
    def react(self, stimuliVector):
        reaction=0
        for i in range(len(self.emotions)-1,-1,-1):
            goalReaction=self.emotions[i].rerun(stimuliVector)
            reaction+=(i**self.discountFactor)*goalReaction
            self.reactionStorage[i].append(goalReaction)

        self.prevReaction=self.reaction
        self.reaction=reaction
    
    def checkEmotionalStates(self,cuttoff):
        for i in range(len(self.reactionStorage)):
            avgDR=0
            count=0
            for j in range(len(self.reactionStorage)-1,max(0,len(self.reactionStorage)-150),-1):
                dR=self.reactionStorage[i][j]-self.reactionStorage[i][j-1]
                avgDR+=dR
                count+=1
            avgDR=avgDR/count
            if avgDR<=cuttoff and i>len(self.emotions)-4 and len(self.emotions)>self.baseEmotionCount:
                self.emotions.remove(self.emotions.index(i))

    def getEmotions(self):
        return self.emotions