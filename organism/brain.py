from organism.ann import Network
from organism.hippocampus import Hippocampus
import random


'''dataAnalysis=Network([784,128,128,dataOutputs])
emotion=Network([dataOutputs,128,128,emotionalOutputs])
logic=Network([dataOutputs+emotionalOutputs,128,128,limbs])
memory=Hippocampus()
cameraData=None

data=dataAnalysis.rerun(cameraData)
emotionalOutput=emotion.rerun(data)
movementValues=logic.rerun([data,emotionalOutput])
limbs.run(movementValues) #placeholder

initVals=random.randint(0,1)
actionChange=[random.randint(0,1) for i in range(len(logic.layers[len(logic.layers)-1].getOutputs()))]

while True:
    data=dataAnalysis.rerun(cameraData)
    prevEmotionalOutput=emotionalOutput
    movementValues=logic.rerun([data,emotionalOutput])
    limbs.run(movementValues) #placeholder

    emotionalOutput=emotionalOutput=emotion.rerun(data)

    if emotionalOutput[0]>prevEmotionalOutput[0]:
        rlValue=actionChange
    else:
        rlValues=[-actionChangeValue for actionChangeValue in actionChange]
    
    logic.train(0.01,rlValues)
    break'''

#MODIFIED FOR NATURAL SELECTION
class Organism: 
    def __init__(self, id, pos, limbs=2, emotionalOutputs=1, dataOutputs=1):
        self.dataAnalysis=Network([784,128,128,dataOutputs])
        self.emotion=Network([dataOutputs,128,128,emotionalOutputs])
        self.logic=Network([dataOutputs+emotionalOutputs,128,128,limbs])
        self.memory=Hippocampus()
        self.id=id
        self.living=True
        self.pos=pos
    def run(self, pos):
        #data=self.dataAnalysis.rerun(cameraData)
        emotionalOutput=self.emotion.rerun(pos)
        movementValues=self.logic.rerun([pos,emotionalOutput])
        self.pos=movementValues
    def kill(self):
        self.living=False
        self.memory.update("asdfasdf") #placeholder

test=Network([784,128,128,1])
print(test.getWeights())