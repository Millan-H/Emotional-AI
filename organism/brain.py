from ann import Network
from hippocampus import Hippocampus
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
    def __init__(self, id, pos, limbs=2, emotionalOutputs=1, dataOutputs=2):
        self.dataAnalysis=Network([784,128,128,dataOutputs])
        self.emotion=Network([dataOutputs,128,128,emotionalOutputs])
        self.logic=Network([dataOutputs+emotionalOutputs,128,128,limbs])
        self.memory=Hippocampus()
        self.priorEmotionalOutput=None
        self.currentEmotionalOutput=None
        self.eta=1
        self.changeValue=50
        self.trainEpochs=0
        self.trainingMemory=[]
        self.direction=None
        self.id=id
        self.living=True
        self.pos=pos
        self.pointPos=[0,0]
    def run(self):
        #data=self.dataAnalysis.rerun(cameraData)
        distToPoint=(((self.pointPos[0]-self.pos[0])**2)+((self.pointPos[1]-self.pos[1])**2))**0.5
        dists=[self.pos[0]-self.pointPos[0], self.pos[1]-self.pointPos[1]]


        #running
        '''self.priorEmotionalOutput=self.currentEmotionalOutput
        emotionalOutput=self.emotion.rerun(dists)
        self.currentEmotionalOutput=emotionalOutput[0]'''
        self.priorEmotionalOutput=self.currentEmotionalOutput
        emotionalOutput=distToPoint
        self.currentEmotionalOutput=emotionalOutput
        moveInputs=dists
        moveInputs.append(emotionalOutput)
        movementValues=self.logic.rerun(moveInputs)
        self.pos[0]+=movementValues[0]
        self.pos[1]+=movementValues[1]
        self.trainEpochs+=1


        if self.trainEpochs>1:
            deltaE=self.currentEmotionalOutput-self.priorEmotionalOutput


            #training
            self.logic.train(data=[[moveInputs,[movementValues[0]+deltaE*self.changeValue, movementValues[1]+deltaE*self.changeValue]]], penaltyfactor=self.eta)


            #updating training values
            if deltaE<0 and self.direction==None:
                changeValue*=-1
            relevantMemory=range(len(self.trainingMemory)-len(self.trainingMemory)//5, len(self.trainingMemory))


            if 1 not in relevantMemory:
                self.direction=1
            if -1 not in relevantMemory:
                self.direction=-1


            if deltaE<0 and self.direction!=None:
                if deltaE<0:
                    self.changeValue*=-0.7
                    self.direction*=-1
                    self.eta*=0.1
           
    def getPos(self):
        return self.pos
    def kill(self):
        self.living=False
        self.memory.update("asdfasdf")


test=Organism(1, [74,86.1562])


for i in range(1,10000):
    test.run()
print(test.getPos())