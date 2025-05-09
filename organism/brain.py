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
    def __init__(self, id, pos, limbs=2, emotionalOutputs=1, dataOutputs=1):
        self.dataAnalysis=Network([784,128,128,dataOutputs])
        self.emotion=Network([dataOutputs,128,128,emotionalOutputs])
        self.logic=Network([dataOutputs+emotionalOutputs,128,128,limbs])
        self.memory=Hippocampus()
        self.evoMem=[]
        self.id=id
        self.living=True
        self.pos=pos
    def run(self, pos):
        #data=self.dataAnalysis.rerun(cameraData)
        emotionalOutput=self.emotion.rerun(pos)
        movementValues=self.logic.rerun([pos,emotionalOutput])
        self.pos=movementValues
    def getPos(self):
        return self.pos
    def kill(self):
        self.living=False
        self.memory.update("asdfasdf") #placeholder

test=Network([784,128,128,1])
print(test.getWeights())

class EnvClassifier:
    def __init__(self):
        pass
    def getObjs(self, camData, rangeOfVisionPx):
        camDataBW=[]
        for i in range(len(camData)):
            camDataBW.append((camData[i][0]+camData[i][1]+camData[i][2])/3)
        camDataSpliced=[]
        spliceCount=-1
        for i in range(len(camDataBW)):
            if i%rangeOfVisionPx==0:
                camDataSpliced.append([])
                spliceCount+=1
            camDataSpliced[spliceCount].append(camData[i])
        colorGrads=[]
        for i in range(len(camDataSpliced)):
            for j in range(len(camDataSpliced[i])):
                colorGrads.append((camDataSpliced[i][j+1]-camDataSpliced[i][j])+(camDataSpliced[i+1][j]-camDataSpliced[i][j+1]))
        edges=[[0 for j in range(len(colorGrads[i]))] for i in range(len(colorGrads))]
        for i in range(len(colorGrads)):
            for j in range(len(colorGrads)):
                if colorGrads[i][j]>0.3:
                    edges[i][j]=1
        print(edges)
    def classifyObjs(self, objs):
        pass
    def classifyEnv(self, envData):
        pass
    def updateObjClassifier(self, trainData):
        pass
    def updateEnvClassifier(self, trainData):
        pass

envClassifier=EnvClassifier()
envClassifier.getObjs([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,84,185,159,151,60,36,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,222,254,254,254,254,241,198,198,198,198,198,198,198,198,170,52,0,0,0,0,0,0,0,0,0,0,0,0,67,114,72,114,163,227,254,225,254,254,254,250,229,254,254,140,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,17,66,14,67,67,67,59,21,236,254,106,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,83,253,209,18,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,22,233,255,83,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,254,238,44,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,59,249,254,62,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,133,254,187,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,205,248,58,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,126,254,182,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,75,251,240,57,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,19,221,254,166,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,203,254,219,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,38,254,254,77,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,31,224,254,115,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,133,254,254,52,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,61,242,254,254,52,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,121,254,254,219,40,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,121,254,207,18,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])