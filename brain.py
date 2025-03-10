from ann import Network
import random

dataOutputs=1
emotionalOutputs=2
limbs=2

dataAnalysis=Network([784,128,128,dataOutputs])
emotion=Network([dataOutputs,128,128,emotionalOutputs])
logic=Network([dataOutputs+emotionalOutputs,128,128,limbs])
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