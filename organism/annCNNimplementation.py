import numpy as np
import pandas as pd
       
class Node:
    def __init__(self, name, connections, weights, bias, type="h"):
        self.name=name
        self.connections=connections
        self.weights=weights
        self.preactivate=0
        self.bias=bias
        self.type=type
    def fire(self, connections, update=False):
        if  self.type=="h":
            val=np.dot(np.array(connections,dtype=float).flatten(),np.array(self.weights,dtype=float).flatten())+self.bias
            if update:
                self.preactivate=val
                self.connections=connections
            return self.activate(val)
        if self.type=='o':
            val=np.dot(np.array(connections,dtype=float).flatten(),np.array(self.weights,dtype=float).flatten())+self.bias
            if update:
                self.preactivate=val
                self.connections=connections
            return val
        else:
            if update:
                self.connections=connections
            return self.connections
    def activate(self, val):
        return max(0.1*val,val)
    def updateBias(self,updateval):
        self.bias+=updateval
    def updateWeights(self,index,updateval):
        self.weights[index]+=updateval
    def getPreAc(self):
        return self.preactivate
class Layer:
    def __init__(self, nodecount, prevlayer,layertype='h', weights=[], biases=[]):
        self.nodecount=nodecount
        self.prevlayer=prevlayer
        self.layertype=layertype
        self.nodes=[]
        self.outputs=[]
        self.base=0
        self.weightsInputted=weights
        self.biasesInputted=biases
        if self.prevlayer!=None and self.layertype=='h':
            if self.weightsInputted==[] and self.biasesInputted==[]:
                for i in range(nodecount):
                    weightli=np.random.randn(len(self.prevlayer.nodes))*np.sqrt(2/len(self.prevlayer.nodes))
                    self.nodes.append(Node(f"node {i}",[0 for node in self.prevlayer.nodes],weightli,0.001))
            elif self.weightsInputted==[] and self.biasesInputted!=[]:
                for i in range(nodecount):
                    weightli=np.random.randn(len(self.prevlayer.nodes))*np.sqrt(2/len(self.prevlayer.nodes))
                    self.nodes.append(Node(f"node {i}",[0 for node in self.prevlayer.nodes],weightli,self.biasesInputted[i]))
            elif self.weightsInputted!=[] and self.biasesInputted==[]:
                for i in range(nodecount):
                    self.nodes.append(Node(f"node {i}",[0 for node in self.prevlayer.nodes],self.weightsInputted[i],0.001))
            else:
                self.nodes.append(Node(f"node {i}",[0 for node in self.prevlayer.nodes],self.weightsInputted[i],self.biasesInputted[i]))
        elif self.prevlayer!=None and self.layertype=='o':
            for i in range(nodecount):
                weightli=np.random.randn(len(self.prevlayer.nodes))*np.sqrt(2/len(self.prevlayer.nodes))
                self.nodes.append(Node(f"node {i}",[0 for node in self.prevlayer.nodes],weightli,0.001,type='o'))
        else:
            for i in range(nodecount):
                self.nodes.append(Node(f"node {i}",0, 0, 0, 'i'))
    def getWeights(self):
        nodeWeights=[]
        for i in range(len(self.nodes)):
            nodeWeights.append(self.nodes[i].weights)
        return nodeWeights
    def getNodes(self):
        return self.nodes
    def getBase(self):
        return self.base
    def rerun(self, connections=[],update=False):
        outputs=[]
        if self.layertype=='i':
            outputs=[self.nodes[i].fire(connections[i],update) for i in range(len(connections))]
        elif self.layertype=='o':
            nodeOutputs=[node.fire(connections,update) for node in self.nodes]
            nodeExps=[np.exp(output) for output in nodeOutputs]
            self.base=np.sum(nodeExps)
            outputs=[(nodeExp/self.base) for nodeExp in nodeExps]
        else:
           outputs=[node.fire(connections,update) for node in self.nodes]
        self.outputs=outputs
        return outputs
    def getPrevLayer(self):
        return self.prevlayer
    def getOutputs(self):
        return self.outputs
   
class Network:
    def __init__(self, nodecounts, convolutions=0, type='ff', kernelDimensions=[3,3], weights=[], biases=[]):
        self.nodecounts=nodecounts
        self.layers=[Layer(self.nodecounts[0],None,'i')]
        self.weightsInputted=weights
        self.biasesInputted=biases
        self.kernel=[]
        self.convolutions=convolutions
        self.output=None
        for i in range(1,len(nodecounts)):
            if self.weightsInputted!=[] and self.biasesInputted!=[]:
                if i!=len(nodecounts)-1:
                    self.layers.append(Layer(nodecounts[i],self.layers[i-1], weights=self.weightsInputted[i-1], biases=self.biasesInputted[i-1]))
                else:
                    self.layers.append(Layer(nodecounts[i],self.layers[i-1],layertype='o', weights=self.weightsInputted[i], biases=self.biasesInputted[i]))
            else:
                if i!=len(nodecounts)-1:
                    self.layers.append(Layer(nodecounts[i],self.layers[i-1]))
                else:
                    self.layers.append(Layer(nodecounts[i],self.layers[i-1],layertype='o'))
        if type=='cnn':
            self.kernel=[np.random.normal(0,np.sqrt(2/(3*kernelDimensions[0]*kernelDimensions[1])),size=(3*kernelDimensions[0]*kernelDimensions[1])) for i in range(convolutions)]
    def rerun(self, ins=[],update=False, updateRL=False):
        if self.type=='cnn':
            for i in range(self.convolutions):
                ins=self.convStack(ins,self.kernel[i])
        for i in range(0,len(self.layers)):
            if self.layers[i].layertype=='i':
                self.layers[i].rerun(ins,update)
            else:
                output=self.layers[i].rerun(self.layers[i].getPrevLayer().getOutputs(),update)
        self.output=output
        return output
    def convolution(self, data, kernel, padding=0, stride=1):
        output=[]
        paddedData=np.pad(data, ((padding, padding), (padding, padding)), 'constant', constant_values=(0, 0))
        for i in range(len(kernel)//2,len(paddedData)-1):
            for j in range(len(kernel)//2,len(paddedData[i])-1,stride):
                output.append(np.sum(data[i-1:i+2][j-1:j+2]*kernel))
        return output
    def cnnReLU(self, data):
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j]=max(0,data[i][j])
        return data
    def pooling(self, data, poolSize=2, stride=2):
        output=[]
        for i in range(0,len(data)-poolSize+1,stride):
            for j in range(0,len(data[i])-poolSize+1,stride):
                output.append(np.max(data[i:i+poolSize][j:j+poolSize]))
        return data
    def convStack(self, data, kernel, poolSize=2,padding=0, poolStride=2, convStride=1):
        pass
    @staticmethod
    def deriv(x):
        return 1 if x>0 else 0.1
    def getWeights(self):
        networkWeights=[]
        for layer in self.layers:
            networkWeights.append(layer.getWeights())
        return networkWeights
    def getNodes(self):
        networkNodes=[]
        for layer in self.layers:
            networkNodes.append(layer.getNodes())
        return networkNodes
    def getErrTerms(self, outputs, nodes, weightsnext, errtermsnext, ltype, targets=[]):
        errtermsli=[]
        if ltype=="out":
            for i in range(len(outputs)):
                errtermsli.append(outputs[i]-targets[i])
        if ltype=="h":
            for i in range(len(outputs)):
                sums=[errtermsnext[j]*weightsnext[j][i] for j in range(len(errtermsnext))]
                errtermsli.append(sum(sums)*self.deriv(nodes[i].getPreAc()))
        return errtermsli
    def getConnections(self):
        connections=[layer.getOutputs() for layer in self.layers]
        return connections
    def train(self, penaltyfactor, data, epochs=4, rl=False):
        for epoch in range(epochs):
            costli=[]
            ylist=[]
            cost=0
            for i in range(len(data)):
                networkOutput=self.output
                if not rl:
                    networkOutput=self.rerun(ins=data[i][0],update=True)
                networkOutputAdjusted=[(max(min(nodeOutput,1-1e-15),1e-15)) for nodeOutput in networkOutput]
                ylist=data[i][1]
                costli.append(-sum([(ylist[i]*np.log(networkOutputAdjusted[i])+(1-ylist[i])*np.log(1-networkOutputAdjusted[i])) for i in range(len(ylist))]))


                nodes=self.getNodes()
                errTerms=[self.getErrTerms(networkOutput,self.layers[len(self.layers)-1].getNodes(),[],[],"out",targets=ylist)]
                for i in range(len(nodes)-2,0,-1):
                    errTerms.append(self.getErrTerms(self.layers[i+1].getOutputs(),self.layers[i].getNodes(),self.layers[i+1].getWeights(),errTerms[len(nodes)-i-2],"h"))
                errTerms=errTerms[::-1]




                for i in range(len(errTerms)):
                    for j in range(len(errTerms[i])):
                        deltaB=penaltyfactor*errTerms[i][j]
                        nodes[i+1][j].updateBias(-1*deltaB)
                        for k in range(len(nodes[i])):
                            deltaW=penaltyfactor*errTerms[i][j]*self.getConnections()[i][k]
                            nodes[i+1][j].updateWeights(k,-1*deltaW)
            mse=sum(costli)/len(costli)
    def getOutput(self):
        return self.output
    def test(self, data):
        networkOutputs=[]
        correct=0
        incorrect=0
        total=len(data)
        for i in range(len(data)):
            networkOutputs.append(self.rerun(ins=data[i][0]).index(max(self.rerun(ins=data[i][0]))))
        for i in range(len(networkOutputs)):
            if networkOutputs[i]==data[i][1]:
                correct+=1
            else:
                incorrect+=1
        print(f"\nCorrect: {correct}")
        print(f"Incorrect: {incorrect}")
        print(f"Accuracy: {correct/total*100}%")