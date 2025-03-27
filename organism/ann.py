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
    def __init__(self, nodecounts, weights=[], biases=[]):
        self.nodecounts=nodecounts
        self.layers=[Layer(self.nodecounts[0],None,'i')]
        self.weightsInputted=weights
        self.biasesInputted=biases
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
    def rerun(self, ins=[],update=False):
        for i in range(0,len(self.layers)):
            if self.layers[i].layertype=='i':
                self.layers[i].rerun(ins,update)
            else:
                output=self.layers[i].rerun(self.layers[i].getPrevLayer().getOutputs(),update)
        return output
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
    def train(self, penaltyfactor, data, epochs=4):
        for epoch in range(epochs):
            costli=[]
            ylist=[[0,0,0,0,0,0,0,0,0,0] for i in range(len(data))]
            cost=0
            for i in range(1000):

                networkOutput=self.rerun(ins=data[i][0],update=True)
                networkOutputAdjusted=[(max(min(nodeOutput,1-1e-15),1e-15)) for nodeOutput in networkOutput]
                ylist=[0 for i in range(10)]
                ylist[data[i][1]]=1
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
            print(f"\nLoss: {mse}")
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