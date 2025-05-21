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
    def __init__(self, nodecounts, convolutions=0, type='ff', maintainDimensions=True, kernelDimensions=[3,3], weights=[], biases=[], name=''):
        self.nodecounts=nodecounts
        self.layers=[Layer(self.nodecounts[0],None,'i')]
        self.weightsInputted=weights
        self.biasesInputted=biases
        self.kernels=[]
        self.convolutions=convolutions
        self.output=None
        self.type=type
        self.convOutputs=[]
        self.maintainDimensions=maintainDimensions
        self.name=name
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
            insR=[ins[i][j][0] for i in range(len(ins)) for j in range(len(ins[i]))]
            insG=[ins[i][j][1] for i in range(len(ins)) for j in range(len(ins[i]))]
            insB=[ins[i][j][2] for i in range(len(ins)) for j in range(len(ins[i]))]
            self.convOutput=[[insR,insG,insB]]
            for kernel in self.kernels:
                insR=self.convStack(insR,kernel[0])
                insG=self.convStack(insG,kernel[1])
                insB=self.convStack(insB,kernel[2])
                self.convOutput.append([insR,insG,insB])
            
            ins=[]
            for i in range(len(insR)):
                for j in range(len(insR[i])):
                    ins.append(insR[i][j])
                    ins.append(insG[i][j])
                    ins.append(insB[i][j])

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
                region = [row[j-1:j+len(kernel)-1] for row in data[i-1:i+len(kernel)-1]]
                output.append(np.sum(np.array(region)*np.array(kernel)))
        return output
    def cnnReLU(self, data):
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j]=max(0,data[i][j])
        return data
    def pooling(self, data, poolSize=2, stride=2):
        output=[]
        for i in range(0, len(data)-poolSize+1, stride):
            for j in range(0, len(data[0])-poolSize+1, stride):
                region = [row[j:j+poolSize] for row in data[i:i+poolSize]]
                output.append(np.max(region))
        return data
    def convStack(self, data, kernel, poolSize=2, padding=0, poolStride=2, convStride=1):
        if self.maintainDimensions:
            padding=(len(kernel))//2
        convoluted=self.convolution(data,kernel,padding=padding,stride=convStride)
        relued=self.cnnReLU(convoluted)
        pooled=self.pooling(relued,poolSize=poolSize,stride=poolStride)
        return pooled
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
    
    def getKernelGrads(self, data, errTerms, connections, kernel):
        rWs=[]
        rHs=[]
        for i in range(1,len(errTerms)+1):
            if (len(errTerms)/i)%1==0:
                rWs.append(max(i,len(errTerms)/i))
                rHs.append(min(i,len(errTerms)/i))
        diffs=[]
        for i in range(len(rWs)):
            diffs.append(abs(rWs[i]-rHs[i]))
        rW=rWs[diffs.index(min(diffs))]
        rH=rHs[diffs.index(min(diffs))]
        neededGrads=np.array(np.array(errTerms)*np.array(connections)).reshape(rH,rW)
        paddingVal=((len(kernel)+len(data)-1)-len(neededGrads))//2
        kernelGrads=self.convolution(neededGrads,kernel,padding=paddingVal)
        return kernelGrads
    
    def getName(self):
        return self.name
    
    def ddgp(self, discountFactor, rewardValue, qFunction, qTarg, actorNetwork, policyNetwork, policyOutput, state, targetEta):
        actorOutput=actorNetwork.rerun(state)
        qFuncOutput=qFunction.rerun([state,policyOutput])
        qFuncMax=qTarg.rerun([state,actorOutput])
        qFuncBellman=rewardValue+discountFactor*qFuncMax
        qFuncLoss=(qFuncOutput-qFuncBellman)**2
        qFuncGrads=qFunction.getGrad([state,policyOutput])
        qFuncGradsPolicy=qFunction.getGrad([state,actorOutput],ylist=False)
        for i in range(len(qFunction.getWeights())):
            for j in range(len(qFunction.getWeights()[i])):
                for k in range(len(qFunction.getWeights()[i][j])):
                    qTarg.getWeights()[i][j][k]=targetEta*qTarg.getWeights()[i][j][k]+(1-targetEta)*qFunction.getWeights()[i][j][k]
        for i in range(len(policyNetwork.getWeights())):
            for j in range(len(policyNetwork.getWeights()[i])):
                for k in range(len(policyNetwork.getWeights()[i][j])):
                    actorNetwork.getWeights()[i][j][k]=targetEta*actorNetwork.getWeights()[i][j][k]+(1-targetEta)*policyNetwork.getWeights()[i][j][k]
        qFunction.backpropFromGrads(qFuncGrads)
        policyNetwork.backpropFromGrads(qFuncGradsPolicy)

    def backpropFromGrads(self, grads):
        for i in range(len(grads)):
            for j in range(len(grads[i])):
                for k in range(len(grads[i][j])):
                    self.layers[i+1].getNodes()[j].updateWeights(k,grads[i][j][k])
                    self.layers[i+1].getNodes()[j].updateBias(grads[i][j][k]/self.getConnections()[i][j])

    def getGrad(self, data, ylistImplemented=True):
        networkOutput=self.rerun(ins=data[0],update=True)
        networkOutputAdjusted=[(max(min(nodeOutput,1-1e-15),1e-15)) for nodeOutput in networkOutput]
        ylist=[]
        if ylistImplemented:
            ylist=[0 for i in range(len(networkOutput))]
            ylist[data[i][1]-1]=1

        nodes=self.getNodes()
        errTerms=[self.getErrTerms(networkOutput,self.layers[len(self.layers)-1].getNodes(),[],[],"out",targets=(ylist))]
        for i in range(len(nodes)-2,0,-1):
            errTerms.append(self.getErrTerms(self.layers[i+1].getOutputs(),self.layers[i].getNodes(),self.layers[i+1].getWeights(),errTerms[len(nodes)-i-2],"h"))
        errTerms=errTerms[::-1]
        gradList=[]
        for i in range(len(errTerms)):
            gradList.append([])
            for j in range(len(errTerms[i])):
                gradList[i].append([])
                for k in range(len(nodes[i])):
                    gradList[i][j].append(-1*errTerms[i][j]*self.getConnections()[i][k])
        return gradList
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
                ylist=[0 for i in range(len(networkOutput))]
                ylist[data[i][1]-1]=1
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
                if self.type=='cnn':
                    for i in range(len(self.kernels)):
                        for j in range(len(self.kernels[i])):
                           kernelGrads=self.getKernelGrads(self.convOutputs[i][j],errTerms[0],self.getConnections()[0],self.kernels[i][j],padding=0,stride=1)
                           self.kernels[i][j]=self.kernels[i][j]-penaltyfactor*kernelGrads
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