a=35625
rWs=[]
rHs=[]
for i in range(1,a+1):
    if (a/i)%1==0:
        rWs.append(max(i,a/i))
        rHs.append(min(i,a/i))


diffs=[]
for i in range(len(rWs)):
    diffs.append(abs(rWs[i]-rHs[i]))


rW=rWs[diffs.index(min(diffs))]
rH=rHs[diffs.index(min(diffs))]
        

print(rW,rH)