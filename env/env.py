import sys
import os
import math

# Add the project root directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Project root
sys.path.append(parent_dir)

# Now import using the directory structure
from organism.brain import Organism

class Environment:
    def __init__(self, popStrtCount):
        self.population=popStrtCount
        self.orgs=[]
        self.successes=[]
        for i in range(popStrtCount):
            self.orgs.append(Organism(i,[math.cos(i),math.sin(i)]))
    def evolve(self, epochs):
        for i in range(epochs):
            epochSuccesses=[]
            for org in self.orgs:
                org.run()
                org.train()
            
            for org in self.orgs:
                if -0.01<org.pos[0]<0.01 and -0.01<org.pos[1]<0.01:
                    epochSuccesses.append(org)
            
            if len(epochSuccesses)>=4:
                for org in self.orgs:
                    if org not in epochSuccesses:
                        org.kill()
                self.successes.append(epochSuccesses)