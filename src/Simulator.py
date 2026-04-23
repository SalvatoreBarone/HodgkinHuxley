import numpy as np
import warnings
from tqdm import tqdm


class Simulator:

    def __init__(self, neuron):
        self.model = neuron
        self.createArrays(0, 0)
        pass

    def createArrays(self, timeSteps, deltaTms):
        self.times = np.arange(timeSteps) * deltaTms
        self.Vm = np.empty(timeSteps)
        self.INa = np.empty(timeSteps)
        self.IK = np.empty(timeSteps)
        self.IL = np.empty(timeSteps)
        self.StateN = np.empty(timeSteps)
        self.StateM = np.empty(timeSteps)
        self.StateH = np.empty(timeSteps)

    def run(self, Iin, deltaTms):
        if (deltaTms > 0.05):
            warnings.warn("step sizes < 0.05 ms are recommended")
            
        assert isinstance(Iin, np.ndarray)
        
        self.createArrays(len(Iin), deltaTms)
        print(f"simulating {len(Iin)} time points...")
        
        for i in tqdm(range(len(Iin))):
            self.model.step(Iin[i], deltaTms)
            self.Vm[i] = self.model.Vm
            self.INa[i] = self.model.INa
            self.IK[i] = self.model.IK
            self.IL[i] = self.model.IL
            self.StateH[i] = self.model.h.state
            self.StateM[i] = self.model.m.state
            self.StateN[i] = self.model.n.state
        print("Simulation complete")

assert __name__ != "__main__", "do not execute this module (only import it)"