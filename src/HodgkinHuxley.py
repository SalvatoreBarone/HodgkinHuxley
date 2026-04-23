import numpy as np, matplotlib.pyplot as plt
from abc import ABC

class Gate(ABC):
    """The Gate object is the base class that manages a channel's common kinetics and open state"""
    
    def __init__(self, Vrest : float):
        super().__init__()
        self.alpha = 0
        self.beta = 0
        self.Vrest = Vrest
        self.state = 0
        self.updateRateVariables(self.Vrest)
        self.setInfiniteState()
        self.setInfiniteTau()

    def update(self, deltaTms):
        alphaState = self.alpha * (1-self.state)
        betaState = self.beta * self.state
        self.state += deltaTms * (alphaState - betaState)

    def setInfiniteState(self):
        self.state = self.alpha / (self.alpha + self.beta)
        
    def setInfiniteTau(self):
        self.tau = 1 / (self.alpha + self.beta)
        
    def updateRateVariables(self, Vm : float):
        pass
    
class NGate(Gate):
    @staticmethod  
    def get_alpha(Vrest, Vm):
        return 0.01 * (Vrest - Vm + 10) / (np.exp( 0.1 * (Vrest - Vm + 10 ) ) -1 )
    
    @staticmethod
    def get_beta(Vrest, Vm):
        return 0.125* np.exp(0.0125 * (Vrest - Vm))

    def __init__(self, Vrest : float):
        super().__init__(Vrest)
        
    def updateRateVariables(self, Vm):
        self.alpha = NGate.get_alpha(self.Vrest, Vm)
        self.beta = NGate.get_beta(self.Vrest, Vm)
        
class MGate(Gate):
    @staticmethod
    def get_alpha(Vrest : float, Vm: float): 
        return 0.1*(Vrest-Vm+25)/(np.exp(0.1*(Vrest-Vm+25))-1)
    
    @staticmethod
    def get_beta(Vrest : float, Vm: float): 
        return 4*np.exp((Vrest-Vm)/18)
    
    def __init__(self, Vrest : float):
        super().__init__(Vrest)
        
    def updateRateVariables(self, Vm):
        self.alpha = MGate.get_alpha(self.Vrest, Vm)
        self.beta = MGate.get_beta(self.Vrest, Vm)
    
class HGate(Gate):   
    @staticmethod
    def get_alpha(Vrest : float, Vm: float): 
        return 0.07*np.exp((Vrest-Vm)/20)
    
    @staticmethod
    def get_beta(Vrest : float, Vm: float): 
        return 1/(1+np.exp((Vrest-Vm+30)/10))
    
    def __init__(self, Vrest : float):
        super().__init__(Vrest)
        
    def updateRateVariables(self, Vm):
        self.alpha = HGate.get_alpha(self.Vrest, Vm)
        self.beta = HGate.get_beta(self.Vrest, Vm)
    
class HodgkinHuxley:
    EL=-50
    EK=-75.
    ENa=55.
    gL=0.3
    gK=36.
    gNa=120.
    C=1.
    
    def __init__(self, V0 : float = -50.):
        self.Vm = V0
        self.n = NGate(self.Vm)
        self.m = MGate(self.Vm)
        self.h = HGate(self.Vm)
        
    def step(self, Iin, deltaTms):
        self.n.updateRateVariables(self.Vm)
        self.m.updateRateVariables(self.Vm)
        self.h.updateRateVariables(self.Vm)
        
        """calculate channel currents using the latest gate time constants"""
        self.INa = self.gNa * (self.Vm - self.ENa) * np.power(self.m.state, 3) * self.h.state 
        self.IK  = self.gK  * (self.Vm - self.EK)  * np.power(self.n.state, 4)                
        self.IL  = self.gL  * (self.Vm - self.EL)
        Im = Iin - self.INa - self.IK - self.IL
        self.Vm += deltaTms * Im / self.C
        
        """calculate new channel open states using latest Vm"""
        self.n.update(deltaTms)
        self.m.update(deltaTms)
        self.h.update(deltaTms)
    
    @staticmethod
    def ninf(Vm):
        return NGate.get_alpha(HodgkinHuxley.EL, Vm) / ( NGate.get_alpha(HodgkinHuxley.EL, Vm) + NGate.get_beta(HodgkinHuxley.EL, Vm))
    
    @staticmethod
    def taun(Vm):
        return 1 / (NGate.get_alpha(HodgkinHuxley.EL, Vm) + NGate.get_beta(HodgkinHuxley.EL, Vm))
    
    @staticmethod
    def n(V0, Vm : float, t):
        return HodgkinHuxley.ninf(Vm) - (HodgkinHuxley.ninf(Vm) - HodgkinHuxley.ninf(V0)) * np.exp(-t / HodgkinHuxley.taun(Vm) )
    
    @staticmethod
    def minf(Vm): 
        return MGate.get_alpha(HodgkinHuxley.EL, Vm)/(MGate.get_alpha(HodgkinHuxley.EL, Vm)+MGate.get_beta(HodgkinHuxley.EL, Vm))
    
    @staticmethod
    def taum(Vm): 
        return 1/(MGate.get_alpha(HodgkinHuxley.EL, Vm)+MGate.get_beta(HodgkinHuxley.EL, Vm))
    
    @staticmethod
    def m(V0, Vm, t):
        return HodgkinHuxley.minf(Vm) - (HodgkinHuxley.minf(Vm) - HodgkinHuxley.minf(V0)) * np.exp(-t / HodgkinHuxley.taum(Vm))
    
    @staticmethod
    def hinf(Vm): 
        return HGate.get_alpha(HodgkinHuxley.EL, Vm)/(HGate.get_alpha(HodgkinHuxley.EL, Vm)+HGate.get_beta(HodgkinHuxley.EL, Vm))
    
    @staticmethod
    def tauh(Vm): 
        return 1/(HGate.get_alpha(HodgkinHuxley.EL, Vm)+HGate.get_beta(HodgkinHuxley.EL, Vm))
    
    
    @staticmethod
    def h(V0, Vm, t):
        return HodgkinHuxley.hinf(Vm) - ( HodgkinHuxley.hinf(Vm) - HodgkinHuxley.hinf(V0)) * np.exp(-t / HodgkinHuxley.tauh(Vm))
    
    @staticmethod
    def IL(Vm):
        return HodgkinHuxley.gL * (Vm - HodgkinHuxley.EL)
    
    @staticmethod
    def GK(V0, Vm, t):
        return np.multiply(np.power(HodgkinHuxley.n(V0, Vm, t), 4), HodgkinHuxley.gK)
    
    @staticmethod
    def IK(V0, Vm, t):
        return HodgkinHuxley.GK(V0, Vm, t)*(Vm - HodgkinHuxley.EK)
    
    @staticmethod
    def GNa(V0,Vm,t): 
        return HodgkinHuxley.gNa*HodgkinHuxley.h(V0,Vm,t) * HodgkinHuxley.m(V0,Vm,t)**3
    
    @staticmethod
    def INa(V0,Vm,t): 
        return HodgkinHuxley.GNa(V0,Vm,t)*(Vm-HodgkinHuxley.ENa)
    
    @staticmethod
    def Gm(V0,Vm,t): 
        return HodgkinHuxley.GK(V0,Vm,t)+HodgkinHuxley.GNa(V0,Vm,t)
    
    @staticmethod
    def Im(V0,Vm,t): 
        return HodgkinHuxley.IK(V0, Vm, t) + HodgkinHuxley.INa(V0,Vm,t) + HodgkinHuxley.IL(Vm)
 
    
assert __name__ != "__main__", "do not execute this module (only import it)"
    