#!python3
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
    
    def __init__(self, Vrest : float, Vm : float):
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
    
    def __init__(self, Vrest : float, Vm: float):
        super().__init__(Vrest)
        
    def updateRateVariables(self, Vm):
        self.alpha = HGate.get_alpha(self.Vrest, Vm)
        self.beta = HGate.get_beta(self.Vrest, Vm)
    
class HodgkinHuxley:
    EL=-50.
    EK=-75.
    ENa=55.
    gL=0.3
    gK=36.
    gNa=120.
    C=1.
    
    def __init__(self):
        self.n = NGate(self.EL)
        self.m = NGate(self.EL)
        self.h = NGate(self.EL)
        self.Vm = [self.EL]
        
    def iterate(self, Iin, deltaTms):
        self.n.updateRateVariables(self.Vm[-1])
        self.m.updateRateVariables(self.Vm[-1])
        self.h.updateRateVariables(self.Vm[-1])
        
        """calculate channel currents using the latest gate time constants"""
        self.INa = self.gNa * (self.Vm[-1] - self.ENa) * np.power(self.m.state, 3) * self.h.state 
        self.IK  = self.gK  * (self.Vm[-1] - self.EK)  * np.power(self.n.state, 4)                
        self.IL  = self.gL  * (self.Vm[-1] - self.EL)
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
    
    @staticmethod
    def plot_ninfty_vm():
        Vm = np.arange(-100., 100., 0.01)
        alpha = NGate.get_alpha(HodgkinHuxley.EL, Vm)
        beta = NGate.get_beta(HodgkinHuxley.EL, Vm)
        n = HodgkinHuxley.ninf(Vm)
        
        fig, ax = plt.subplots()
        ax.plot(Vm, alpha, 'g', label=r'$\alpha_n$')
        ax.plot(Vm, beta, 'b', label=r'$\beta_n$')
        ax.plot(Vm, n, 'r', label=r'$n_\infty$')
        legend = ax.legend(loc='upper left', frameon=False)
        plt.title("Steady-state voltage dependence of the n gating variable")
        plt.xlabel(r"$V_m \;\; (mV)$")
        plt.ylabel(r"ms^{-1}$")
        plt.savefig('ninfty_varying_vinfty.pdf')
        
    @staticmethod
    def plot_minfty_vm():
        Vm = np.arange(-100., 100., 0.01)
        alpha = MGate.get_alpha(HodgkinHuxley.EL, Vm)
        beta = MGate.get_beta(HodgkinHuxley.EL, Vm)
        m = HodgkinHuxley.minf(Vm)
        
        fig, ax = plt.subplots()
        ax.plot(Vm, alpha, 'g', label=r'$\alpha_m$')
        ax.plot(Vm, beta, 'b', label=r'$\beta_m$')
        ax.plot(Vm, m, 'r', label=r'$m_\infty$')
        legend = ax.legend(loc='upper right', frameon=False)
        plt.title("Steady-state voltage dependence of the m gating variable")
        plt.xlabel(r"$V_m \;\; (mV)$")
        plt.ylabel(r"ms^{-1}$")
        plt.savefig('minfty_varying_vinfty.pdf')
        
    @staticmethod
    def plot_hinfty_vm():
        Vm = np.arange(-100., 100., 0.01)
        alpha = HGate.get_alpha(HodgkinHuxley.EL, Vm)
        beta = HGate.get_beta(HodgkinHuxley.EL, Vm)
        h = HodgkinHuxley.hinf(Vm)
        
        fig, ax = plt.subplots()
        ax.plot(Vm, alpha, 'g', label=r'$\alpha_h$')
        ax.plot(Vm, beta, 'b', label=r'$\beta_h$')
        ax.plot(Vm, h, 'r', label=r'$h_\infty$')
        legend = ax.legend(loc='center right', frameon=False)
        plt.title("Steady-state voltage dependence of the h gating variable")
        plt.xlabel(r"$V_m \;\; (mV)$")
        plt.ylabel(r"ms^{-1}$")
        plt.savefig('hinfty_varying_vinfty.pdf')
        
    @staticmethod
    def plot_m3h_vm():
        Vm = np.arange(-100., 100., 0.01)
        m = HodgkinHuxley.minf(Vm)
        h = HodgkinHuxley.hinf(Vm)
        
        fig, ax = plt.subplots()
        ax.plot(Vm, m, 'g', label=r'$m_\infty$')
        ax.plot(Vm, h, 'b', label=r'$h_\infty$')
        ax.plot(Vm, np.multiply(h, np.power(m,3)), 'r', label=r'$m_\infty ^3 h_\infty$')
        legend = ax.legend(loc='center right', frameon=False)
        plt.title(r"Steady-state voltage dependence of the $m^3h$ gating variable")
        plt.xlabel(r"$V_m \;\; (mV)$")
        plt.ylabel(r"$ms^{-1}$")
        plt.savefig('m3h_varying_vinfty.pdf')
        
    @staticmethod
    def plot_GK_t_vm():
        t = np.arange(0., 10., 0.1)
        fig, ax = plt.subplots()
        for v, c in zip([109, 76, 38, 19], ['g', 'b', 'r', 'y']):
            ax.plot(t, HodgkinHuxley.GK(0, v, t), c, label=f'{v} mV')
        legend = ax.legend(loc='lower right', frameon=False)
        plt.title(r"Time dependence of $G_{K}$ as $V_m$ varies")
        plt.xlabel("t (ms)")
        plt.ylabel(r"$G_{K} \;\; (mS/cm^2)$")
        plt.savefig('GKt_varying_vinfty.pdf')
        
    @staticmethod
    def plot_GNa_t_vm():
        t = np.arange(0., 10., 0.1)
        fig, ax = plt.subplots()
        for v, c in zip([-50, -40, 0, 40], ['g', 'b', 'r', 'y']):
            ax.plot(t, HodgkinHuxley.GNa(0, v, t), c, label=f'{v} mV')
        legend = ax.legend(loc='center right', frameon=False)
        plt.title(r"Time dependence of $G_{Na}$ as $V_m$ varies")
        plt.xlabel("t (ms)")
        plt.ylabel(r"$G_{Na} \;\; (mS/cm^2)$")
        plt.savefig('GNat_varying_vinfty.pdf')
        
    @staticmethod
    def plot_IK_t_vm():
        t = np.arange(0., 10., 0.1)
        fig, ax = plt.subplots()
        for v, c in zip([109, 76, 38, 19], ['g', 'b', 'r', 'y']):
            ax.plot(t, HodgkinHuxley.IK(0, v, t), c, label=f'{v} mV')
        legend = ax.legend(loc='center right', frameon=False)
        plt.title(r"Time dependence of $I_{K}$ as $V_m$ varies")
        plt.xlabel("t (ms)")
        plt.ylabel(r"$I_{Na} \;\; (\mu A/cm^2)$")
        plt.savefig('IKt_varying_vinfty.pdf')
        
    @staticmethod
    def plot_INa_t_vm():
        t = np.arange(0., 10., 0.1)
        fig, ax = plt.subplots()
        for v, c in zip([-50, -40, 0, 40], ['g', 'b', 'r', 'y']):
            ax.plot(t, HodgkinHuxley.INa(0, v, t), c, label=f'{v} mV')
        legend = ax.legend(loc='lower right', frameon=False)
        plt.title(r"Time dependence of $I_{Na}$ as $V_m$ varies")
        plt.xlabel("t (ms)")
        plt.ylabel(r"$I_{Na} \;\; (\mu A/cm^2)$")
        plt.savefig('INat_varying_vinfty.pdf')
        
    @staticmethod
    def plot_K_gating_t():
        run_time = np.linspace(0., 25., 100)
        real_time = np.linspace(0., 50., 200)
        
        n = np.concatenate((
            HodgkinHuxley.n(HodgkinHuxley.EL, 0, run_time),
            HodgkinHuxley.n(0, HodgkinHuxley.EL, run_time)
        ))
        
        fig, ax = plt.subplots()
        
        ax.plot(real_time, n, 'g', label=r'$n$')
        ax.plot(real_time, np.power(n, 4), 'b', label=r'$n^4$')
        legend = ax.legend(loc='upper right', frameon=False)
        plt.title(r"Time dependence of $K$ gating variable as $V_m$ goes from -50 to 0 mV")
        plt.xlabel("t (ms)")
        plt.ylabel(r"$ms^{-1}$")
        plt.savefig('K_gating_varying_t.pdf')
        
    @staticmethod
    def plot_Na_gating_t():
        run_time = np.linspace(0., 25., 100)
        real_time = np.linspace(0., 50., 200)
        
        m = np.concatenate((
            HodgkinHuxley.m(HodgkinHuxley.EL, 0, run_time),
            HodgkinHuxley.m(0, HodgkinHuxley.EL, run_time)
        ))
        h = np.concatenate((
            HodgkinHuxley.h(HodgkinHuxley.EL, 0, run_time),
            HodgkinHuxley.h(0, HodgkinHuxley.EL, run_time)
        ))
        
        fig, ax = plt.subplots()
        
        ax.plot(real_time, m, 'g', label=r'$m$')
        ax.plot(real_time, h, 'b', label=r'$h$')
        ax.plot(real_time, np.multiply(h, np.power(m, 3)), 'r', label=r'$m^3h$')
        legend = ax.legend(loc='upper right', frameon=False)
        plt.title(r"Time dependence of $Na$ gating variables as $V_m$ goes from -50 to 0 mV")
        plt.xlabel("t (ms)")
        plt.ylabel(r"$ms^{-1}$")
        plt.savefig('Na_gating_varying_t.pdf')
        
    @staticmethod
    def plot_G_t():
        run_time = np.linspace(0., 25., 100)
        real_time = np.linspace(0., 50., 200)
        
        GK = np.concatenate((
            HodgkinHuxley.GK(HodgkinHuxley.EL, 0, run_time),
            HodgkinHuxley.GK(0, HodgkinHuxley.EL, run_time)
        ))
        GNa = np.concatenate((
            HodgkinHuxley.GNa(HodgkinHuxley.EL, 0, run_time),
            HodgkinHuxley.GNa(0, HodgkinHuxley.EL, run_time)
        ))
        
        G = np.concatenate((
            HodgkinHuxley.Gm(HodgkinHuxley.EL, 0, run_time),
            HodgkinHuxley.Gm(0, HodgkinHuxley.EL, run_time)
        ))
        
        
        fig, ax = plt.subplots()
        ax.plot(real_time, GK, 'g', label=r'$G_K$')
        ax.plot(real_time, GNa, 'b', label=r'$G_{Na}$')
        ax.plot(real_time, G, 'r', label=r'$G_m$')
        legend = ax.legend(loc='upper right', frameon=False)
        plt.title(r"Time dependence of $G_{m}$ as $V_m$ goes from -50 to 0 mV")
        plt.xlabel("t (ms)")
        plt.ylabel(r"$mS/cm^2$")
        plt.savefig('Gm_varying_t.pdf')
        
    @staticmethod
    def plot_I_t():
        run_time = np.linspace(0., 25., 100)
        real_time = np.linspace(0., 50., 200)
        
        IK = np.concatenate((
            HodgkinHuxley.IK(HodgkinHuxley.EL, 0, run_time),
            HodgkinHuxley.IK(0, HodgkinHuxley.EL, run_time)
        ))
        INa = np.concatenate((
            HodgkinHuxley.INa(HodgkinHuxley.EL, 0, run_time),
            HodgkinHuxley.INa(0, HodgkinHuxley.EL, run_time)
        ))
        
        I = np.concatenate((
            HodgkinHuxley.Im(HodgkinHuxley.EL, 0, run_time),
            HodgkinHuxley.Im(0, HodgkinHuxley.EL, run_time)
        ))
        
        fig, ax = plt.subplots()
        ax.plot(real_time, IK, 'g', label=r'$I_K$')
        ax.plot(real_time, INa, 'b', label=r'$I_{Na}$')
        ax.plot(real_time, I, 'r', label=r'$I_m$')
        legend = ax.legend(loc='upper right', frameon=False)
        plt.title(r"Time dependence of $I_{m}$ as $V_m$ goes from -50 to 0 mV")
        plt.xlabel("t (ms)")
        plt.ylabel(r"$\mu A/cm^2$")
        plt.savefig('Im_varying_t.pdf')
        
if __name__ == "__main__":
    HodgkinHuxley.plot_ninfty_vm()
    HodgkinHuxley.plot_minfty_vm()
    HodgkinHuxley.plot_hinfty_vm()
    HodgkinHuxley.plot_m3h_vm()
    HodgkinHuxley.plot_GK_t_vm()
    HodgkinHuxley.plot_IK_t_vm()
    HodgkinHuxley.plot_GNa_t_vm()
    HodgkinHuxley.plot_INa_t_vm()
    HodgkinHuxley.plot_K_gating_t()
    HodgkinHuxley.plot_Na_gating_t()
    HodgkinHuxley.plot_G_t()
    HodgkinHuxley.plot_I_t()
    
    
    