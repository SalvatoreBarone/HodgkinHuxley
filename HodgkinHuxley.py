#!python3
import numpy as np, matplotlib.pyplot as plt


class HodgkinHuxley:
    El=-50.
    EK=-75.
    ENa=55.
    gK=24.31
    gNa=70.7
    
    def __init__(self):
        pass
      
    @staticmethod  
    def alphan(vm_inf):
        return 0.01 * (HodgkinHuxley.El - vm_inf + 10) / (np.exp( 0.1 * (HodgkinHuxley.El - vm_inf + 10 ) ) -1 )
    
    @staticmethod
    def betan(vm_inf):
        return 0.125* np.exp(0.0125 * (HodgkinHuxley.El - vm_inf))
    
    @staticmethod
    def ninf(vm_inf):
        return HodgkinHuxley.alphan(vm_inf) / ( HodgkinHuxley.alphan(vm_inf) + HodgkinHuxley.betan(vm_inf))
    
    @staticmethod
    def taun(vm_inf):
        return 1 / (HodgkinHuxley.alphan(vm_inf) + HodgkinHuxley.betan(vm_inf))

    @staticmethod
    def alpham(vm_inf): 
        return 0.1*(HodgkinHuxley.El-vm_inf+25)/(np.exp(0.1*(HodgkinHuxley.El-vm_inf+25))-1)
    
    @staticmethod
    def betam(vm_inf): 
        return 4*np.exp((HodgkinHuxley.El-vm_inf)/18)
    
    @staticmethod
    def alphah(vm_inf): 
        return 0.07*np.exp((HodgkinHuxley.El-vm_inf)/20)
    
    @staticmethod
    def betah(vm_inf): 
        return 1/(1+np.exp((HodgkinHuxley.El-vm_inf+30)/10))
    
    @staticmethod
    def minf(vm_inf): 
        return HodgkinHuxley.alpham(vm_inf)/(HodgkinHuxley.alpham(vm_inf)+HodgkinHuxley.betam(vm_inf))
    
    @staticmethod
    def taum(vm_inf): 
        return 1/(HodgkinHuxley.alpham(vm_inf)+HodgkinHuxley.betam(vm_inf))
    
    @staticmethod
    def hinf(vm_inf): 
        return HodgkinHuxley.alphah(vm_inf)/(HodgkinHuxley.alphah(vm_inf)+HodgkinHuxley.betah(vm_inf))
    
    @staticmethod
    def tauh(vm_inf): 
        return 1/(HodgkinHuxley.alphah(vm_inf)+HodgkinHuxley.betah(vm_inf))
    
    @staticmethod
    def n(vzero : float, vm_inf : float, t):
        return HodgkinHuxley.ninf(vm_inf) - (HodgkinHuxley.ninf(vm_inf) - HodgkinHuxley.ninf(vzero)) * np.exp(-t / HodgkinHuxley.taun(vm_inf) )
    
    @staticmethod
    def m(vzero, vm_inf, t):
        return HodgkinHuxley.minf(vm_inf) - (HodgkinHuxley.minf(vm_inf) - HodgkinHuxley.minf(vzero)) * np.exp(-t / HodgkinHuxley.taum(vm_inf))
    
    @staticmethod
    def h(vzero, vm_inf, t):
        return HodgkinHuxley.hinf(vm_inf) - ( HodgkinHuxley.hinf(vm_inf) - HodgkinHuxley.hinf(vzero)) * np.exp(-t / HodgkinHuxley.tauh(vm_inf))
    
    @staticmethod
    def GK(vzero, vm_inf, t):
        return np.multiply(np.power(HodgkinHuxley.n(vzero, vm_inf, t), 4), HodgkinHuxley.gK)
    
    @staticmethod
    def IK(vzero, vm_inf, t):
        return HodgkinHuxley.GK(vzero, vm_inf, t)*(vm_inf - HodgkinHuxley.EK)
    
    @staticmethod
    def GNa(vzero,vm_inf,t): 
        return HodgkinHuxley.gNa*HodgkinHuxley.h(vzero,vm_inf,t) * HodgkinHuxley.m(vzero,vm_inf,t)**3
    
    @staticmethod
    def INa(vzero,vm_inf,t): 
        return HodgkinHuxley.GNa(vzero,vm_inf,t)*(vm_inf-HodgkinHuxley.ENa)
    
    @staticmethod
    def Gm(vzero,vm_inf,t): 
        return HodgkinHuxley.GK(vzero,vm_inf,t)+HodgkinHuxley.GNa(vzero,vm_inf,t)
    
    @staticmethod
    def Im(vzero,vm_inf,t): 
        return HodgkinHuxley.IK(vzero,vm_inf,t)+HodgkinHuxley.INa(vzero,vm_inf,t)
    
    @staticmethod
    def plot_ninfty_vm():
        vm_inf = np.arange(-100., 100., 0.01)
        alpha = HodgkinHuxley.alphan(vm_inf)
        beta = HodgkinHuxley.betan(vm_inf)
        n = HodgkinHuxley.ninf(vm_inf)
        
        fig, ax = plt.subplots()
        ax.plot(vm_inf, alpha, 'g', label=r'$\alpha_n$')
        ax.plot(vm_inf, beta, 'b', label=r'$\beta_n$')
        ax.plot(vm_inf, n, 'r', label=r'$n_\infty$')
        legend = ax.legend(loc='upper left', frameon=False)
        plt.title("Steady-state voltage dependence of the n gating variable")
        plt.xlabel(r"$V_m \;\; (mV)$")
        plt.ylabel(r"ms^{-1}$")
        plt.savefig('ninfty_varying_vinfty.pdf')
        
    @staticmethod
    def plot_minfty_vm():
        vm_inf = np.arange(-100., 100., 0.01)
        alpha = HodgkinHuxley.alpham(vm_inf)
        beta = HodgkinHuxley.betam(vm_inf)
        m = HodgkinHuxley.minf(vm_inf)
        
        fig, ax = plt.subplots()
        ax.plot(vm_inf, alpha, 'g', label=r'$\alpha_m$')
        ax.plot(vm_inf, beta, 'b', label=r'$\beta_m$')
        ax.plot(vm_inf, m, 'r', label=r'$m_\infty$')
        legend = ax.legend(loc='upper right', frameon=False)
        plt.title("Steady-state voltage dependence of the m gating variable")
        plt.xlabel(r"$V_m \;\; (mV)$")
        plt.ylabel(r"ms^{-1}$")
        plt.savefig('minfty_varying_vinfty.pdf')
        
    @staticmethod
    def plot_hinfty_vm():
        vm_inf = np.arange(-100., 100., 0.01)
        alpha = HodgkinHuxley.alphah(vm_inf)
        beta = HodgkinHuxley.betah(vm_inf)
        h = HodgkinHuxley.hinf(vm_inf)
        
        fig, ax = plt.subplots()
        ax.plot(vm_inf, alpha, 'g', label=r'$\alpha_h$')
        ax.plot(vm_inf, beta, 'b', label=r'$\beta_h$')
        ax.plot(vm_inf, h, 'r', label=r'$h_\infty$')
        legend = ax.legend(loc='center right', frameon=False)
        plt.title("Steady-state voltage dependence of the h gating variable")
        plt.xlabel(r"$V_m \;\; (mV)$")
        plt.ylabel(r"ms^{-1}$")
        plt.savefig('hinfty_varying_vinfty.pdf')
        
    @staticmethod
    def plot_m3h_vm():
        vm_inf = np.arange(-100., 100., 0.01)
        m = HodgkinHuxley.minf(vm_inf)
        h = HodgkinHuxley.hinf(vm_inf)
        
        fig, ax = plt.subplots()
        ax.plot(vm_inf, m, 'g', label=r'$m_\infty$')
        ax.plot(vm_inf, h, 'b', label=r'$h_\infty$')
        ax.plot(vm_inf, np.multiply(h, np.power(m,3)), 'r', label=r'$m_\infty ^3 h_\infty$')
        legend = ax.legend(loc='center right', frameon=False)
        plt.title(r"Steady-state voltage dependence of the $m^3h$ gating variable")
        plt.xlabel(r"$V_m \;\; (mV)$")
        plt.ylabel(r"ms^{-1}")
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
        plt.ylabel(r"$I_{K}$ \;\; ($\mu A/cm^2$)")
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
        plt.ylabel(r"$I_{Na}$ \;\; ($\mu A/cm^2$)")
        plt.savefig('INat_varying_vinfty.pdf')
        
    @staticmethod
    def plot_K_gating_t():
        v_zero = (HodgkinHuxley.El, 0.)
        v_set = (0., HodgkinHuxley.El)
        time = ( np.arange(0., 10., 0.01), np.arange(10., 20., 0.01) )
        
        n = np.concatenate([HodgkinHuxley.n(v0, v2, t - 0.995*t[0]) for v0, v2, t in zip (v_zero, v_set, time) ])
        t = np.concatenate(time)
        
        fig, ax = plt.subplots()
        
        ax.plot(t, n, 'g', label=r'$n$')
        ax.plot(t, np.power(n, 4), 'b', label=r'$n^4$')
        legend = ax.legend(loc='upper right', frameon=False)
        plt.title(r"Time dependence of $K$ gating variable as $V_m$ goes from -50 to 0 mV")
        plt.xlabel("t (ms)")
        plt.ylabel(r"$ms^{-1}$")
        plt.savefig('K_gating_varying_t.pdf')
        
    @staticmethod
    def plot_Na_gating_t():
        v_zero = (HodgkinHuxley.El, 0)
        v_set = (0, HodgkinHuxley.El)
        time = ( np.arange(0., 10., 0.1), np.arange(10., 20., 0.1) )
        
        m = np.concatenate([ HodgkinHuxley.m(v0, v2, t - 0.995*t[0]) for v0, v2, t in zip (v_zero, v_set, time) ])
        h = np.concatenate([ HodgkinHuxley.h(v0, v2, t-  0.995*t[0]) for v0, v2, t in zip (v_zero, v_set, time) ])
        t = np.concatenate(time)
        
        fig, ax = plt.subplots()
        
        ax.plot(t, m, 'g', label=r'$m$')
        ax.plot(t, h, 'b', label=r'$h$')
        ax.plot(t, np.multiply(h, np.power(m, 3)), 'r', label=r'$m^3h$')
        legend = ax.legend(loc='upper right', frameon=False)
        plt.title(r"Time dependence of $Na$ gating variables as $V_m$ goes from -50 to 0 mV")
        plt.xlabel("t (ms)")
        plt.ylabel(r"$ms^{-1})$")
        plt.savefig('Na_gating_varying_t.pdf')
        
    @staticmethod
    def plot_G_t():
        run_time = np.linspace(0., 25., 100)
        real_time = np.linspace(0., 50., 200)
        
        GK = np.concatenate((
            HodgkinHuxley.GK(HodgkinHuxley.El, 0, run_time),
            HodgkinHuxley.GK(0, HodgkinHuxley.El, run_time)
        ))
        GNa = np.concatenate((
            HodgkinHuxley.GNa(HodgkinHuxley.El, 0, run_time),
            HodgkinHuxley.GNa(0, HodgkinHuxley.El, run_time)
        ))
        
        G = np.concatenate((
            HodgkinHuxley.Gm(HodgkinHuxley.El, 0, run_time),
            HodgkinHuxley.Gm(0, HodgkinHuxley.El, run_time)
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
            HodgkinHuxley.IK(HodgkinHuxley.El, 0, run_time),
            HodgkinHuxley.IK(0, HodgkinHuxley.El, run_time)
        ))
        INa = np.concatenate((
            HodgkinHuxley.INa(HodgkinHuxley.El, 0, run_time),
            HodgkinHuxley.INa(0, HodgkinHuxley.El, run_time)
        ))
        
        I = np.concatenate((
            HodgkinHuxley.Im(HodgkinHuxley.El, 0, run_time),
            HodgkinHuxley.Im(0, HodgkinHuxley.El, run_time)
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
    
    
    