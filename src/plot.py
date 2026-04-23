import numpy as np, matplotlib.pyplot as plt
from .HodgkinHuxley import NGate, MGate, HGate, HodgkinHuxley


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