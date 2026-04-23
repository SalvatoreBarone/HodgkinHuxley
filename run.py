#!python3
from src.plot import *
from src.Simulator import Simulator

        
def plot_hh():
    neuron = HodgkinHuxley()
    
    # Defining a stimulus current
    Iin = np.zeros(30000)
    Iin[1000:3000] = 25  # add a square pulse of 50 uA / cm^2
    Iin[5000:7000] = 50  # add a square pulse of 50 uA / cm^2
    Iin[10000:12000] = 100
    Iin[15000:17000] = 200
    Iin[20000:22000] = 400
    Iin[25000:27000] = 600

    # simulate the neuron using the custom waveform
    sim = Simulator(neuron)
    sim.run(Iin, 0.01)
    
    
    # plot the results with MatPlotLib
    plt.figure(figsize=(10, 8))

    ax1 = plt.subplot(411)
    ax1.plot(sim.times, sim.Vm, color='r')
    ax1.set_ylabel("Membrane Voltage (mV)")
    ax1.set_title("Hodgkin-Huxley Spiking Neuron Model")

    ax2 = plt.subplot(412)
    ax2.plot(sim.times, Iin, color='b')
    ax2.set_ylabel(r'$I_{in} (\mu A / cm^2)$')

    ax3 = plt.subplot(413, sharex=ax1)
    ax3.plot(sim.times, sim.StateN, label='n', color='g')
    ax3.plot(sim.times, sim.StateM, label='m', color='b')
    ax3.plot(sim.times, sim.StateH, label='h', color='r' )
    ax3.set_ylabel(r"Activation $ms^-1$")
    ax3.legend()

    ax4 = plt.subplot(414, sharex=ax1)
    ax4.plot(sim.times, sim.IL, label=r'$I_{L}$', color='g')
    ax4.plot(sim.times, sim.IK, label=r'$I_{K}$', color='b')
    ax4.plot(sim.times, sim.INa, label=r'$I_{Na}$', color='r')
    ax4.set_ylabel(r'$\mu A / cm^2$')
    ax4.set_xlabel("Simulation Time (ms)")
    ax4.legend()

    plt.tight_layout()
    plt.savefig("Hodgkin_Huxley.pdf")

if __name__ == "__main__":
    plot_ninfty_vm()
    plot_minfty_vm()
    plot_hinfty_vm()
    plot_m3h_vm()
    plot_GK_t_vm()
    plot_IK_t_vm()
    plot_GNa_t_vm()
    plot_INa_t_vm()
    plot_K_gating_t()
    plot_Na_gating_t()
    plot_G_t()
    plot_I_t()
    
    plot_hh()
    