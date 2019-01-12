import matplotlib.pyplot as plt
import numpy as np

# Basic simulation Properties
dt = 10e-6  # 10 us timestep

# Basic Cell Properties
Cm = 100e-12   # Membrane Capacitance = 100 pF
v_init = -70e-3    # Initial membrane potential
# leak
Gleak = 5e-9       # 5 nS conductance
Eleak = -70e-3     # Reversal potential of -90 mV
# voltage-gated potassium (HH)
Gbar_k = 1e-6  # max conductance
Ek = -80e-3
# voltage-gated sodium (HH)
Gbar_na = 9e-6
Ena = 40e-3


def Gk(v, n):
    return Gbar_k * n**4


def Gna(v, m, h):
    return Gbar_na * m**3 * h


def alpha_n(v):  # channel opening rate
    return .01*(-v - 55) / (np.exp((-v - 55)/10) - 1) * 1000


def beta_n(v):  # channel closing rate
    return .125*np.exp((-v - 65)/80) * 1000


def stateK(v, n):  # calc open/close state of channel population
    v = v*1000  # work in mV instead of volts
    delta = (alpha_n(v)*(1 - n) - beta_n(v)*n) * dt
    return n + delta


def alpha_m(v):
    return .1*(-v - 40) / (np.exp((-v-40)/10.0) - 1) * 1000


def beta_m(v):
    return 4*np.exp((-v - 65)/18.0) * 1000


def alpha_h(v):
    return .07*np.exp((-v - 65)/20.0) * 1000


def beta_h(v):
    return 1/(np.exp((-v - 35)/10.0) + 1) * 1000


def stateNa(v, m, h):
    v = v*1000
    # activation (m particle)
    delta_m = (alpha_m(v)*(1 - m) - beta_m(v)*m) * dt
    # inactivation (h particle)
    delta_h = (alpha_h(v)*(1 - h) - beta_h(v)*h) * dt
    return m + delta_m, h + delta_h


# initialize channel states
nK = alpha_n(v_init)/(alpha_n(v_init)+beta_n(v_init))
mNa = alpha_m(v_init)/(alpha_m(v_init)+beta_m(v_init))
hNa = alpha_h(v_init)/(alpha_h(v_init)+beta_h(v_init))

# Injected Current step
current_magnitude = 100E-12  # 100 pA

# Injected current, 0.2 seconds of 0 current, 0.3 seconds of some current,
# and 0.5 seconds of no current
i_inj = np.concatenate((np.zeros([round(0.2/dt), 1]),
                        current_magnitude*np.ones([round(0.3/dt), 1]),
                        np.zeros([round(0.5/dt), 1])))

# Preallocate the voltage output
v_out = np.zeros(i_inj.size)

# The real computational meat
for t in range(v_out.size):
    if not t:
        v_out[t] = v_init  # initialize voltage
    else:
        # voltage gated potassium
        nK = stateK(v_out[t-1], nK)
        i_k = Gk(v_out[t-1], nK) * (v_out[t-1] - Ek)
        # voltage gated sodium
        mNa, hNa = stateNa(v_out[t-1], mNa, hNa)
        i_na = Gna(v_out[t-1], mNa, hNa) * (v_out[t-1] - Ena)
        # leak and total current
        i_leak = Gleak * (v_out[t-1] - Eleak)
        i_cap = i_inj[t] - i_leak - i_k - i_na  # total current
        # calculate voltage
        dv = i_cap/Cm * dt  # change in voltage
        v_out[t] = v_out[t-1] + dv  # add dv on to our last known voltage

# Make the graph
t_vec = np.linspace(0, 1, np.size(v_out))
plt.plot(t_vec, v_out)
plt.xlabel('Time (s)')
plt.ylabel('Membrane Potential (V)')
plt.show()
