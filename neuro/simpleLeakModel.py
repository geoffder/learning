import matplotlib.pyplot as plt
import numpy as np

# Basic simulation Properties
dt = 10E-6  # 10 us timestep

# Basic Cell Properties
Cm = 100E-12   # Membrane Capacitance = 100 pF
v_init = -70E-3    # Initial membrane potential -70 mV
Gk = 5E-9       # 5 nS conductance
Ek = -90E-3     # Reversal potential of -90 mV

# Injected Current step
current_magnitude = 100E-12  # 100 pA

# Injected current, 0.2 seconds of 0 current, 0.3 seconds of some current,
# and 0.5 seconds of no current
i_inj = np.concatenate((np.zeros([round(0.2/dt), 1]),
                        current_magnitude*np.ones([round(0.3/dt), 1]),
                        np.zeros([round(0.5/dt), 1])))

# Preallocate the voltage output
v_out = np.zeros(np.size(i_inj))

# The real computational meat
for t in range(v_out.size):
    if not t:
        v_out[t] = v_init  # initialize voltage
    else:
        i_ion = Gk * (v_out[t-1] - Ek)  # current through ion channels
        i_cap = i_inj[t] - i_ion  # total current
        dv = i_cap/Cm * dt  # Calculate dv
        v_out[t] = v_out[t-1] + dv  # add dv on to our last known voltage

# Make the graph
t_vec = np.linspace(0, 1, np.size(v_out))
plt.plot(t_vec, v_out)
plt.xlabel('Time (s)')
plt.ylabel('Membrane Potential (V)')
plt.show()
