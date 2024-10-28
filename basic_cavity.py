# Importing the libraries
import finesse
from finesse.components.laser import Laser
from finesse.components.gauss import Gauss
from finesse.components.mirror import Mirror
from finesse.components.space import Space
from finesse.components.beamsplitter import Beamsplitter
from finesse.components.cavity import Cavity
from finesse.components.modulator import Modulator
from finesse.analysis.actions.axes import Xaxis
from finesse.detectors.powerdetector import PowerDetector
from finesse.detectors.powerdetector import PowerDetectorDemod1
from finesse.locks import Lock
from finesse.analysis.actions.locks import RunLocks
import numpy as np
import matplotlib.pyplot as plt

# Defining variables
wavelength = 1064e-9
cav_len = 0.5

# Defining the model
model = finesse.Model()
# model.lambda0 = wavelength

# Adding laser source
laser = Laser("source", 1, 2.8167763157e14)
model.add(laser)

# Adding modulation to the laser
mod = Modulator("mod", 30e6, 0.7, 3)
model.add(mod)

# Defining the mirrors
m1 = Mirror("m1", R=0.9, L=0.0, Rc=0.7)
m2 = Mirror("m2", R=0.9, L=0.0, Rc=0.7)
model.add(m1)
model.add(m2)

# Adding 6 dof's to the mirrors
# In finesse the optical path is along z axis

# Defining the detectors
pd_trans = PowerDetector("pd_trans", m2.p2.o)
model.add(pd_trans)
pd_ref = PowerDetector("pd_ref", m1.p1.o)
model.add(pd_ref)

# Defining the spaces
model.add(Space("s0", laser.p1, mod.p1))
model.add(Space("s1", mod.p2, m1.p1))
model.add(Space("s2", m1.p2, m2.p1, L=cav_len))

# Adding the cavities
cav = Cavity("cav", m1.p2.o)
model.add(cav)
print(f"Cavity g parameters are {cav.g}")

# First we plot the modulated signal (reflected and transmitted) v/s m2.phi
xlimits = [-100, 0, 400]
xaxis = Xaxis(m2.phi, 'lin', xlimits[0], xlimits[1], xlimits[2])
x = np.linspace(xlimits[0], xlimits[1], xlimits[2]+1)

mod.f = 10e6
output = model.run(xaxis)

plt.subplot(2, 3, 1)
plt.title("Without Modulation")
plt.plot(x, output["pd_trans"], label="trans", color="blue")
plt.xlabel("m2.phi")
plt.ylabel("Power")
plt.legend()
plt.plot()

mod.f = 30e6
output = model.run(xaxis)

plt.subplot(2, 3, 2)
plt.title("With Modulation")
plt.plot(x, output["pd_trans"], label="trans", color="blue")
plt.plot(x, output["pd_ref"], label="refl", color="black")
plt.xlabel("m2.phi")
plt.ylabel("Power")
plt.legend()
plt.plot()


# Demodulating the reflected signal

outputs = []
model.remove(pd_ref)
for phase in np.linspace(0, 90, endpoint=True, num=6):
    pd = PowerDetectorDemod1("pd", m1.p1.o, mod.f, phase)
    model.add(pd)
    output = model.run(xaxis)
    outputs.append([output, phase])
    model.remove(pd)

plt.subplot(2, 3, 3)
plt.title("Error Signal")
for output in outputs:
    plt.plot(x, output[0]["pd"], label=f"{output[1]}")
plt.xlabel("m2.phi")
plt.ylabel("Demodulated Power")
plt.legend()
plt.plot()

# Checking for max error gradient

phases = np.arange(-90, 30, 1)

gradients = []
for phase in phases:
    testpd = PowerDetectorDemod1("pd", m1.p1.o, mod.f, phase)
    model.add(testpd)
    output = model.run("xaxis(m2.phi, lin, -1, 1, 2)")
    var = output["pd"]
    gradients.append((var[0] - var[-1])/2)
    model.remove(testpd)

# Find the maximum gradient for the error signal

gradients = np.array(gradients)
idxmax = np.argmax(np.abs(gradients))
print(f'Maximum error signal gradient occurs at {phases[idxmax]} degrees')

plt.subplot(2, 3, 4)
plt.title("Error Gradients")
plt.plot(phases,gradients)
plt.xlabel('Demodulation Phase (°)')
plt.ylabel('Error Signal Gradient (arb)');

# Locking the cavity

pdh = PowerDetectorDemod1("pdh", m1.p1.o, mod.f, phases[idxmax])
model.add(pdh)
lock = Lock("lock", pdh, m1.phi, -10, 1e-8)
model.add(lock)

# Checking if cavity is locked

pd_ref = PowerDetector("pd_ref", m1.p1.o)
model.add(pd_ref)
output = model.run(Xaxis(m2.phi, 'lin', 0, 100, 400, pre_step=RunLocks()))

plt.subplot(2, 3, 5)
plt.title("Checking Locking")
plt.plot(x, output["pd_trans"], label="trans", color="blue")
plt.plot(x, output["pd_ref"], label="refl", color="black")
plt.legend()
plt.xlabel("m2.phi")
plt.ylabel("Power")
plt.plot()
plt.show()
