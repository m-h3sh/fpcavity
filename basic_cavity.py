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
print(cav.g)

# First we plot the modulated signal (reflected and transmitted) v/s m2.phi
xaxis = Xaxis(m2.phi, 'lin', -100, 0, 400)

mod.f = 10e6
output = model.run(xaxis)
plt.figure("Without Modulation")
plt.plot(output["pd_trans"], label="Transmitted Power", color="blue")
plt.legend()
plt.plot()
plt.xlabel("m2.phi")

mod.f = 30e6
output = model.run(xaxis)

print(output)
plt.figure("With Modulation")
plt.plot(output["pd_trans"], label="Transmitted Power", color="blue")
plt.plot(output["pd_ref"], label="Reflected Power", color="black")
plt.legend()
plt.plot()
plt.xlabel("m2.phi")
# plt.yscale('log')


# Demodulating the reflected signal

outputs = []
model.remove(pd_ref)
for phase in np.linspace(0, 90, endpoint=True, num=6):
    pd = PowerDetectorDemod1("pd", m1.p1.o, mod.f, phase)
    model.add(pd)
    output = model.run(xaxis)
    outputs.append([output, phase])
    model.remove(pd)

plt.figure("Demodulated Reflected Power (Error Signal)")
for output in outputs:
    plt.plot(output[0]["pd"], label=f"{output[1]}")

plt.legend()
plt.plot()
plt.xlabel("m2.phi")
plt.show()
