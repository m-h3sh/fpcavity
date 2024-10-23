# Importing the libraries
import finesse
from finesse.components.laser import Laser
from finesse.components.gauss import Gauss
from finesse.components.mirror import Mirror
from finesse.components.space import Space
from finesse.components.beamsplitter import Beamsplitter
from finesse.components.cavity import Cavity
from finesse.analysis.actions.axes import Xaxis
from finesse.detectors.powerdetector import PowerDetector
import numpy as np
import matplotlib.pyplot as plt

# Defining variables
wavelength = 1064e-9
cav_len = 0.49999967 # resonant length for wavelength 1064 nm

# Defining the model
model = finesse.Model()
# model.lambda0 = wavelength

# Adding laser source
laser = Laser("source", 1.0, 2.8167763157e14)
model.add(laser)

# Adding Gaussian parameters to the beam
w0 = np.sqrt((3e8/laser.f) * ((93.15 + 26)*1e-3 / 2) / np.pi)
glaser = Gauss("glaser", laser.p1.o, w0=w0, z=0)
model.add(glaser)

# Defining the mirrors
mx1 = Mirror("mx1", R=0.99, L=0.0)
mx2 = Mirror("mx2", R=0.999, T=0.001)
my1 = Mirror("my1", R=0.99, L=0.0)
my2 = Mirror("my2", R=0.999, T=0.001)
model.add(mx1)
model.add(mx2)
model.add(my1)
model.add(my2)

# Adding 6 dof's to the mirrors
# In finesse the optical path is along z axis


# Defining the beamsplitter
bs = Beamsplitter("bs", 0.5, 0.5, alpha=45.0)
model.add(bs)

# Defining the detector
pd = PowerDetector("pd", bs.p4.o)
model.add(pd)

# Defining the spaces
model.add(Space("x0", laser.p1, bs.p1, L=cav_len/5))
model.add(Space("x1", bs.p3, mx1.p1, L=cav_len/5))
model.add(Space("x2", mx1.p2, mx2.p1, L=cav_len))
model.add(Space("y0", bs.p2, my1.p1, L=cav_len/5))
model.add(Space("y1", my1.p2, my2.p1, L=cav_len))

# Adding the cavities
model.add(Cavity("xcav", mx1.p2.o))
model.add(Cavity("ycav", my1.p2.o))

xaxis = Xaxis(mx1.phi, 'lin', -180 , 180, 400)
# xaxis = Xaxis(laser.f, 'lin', 2.44e14, 2.84e14, 300)
# xaxis = Xaxis(x2.L, 'lin', 0.5, 0.6, 200)
output = model.run(xaxis)

print(output)
plt.figure("Analyzing Output")
x = np.linspace(-180, 180, 401)
plt.plot(x, output["pd"], label="Output Power")
plt.xticks(np.arange(-180, 180, step=50))
plt.xlabel("angle (phi)")
plt.yscale('log')
plt.show()


# plt.plot(output["pd"], label="Output Power")
# plt.ylabel("Output power (W)")
# plt.yscale("log")
# plt.show()
