# Importing the libraries
import finesse
from finesse.components.laser import Laser
from finesse.components.gauss import Gauss
from finesse.components.mirror import Mirror
from finesse.components.space import Space
from finesse.components.beamsplitter import Beamsplitter
from finesse.components.cavity import Cavity
from finesse.analysis.actions.axes import Xaxis
import numpy as np
import matplotlib.pyplot as plt

# Defining variables
wavelength = 1064e-9
cav_length = 0.5

# Defining the model
model = finesse.Model()

# Add laser source
laser = Laser("source", 1.0, 2.82e14)
model.add(laser)

# TODO: add gaussian parameters to laser

# Defining the mirrors
mx1 = Mirror("mx1", R=0.99, T=0.01)
mx2 = Mirror("mx2", R=0.99, T=0.01)
my1 = Mirror("my1", R=0.99, T=0.01)
my2 = Mirror("my2", R=0.99, T=0.01)
model.add(mx1)
model.add(mx2)
model.add(my1)
model.add(my2)

# Defining the beamsplitter
bs = Beamsplitter("bs", 0.5, 0.5, alpha=45.0)
model.add(bs)

# Defining the detector

# Defining the spaces

print(model.components[len(model.components)-1].parameters)
