

<h3 align="center">FABRY-PÉROT CAVITY SIMULATION</h3>

<p align="center"> documentation <br></p>

# final_cavity.py <a name = "final_cavity"></a>

Script to simulate Pound-Drever-Hall Locking for a Fabry-Pérot cavity.

We are using two mirrors -
```
mirror 1 : Reflectivity = 0.85, Radius = 1.5 m
mirror 2 : Reflectivity = 0.85, Radius = 1.5 m
```
The cavity length is <b>1m</b> 

Some cavity parameters - 
```
FSR = 149896229.0
Finesse = 23.437748716808976
Linewidth (FWHM) = 6395504.568768506
```
First we vary m2.phi to obtain resonance peaks without modulation - <br><br>
![without_mod](/plots/withoutmod.png)

Now we add a modulator before the first mirror with a modulation frequency of around <b>50 MHz</b>

![with_mod](/plots/withmod.png)

This adds two sidebands to the carrier wave. 
- <b>NOTE</b>: To get proper locking, ensure that the sidebands lie <b>outside</b> the FWHM of the carrier wave.

To get the error signal, we measure the demodulated power that is reflected from mirror 1. Plotted for different demodulation phases

![error](/plots/error.png)

For locking, we choose the phase that gives the highest gradient in the error signal at the occurence of a peak. This is a plot of the gradients v/s the phase angles. For this configuration, the phase comes out to be <b>3 degrees</b>.

![gradients](/plots/gradients.png)

Now, we lock m1.phi (can lock any parameter), and to check locking, we plot transmitted and reflected intensities.
The lock maintains resonance, hence transmitted power is maximum and reflected power minimum.

![lock_check](/plots/lockcheck.png)

# mi.py <a name = "mi"></a>
```
TODO: integrate this cavity into a michelson interferometer
```