# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 11:20:30 2017

@author: mhayman
"""

import numpy as np
import MieLibrary as mie
import matplotlib.pyplot as plt

"""
Calculate the scattered angular distribution of a particle
"""
wavelen = 780e-9  # incident/scattered wavelength
#x_part = 10**np.arange(-9,-5,0.01)*np.pi*2/780e-9
x_part = 1e-6*np.pi*2/wavelen
ang = np.linspace(0,np.pi,180)
#Fbs = mie.Mie_PhaseMatrix(1.5, x_part, np.pi)
Fbs = mie.Mie_PhaseMatrix(1.5, x_part, ang)

# plot angular dependence of scattering cross section
plt.figure()
plt.semilogy(ang*180/np.pi,Fbs[0,:]/(2*np.pi/wavelen)**2)
plt.grid(b=True)
plt.xlabel('Scattering Angle [deg.]')
plt.ylabel('Scattering Cross-Section [$m^{-1} sr^{-1}$]')


"""
# Calculate backscatter cross section of different sized particles
"""
n_part = 1.3  # index of refraction
x_part = np.logspace(-9,-5,1000)*np.pi*2 # modified scattering size parameter (2*pi*d) - wavelength not included
sigBS_780 = np.zeros(x_part.size)
sigBS_532 = np.zeros(x_part.size)
sigBS_1560 = np.zeros(x_part.size)
for ai, x in zip(np.arange(x_part.size),x_part):
    sigBS_780[ai] = np.abs(mie.Mie_PhaseMatrix(1.3, x/780e-9, np.pi)[0][0])/(2*np.pi/780e-9)**2
    sigBS_532[ai] = np.abs(mie.Mie_PhaseMatrix(1.3, x/532e-9, np.pi)[0][0])/(2*np.pi/532e-9)**2
    sigBS_1560[ai] = np.abs(mie.Mie_PhaseMatrix(1.3, x/1560e-9, np.pi)[0][0])/(2*np.pi/1560e-9)**2

# plot backscatter cross sections
plt.figure()
plt.loglog(x_part/np.pi,sigBS_780,'r',label='$\lambda$=780 nm')
plt.loglog(x_part/np.pi,sigBS_532,'g',label='$\lambda$=532 nm')
plt.loglog(x_part/np.pi,sigBS_1560,'k',label='$\lambda$=1560 nm')
plt.grid(b=True)
plt.xlabel('Particle Diameter [m]')
plt.ylabel('Backscatter Cross-Section [$m^{-1} sr^{-1}$]')


# plot difference in backscatter cross section between 780 and 532 nm
plt.figure()
plt.semilogx(x_part/np.pi,sigBS_532/sigBS_780,'r')
plt.grid(b=True)