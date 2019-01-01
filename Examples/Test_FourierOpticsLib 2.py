# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 12:54:07 2016

@author: mhayman
"""

import FourierOpticsLib as FO
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


"""
List of important test cases:
Lens Propagation through FT and Phase mask
Wave propagation in medium where n != 1

Does direction work correctly?
    Check agains gaussian beams
    different masks
"""
plt.rcParams['mathtext.fontset']="stix"


#lam = 830e-9;
lam = 828.199e-9

#grid0 = FO.Coordinate_Grid((6e-3,30e-6))
grid0 = FO.Coordinate_Grid((1e-3,10e-6))

g1D = FO.Coordinate_Grid((1e-2,100e-6),yset=0)

##PW1D = FO.Efield(lam,g1D)
#PW1 = FO.Efield(lam,grid0)
#
#part1 = FO.SphericalParticle(30e-6,1.3)
#
#Fpart1 = part1.freq_mask(grid0,lam);
#
#Spart1 = part1.space_mask(grid0,lam);
#
##PW1.mask(Spart1[0]+1)
#
#PW1.mask(FO.Window(grid0.r,100e-6))
#print('Initial Power: %f' %np.sum(np.abs(PW1.field)**2));
#
#PW2 = PW1.copy();
#
#PW1.propagate_Fresnel(0.1,trim=True)
#PW2.propagate(0.1)
#
#print('Final Power Fresnel: %f' %np.sum(np.abs(PW1.field)**2));
#print('Final Power FFT: %f' %np.sum(np.abs(PW2.field)**2));
#
#PW1.plot(fignum=2);
#PW2.plot(fignum=2)
#
#PW2.imshow(savefile='/h/eol/mhayman/PythonScripts/Optics/BeamPattern_Test.png')

#plt.figure();
#plt.contourf(grid0.fx*lam,grid0.fy*lam,np.log(np.abs(Fpart1[0])))

#Ap1 = FO.CircFunc(grid0,0.5e-3,invert=True)

#Filter = FO.FP_Etalon(0.01e-9,0.5e-9,lam,InWavelength=True)
#Filter = FO.FP_Etalon(10e9,500e9,FO.c/lam)

#Ax1 = FO.Axicon(5.0*np.pi/180,n=1.45,z=0.0)
#Ax2 = Ax1.copy();
#Ax2.z = Ax2.z+10e-2
#
#L1 = FO.ThinLens(200e-3,z=200e-3)


#PW1D.mask(FO.Window(g1D.x,0.1e-2))

#PW1.mask(FO.Window(grid0.x,0.2e-3))
#PW1.mask(FO.Window(grid0.y,0.7e-3))
#
#PW2= PW1.copy();
#
#PW2.propagate_Fresnel(100e-3)
#PW1.propagate(20e-3)

#L1.propagate(PW1D)

#PW1.mask(Ap1)
#PW2 = PW1.copy()

#GB2 = FO.GaussianBeam(grid0,0.4e-3/2,lam)

#Ax1.propagate(GB2)
#GB2.propagate(1e-2)
#Ax2.propagate(GB2)
#GB2.mask(Ap1)

#GB2.contourf()5e

GB1 = FO.GaussianBeam(grid0,[0.1e-3,0.4e-3],lam,divergence=[0.04,0.01],Norder=4)
for ai in range(4):
    GB1 = FO.GaussianBeam(grid0,[0.1e-3,0.4e-3],lam,Norder=(ai+1))
    #GB1.imshow();
    GB1.plot(fignum=1)
    GB1.plot(axis='fx',fignum=2)

GB1.imshow(coord='angle')

GB1 = FO.GaussianBeam(grid0,[0.1e-3/4.0,0.4e-3/4.0],lam,Norder=4)
#GB1_BS = FO.backscatter(GB1,AngleLim=0.1)
#GB1_BS.spatial_filter(FO.Window(GB1_BS.grid.fr*GB1_BS.wavelength,0.06))
#GB1_BS.imshow(coord='angle')


EtalonFWHM = 2.5e-12;
EtalonFSR = 0.1e-9;
Etalon_center_freq = 828.2e-9
zEtalon = 2e-2;

# define FP etalon filter
Filter1 = FO.FP_Etalon(EtalonFWHM,EtalonFSR,Etalon_center_freq,efficiency=1.0,tilt=[0,0],z=zEtalon,InWavelength=True);

GB1.imshow();
GB1.imshow(coord='angle')
print('%f'%GB1.power())
Filter1.propagate(GB1)
GB1.imshow()
GB1.imshow(coord='angle')
print('%f'%GB1.power())

#GB1.imshow()
#GB1_BS.imshow()
#for ai in range(5):
#    GB1_BS.propagate(0.001)
#    GB1_BS.imshow()

#FO.AnimatePropagation(GB2,distance=10e-2,Num=30)

#PW1.propagate(3e-2)
#L1.propagate_FT(PW1)
#PW1.propagate_to(60e-2)



#L1.propagate(PW2)
#FO.AnimatePropagation(PW2,distance=200e-3)

#plt.figure();
#plt.pcolor(np.angle(L1.mask(grid0,lam)))
#plt.show();

#PW2.propagate_to(100e-2)

#plt.figure(); 
#plt.contourf(PW1.grid.x,PW1.grid.y,np.abs(PW1.field)**2);
#
#plt.figure(); 
#plt.contourf(PW2.grid.x,PW2.grid.y,np.abs(PW2.field)**2);
