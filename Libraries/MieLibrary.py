# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:24:35 2016

@author: mhayman
"""

import numpy as np
import scipy.special as spf

"""
Calculate the total scattering matrix components of a spherical particle
Mie_PhaseMatrix(m, x, ang)
m -index of refraction
x -size parameter 2*pi*r/lambda
ang - scattering angle in radians

Collected Power are given by
Pc = A/(k^2*r^2)*F(Theta,Phi)*I0
where
    A - collection aperture
    k = 2pi/lambda
    r - particle radius
    I0 - incident stokes vector in intensity [W/m^2]
    
Scattering Cross Section in a solid angle is given by:
sig_s(theta) = 1/k^2*2*pi*Integral(F11.*sin(theta)dtheta)
where the integral is evaluated from 0 to the maximum scattering angle.

"""
def Mie_PhaseMatrix(m, x, ang):
    nc = np.ceil(x+4.05*(x**(1.0/3))+2).astype(np.int);
    n=np.arange(1,nc+1)  # (1:nc)';
    E = (2.0*n+1.0)/(n*(n+1.0))
    p,t = ALegendr(ang,nc);
    a,b = ScatCoef(m,x,nc);
    a = a*E
    b = b*E
    S1 = np.sum(a[:,np.newaxis]*p,axis=0) + np.sum(b[:,np.newaxis]*t,axis=0);
    S2 = np.sum(a[:,np.newaxis]*t,axis=0) + np.sum(b[:,np.newaxis]*p,axis=0);
    S11 = ((S2*np.conj(S2))+(S1*np.conj(S1)))/2;
    S12 = ((S2*np.conj(S2))-(S1*np.conj(S1)))/2;
    S33 = ((S1*np.conj(S2))+(S2*np.conj(S1)))/2;
    S34 = 1j*((S1*np.conj(S2))-(S2*np.conj(S1)))/2;
    F = np.vstack((S11, S12, S33, S34));
    return F
    # F = [S11 S12 0 0; S12 S11 0 0; 0 0 S33 S34; 0 0 -S34 S33];
    
"""
Calculate the total scattering amplitude components of a spherical particle
Mie_PhaseMatrix(m, x, ang)
m -index of refraction
x -size parameter 2*pi*r/lambda
ang - scattering angle in radians - accepts arrays

S1 is the horizontal polarization coefficient
S2 is the vertical polarization coefficient

Collected Power are given by
Pc = A/(k^2*r^2)*F(Theta,Phi)*I0
where
    A - collection aperture
    k = 2pi/lambda
    r - particle radius
    I0 - incident stokes vector in intensity [W/m^2]
    
Scattering Cross Section in a solid angle is given by:
sig_s(theta) = 1/k^2*2*pi*Integral(F11.*sin(theta)dtheta)
where the integral is evaluated from 0 to the maximum scattering angle.

"""
def Mie_AmplitudeMatrix(m, x, ang):
    nc = np.int(np.ceil(x+4.05*(x**(1.0/3))+2));
    n=np.arange(1,nc+1)  # (1:nc)';
    E = (2.0*n+1.0)/(n*(n+1.0))
    p,t = ALegendr(ang,nc);
    a,b = ScatCoef(m,x,nc);
    a = a*E
    b = b*E
    S1 = np.sum(a[:,np.newaxis]*p,axis=0) + np.sum(b[:,np.newaxis]*t,axis=0);
    S2 = np.sum(a[:,np.newaxis]*t,axis=0) + np.sum(b[:,np.newaxis]*p,axis=0);
    F = np.vstack((S1, S2));
    return F
    # F = [S11 S12 0 0; S12 S11 0 0; 0 0 S33 S34; 0 0 -S34 S33];    
    
"""
Calculate the scattering and extinction efficiency of a spherical particle
Mie_PhaseMatrix(m, x, ang)
m -index of refraction
x -size parameter 2*pi*r/lambda

Returns:
    eta_s,eta_e (scattering efficiency, extinction efficiency)
    which convert to cross sections by multiplying by particle area (pi*r^2)

"""    
def CrossSections(m,x):
    nc = np.int(np.ceil(x+4.05*(x**(1.0/3))+2));
    a,b = ScatCoef(m,x,nc);
    nvec = np.arange(1,np.size(a)+1);
    eta_s = 2/x**2*np.sum((2*nvec+1)*(a*np.conj(a)+b*np.conj(b)));
    eta_e = 2/x**2*np.sum((2*nvec+1)*np.real(a+b));
    return eta_s,eta_e

"""
Calculates the scattering matrix for a given range of collection angles.
For actual collected power, the collection aperture area (A), range (r)
and wavelength are needed.
I0 is the incident Stokes vector in intensity.

Pc = A/(k^2*r^2)*FscaT*I0

"""
def Mie_PhaseMatrix_Forward(m,x,MaxAng,MinAng=0.0,points=100):
    theta = np.linspace(MinAng,MaxAng,points)
    dtheta = np.abs(np.mean(np.diff(theta)))
    Fsca = Mie_PhaseMatrix(m,x,theta)
    FscaT = 2*np.pi*np.sum(Fsca*np.sin(theta+dtheta/2),axis=1)[:,np.newaxis]*dtheta
    
    return FscaT

"""
Calculate Legender polynomial coefficient expansions for VWSM
"""
def ALegendr(ang, nmax):

    p = np.ones((nmax,np.size(ang)))
    t = np.zeros((nmax,np.size(ang)))
    t[0,:] = np.cos(ang);
    p[1,:] = 3*np.cos(ang);
    t[1,:] = 2*np.cos(ang)*p[1,:]-3;
    for n in range(3,nmax+1):
        p[n-1,:] = ((2*n-1)*np.cos(ang)*p[n-2,:] - n*p[n-3,:])/(n-1);
        t[n-1,:] = n*np.cos(ang)*p[n-1,:] - (n+1)*p[n-2,:];
    return p,t

#for n=3:nmax
#	p(n,:) = ((2*n-1)*cos(ang).*p(n-1,:) - n*p(n-2,:))/(n-1);
#	t(n,:) = n*cos(ang).*p(n,:) - (n+1)*p(n-1,:);

"""
Calculate VWSM coefficients
"""
def ScatCoef(m,x,nmax):    
#function [a,b] = ScatCoef(m,x,nmax)
    N = np.arange(1,nmax+1)
    phi = RB1(x, nmax);
    phim = RB1(m*x, nmax);
    zeta = RB2(x, nmax);
    xi = phi + 1j * zeta;
    phin_1 = np.concatenate(([np.sin(x)],phi[0:nmax-1]));
    phimn_1 = np.concatenate(([np.sin(m*x)],phim[0:nmax-1]));
    zetan_1 = np.concatenate(([-np.cos(x)],zeta[0:nmax-1]));
    dphi = phin_1-N*phi/x;
    dphim = phimn_1-N*phim/(m*x);
    dzeta = zetan_1-N*zeta/x;
    dxi = dphi + 1j * dzeta;
    a = (m*phim*dphi - phi*dphim) / (m*phim*dxi - xi*dphim);
    b = (phim*dphi - m*phi*dphim) / (phim*dxi - m*xi*dphim);
    return a,b
    
def RB1(rho, nmax):
#function phi = RB1(rho, nmax)
    nst = np.int(np.ceil(nmax + np.sqrt(101+np.real(rho))));
    phi = np.zeros(nst)
    #phi[-1] = 0;
    phi[-2] = 1e-10;    
    for n in range(nst-2,0,-1):
    	phi[n-1] = (2*n+3)*phi[n]/rho - phi[n+1];

    phi0 = 3*phi[0]/rho - phi[1];
    phi0 = np.sin(rho)/phi0;
    phi = phi[0:nmax] * phi0;
    return phi

def RB2(rho, nmax):
    #function zeta = RB2(rho, nmax)
    zeta = np.zeros(nmax)
    zeta[0] = -np.cos(rho)/rho - np.sin(rho);
    zeta[1] = 3*zeta[0]/rho + np.cos(rho);
    for n in range(3,nmax):
        zeta[n-1] = (2*n-1)*zeta[n-2]/rho - zeta[n-3];
    return zeta
    
    
#def sph_hn2(n,z):
#    h2 = spf.sph_jn(n,z)-1j*spf.sph_yn(n,z)
#    return h2
#    
#    
#def VSM_M(theta,parity,n):
#    phi = 0;  # use 
#    if parity == 'e':
#        Me1nTheta=-1/np.sin(theta)*sph_hn2(k*r)*spf.sph_harm(1,n,0,theta)  
#        Me1nPhi= -sph_hn2(k*r)*spf.sph_harm(1,n,0,theta)  
#    elif parity == 'o':
#        
#    else:
#        print "Error in VSM_M:  Unrecognized parity.  Pass a char e or o only'
    