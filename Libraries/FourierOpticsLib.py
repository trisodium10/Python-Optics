# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:11:57 2016

@author: mhayman
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib
import time
import scipy.signal
import MieLibrary as mie
from scipy.ndimage import convolve

# Speed of light in a vaccuum
c = 299792458.0

"""
xset - spatial grid defining parameters dependent on the inputType
inputType - accepted strings:
    'spatial' form a grid based on max dimension and step size favoring more dimensions
        xset = (Xmax,step size)
        yset = (Ymax,step_size) : optional argument.  If not supplied uses xset.
    'frequency' form a grid based on max dimension and step size favoring more spatial frequencies
        xset = (Xmax,step_size)
        yset = (Ymax,step_size) : optional argument.  If not supplied uses xset.
    'ccd' form a grid based on passed CCD np.shape with given pixel size 
        xset = ([number_pixels_rows,number_pixels_columns],[pixel_width_row,pixel_width_col])
        note that the pixel dimension is setup to accept output from np.shape(CCD_ARRAY)
        this means that the y dimension is first (rows) and the x dimension is second
    'exact' form a grid that uses the exact size dimensions passed to the initialization
        do not rescale based on powers of 2.
        xset = (Xmax,step size)
        yset = (Ymax,step_size) : optional argument.  If not supplied uses xset.
    'array' form a grid that uses the input array for the spatial grid
        xset = 1D array of x values
        yset = 1D array of y values
yset - same as xset.  If not defined by user, it will be the same as xset.
        if user sets yset = 0, a 1D coordinate grid (number of y points = 1) will be created
Nmax - set an absolute maximum number of grid points along a dimension.  Should be set to a power of 2
        has no effect if inputType == 'ccd'
        
Work Needed:
    copy() needs to be corrected so the grid is passed identically to new grid
        right now 1D data is broken in the copy processes
"""

class Coordinate_Grid:
    def __init__(self,xset,yset=np.nan,inputType='frequency',Nmax=np.nan):
        self.dimensions = 2 # set number of dimensions to 2
        if inputType == 'spatial':
            Nx1D = np.nanmin([2**(np.ceil(np.log2(xset[0]/xset[1]))),Nmax/2])
            x1D = np.arange(-Nx1D,(Nx1D))*xset[1]
            dfx1D = 1.0/(2*Nx1D*np.double(xset[1]));
            fx1D = np.arange(-Nx1D,Nx1D)*dfx1D
            if yset == 0:
                Ny1D = 1
                y1D = np.array([0])
                dy1D = 1
                dfy1D = 1
                fy1D = np.array([0])
            elif not np.isnan(yset).any():
                Ny1D = np.nanmin([2**(np.ceil(np.log2(yset[0]/yset[1]))),Nmax/2])
                y1D = np.arange(-Ny1D,(Ny1D))*yset[1]
                dy1D = yset[0]
                dfy1D = 1.0/(2*Ny1D*np.double(yset[1]));
                fy1D = np.arange(-Ny1D,Ny1D)*dfy1D
            else:
                y1D = x1D
                dy1D = xset[0]
                dfy1D = dfx1D
                fy1D = fx1D
            self.x,self.y = np.meshgrid(x1D,y1D)
            self.fx,self.fy = np.meshgrid(fx1D,fy1D)
            self.dx = xset[1]
            self.dy = dy1D
            self.dfx = dfx1D
            self.dfy = dfy1D
            self.Nx = np.shape(self.x)[1]
            self.Ny = np.shape(self.x)[0]
            
            # radial coordinates
            self.r = np.sqrt(self.x**2+self.y**2)
            self.phi = np.arctan2(self.y,self.x)
            self.fr = np.sqrt(self.fx**2+self.fy**2)          
            self.fphi = np.arctan2(self.fy,self.fx)
        elif inputType == 'frequency':
            Nx1D = np.nanmin([2**(np.ceil(np.log2(xset[0]/xset[1]))),Nmax/2])
            dx1D = np.double(xset[0])/Nx1D
            x1D = np.arange(-Nx1D,(Nx1D))*dx1D
            dfx1D = 1.0/(2*Nx1D*dx1D);
            fx1D = np.arange(-Nx1D,Nx1D)*dfx1D
            if yset == 0:
                Ny1D = 1
                y1D = np.array([0])
                dy1D = 1
                dfy1D = 1
                fy1D = np.array([0])
            elif not np.isnan(yset).any():
                Ny1D = np.nanmin([2**(np.ceil(np.log2(yset[0]/yset[1]))),Nmax/2])
                dy1D = np.double(yset[0])/Ny1D
                y1D = np.arange(-Ny1D,(Ny1D))*dy1D
                dfy1D = 1.0/(2*Ny1D*dy1D);
                fy1D = np.arange(-Ny1D,Ny1D)*dfy1D
            else:
                y1D = x1D
                dy1D = dx1D
                dfy1D = dfx1D
                fy1D = fx1D
            self.x,self.y = np.meshgrid(x1D,y1D)
            self.fx,self.fy = np.meshgrid(fx1D,fy1D)
            self.dx = dx1D
            self.dy = dy1D
            self.dfx = dfx1D
            self.dfy = dfy1D
            self.Nx = np.shape(self.x)[1]
            self.Ny = np.shape(self.x)[0]
            
            # radial coordinates
            self.r = np.sqrt(self.x**2+self.y**2)
            self.phi = np.arctan2(self.y,self.x)
            self.fr = np.sqrt(self.fx**2+self.fy**2)          
            self.fphi = np.arctan2(self.fy,self.fx)
        elif inputType == 'ccd':
            Nx = xset[0][0]
            Ny = xset[0][1]
            
            x1D = (np.arange(Nx)-np.floor(Nx/2.0))*xset[1][1]
#            x1D = np.arange(-np.ceil((Nx-1)/2),np.floor((Nx-1)/2)+1)*xset[1][1];
            dfx1D =  1.0/(Nx*np.double(xset[1][1]))
            fx1D = (np.arange(Nx)-np.floor(Nx/2.0))*dfx1D
#            fx1D = np.arange(-np.ceil((Nx-1)/2),np.floor((Nx-1)/2))*dfx1D
            
            y1D = (np.arange(Ny)-np.floor(Ny/2.0))*xset[1][0]
#            y1D = np.arange(-np.ceil((Ny-1)/2),np.floor((Ny-1)/2)+1)*xset[1][0];
            dfy1D =  1.0/(Ny*np.double(xset[1][0]))
            fy1D = (np.arange(Ny)-np.floor(Ny/2.0))*dfy1D
#            fy1D = np.arange(-np.ceil((Ny-1)/2),np.floor((Ny-1)/2))*dfy1D
            
            self.x,self.y = np.meshgrid(x1D,y1D)
            self.fx,self.fy = np.meshgrid(fx1D,fy1D)
            self.dx = xset[1][1]
            self.dy = xset[1][0]
            self.dfx = dfx1D
            self.dfy = dfy1D
            self.Nx = Nx
            self.Ny = Ny
            
            # radial coordinates
            self.r = np.sqrt(self.x**2+self.y**2)
            self.phi = np.arctan2(self.y,self.x)
            self.fr = np.sqrt(self.fx**2+self.fy**2)          
            self.fphi = np.arctan2(self.fy,self.fx)
        elif inputType == 'exact':
            Nx1D = np.nanmin([np.round(xset[0]/xset[1]),Nmax/2])
            x1D = np.arange(-Nx1D,(Nx1D))*xset[1]
            dfx1D = 1.0/(2*Nx1D*xset[1]);
            fx1D = np.arange(-Nx1D,Nx1D)*dfx1D
            if yset == 0:
                Ny1D = 1
                y1D = np.array([0])
                dy1D = 1
                dfy1D = 1
                fy1D = np.array([0])
            elif not np.isnan(yset).any():
                Ny1D = np.nanmin([np.round(yset[0]/yset[1]),Nmax/2])
                y1D = np.arange(-Ny1D,(Ny1D))*yset[1]
                dy1D = yset[1]
                dfy1D = 1.0/(2*Ny1D*yset[1]);
                fy1D = np.arange(-Ny1D,Ny1D)*dfy1D
            else:
                y1D = x1D
                dy1D = xset[1]
                dfy1D = dfx1D
                fy1D = fx1D
            self.x,self.y = np.meshgrid(x1D,y1D)
            self.fx,self.fy = np.meshgrid(fx1D,fy1D)
            self.dx = xset[1]
            self.dy = dy1D
            self.dfx = dfx1D
            self.dfy = dfy1D
            self.Nx = np.shape(self.x)[1]
            self.Ny = np.shape(self.x)[0]
            
            # radial coordinates
            self.r = np.sqrt(self.x**2+self.y**2)
            self.phi = np.arctan2(self.y,self.x)
            self.fr = np.sqrt(self.fx**2+self.fy**2)          
            self.fphi = np.arctan2(self.fy,self.fx)
#        elif inputType == 'array':
#            x1D = xset.copy()
#            self.dx = np.mean(np.diff(x1D))
#            self.Nx = np.size(xset)
#            self.dfx = 1.0/(self.Nx*self.dx)
#            fx1D = np.arange(self.Nx)-Nx/2.0
#            if not np.isnan(yset).any():
#                y1D = yset.copy()
#                self.dy = np.mean(np.diff(y1D))
#                self.Ny = np.size(yset)
#                self.dfy = 1.0/(self.Ny*self.dy)
                
        else:
            print ('Error: inputType in Coordinate_Grid initialization not recognized.')
            print ('    It accepts string arguments spatial, frequency, exact or ccd')
        
#        self.dx  = np.mean(np.diff(self.x))
#        self.dy = np.mean(np.diff(self.y))
#        self.dfx = np.mean(np.diff(self.fx))
#        self.dfy = np.mean(np.diff(self.fy))
        
    def rescale(self,factor):
        # rescale grid to maintain consistent spatial and frequency grids
        self.x = self.x*factor
        self.y = self.y*factor
        self.fx = self.fx/factor
        self.fy = self.fy/factor

        self.dx = self.dx*factor
        self.dy = self.dy*factor
        self.dfx = self.dfx/factor
        self.dfy = self.dfy/factor
        
        self.r = np.sqrt(self.x**2+self.y**2)
        self.phi = np.arctan2(self.y,self.x)
        self.fr = np.sqrt(self.fx**2+self.fy**2)          
        self.fphi = np.arctan2(self.fy,self.fx)
    def FTrescale(self,factor):
        temp = self.x
        self.x = self.fx*factor
        self.fx = temp/factor
        
        temp = self.y
        self.y = self.fy*factor
        self.fy = temp/factor
        
        temp = self.dx
        self.dx = self.dfx*factor
        self.dfx = temp/factor
        
        temp = self.dy        
        self.dy = self.dfy*factor
        self.dfy = temp/factor
        
        self.r = np.sqrt(self.x**2+self.y**2)
        self.phi = np.arctan2(self.y,self.x)
        self.fr = np.sqrt(self.fx**2+self.fy**2)          
        self.fphi = np.arctan2(self.fy,self.fx)
    def copy(self):
        if self.Ny == 1:
            NewYset = 0;
        else:
            NewYset = (self.Ny*self.dy/2.0,self.dy)
#            NewYset = (self.y[-1,-1],self.dy)
        if self.Nx == 1:
            NewXset = 0;
        else:
            NewXset = (self.Nx*self.dx/2.0,self.dx)
#            NewXset = (self.x[-1,-1],self.dx)
        NewGrid = Coordinate_Grid(NewXset,yset=NewYset,inputType='exact');
        return NewGrid
"""
Electric field class with physical definitions
wavelength - (in m) wavelength in a vaccuum
n - index of refraction in current medium
z - position along the optic axis
x - grid of x transverse coordinates
y - grid of y transverse coordinates
fx - grid of frequency coordinates in x
fy - grid of frequency coordinates in y
direction - set to 1 (default) for forward propagation
                  -1 for backward propagation (backscattering or reflection)
                  
Work Needed:
    should have a command to return a 1-D stripe from the field along a single dimension
    1D plot command?
    pcolor plot command?
"""

class Efield:
    def __init__(self,wavelength,grid,z=0,n=1.0,direction=1,fielddef=np.nan):
        self.wavelength=wavelength
        self.n = n  # index of refraction of current medium
        self.z = z  # position along the optic axis
#        self.grid = grid.copy()  # grid of transverse coordinates of type Coordinate_Grid
#        self.x = x
#        self.y = y
#        self.fx = fx
#        self.fy = fy
        self.direction = direction
        
        if np.isnan(fielddef).all():
            self.field = np.ones([grid.Ny,grid.Nx])
            self.grid = grid.copy()  # grid of transverse coordinates of type Coordinate_Grid
        else:
            self.field = fielddef
            self.grid = Coordinate_Grid((fielddef.shape,(grid.dy,grid.dx)),inputType='ccd')
            # set x and y grid based on provided field definition
        
    def copy(self):
        # returns a copy of the current field
        Enew = Efield(self.wavelength,self.grid,z=self.z,n=self.n,direction=self.direction,fielddef=self.field)
        return Enew
    def propagate(self,distance):
        # propagate a distance along the current direction (positive forward, negative backward)
        # using FFT
        if np.abs(distance) > self.wavelength:
            Hcirc = Window(self.grid.fr,2.0/self.wavelength)
            H = Hcirc*np.exp(1j*2*np.pi*distance*self.direction*self.n/self.wavelength*np.sqrt(1-np.complex_((self.wavelength*self.grid.fr)**2)))
            self.field = OpticsIFFT(H*OpticsFFT(self.field))
            self.z = self.z+distance*self.direction
    def propagate_to(self,zpos):
        # propagate to a specific location on the optic axis
        self.propagate(self.direction*(zpos-self.z))
    def propagate_Fresnel(self,distance,trim=True):
        # propagate using Fresnel convolution
        # set trim=False if you want the grid to grow with propagation.  This requires rescaling the grid through and has not been written yet.
        print('Warning: Efield.propagate_Fresnel() is slow due to slow 2D convolution operation in scipy.signal.convolve2d()')
        Nxh = np.min([np.round(self.wavelength*self.grid.fx[-1,-1]*distance/self.grid.dx),self.grid.Nx/2])
        Nyh = np.min([np.round(self.wavelength*self.grid.fy[-1,-1]*distance/self.grid.dy),self.grid.Ny/2]);
        xh1D = np.arange(-Nxh,Nxh+1)*self.grid.dx
        yh1D = np.arange(-Nyh,Nyh+1)*self.grid.dy
        xh,yh = np.meshgrid(xh1D,yh1D)
#        gridh = Coordinate_Grid((2*self.wavelength*self.grid.fx[-1,-1]*distance,self.grid.dx),yset=(2*self.wavelength*self.grid.fy[-1,-1]*distance,self.grid.dy),inputType='exact')
        hxy = np.exp(1j*2*np.pi/self.wavelength*self.direction*distance)/(1j*self.wavelength*self.direction*distance) \
            *np.exp(1j*np.pi/(self.wavelength*self.direction*distance)*(xh**2+yh**2)) #np.sqrt(1.0*np.min([np.size(self.grid.x),np.size(xh)]))
        
#        ixh0 = np.nonzero(xh1D==0)[0][0]  # index to zero x coordinate in impulse response
#        iyh0 = np.nonzero(yh1D==0)[0][0]  # index to zero y coordinate in impulse response
#        ix0 = self.grid.Nx/2+ixh0  # index to zero x coordinate in resultant output
#        iy0 = self.grid.Ny/2+iyh0  # index to zero y coordinate in resultant output
#        Outfield = scipy.signal.convolve2d(self.field,hxy,mode='full')*self.grid.dx*self.grid.dy        
        if trim == True:
#            self.field = Outfield[(-self.grid.Ny/2+iy0):(self.grid.Ny/2+iy0),(-self.grid.Nx/2+ix0):(self.grid.Nx/2+ix0)]
#            Outfield = convolve(self.field,hxy,mode='constant',cval=0.0)  # use scipy.ndimage convolve library
            Outfield = scipy.signal.convolve2d(self.field,hxy,mode='same')*self.grid.dx*self.grid.dy       
            self.field = Outfield
        else:
#            ixh0 = np.nonzero(xh1D==0)[0][0]  # index to zero x coordinate in impulse response
#            iyh0 = np.nonzero(yh1D==0)[0][0]  # index to zero y coordinate in impulse response
#            ix0 = self.grid.Nx/2+ixh0  # index to zero x coordinate in resultant output
#            iy0 = self.grid.Ny/2+iyh0  # index to zero y coordinate in resultant output
            Lx = np.size(xh1D)+self.grid.Nx-1  # size of x grid in post-convolution output
            Ly = np.size(yh1D)+self.grid.Ny-1  # size of y grid in post-convolution output
            if np.mod(Lx,2) != 0:
                Lx  =Lx-1
                if np.mod(Ly,2) != 0:
                    Outfield = (scipy.signal.convolve2d(self.field,hxy,mode='full')*self.grid.dx*self.grid.dy)[1:,1:]
                    Ly = Ly-1
                else:
                    Outfield = (scipy.signal.convolve2d(self.field,hxy,mode='full')*self.grid.dx*self.grid.dy)[:,1:]
            else:
                if np.mod(Ly,2) != 0:
                    Outfield = (scipy.signal.convolve2d(self.field,hxy,mode='full')*self.grid.dx*self.grid.dy)[1:,:]
                    Ly = Ly-1
                else:
                    Outfield = (scipy.signal.convolve2d(self.field,hxy,mode='full')*self.grid.dx*self.grid.dy)
#            Outfield = scipy.signal.convolve2d(self.field,hxy,mode='full')*self.grid.dx*self.grid.dy 
            NewYset = (Ly*self.grid.dy/2.0,self.grid.dy)
            NewXset = (Lx*self.grid.dx/2.0,self.grid.dx)
            self.grid=Coordinate_Grid(NewXset,yset=NewYset,inputType='exact')
            self.field = Outfield
            print('trim option set to False, but this feature is not validated')
            
        # if trim =False, convolution mode should be 'full' (default) and the grid needs to be redefined using the 'exact' option
    def spatial_filter(self,filtermask):
        # apply a mask in the frequency domain
        self.field = OpticsIFFT(filtermask*OpticsFFT(self.field))
    def mask(self,mask):
        # apply a mask in the spatial domain
        self.field = self.field*mask
    def angular_spectrum(self):
        # returns angular spectrum of the field
        return OpticsFFT(self.field) #,self.grid.fx*self.wavelength,self.grid.fy*self.wavelength
    def translate_FT(self,xshift,yshift):
        self.field = self.spatial_filter(np.exp(1j*2*np.pi*(xshift*self.grid.fx+yshift*self.grid.fy)))
        # use Fourier Transform to translate the field spatially
    def contourf(self,Clevels=np.nan,savefile=''):
        # plot the intensity of the electric field
        fig1 = plt.figure();
        Ifield = np.abs(self.field)**2
        if np.isnan(Clevels).any():
            plt.contourf(self.grid.x,self.grid.y,Ifield)
#            if np.median(Ifield)/np.mean(Ifield) < 0.01:
#                plt.contourf(self.grid.x,self.grid.y,np.log10(Ifield))
#                plt.title('$Log_{10}$ Intensity')
#            else:
#                plt.contourf(self.grid.x,self.grid.y,Ifield)
#                plt.title('Intensity')
        else:
            plt.contourf(self.grid.x,self.grid.y,Ifield,levels=Clevels)
        plt.xlabel('x');
        plt.ylabel('y');
        plt.axis('equal')
        if len(savefile) > 0:
            plt.savefig(savefile)
        else:
            plt.show();
        return fig1
    def imshow(self,coord='spatial',savefile='',title='',scale='linear',colorbar=True):
        # plot the intensity on an implot
        if coord=='spatial':
            # spatial intensity
            fig1 = plt.figure();
            Ifield = np.abs(self.field)**2
            if scale == 'log':
                plt.imshow(Ifield[::-1,:],extent=(self.grid.x.min(),self.grid.x.max(),self.grid.y.min(),self.grid.y.max()),norm=matplotlib.colors.LogNorm())
            else:
                plt.imshow(Ifield[::-1,:],extent=(self.grid.x.min(),self.grid.x.max(),self.grid.y.min(),self.grid.y.max()))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(title)
        else:
            # plot frequency spectrum
            fig1 = plt.figure();
            Ifield = np.abs(self.angular_spectrum())
            if coord=='angle':
#                plt.rcParams['mathtext.fontset']="stix"
                if scale == 'log':
                    plt.imshow(Ifield[::-1,:],extent=(self.grid.fx.min()*self.wavelength,self.grid.fx.max()*self.wavelength,self.grid.fy.min()*self.wavelength,self.grid.fy.max()*self.wavelength),norm=matplotlib.colors.LogNorm())
                else:    
                    plt.imshow(Ifield[::-1,:],extent=(self.grid.fx.min()*self.wavelength,self.grid.fx.max()*self.wavelength,self.grid.fy.min()*self.wavelength,self.grid.fy.max()*self.wavelength))
                plt.xlabel(r'$\alpha_x$')
                plt.ylabel(r'$\alpha_y$')
            else:
                if scale == 'log':
                    plt.imshow(Ifield[::-1,:],extent=(self.grid.fx.min(),self.grid.fx.max(),self.grid.fy.min(),self.grid.fy.max()),norm=matplotlib.colors.LogNorm())
                else:
                    plt.imshow(Ifield[::-1,:],extent=(self.grid.fx.min(),self.grid.fx.max(),self.grid.fy.min(),self.grid.fy.max()))
                plt.xlabel('$f_x$')
                plt.ylabel('$f_y$')            
            plt.title(title)
        
        if colorbar:
            plt.colorbar()

        if len(savefile) > 0:
            plt.savefig(savefile)
        else:
            plt.show();
        
            
        return fig1
    
    def line(self,axis='x',position=0):
        # generate a line section of the electric field
        # default returns section along x at y = 0
        # accepts axis = 0 (y), 1 (x), 'x','y','fx','fy'
        if axis == 1 or axis == 'x':
            iposition = np.argmin(np.abs(self.grid.y[:,0]-position))
            return self.field[iposition,:]
        elif axis == 0 or axis == 'y':
            iposition = np.argmin(np.abs(self.grid.x[0,:]-position))
            return self.field[:,iposition]
        elif axis == 'fx':
            iposition = np.argmin(np.abs(self.grid.fy[:,0]-position))
            return OpticsFFT(self.field)[iposition,:]
        elif axis == 'fy':
            iposition = np.argmin(np.abs(self.grid.fx[0,:]-position))
            return OpticsFFT(self.field)[:,iposition]
        else:
            print('Efield.line:  axis argument not recognized')
    def plot(self,axis='x',position=0,fignum=np.nan,scale='linear'):
        if axis == 1 or axis == 'x':
            pltaxis = self.grid.x[0,:]
            axistxt = 'x'
        elif axis == 0 or axis == 'y':
            pltaxis = self.grid.y[:,0]
            axistxt = 'y'
        elif axis == 'fx':
            pltaxis = self.grid.fx[0,:]
            axistxt = '$f_x$'
        elif axis == 'fy':
            pltaxis = self.grid.fy[:,0]
            axistxt = '$f_y$'
        else:
            print('Efield.plot:  unrecognized axis argument')
            
        Iplt = np.abs(self.line(axis=axis,position=position).flatten())**2
        if np.isnan(fignum):
            fignum = plt.figure()
        else:
            plt.figure(fignum)
        plt.plot(pltaxis,Iplt)
        plt.xlabel(axistxt)
        plt.ylabel('Intensity')
        plt.yscale(scale)
        return fignum
    def power(self):
        # returns the total power in the wave.  The returned power is consistent
        # between spatial and frequency coodinate frames.
        TotalPower = np.sum(np.abs(self.field)**2)
        return TotalPower
"""
Propagate an Electric field through a thin lens
Ein - input electric field
f   - lens focal length
FFz - (optional)
    - True use Fourier Transform method (default)
    - False multiply by complex phase
z   - give the z location of the lens.  If defined, propagation will 
        automatically be applied to perform operations in the correct plane
outputs:
Eout - resulting electric field
"""

"""
Thin lens object - FT or propagate an electric field with the object
Definition:
    ThinLens(f,radius=np.nan,FFL=np.nan,BFL=np.nan,z=np.nan,dFPP=np.nan,dBPP=np.nan)
    Inputs (Required):
        f - focal length of lens
    Inputs (Optional):
        radius - apperture radius of the lens (otherwise it is assumed infinite)
        z - z position of the lens
        FFL - front focal length (absolute distance from z position of lens to front focal plane)
        BFL - back focal length (absolute distance from z position of lens to the back focal plane)
        dFPP - distance from the z position of the lens to the front principle plane
        dBPP - distance from the z position of the lens to the back principle plane
"""
class ThinLens:
    def __init__(self,f,radius=np.nan,FFL=np.nan,BFL=np.nan,z=np.nan,dFPP=np.nan,dBPP=np.nan):
        self.f=f
        self.radius = radius # radius of lens aperature (if it has one)
        self.z = z          # position of lens along optic axis (nan if undefined)
        if np.isnan(FFL):
            # if front focal length is not defined check if front principal plane is defined
            if np.isnan(dFPP):
                # if front principle plane is also not defined assume
                self.FFL = self.f   # front focal length is equal to the effective focal length
                self.dFPP = 0;      # the front principle plane is at the lens location
            else:
                # if front focal length is not defined but the front principle plane is
                # use the front PP definition to also obtain the front focal length
                self.FFL = self.f+dFPP
                self.dFPP = dFPP
        else:
            # if the front focal length is defined use it to define the front principle plane
            self.FFL = FFL  # distance from front focal plane to lens surface
            self.dFPP = FFL-self.f
        if np.isnan(BFL):
            if np.isnan(dBPP):
                self.BFL = self.f
                self.dBPP = 0
            else:
                self.BFL = self.f+dBPP
                self.dBPP = dBPP
        else:
            self.BFL = BFL  # distance from back focal plane to lens surface
            self.dBPP = BFL-self.f
            
    def propagate_FT(self,Ein,update=True):
        # uses Fourier Transform to propagate from front focal plane to back focal plane
    
        # check propagation direction to determine which focal plane and principal planes
        # to use as front and back.
        if Ein.direction < 0:
            FFz = -self.BFL
#            FPP = self.zBPP
#            BPP = self.zFPP
        else:
            FFz = self.FFL
#            FPP = self.zFPP
#            BPP = self.zBPP
        if update:
            if not np.isnan(self.z):
                if not np.isnan(self.radius):
                    # if the lens aperature is defined, first apply the aperture
                    # function of the lens before the FFT
                    Ein.propagate_to(self.z)
                    Ein.mask(Window(Ein.grid.r,2*self.radius))
                    
                # if the lens z position is defined, move the input wave to the
                # lens fourier plane first
                Ein.propagate_to(self.z-FFz)
            Ein.field = OpticsFFT(Ein.field);
            Ein.z = Ein.z+Ein.direction*(self.FFL+self.BFL)
            Ein.grid.FTrescale(Ein.wavelength*self.f)
        else:
            Enew = Ein.copy()
            if not np.isnan(self.z):
                if not np.isnan(self.radius):
                    # if the lens aperature is defined, first apply the aperture
                    # function of the lens before the FFT
                    Enew.propagate_to(self.z)
                    Enew.mask(Window(Enew.grid.r,2*self.radius))
                    
                # if the lens z position is defined, move the input wave to the
                # lens fourier plane first
                Enew.propagate_to(self.z-FFz)
            
            Enew.field = OpticsFFT(Enew.field)
            Enew.z = Enew.z+Enew.direction*(self.FFL+self.BFL)
            Enew.grid.FTrescale(Enew.wavelength*self.f)
            return Enew
            
#        Ein.x = Ein.grid.fx*Ein.wavelength*self.f
#        Ein.y = Ein.grid.fy*Ein.wavelength*self.f
#        Ein.fx = Ein.grid.x/(Ein.wavelength*self.f)
#        Ein.fy = Ein.grid.y/(Ein.wavelength*self.f)
        
    def propagate(self,Ein,update=True):
        # uses a phase mask to propagate from lens front surface to lens back surface
        if update:
            if not np.isnan(self.z):
                Ein.propagate_to(self.z)
            Ein.mask(self.mask(Ein.grid,Ein.wavelength))
        else:
            Enew = Ein.copy()
            if not np.isnan(self.z):
                Enew.propagate_to(self.z)
            Enew.mask(self.mask(Enew.grid,Enew.wavelength))
            return Enew
    def mask(self,grid,wavelength):
        if not np.isnan(self.radius):
            MaskOut = Window(grid.r,2*self.radius)*np.exp(-1j*np.pi*(grid.r**2)/(wavelength*self.f))
        else:
            MaskOut = np.exp(-1j*np.pi*(grid.r**2)/(wavelength*self.f))
        return MaskOut
        
"""
Define an Axicon optical element
required inputs:
    angle - angle of axicon cut in radians.  If index of refraction (n) is not provided,
        this is the divergence angle created by the axicon.  If it is provided
        it is the angle of the actual material.
optional inputs:
    n - index of refraction contrast.  For a freespace optic, this is the actual
        index of refraction.  However if immersed in something else, it is the ratio
        of the axicon index to the external medium.  e.g. a glass axicon in water
        would have and approximate n = 1.5/1.3
    radius - the aperature radius of the axicon.  Inifinite if not specified.
    z - the position of the axicon along the optic axis
"""             
class Axicon:
    def __init__(self,angle,n=np.nan,radius=np.nan,z=np.nan):
        self.angle=angle    # cut angle of axicon
        self.radius=radius  # aperture radius of axicon
        self.z = z          # z position of axicon
        self.n = n          # index contrast of axicon
        
    def propagate(self,Ein,update=True):
        if update:
            if not np.isnan(self.z):
                Ein.propagate_to(self.z)
            Ein.mask(self.mask(Ein.grid,Ein.wavelength,n_medium=Ein.n))
        else:
            Enew = Ein.copy()
            if not np.isnan(self.z):
                Enew.propagate_to(self.z)
            Enew.mask(self.mask(Enew.grid,Enew.wavelength,n_medium=Enew.n))
            return Enew
        
    def mask(self,grid,wavelength,n_medium=1.0):
        m_ax = self.slope(n_medium) # determine phase slope of axicon
        if np.isnan(self.radius):
            AxiconMask = np.exp(-1j*2*np.pi*grid.r*m_ax/wavelength)
        else:
            AxiconMask = Window(grid.r,2*self.radius)*np.exp(-1j*2*np.pi*grid.r*m_ax/wavelength)
        
        return AxiconMask
        
    def slope(self,n_medium):
        if np.isnan(self.n):
            AxiconSlope = np.tan(self.angle)
        else:
            AxiconSlope = np.tan(self.angle)*(self.n-n_medium)
        return AxiconSlope
    def copy(self):
        AxNew = Axicon(self.angle,n=self.n,radius=self.radius,z=self.z)
        return AxNew
""" 
Propagate an Electric field through a thick lens
Ein - input electric field
C1  - radius of curvature of the first surface
C2  - radius of curvature of the second surface
nL  - index of refraction of lens
nOut- (optional) index of refraction of exit medium (defaults to 1.0)
outputs:
Eout - resulting electric field
"""    
class ThickLens:
    def __init__(self,Curvature,nL):
        print ('ThickLens has not been written')
    
"""
A flat surface interface between two refractive indicies
Surface(self,n1,n2,tilt=[0,0],z=np.nan)
Inputs:
    n1 - front index of refraction
    n2 - back index of refraction
    tilt - slope in [mx,my] of surface
    z - position of surface
"""
class Surface:
    def __init__(self,n1,n2,tilt=[0,0],z=np.nan):
        self.n1 = n1
        self.n2 = n2
        self.z = z
        self.tiltx= tilt[0]
        self.tilty= tilt[1]
    def mask(self,grid,wavelength):
        return np.exp(1j*2*(self.n2-self.n1)*np.pi/wavelength*(self.tiltx*grid.x+self.tilty*grid.y))
    def propagate(self,Ein):
        if Ein.direction == 1:
            if Ein.n == self.n1:
                if not np.isnan(self.z):
                    Ein.propagate_to(self.z)
                Ein.mask(self.mask(Ein.grid,Ein.wavelength)) # apply phase shift due to tilt of surface
                Ein.n = self.n2     # Update index of refraction for Ein 
            else:
                print('Index mismatch between surface (n=%f) and input field (n=%f)' %(self.n1,Ein.n))
        else:
            if Ein.n == self.n2:
                if not np.isnan(self.z):
                    Ein.propagate_to(self.z)
                Ein.mask(np.conjugate(self.mask(Ein.grid,Ein.wavelength))) # apply phase shift due to tilt of surface
                Ein.n = self.n1      # Update index of refraction for Ein 
            else:
                print('Index mismatch between surface (n=%f) and input field (n=%f)' %(self.n2,Ein.n))
    def copy(self):
        Snew = Surface(self.n1,self.n2,tilt = [self.tiltx,self.tilty],z=self.z)
        return Snew
    def reverse(self):
        ntemp = self.n1
        self.n1 = self.n2
        self.n2 = ntemp
        
        
class SphericalParticle:
    def __init__(self,radius,n):
        self.radius=radius
        self.n = n
    def freq_mask(self,grid,wavelength,angle_offset=[0.0,0.0]):
        if angle_offset==[0,0]:
            scat_ang = grid.fr*wavelength
        else:
            scat_ang = np.sqrt((grid.fx*wavelength-angle_offset[0])**2 + \
                (grid.fy*wavelength-angle_offset[1])**2)
                
        Sp = mie.Mie_AmplitudeMatrix(self.n, 2*np.pi*self.radius/wavelength, scat_ang.flatten())
#        print('Sp is %s long with %s rows and 0 cols'%(np.size(Sp[0,:]),np.shape(Sp[0,:]))) #[0],np.shape(Sp[0,:])[1]))
        Sp1 = np.reshape(Sp[0,:],(grid.Ny,grid.Nx))
        Sp2 = np.reshape(Sp[1,:],(grid.Ny,grid.Nx))
        
        SpH = Sp1*np.cos(grid.fphi)**2+Sp2*np.sin(grid.fphi)**2
        SpV = Sp2*np.cos(grid.fphi)**2+Sp1*np.sin(grid.fphi)**2
        SpHV = (Sp1-Sp2)*np.cos(grid.fphi)*np.sin(grid.fphi)
        
        # Power normalization term for conversion to cartesian coordinates
        Snormalize = -1/(np.cos(grid.fx*wavelength)*np.cos(grid.fy*wavelength))*np.sqrt(grid.dfx*grid.dfy)*wavelength**2/(2*np.pi)    
        
        SpH = SpH*Snormalize
        SpV = SpV*Snormalize
        SpHV = SpHV*Snormalize
        
        return SpH,SpV,SpHV
        
    def space_mask(self,grid,wavelength,offset=[0.0,0.0],angle_offset=[0.0,0.0]):
        """
        % [Sh,Sv,Shv] = space_mask(grid,wavlength,offset=[0.0,0.0],angle_offset=[0.0,0.0])
        % Calculates a virtual complex transmission mask for a particle from Mie
        % theory.  The resulting data should not be assumed to be accurate near the
        % particle.  It should be used to produce a reliable angular spectrum from
        % the particle.  The output of this function provides polarization
        % information in a global coordinate frame such that
        % T = [Sh  Shv]
        %     [Shv Sv ]
        %
        % Interpreting Output:
        %  ExOut = Sh*ExIn+Shv*EyIn + ExIn
        %  EyOut = Shv*Ex+Sv*EyIn + EyIn
        % Note that the incident field needs to be added to the result Tpart*E
        %
        %  For unpolarized light detection add the intensities of the output
        %    Iout = |ExOut|^2 + |EyOut|^2
        
        """        
                
        Shf,Svf,Shvf = self.freq_mask(grid,wavelength,angle_offset=angle_offset)
        LinearPhaseOffset = np.exp(-1j*2*np.pi*(offset[0]*grid.fx+offset[1]*grid.fy))
        Shf = Shf*LinearPhaseOffset
        Svf = Svf*LinearPhaseOffset
        Shvf = Shvf*LinearPhaseOffset
        Sh = OpticsIFFT(Shf);
        Sv = OpticsIFFT(Svf);
        Shv = OpticsIFFT(Shvf);
        return Sh,Sv,Shv

    def scatter(self,Ein,offset=[0.0,0.0]):
        """
        Propagate horizontal polarized light
        """
        SpH,_,_ = self.space_mask(Ein.grid,Ein.wavelength,offset=offset)
        Ein.field = Ein.field*(SpH+1)
        

class ShearPlate:
    def __init__(self,n,wedge,thickness):
        print('ShearPlate still needs to be written')

"""
Detector converts an electric field at a position to a time domain signal.
Maybe that doesn't make any sense.
"""
class Detector:
    def __init__(self,radius=np.nan,active_area=np.nan):
        print('Detector has not been written')        

"""
FP_Etalon:
Fabret-Perot Etalon with
FWHM - transmission FWHM
FSR - Free spectral range (mode spacing)
center_frequency - design frequency of etalon
efficiency - peak transmission efficiency (defaults to 1.0)
tilt - offset tilt angle relative to incident beam (defaults to [0,0]=[xtilt,ytilt])
radius - apperture function of etalon
nE - etalon index of refraction (defaults to 1.5)
InWavelength - if true, all specifications are in vaccuum wavelength, otherwise in frequency (defaults to False)
z - z position on the optic axis


"""   

class FP_Etalon:
    def __init__(self,FWHM,FSR,center_frequency,efficiency=1.0,tilt=[0,0],radius=np.nan,nE = 1.5,InWavelength=False,z=np.nan):
        
        if InWavelength:        
            self.FWHM = c/(center_frequency-FWHM/2)-c/(center_frequency+FWHM/2)
            self.FSR = c/(center_frequency)-c/(center_frequency+FSR)
            self.center_frequency=c/center_frequency
            self.center_wavelength = center_frequency
        else:
            self.FWHM = FWHM
            self.FSR = FSR
            self.center_frequency = center_frequency
            self.center_wavelength = c/center_frequency
        
        self.nE = nE
        self.order = np.round(self.center_frequency/self.FSR)
        self.FSR = self.center_frequency/self.order  # force center frequency
        self.finess = self.FSR/self.FWHM
        self.length = self.order*c/(2*self.center_frequency)/self.nE
        self.refl = ((np.pi/self.finess-np.sqrt(np.pi**2/self.finess**2+4))/-2)**2
#        self.finess_coeff = 1.0/np.sin(np.pi/(2*self.finess))
        self.radius = radius
        self.tilt = tilt
        self.efficiency = efficiency
        self.z = z

    def propagate(self,Ein,transmit=True,update=True):
        """
        propagate(Ein)
        propagates the electric field Ein through the etalon
        """
        if update:
            if not np.isnan(self.z):
                Ein.propagate_to(self.z)      
            
            Tmask = self.fourier_mask(Ein.grid,Ein.wavelength,transmit=transmit)
            Ein.spatial_filter(Tmask)
            
            if not np.isnan(self.radius):
                Etalon_Aperture = Window(Ein.grid.r,2*self.radius)
                Ein.mask(Etalon_Aperture)
        
        else:
            Enew = Ein.copy()
            if not np.isnan(self.z):
                Enew.propagate_to(self.z)
            
            Tmask = self.fourier_mask(Enew.grid,Enew.wavelength,transmit=transmit)
            Enew.spatial_filter(Tmask)
            
            if not np.isnan(self.radius):
                Etalon_Aperture = Window(Ein.grid.r,2*self.radius)
                Ein.mask(Etalon_Aperture)
            return Enew
            
    def fourier_mask(self,grid,wavelength,transmit=True):
        """
        fourier_mask(grid,wavelength)
        Returns the fourier transmission function of the etalon
        for a given coordinate grid and wavelength
        """

        etalon_phase = 2*np.pi/wavelength*2*self.nE*self.length*np.cos(np.sqrt((grid.fx*wavelength+self.tilt[0])**2+(grid.fy*wavelength+self.tilt[1])**2))
        if transmit:
            # transmitted spectrum
            Tmask = np.sqrt(self.efficiency)*(1-self.refl)*np.exp(-1j*etalon_phase/2.0)/(1-self.refl*np.exp(-1j*etalon_phase))
        else:
            # reflected spectrum
            Tmask = np.sqrt(self.efficiency)*np.sqrt(self.refl)*(1-np.exp(-1j*etalon_phase))/(1-self.refl*np.exp(-1j*etalon_phase))
            
        return Tmask
    def spectrum(self,lam_range0,InWavelength=True,aoi=0.0,transmit=True):
        """
        outputs the transmission spectram of the etalon over the supplied input wavelength
        Transmission is in Intensity
        lam_range - array of of wavelengths or frequencies
        InWavelength - lam_range is in wavelength.  If set to false it is in frequency
        aoi - angle of incidence in radians
        transmit - if true, outputs transmission spectrum.
                    if false, outputs reflected spectrum.
        """
        
        if InWavelength:     
            lam_range = lam_range0
        else:
            lam_range = c/lam_range0
            
        aoi_glass = np.arcsin(1.0/self.nE*np.sin(aoi))            
        
        etalon_phase = 2*np.pi/lam_range*2*self.nE*self.length*np.cos(aoi_glass)
        if transmit:
#            Spectrum = self.efficiency*1.0/(1+self.finess_coeff*np.sin(etalon_phase/2.0)**2)
            Spectrum = np.abs(np.sqrt(self.efficiency)*(1-self.refl)*np.exp(-1j*etalon_phase/2.0)/(1-self.refl*np.exp(-1j*etalon_phase)))**2
        else:
#            Spectrum = self.efficiency*(1.0-1.0/(1+self.finess_coeff*np.sin(etalon_phase/2.0)**2))
            Spectrum = self.efficiency*(1-np.abs((1-self.refl)*np.exp(-1j*etalon_phase/2.0)/(1-self.refl*np.exp(-1j*etalon_phase)))**2)
        
        return Spectrum

#deltaEtalon0 = 2*pi*nuE/nu0*cos((AngX+thetaFilter1)/FilterIndex)*FilterOrder;
#deltaEtalon1 = 2*pi*nuE/nu1*cos((AngX+thetaFilter1)/FilterIndex)*FilterOrder;
#Tetalon0 = ((1-RefEtalon)^2./((1-RefEtalon)^2+4*RefEtalon*sin(0.5*deltaEtalon0).^2)).^2;
#Tetalon1 = ((1-RefEtalon)^2./((1-RefEtalon)^2+4*RefEtalon*sin(0.5*deltaEtalon1).^2)).^2;    

"""
Generates a Gaussian beam field for 
inputs (required):
    grid - coordinate grid for the field
    radius - list containing x radius and y radius e.g. [Wx,Wy]
        if it is of length 1, the same waist will be applied to all dimensions
    wavelength - wavelength of light in m
inputs (optional):
    divergence - list containing x and y divergence in radians [Divx,Divy]
        if it is of length 1, the same divergence will be applied to all dimensions
        if it is nan, the code will place the waist at this position
    angle - rotation angle in radians to change the major and minor axis orientation
    n - index of refraction of the medium.  Assumed to be 1 if not specified
    z - specifiy a z position of the beam
    direction - specify a direction the beam is propagating (1 -forward, -1 backward)
    Norder - Gaussian order (to create flat topped super Gaussians)
returns
    Efield containing the Gaussian Beam
    
Work needed:
    Make this work for 1D coodinate grids
"""

def GaussianBeam(grid,radius,wavelength,divergence=np.nan,angle=0.0,n=1.0,z=0,direction=1,Norder=1.0):
    
    wavelen = wavelength/n  # use a wavelength adjusted for index of refraction
    # define coordinate system for case where beam principal axes are not aligned
    # to the grid coordinates
    # e.g. elliptical beam at 30 degrees
    if angle != 0.0:
        xb = grid.x*np.cos(angle)+grid.y*np.sin(angle)
        yb = grid.x*np.sin(angle)-grid.y*np.cos(angle)   
    else:
        xb = grid.x
        yb = grid.y   
    
    radius = radius/np.sqrt(np.double(Norder))
    
    # check for radius with length = 1
    if np.size(radius) == 1:
        radius = [radius,radius]
    if np.size(divergence) == 1:
        divergence = [divergence,divergence]
        
    #define the x any y components spearately
    #  need to add a segement checking divergence against radius for physical consistency
    if np.isnan(divergence[0]):
        w0x = radius[0];  # beam waist definition
        zGx = 0;            # rayleigh range definition
        EbeamX = np.exp((1j*2*np.pi/wavelen*zGx-1j*0.5*np.arctan(wavelen*zGx/(np.pi*w0x**2)))-(xb**2/w0x**2))
    else:
        w0x = wavelen/(np.pi*divergence[0]);        # waist definition
        if divergence[0] > 0:
            zGx = -direction*np.pi*w0x**2/wavelen*np.sqrt(np.complex_(radius[0]**2/w0x**2-1));
        else:
            zGx = direction*np.pi*w0x**2/wavelen*np.sqrt(np.complex_(radius[0]**2/w0x**2-1));

        Rgx = zGx*(1+(np.pi*w0x**2/(wavelen*zGx))**2);
        EbeamX = np.sqrt(np.abs(w0x)/radius[0])*np.exp((1j*2*np.pi/wavelen*zGx-1j*0.5*np.arctan(wavelen*zGx/(np.pi*w0x**2)))-(xb**2)*(1/radius[0]**2+1j*2*np.pi/(wavelen*2*Rgx)));

    if np.isnan(divergence[1]):
        w0y = radius[1];  # beam waist definition
        zGy = 0;            # rayleigh range definition
        EbeamY = np.exp((1j*2*np.pi/wavelen*zGy-1j*0.5*np.arctan(wavelen*zGy/(np.pi*w0y**2)))-(yb**2/w0y**2))
    else:
        w0y = wavelen/(np.pi*divergence[1]);        # waist definition
        if divergence[1] > 0:
            zGy = -direction*np.pi*w0y**2/wavelen*np.sqrt(np.complex_(radius[1]**2/w0y**2-1));
        else:
            zGy = direction*np.pi*w0y**2/wavelen*np.sqrt(np.complex_(radius[1]**2/w0y**2-1));

        Rgy = zGy*(1+(np.pi*w0y**2/(wavelen*zGy))**2);
        EbeamY = np.sqrt(np.abs(w0y)/radius[1])*np.exp((1j*2*np.pi/wavelen*zGy-1j*0.5*np.arctan(wavelen*zGy/(np.pi*w0y**2)))-(yb**2)*(1/radius[1]**2+1j*2*np.pi/(wavelen*2*Rgy)));

    Gbeam = Efield(wavelength,grid,z=z,n=1.0,direction=direction,fielddef=EbeamX*EbeamY)
        
    if Norder > 1:
        rb = np.sqrt((xb/radius[0])**2+(yb/radius[1])**2)
        Sg = np.zeros(np.shape(rb))
        for ai in range(Norder):
            Sg = Sg+rb**(2*ai)/sp.misc.factorial(ai)
        Gbeam.mask(Sg)
        
    return Gbeam
    
#function Eg = FlatGaussian(BeamRadiusX,BeamRadiusY,ElAng,lambda,x,y,Norder)
#% Eg = FlatGaussian(BeamRadiusX,BeamRadiusY,ElAng,lambda,x,y,Norder)
#%  Provides the complex Electric field description of an Elliptical
#%  Gaussian beam.
#%  Description is generated at the waist of the beam only
#%  BeamRadiusX - beam radial width in x at waist
#%  BeamRadiusY - beam radial width in y at waist
#%  ElAng - elevation of x axis of elliptical beam (rotates the beam)
#%  lambda - beam wavelength
#%  x - first transverse coordinate
#%  y - second transverse coordinate
#%  Norder - order of flat top Gaussian (higher -> wider)
#
#if ElAng ~= 0
#    xb = x*cos(ElAng)+y*sin(ElAng);
#    yb = x*sin(ElAng)-y*cos(ElAng);
#else
#    xb = x;
#    yb = y;
#end
#
#r = sqrt((xb/BeamRadiusX).^2+(yb/BeamRadiusY).^2);
#
#E0 = EllipticalGaussianBeam(BeamRadiusX,BeamRadiusY,ElAng,lambda,x,y);
#
#Sg = zeros(size(r));
#for ai =0:Norder
#   Sg = Sg+r.^(2*ai)/factorial(ai); 
#end
#Eg = E0.*Sg;
        
"""
Perform a 2DFFT in a way that is conducive with Fourier Optics methods
    e.g. retains centering and scales conserves energy
input:
    Ain - 2D array to be Fourier Transformed
output:
    Aout - 2D Fourier Transform of Ain
    
Work needed:
    Should have an axis argument (0 or 1, 'x' or 'y') for 1D FFT
"""
        
def OpticsFFT(Ain):
    Aout = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Ain)))/np.sqrt(np.size(Ain))
    return Aout

def OpticsIFFT(Ain):
    Aout = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(Ain)))*np.sqrt(np.size(Ain))
    return Aout
    
"""
Mask = CircFunc(gridset,Radius,invert=False)
Generates a circ function on the coodinate grid set provided with radius Radius


work needed:
    This is depricated and replaced with Window()
    Remove calls to this function.


"""
def CircFunc(gridset,Radius,invert=False):
    Mask = np.ones([gridset.Ny,gridset.Nx])
    Mask[np.nonzero(gridset.r**2>Radius**2)] = 0;
    if invert:
        Mask = 1-Mask
    return Mask
    
"""
Generates a rect function mask with full-width of Width on the coordinate set gridset
The function accepts either a 1D or 2D gridset
inputs:
    gridset - 1D or 2D coordinate grid
    Width - full width of the rect function
optional inputs:
    freq - False - apply rect function in spatial coordinates
         - True  - apply rect function in frequency coordinates
    invert - set rect function to an opaque mask (zero inside Width,one outside)
    axis - used only for 2D gridset.  Sets the axis to which the rect is applied.
             if axis = 1 - x axis
                axis = 0 - y axis (for row/column consistency)

work needed:
    This is depricated and replaced with Window()
    Remove calls to this function.
"""
    
def RectFunc(gridset,Width,freq=False,invert=False,axis=1):    
    if gridset.Ny == 1:
        Mask = np.ones([gridset.Ny,gridset.Nx])
        if freq:     
            Mask[np.nonzero(np.abs(gridset.fx)>Width/2.0)] = 0;
            if invert:
                Mask = 1-Mask
            return Mask
        else:
            Mask[np.nonzero(np.abs(gridset.x)>Width/2.0)] = 0;
            if invert:
                Mask = 1-Mask
            return Mask
    elif gridset.dimensions == 2:
        Mask = np.ones([gridset.Ny,gridset.Nx])
        if freq:
            if axis == 0:
                Mask[np.nonzero(np.abs(gridset.fy)>Width/2.0)] = 0;
                if invert:
                    Mask = 1-Mask
                return Mask
            else:
                Mask[np.nonzero(np.abs(gridset.fx)>Width/2.0)] = 0;
                if invert:
                    Mask = 1-Mask
                return Mask
        else:
            if axis == 0:
                Mask[np.nonzero(np.abs(gridset.y)>Width/2.0)] = 0;
                if invert:
                    Mask = 1-Mask
                return Mask
            else:
                Mask[np.nonzero(np.abs(gridset.x)>Width/2.0)] = 0;
                if invert:
                    Mask = 1-Mask
                return Mask
    else:
        print('Error in RectFunc: Unrecognized gridset dimension')
        
"""
Window(gridDim,Width,invert=False,shift=0)

Generates a window function on the grid dimension supplied (e.g. grid.fx, grid.r, etc.)
Replaces CircFunc and RectFunc which are overly specific.  This function can generate
all those features with less code.

Inputs:
    gridDim - array of dimension to operate on.
                for circ function use grid.r, grid.fr
                    rect function use grid.x, grid.fx
                    2D rect function, multiply two window functions together.
    Width - Full width of the window function in coordinate grid space
    shift - shift the window function off zero centering by this amount
    invert - create mask with zeros inside the window limits and ones outside
Output:
    array mask matched to dimensions of supplied coordinate dimension with
    transmitting ones and opaque zeros.
"""
        
def Window(gridDim,Width,shift=0,invert=False):
    Mask = np.ones(np.shape(gridDim))
    Mask[np.nonzero(np.logical_or(gridDim<-(Width/2.0+shift),gridDim>(Width/2.0+shift)))] = 0;
#    Mask[np.nonzero(gridDim>(Width/2.0+shift))] = 0;
    if invert:
        Mask = 1.0-Mask
    return Mask

"""
TelescopeSpider(grid,Router,Rinner,width=0.0)
Create a mask of a telescope spider on the supplied coordinate grid
Router - outer radius of the telescope
Rinner - inner radius of the center mirror
offset(optional = [0,0]) - offset of secondary mirror in the telescope in [x,y]
width (optional = 0.0) - width of supports holding the secondary mirror
rot (optional = 0.0) - Not Implimented yet
"""
def TelescopeSpider(grid,Router,Rinner,offset = [0.0,0.0],width=0.0,rot=0.0):
    if rot != 0.0:
        print('rotation not yet implemented in TelescopeSpider')    
    SpMask = Window(grid.r,2*Router)* \
        Window(np.sqrt((grid.x-offset[0])**2+(grid.y-offset[1])**2),2*Rinner,invert=True)
    SpMask[np.nonzero(np.abs(grid.x-offset[0])<=width/2.0)] = 0;
    SpMask[np.nonzero(np.abs(grid.y-offset[1])<=width/2.0)] = 0;
    return SpMask

"""
Create a figure that shows the progressive intensity of a propagating wave.
AnimatePropagation(Ein,distance=np.nan,z=np.nan,Num=0,dz=0,z0=np.nan,fignum=np.nan)
Inputs:
    Ein - electric field to be propagated
Optional Inputs:
    distance - total distance the user wants the wave propagated.  Do not use with z.
    z - propagate the wave to a position z.  Do not use with distance.
    Num - number of animation frames to create over the propagation distance or z
    dz - propagation step size between frames
    z0 - if using z to specify wave's final positon, z0 can be set to set the initial z postion
    fignum - sets the figure that the animation is placed in.
"""
    
def AnimatePropagation(Ein,distance=np.nan,z=np.nan,Num=0,dz=0,z0=np.nan,fignum=np.nan):
    # fignum optional input does not currently work.
    
    # First Option:  plot based on z position using final position z
    # and either increment dz or number of points Num.  Initial position
    # z0 can be defined or left undefined (uses current wave position)
    if not np.isnan(z):
        if np.isnan(z0):
            z0 = Ein.z
        if Num != 0:
            z_array = np.linspace(z0,z,num=Num)
        elif dz != 0:
            z_array = np.arange(z0,z+dz,dz)
        else:
            z_array = np.linspace(z0,z,num=10)
        # check if user has defined a figure number.  If not
        # create a new figure
        if np.isnan(fignum):
            fignum = plt.figure()
        else:
            fignum = plt.figure(fignum)
        
        #for zi in range(np.size(z_array)):
        plt.xlabel('x position [m]')
        plt.ylabel('y position [m]')
        anim = animation.FuncAnimation(fignum, animate_z, frames=np.size(z_array),fargs=(Ein,z_array),repeat=False)
        plt.show()
        
        # Second Option:  plot based on a propagation distance
    elif not np.isnan(distance):
        z0 = Ein.z
        if Num != 0:
            z_array = z0+np.linspace(0,distance,num=Num)
        elif dz != 0:
            z_array = z0+np.arange(0,distance+dz,dz)
        else:
            z_array = z0+np.linspace(0,distance,num=10)
        
        if np.isnan(fignum):
            fignum = plt.figure()
        else:
            plt.figure(fignum)
        #for di in range(np.size(d_array)):
        plt.xlabel('x position [m]')
        plt.ylabel('y position [m]')
        anim = animation.FuncAnimation(fignum, animate_d, frames=np.size(z_array),fargs=(Ein,z_array,z0),repeat=False)
        plt.show()

    else:
        print ('AnimatePropagation requires more input arguments.  Either z or distance must be defined.')

"""
backscatter(Field,number=1,AngleLim=np.pi,PhaseFunc=np.nan)
Simulate incoherent backscatter from an incident electric field by adding
random phase to the field.
The scattering phase function can be imparted if it is known.  Also an angle
limit can be used to avoid over simulation of scattering angles that are not relevant
to the system detecting the scattering (e.g. set angle limit to 1.5*FS or AS acceptance)
Field - incident electric field
number - number of backscatter profiles to create (for MC modeling)
AngleLim - maximum scattered angle (Half Angle)
PhaseFunc - backscatter phase function of the particles.  Must have the same
        coordinates/dimensions as the input Field
        
Returns the backscattered electric field (including direction reversal)
"""
def backscatter(Field,number=1,AngleLim=np.pi,PhaseFunc=np.nan):
    # reverse propagation direction
    BS_Field = Field.copy();
    BS_Field.direction = -1*BS_Field.direction
    
    AngleFilter = Window(Field.grid.fr*Field.wavelength,1.5*AngleLim)
    if not np.isnan(PhaseFunc):
        AngleFilter = Window(Field.grid.fr*Field.wavelength,1.5*AngleLim)*PhaseFunc
    else:
        AngleFilter = Window(Field.grid.fr*Field.wavelength,1.5*AngleLim)
        
    RandPhase = np.exp(1j*np.random.rand(BS_Field.grid.Ny,BS_Field.grid.Nx)*2*np.pi)
    BS_Field.mask(RandPhase)
    BS_Field.spatial_filter(AngleFilter)
    return BS_Field

# animation routines for wave visualization

def animate_z(i,Ein,z_array):
    Ein.propagate_to(z_array[i])
    cont = plt.contourf(Ein.grid.x, Ein.grid.y, np.abs(Ein.field)**2)
    
    plt.title(r'z = %1.2e m' % z_array[i] )

    return cont

def animate_d(i,Ein,z_array,z0):
    Ein.propagate_to(z_array[i])
    cont = plt.contourf(Ein.grid.x, Ein.grid.y, np.abs(Ein.field)**2)
    
    plt.title(r'd = %1.2e' % (z_array[i]-z0) )
    
    return cont

"""
1-D Library Components
These routines will be removed.  1D coodinates and fields should be folded
into the existing 2D archetecture.

"""

class Coordinate_Grid1D:
    def __init__(self,xset,inputType='frequency',Nmax=np.nan):
        if inputType == 'spatial':
            Nx1D = np.nanmin([2**(np.ceil(np.log2(xset[0]/xset[1]))),Nmax/2])
            x1D = np.arange(-Nx1D,(Nx1D))*xset[1]
            dfx1D = 1/(2*Nx1D*xset[1]);
            fx1D = np.arange(-Nx1D,Nx1D)*dfx1D
            
            self.x=x1D
            self.fx=fx1D,fy1D
            self.dx = xset[1]
            self.dfx = dfx1D
            self.Nx = np.size(self.x)
            self.dimensions = 1
            
        elif inputType == 'frequency':
            Nx1D = np.nanmin([2**(np.ceil(np.log2(xset[0]/xset[1]))),Nmax/2])
            dx1D = xset[0]/Nx1D
            x1D = np.arange(-Nx1D,(Nx1D))*dx1D
            dfx1D = 1/(2*Nx1D*dx1D);
            fx1D = np.arange(-Nx1D,Nx1D)*dfx1D

            self.x=x1D
            self.fx=fx1D,fy1D
            self.dx = xset[1]
            self.dfx = dfx1D
            self.Nx = np.size(self.x)
            
        elif inputType == 'ccd':
            Nx = xset[0]
            
            x1D = np.arange(-np.ceil((Nx-1)/2),np.floor((Nx-1)/2)+1)*xset[1];
            dfx1D =  1.0/(Nx*xset[1])
            fx1D = np.arange(-np.ceil((Nx-1)/2),np.floor((Nx-1)/2))*dfx1D
            
            self.x=x1D
            self.fx=fx1D,fy1D
            self.dx = xset[1]
            self.dfx = dfx1D
            self.Nx = np.size(self.x)
            
        else:
            print ('Error: inputType in Coordinate_Grid initialization not recognized.')
            print ('    It accepts string arguments spatial, frequency or ccd')
        
#        self.dx  = np.mean(np.diff(self.x))
#        self.dy = np.mean(np.diff(self.y))
#        self.dfx = np.mean(np.diff(self.fx))
#        self.dfy = np.mean(np.diff(self.fy))
        
    def rescale(self,factor):
        # rescale grid to maintain consistent spatial and frequency grids
        self.x = self.x*factor
        self.fx = self.fx/factor

        self.dx = self.dx*factor
        self.dfx = self.dfx/factor

    def FTrescale(self,factor):
        self.x = self.fx*factor
        self.fx = self.x/factor
        
        self.dx = self.dfx*factor
        self.dfx = self.dx/factor
        
    def copy(self):
        NewGrid = Coordinate_Grid1D((self.x[-1,-1],self.dx));
        return NewGrid

"""
1 Dimensional Electric field class with physical definitions
wavelength - (in m) wavelength in a vaccuum
n - index of refraction in current medium
z - position along the optic axis
x - grid of x transverse coordinates
fx - grid of frequency coordinates in x
direction - set to 1 (default) for forward propagation
                  -1 for backward propagation (backscattering or reflection)
"""
class Efield1D:
    def __init__(self,wavelength,grid,z=0,n=1.0,direction=1,fielddef=np.nan):
        self.grid = grid.copy()        
        self.wavelength=wavelength
        self.n = n  # index of refraction of current medium
        self.z = z  # position along the optic axis
        self.direction = direction
        self.description = '1-D Electric Field'
        if np.isnan(fielddef):
            self.field = np.ones(self.grid.Nx)
        elif np.size(fielddef) == self.grid.Nx:
            self.field == fielddef.flatten()
        else:
            print ('Efield1D:  Cannot initialize electric fielddef.  Grid has size %d and fielddef has size %d' %(self.grid.Nx,np.size(fielddef)))
            self.field = np.ones(self.grid.Nx)
    def copy(self):
        NewE = Efield1D(self.wavelength,self.grid,z=self.z,direction=self.direction,fielddef=self.field)
        return NewE
    def propagate(self,distance):
        # propagate a distance along the current direction (positive forward, negative backward)
        # using FFT
        if np.abs(distance) > self.wavelength:
            Hcirc = 1.0*(np.abs(self.grid.fx) < 1/self.wavelength);
            H = Hcirc*np.exp(1j*2*np.pi*distance*self.n/self.wavelength*np.sqrt(1-np.complex_((self.wavelength*self.grid.fx)**2)))
            self.field = OpticsIFFT1D(H*OpticsFFT1D(self.field))
            self.z = self.z+distance*self.direction
    def propagate_to(self,zpos):
        # propagate to a specific location on the optic axis
        self.propagate(self.direction*(zpos-self.z))
    def propagate_Fresnel(self,distance):
        print ('propagate_Fresnel not written')
    def spatial_filter(self,filter):
        print ('spatial filter not written')
    def plot(self,linewidth=1.0,fig=np.nan,option=np.nan):
        """
        Plots intensity of 1D electric field.
        
        """
        if np.isnan(fig):
            fig = plt.figure()
        else:
            plt.figure(fig)
        if np.isnan(option):
            plt.plot(self.grid.x,np.abs(self.field)**2)
        else:
            plt.plot(self.grid.x,np.abs(self.field)**2,option)
            
        plt.xlabel('x');
        plt.ylabel('Intensity');
        plt.show();
        return fig
def OpticsFFT1D(Ain):
    Aout = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Ain))) # /np.sqrt(np.size(Ain))
    return Aout

def OpticsIFFT1D(Ain):
    Aout = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(Ain))) # *np.sqrt(np.size(Ain))
    return Aout


"""
MedianSort(NumList,n)
Takes a list of numbers and returns levels of n submedian levels.
"""
def MedianSort(NumList,n):
    med = np.median(NumList)
    if n > 1 and np.size(NumList) > 1:
        NumListU = NumList[np.nonzero(NumList>med)]
        NumListL = NumList[np.nonzero(NumList<med)]
        medU = MedianSort(NumListU,n/2.0)
        medL = MedianSort(NumListL,n/2.0)
        medRet = np.concatenate((medL,medU))
    else:
        medRet = np.array([med])
    return medRet
        
    