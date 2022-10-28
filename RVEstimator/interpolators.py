#!/usr/bin/env python
""" This module contains various interpolator objects """
import numpy as np
import scipy.interpolate as interp
import logging
try:
    from functools32 import partial
except ModuleNotFoundError:
    from functools import partial

def remove_nans(Y,X=None,method='drop'):
    """ Returns a clean Y data after removing nans in it.
    If X is provided, the corresponding values form X are also matched and (Y,X) is returned.
    Input:
         Y: numpy array
         X: (optional) 1d numpy array of same size as Y
         method: drop: drops the nan values and return a shorter array
                 any scipy.interpolate.interp1d kind keywords: interpolates the nan values using interp1d 
    Returns:
         if X is provided:  (Y,X,NanMask)
         else: (Y,NanMask)
    """
    NanMask = np.isnan(Y)
    if method == 'drop':
        returnY = Y[~NanMask]
        if X is not None:
            returnX = X[~NanMask]
    else: # Do interp1d interpolation
        if X is not None:
            returnX = X
        else:
            returnX = np.arange(len(Y))
        returnY = interp.interp1d(returnX[~NanMask],Y[~NanMask],kind=method,fill_value='extrapolate')(returnX)

    if X is not None:
        return returnY,returnX,NanMask
    else:
        return returnY,NanMask

# Band limited 1D Interpolator
class BandLimitedInterpolator(object):
    """ Interpolator for doing Band-limited interpolation using windowed Sinc function """
    def __init__(self,filter_size = 23, kaiserB=13):
        """ 
        Input:
             filter_size : total number of pixels in the interpolation window 
                           (keep it odd number), default =23
             kaiserB     : beta value for determiniong the width of Kaiser window function
        """
        self.filter_size = filter_size
        self.kaiserB = kaiserB
        self.Filter = self.create_filter_curve(no_of_points = self.filter_size*21)
        self.pixarray = np.arange(-int(self.filter_size/2), int(self.filter_size/2)+1,dtype=np.int)

    def create_filter_curve(self,no_of_points=None):
        """ Returns a cubit interpolator for windowed sinc Filter curve.
        no_of_points: number of intepolation points to use in cubic inteprolator"""
        if no_of_points is None:
            no_of_points = self.filter_size*21
        x = np.linspace(-int(self.filter_size/2), int(self.filter_size/2), no_of_points)
        Sinc = np.sinc(x)
        Window = np.kaiser(len(x),self.kaiserB)
        FilterResponse = Window*Sinc
        # append 0 to both ends far at the next node for preventing cubic spline 
        # from extrapolating spurious values
        return interp.CubicSpline( np.concatenate(([x[0]-1],x,[x[-1]+1])), 
                                   np.concatenate(([0],FilterResponse,[0])))

    def interpolate(self,newX,oldX,oldY,PeriodicBoundary=False):
        """ Inteprolates oldY values at oldX coordinates to the newX coordinates.
        Periodic boundary conditions set to True can create worse instbailities at edge..
        oldX and oldY should be larger than filter window size self.filter_size"""
        # First clean and remove any nans in the data
        oldY, oldX, NanMask = remove_nans(oldY,X=oldX,method='linear')
        if np.sum(NanMask) > 0:
            logging.warning('Interpolated {0} NaNs'.format(np.sum(NanMask)))
        oXsize = len(oldX)
        # First generate a 2D array of difference in pixel values
        OldXminusNewX = np.array(oldX)[:,np.newaxis] - np.array(newX)
        # Find the minimum position to find nearest pixel for each each newX
        minargs = np.argmin(np.abs(OldXminusNewX), axis=0)
        # Pickout the those minumum values from 2D array
        minvalues = OldXminusNewX[minargs, range(OldXminusNewX.shape[1])]
        sign = minvalues < 0  # True means new X is infront of nearest old X
        # coordinate of the next adjacent bracketing point
        Nminargs = minargs +sign -~sign
        Nminargs = Nminargs % oXsize  # Periodic boundary
        # In terms of pixel coordinates the shift values will be
        shiftvalues = minvalues/np.abs(oldX[minargs]-oldX[Nminargs])
        # Coordinates to calculate the Filter values
        FilterCoords = shiftvalues[:,np.newaxis] + self.pixarray
        FilterValues = self.Filter(FilterCoords)
        # Coordinates to pick the values to be multiplied with Filter and summed
        OldYCoords = minargs[:,np.newaxis] + self.pixarray
        if PeriodicBoundary:
            OldYCoords = OldYCoords % oXsize  # Periodic boundary
        else:   # Extrapolate the last value till end..
            OldYCoords[OldYCoords >= oXsize] = oXsize-1
            OldYCoords[OldYCoords < 0] = 0

        OldYSlices = oldY[OldYCoords] # old flux values to be multipled with filter values
        return np.sum(OldYSlices*FilterValues,axis=1)



class BSplineInterpolator(object):
    """ Interpolator for doing B-Spline interpolation. 
    Warning: This interpolation is just an approximation. For Band limited Inteprolation, 
    use the slower BandLimitedInterpolator"""
    def __init__(self,boundry_ext = 3, order_k=3, smoothing_s = 0):
        """ 
        Input:
             boundry_ext : How the external interpolation needs to be done
                           See the documentation of ext in scipy.interpolate.splev()
             order_k : Order of the Bspline
                           See the documentation of k in scipy.interpolate.splrep()
             smoothing_s : Smoothing of the B-spline
                           See the documentation of s in scipy.interpolate.splrep()
        """
        self.boundry_ext = boundry_ext
        self.order_k = order_k
        self.smoothing_s = smoothing_s

    def interpolate(self,NewX,OldX,OldY):
        """ Inteprolates oldY values at oldX coordinates to the newX coordinates. """
        # First clean and remove any nans in the data
        OldY, OldX, NanMask = remove_nans(OldY,X=OldX,method='drop')
        if np.sum(NanMask) > 0:
            logging.warning('Dropped {0} NaNs'.format(np.sum(NanMask)))

        tck = interp.splrep(OldX, OldY,k=self.order_k, s=self.smoothing_s)
        return interp.splev(NewX, tck, ext=self.boundry_ext)

class CumSumInterpolator(object):
    """ Interpolator for doing Flux preserving interpolation in the cumulative sum space. 
    It creates a sumulative sum data, interpolate, and then differentiate for the final results.
    """
    def __init__(self,boundry_ext = 3, order_k=3, smoothing_s = 0,x_loc='middle',use_pchip=False):
        """ 
        Input:
             boundry_ext : How the external interpolation needs to be done on the cumulative sum
                           See the documentation of ext in scipy.interpolate.splev()
             order_k : Order of the Bspline inteprolation of the cumulative sum
                           See the documentation of k in scipy.interpolate.splrep()
             smoothing_s : Smoothing of the B-spline inteprolation of the cumulative sum
                           See the documentation of s in scipy.interpolate.splrep()
             x_loc : ('middle','start','end')
                           Location of the x values represent the middle of the bin, or start of the bin or end of the bin.
             use_pchip: (bool) default:False
                           If True, in the cumsum space use pchip instead of spline. This constrains the pixel values to be strictly positive.
                           This is highly recommended over Bspline if the values are know to be positive to avoid any ripples.
                           if True, the other Bspline parameters will be ignored.
        """
        self.boundry_ext = boundry_ext
        self.order_k = order_k
        self.smoothing_s = smoothing_s
        self.x_loc = x_loc
        self.use_pchip = use_pchip

    def interpolate(self,NewX,OldX,OldY):
        """ Inteprolates oldY values at oldX coordinates to the newX coordinates. """
        # First convert OldX and NewX to the trail end of the bins
        if self.x_loc == 'middle':  # convert to end
            binwidths = np.diff(OldX)/2.
            OldX = np.array(OldX)+np.concatenate((binwidths,[binwidths[-1]]))
            binwidths_n = np.diff(NewX)/2.
            NewX = np.array(NewX)+np.concatenate((binwidths_n,[binwidths_n[-1]]))
        elif self.x_loc == 'start': # convert to end of the bins
            OldX = np.concatenate((OldX[1:],[ OldX[-1] + (OldX[-1]-OldX[-2]) ]))
            NewX = np.concatenate((NewX[1:],[ NewX[-1] + (NewX[-1]-NewX[-2]) ]))
        elif self.x_loc == 'end': # nothing to do
            pass
        else:
            raise ValueError('x_loc = {0} is not valid in CumSumInterpolator. Allowed x_loc values:{"middle","start","end"}'.format(self.x_loc))

        # Now clean and remove any nans in the data
        OldY, OldX, NanMask = remove_nans(OldY,X=OldX,method='drop')
        if np.sum(NanMask) > 0:
            logging.warning('Dropped {0} NaNs'.format(np.sum(NanMask)))
        # Create Cumsum array to interpolate
        CumSumOldY = np.cumsum(OldY.astype(np.float64))
        if self.use_pchip:
            pchip = interp.PchipInterpolator(OldX, CumSumOldY, extrapolate=True)
            CumSumNewY = pchip(NewX)
        else:
            tck = interp.splrep(OldX, CumSumOldY,k=self.order_k, s=self.smoothing_s)
            CumSumNewY = interp.splev(NewX, tck, ext=self.boundry_ext)
        # Return diff of the cumulative sum
        return np.concatenate(([CumSumNewY[0]],np.diff(CumSumNewY)))


def productfunction(X,Y,FuncX,FuncY):
    """ Returncs product of the dunctions.  FuncX(X)*FuncY(Y) """
    return FuncX(X)*FuncY(Y)

# Band limited 2D Interpolator
class BandLimited2DInterpolator(object):
    """ Interpolator for doing Band-limited interpolation using windowed Sinc function in 2D image """
    def __init__(self,filter_sizeX = 13,filter_sizeY = 13, kaiserBX=7, kaiserBY=7):
        """ 
        Input:
             filter_sizeX : total number of pixels in X axis of the interpolation window 
                          (keep it odd number), default =13 
             filter_sizeY : total number of pixels in Y axis of the interpolation window 
                          (keep it odd number), default =13 
             kaiserBX     : beta value for determiniong the width of Kaiser window function in X axis
             kaiserBY     : beta value for determiniong the width of Kaiser window function in Y axis
        """
        self.filter_sizeX = filter_sizeX
        self.filter_sizeY = filter_sizeY
        self.kaiserBX = kaiserBX
        self.kaiserBY = kaiserBY
        self.Filter = self.create_filter_curve(no_of_subpixelbinning = 21)
        self.pixarrayX = np.arange(-int(self.filter_sizeX/2), int(self.filter_sizeX/2)+1,dtype=np.int)
        self.pixarrayY = np.arange(-int(self.filter_sizeY/2), int(self.filter_sizeY/2)+1,dtype=np.int)

    def create_filter_curve(self,no_of_subpixelbinning=None):
        """ Returns a cubit interpolator for windowed 2D sinc Filter curve.
        ie. Kx*Sinc(x)*Ky*Sinc(y) function for 2D BL interpolation
        no_of_points: number of intepolation points to use in cubic inteprolator"""
        if no_of_subpixelbinning is None:
            no_of_subpixelbinning = 21
        x = np.linspace(-int(self.filter_sizeX/2), int(self.filter_sizeX/2),
                        no_of_subpixelbinning * self.filter_sizeX)
        y = np.linspace(-int(self.filter_sizeY/2), int(self.filter_sizeY/2),
                        no_of_subpixelbinning * self.filter_sizeY)
        SincX = np.sinc(x)
        SincY = np.sinc(y)
        WindowX = np.kaiser(len(x),self.kaiserBX)
        WindowY = np.kaiser(len(y),self.kaiserBY)
        FilterResponseX = WindowX*SincX
        FilterResponseY = WindowY*SincY
        # append 0 to both ends far at the next node for preventing cubic spline 
        # from extrapolating spurious values
        XFunction = interp.CubicSpline( np.concatenate(([x[0]-1],x,[x[-1]+1])), 
                                        np.concatenate(([0],FilterResponseX,[0])))
        YFunction = interp.CubicSpline( np.concatenate(([y[0]-1],y,[y[-1]+1])), 
                                        np.concatenate(([0],FilterResponseY,[0])))
        # Return the Kx*Sinc(x)*Ky*Sinc(y) function for 2D BL interpolation
        return partial(productfunction,FuncX=XFunction,FuncY=YFunction)


    def interpolate(self,newX,newY,oldZ, oldX=None,oldY=None,  PeriodicBoundary=False):
        """ Inteprolates 2D oldZ array values at (oldX,oldY) coordinate grid to the 
             (newX,newY) coordinate grid.
        Periodic boundary conditions set to True can create worse instbailities at edge..
        oldX and oldY should be larger than filter window size self.filter_sizeX,Y respectively
        Input array Dimensions:
          newX : 1D array X coordinates of points to interpolate to
          newY : 1D array Y coordinates of points to interpolate to
          oldZ : 2D array image to interpolate from
          oldX : 1D array default: np.arange(oldZ.shape[0])
          oldY : 1D array default: np.arange(oldZ.shape[0])
        """
        if oldX is None:
            oldX = np.arange(oldZ.shape[0])
        if oldY is None:
            oldY = np.arange(oldZ.shape[1])

        oXsize = len(oldX)
        oYsize = len(oldY)
        # First generate a 2D array of difference in pixel values
        OldXminusNewX = np.array(oldX)[:,np.newaxis] - np.array(newX)
        OldYminusNewY = np.array(oldY)[:,np.newaxis] - np.array(newY)
        # Find the minimum position to find nearest pixel for each each newX
        minargsX = np.argmin(np.abs(OldXminusNewX), axis=0)
        minargsY = np.argmin(np.abs(OldYminusNewY), axis=0)
        # Pickout the those minumum values from 2D array
        minvaluesX = OldXminusNewX[minargsX, range(OldXminusNewX.shape[1])]
        minvaluesY = OldYminusNewY[minargsY, range(OldYminusNewY.shape[1])]
        signX = minvaluesX < 0  # True means new X is infront of nearest old X
        signY = minvaluesY < 0  # True means new Y is infront of nearest old Y
        # coordinate of the next adjacent bracketing point
        NminargsX = minargsX +signX -~signX  
        NminargsX = NminargsX % oXsize  # Periodic boundary
        NminargsY = minargsY +signY -~signY  
        NminargsY = NminargsY % oYsize  # Periodic boundary
        # In terms of pixel coordinates the shift values will be
        shiftvaluesX = minvaluesX/np.abs(oldX[minargsX]-oldX[NminargsX])
        shiftvaluesY = minvaluesY/np.abs(oldY[minargsY]-oldY[NminargsY])
        # Coordinates to calculate the Filter values
        FilterCoordsX = shiftvaluesX[:,np.newaxis] + self.pixarrayX
        FilterCoordsY = shiftvaluesY[:,np.newaxis] + self.pixarrayY
        # Create a 3D mesh grid of the coordinates
        FilterCoordsXmeshgd = np.repeat(FilterCoordsX[:,:,np.newaxis],len(self.pixarrayY),axis=2)
        FilterCoordsYmeshgd = np.repeat(FilterCoordsY[:,np.newaxis,:],len(self.pixarrayX),axis=1)
        FilterValues = self.Filter(FilterCoordsXmeshgd,FilterCoordsYmeshgd)
        # Coordinates to pick the values to be multiplied with Filter and summed
        OldZCoordsX = minargsX[:,np.newaxis] + self.pixarrayX
        OldZCoordsY = minargsY[:,np.newaxis] + self.pixarrayY
        if PeriodicBoundary:
            OldZCoordsX = OldZCoordsX % oXsize  # Periodic boundary
            OldZCoordsY = OldZCoordsY % oYsize  # Periodic boundary
        else:   # Extrapolate the last value till end..
            OldZCoordsX[OldZCoordsX >= oXsize] = oXsize-1
            OldZCoordsX[OldZCoordsX < 0] = 0
            OldZCoordsY[OldZCoordsY >= oYsize] = oYsize-1
            OldZCoordsY[OldZCoordsY < 0] = 0
        # Create a 3D mesh grid of the coordinates
        OldZCoordsXmeshgd = np.repeat(OldZCoordsX[:,:,np.newaxis],len(self.pixarrayY),axis=2)
        OldZCoordsYmeshgd = np.repeat(OldZCoordsY[:,np.newaxis,:],len(self.pixarrayX),axis=1)
        # old flux values to be multipled with filter values
        OldZSlices = oldZ[[OldZCoordsXmeshgd,OldZCoordsYmeshgd]] 
        return np.sum(OldZSlices*FilterValues,axis=(1,2))
