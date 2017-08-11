#!/usr/bin/env python
""" This module contains various interpolator objects """
import numpy as np
import scipy.interpolate as interp

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
        tck = interp.splrep(OldX, OldY,k=self.order_k, s=self.smoothing_s)
        return interp.splev(NewX, tck, ext=self.boundry_ext)
