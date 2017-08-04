#!/usr/bin/env python
""" This module contains various radial velcoty estimation methods """
import numpy as np
from scipy import optimize
from .interpolators import BandLimitedInterpolator, BSplineInterpolator
from .utils import NearestIndex

################################################################
# Fit multiple RVs by Least Square minimisation of pieces of spectrum

class AdaptiveRVshifter(object):
    """ Callable object which can be used to shift input spectrum by radial velocities .
    Useful for scipy.optimise.curv_fit """
    def __init__(self,TemplateXY,V_0=0,interpolator=None):
        """
        TemplateXY : Template spectrum to fit data
        V_0 : the zeroth order fixed velocity (Barycentric Radial Velocity to be be added to RV fit
        interpolator: interpolator to use to interpolating Template to star spectrum
        """
        self.c = 299792.458  # Speed of light in km/s
        self.TemplateXY = TemplateXY
        self.V_0 = V_0
        if interpolator is None:
            self.interp = BandLimitedInterpolator(kaiserB=13)
        else:
            self.interp = interpolator
        
        self.wranges = [] # wavelength ranges to fit RV seperately
        self.fit_only_continuum = True # First fit only continuum
        self.polycoeffs = [1,0] # polynomial coefficients of fitted continuum correction

    def dopplerfactor(self,velocity):
        """ Returns the doppler factor for input velocity """
        return np.sqrt((1 + (-velocity)/self.c) / (1 - (-velocity)/self.c))

    def __call__(self,X,*params):
        """ Returns the Ydata for input Xdata and params """
        if self.fit_only_continuum:
            # print('Fitting only continuum with polynomial order ={0}'.format(len(params)))
            doppler_factor = self.dopplerfactor(self.V_0)
            PolynomialScale = np.polynomial.polynomial.polyval(X, params)
            TemplateIntpFlux = self.interp.interpolate(X,
                                                       self.TemplateXY['wavel'] * doppler_factor, 
                                                       self.TemplateXY['flux'])
            # self.polycoeffs = params  # update the internal variable with latest call
            return TemplateIntpFlux*PolynomialScale
        else:
            doppler_factors = [self.dopplerfactor(self.V_0+v) for v in params]
            segmentedX = []
            for ws,we in zip(self.wranges[:-1],self.wranges[1:]):
                segmentedX.append(X[NearestIndex(X,ws):NearestIndex(X,we)])
            # Last index need to be +1 to get the full array
            segmentedX[-1] = X[NearestIndex(X,ws):NearestIndex(X,we)+1]
            OutputY = []
            for segX,dpf in zip(segmentedX,doppler_factors):
                PolynomialScale = np.polynomial.polynomial.polyval(segX, self.polycoeffs)
                TemplateIntpFlux = self.interp.interpolate(segX,
                                                           self.TemplateXY['wavel'] * dpf, 
                                                           self.TemplateXY['flux'])
                OutputY.append(TemplateIntpFlux*PolynomialScale)
            return np.concatenate(OutputY)
            
def FitRVTemplateAdaptively(TemplateXY,SpectrumXY,V_0=0,interpolator=None,TrimSize=150,minsize=50):
    """ Fits TemplateXY to SpectrumXY optimising piecewise radial velocity and polynomial continnum.
        TemplateXY : Template spectrum to fit data
        SpectrumXY : Star spectrum to fit RVs
        V_0 : the zeroth order fixed velocity (Barycentric Radial Velocity to be be added to RV fit
        interpolator: interpolator to use to interpolating Template to star spectrum.
        TrimSize : the number of data points to discard in minimisation from both ends of data.
        minsize: the minium pixel array size below which we should not adaptively make smaller RV fits.

    Returns:
        pvopt: Fitted RV values
        pvcov: Covarience matrix fo the fitter RV values
        wranges: Wavelength ranges which defines the calculated RV value positions
    """
    FittingFunction = AdaptiveRVshifter(TemplateXY,V_0,interpolator)
    FittingFunction.fit_only_continuum = True # First fit only continuum
    xdata = SpectrumXY['wavel'][TrimSize:-TrimSize]
    ydata = SpectrumXY['flux'][TrimSize:-TrimSize]
    ysigma = np.sqrt(SpectrumXY['fluxVar'][TrimSize:-TrimSize])
    popt, pcov = optimize.curve_fit(FittingFunction, xdata, ydata,p0=[1,0,0],sigma=ysigma)
    FittingFunction.polycoeffs = popt #assign best fitted continuum
    FittingFunction.fit_only_continuum = False # Don't fit continuum again
    NoOfSegments = len(xdata)/minsize
    FittingFunction.wranges = np.linspace(SpectrumXY['wavel'][TrimSize],
                                          SpectrumXY['wavel'][-TrimSize], NoOfSegments+1)
    p0rv = np.zeros(len(FittingFunction.wranges)-1)
    pvopt, pvcov = optimize.curve_fit(FittingFunction, xdata, ydata,p0=p0rv,sigma=ysigma,method='trf')
    return pvopt, pvcov, FittingFunction.wranges


################################################################
