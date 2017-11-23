#!/usr/bin/env python
""" This module contains various radial velcoty estimation methods """
import numpy as np
from scipy import optimize
from .interpolators import BandLimitedInterpolator, BSplineInterpolator
from .utils import NearestIndex

################################################################
# Fit multiple RVs by Least Square minimisation of pieces of spectrum

class SpectralRVtransformer(object):
    """ Object which has methods to transform input spectrum by radial velocities or polynomial conitnuum
    Methods inside are useful for scipy.optimise.curv_fit as well. """
    def __init__(self,SpectrumXY,V_0=0,interpolator=None):
        """
        SpectrumXY : Spectrum to fit transform
        V_0 : the zeroth order fixed velocity (Barycentric Radial Velocity to be be added to RV fit
        interpolator: interpolator to use to interpolating input Spectrum to dopler shifted output spectrum
        """
        self.c = 299792.458  # Speed of light in km/s
        self.SpectrumXY = SpectrumXY
        self.V_0 = V_0
        if interpolator is None:
            self.interp = BandLimitedInterpolator(kaiserB=13)
        else:
            self.interp = interpolator
        
        self.wranges = [] # wavelength ranges to fit RV seperately
        self.polycoeffs = [1,0] # polynomial coefficients of fitted continuum correction

    def dopplerfactor(self,velocity):
        """ Returns the doppler factor for input velocity """
        return np.sqrt((1 + (-velocity)/self.c) / (1 - (-velocity)/self.c))

    def multiply_poly_continuum(self,X,*params):
        """ Returns the Ydata for input Xdata and params after multiplying with the polynomial
        defined by the input params"""
        # print('Fitting only continuum with polynomial order ={0}'.format(len(params)))
        doppler_factor = self.dopplerfactor(self.V_0)
        PolynomialScale = np.polynomial.polynomial.polyval(X, params)
        SpectrumIntpFlux = self.interp.interpolate(X,
                                                   self.SpectrumXY['wavel'] * doppler_factor, 
                                                   self.SpectrumXY['flux'])
        # self.polycoeffs = params  # update the internal variable with latest call
        return SpectrumIntpFlux*PolynomialScale
    def apply_rv_redshift(self,X,*params):
        """ Returns the Ydata for input Xdata and params after applying the radial velocity redshift.
        If the input params are a list of rv, user should make sure self.wranges 
            are correspondingly defined to apply different rv to corrseponding wavelength ranges"""
        doppler_factors = [self.dopplerfactor(self.V_0+v) for v in params]
        segmentedX = []
        if self.wranges:
            for ws,we in zip(self.wranges[:-1],self.wranges[1:]):
                segmentedX.append(X[NearestIndex(X,ws):NearestIndex(X,we)])
            # Last index need to be +1 to get the full array
            segmentedX[-1] = X[NearestIndex(X,ws):NearestIndex(X,we)+1]
        else:
            segmentedX = [X]

        if len(doppler_factors) != len(segmentedX):
            print('WARNING: No: of wavelgnth ranges ({0}) does not match no: of rvs ({1})'.format(len(segmentedX),len(doppler_factors)))

        OutputY = []
        for segX,dpf in zip(segmentedX,doppler_factors):
            PolynomialScale = np.polynomial.polynomial.polyval(segX, self.polycoeffs)
            SpectrumIntpFlux = self.interp.interpolate(segX,
                                                       self.SpectrumXY['wavel'] * dpf, 
                                                       self.SpectrumXY['flux'])
            OutputY.append(SpectrumIntpFlux*PolynomialScale)
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
    FittingFunction = SpectralRVtransformer(TemplateXY,V_0,interpolator)
    xdata = SpectrumXY['wavel'][TrimSize:-TrimSize]
    ydata = SpectrumXY['flux'][TrimSize:-TrimSize]
    ysigma = np.sqrt(SpectrumXY['fluxVar'][TrimSize:-TrimSize])
    popt, pcov = optimize.curve_fit(FittingFunction.multiply_poly_continuum, 
                                    xdata, ydata,p0=[1,0,0],sigma=ysigma)
    FittingFunction.polycoeffs = popt #assign best fitted continuum
    NoOfSegments = len(xdata)/minsize
    FittingFunction.wranges = np.linspace(SpectrumXY['wavel'][TrimSize],
                                          SpectrumXY['wavel'][-TrimSize], NoOfSegments+1)
    p0rv = np.zeros(len(FittingFunction.wranges)-1)
    pvopt, pvcov = optimize.curve_fit(FittingFunction.apply_rv_redshift, 
                                      xdata, ydata,p0=p0rv,sigma=ysigma,method='trf')
    return pvopt, pvcov, FittingFunction.wranges


################################################################
