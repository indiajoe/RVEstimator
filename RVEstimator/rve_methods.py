#!/usr/bin/env python
""" This module contains various radial velcoty estimation methods """
import numpy as np
from scipy import optimize
from .utils import NearestIndex, SpectralRVtransformer

################################################################
# Fit multiple RVs by Least Square minimisation of pieces of spectrum

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
