#!/usr/bin/env python
""" This module contains various radial velcoty estimation methods """
import numpy as np
from scipy import optimize
from astropy.io import fits
from .utils import NearestIndex, SpectralRVtransformer, MultiOrderSpectrum, scale_spectrum

################################################################
# Fit multiple RVs by Least Square minimisation of pieces of spectrum

def FitRVTemplateAdaptively(TemplateXY,SpectrumXY,V_bary=0,V_star=0,interpolator=None,TrimSize=150,minsize=50):
    """ Fits TemplateXY to SpectrumXY optimising piecewise radial velocity and polynomial continnum.
        TemplateXY : Template spectrum to fit data
        SpectrumXY : Star spectrum to fit RVs
        V_bary : Barycentric Radial Velocity to be added to RV fit
        V_star : Zeroth order stellar radial velocity to be added to RV fit
        interpolator: interpolator to use to interpolating Template to star spectrum.
        TrimSize : the number of data points to discard in minimisation from both ends of data.
        minsize: the minium pixel array size below which we should not adaptively make smaller RV fits.

    Returns:
        pvopt: Fitted RV values
        pvcov: Covarience matrix fo the fitter RV values
        wranges: Wavelength ranges which defines the calculated RV value positions
    """
    FittingFunction = SpectralRVtransformer(TemplateXY,V_bary=V_bary,V_star=V_star,interpolator=interpolator)
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

def CreateTemplateFromSpectra(SpecList,BaryVShift=False,StarVShift=False,normalise=True,
                              cmethod='median',interpolator=None, TemplateSpec=None):
    """ Creates a Template by combining List of Spectra objects.
    Inputs:
       SpecList: List of MultiOrderSpectrum dictionary continaing spectrum
       BaryVShift: (default False), list of Barycentric rv shifts, str of the header keyword.
               If False, Template is created by combining without doing any Barycentric RV shift.
       StarVShift: (default False), list of Star rv shifts, str of the header keyword.
               If False, Template is created by combining without doing any Stellar RV shift.
       normalise: (default True)
               If True, each order is normalised before combining.
       cmethod: (default 'median')
               method to combine all Spectra
       interpolator: 
              Interpolator to use for interpolating star spectrum to same Wavelgnth before combining.
       TemplateSpec (optional): 
              An optional MultiOrderSpectrum dictionary continaing Template wavelengths to
              interpolate to for each order.
              If None provided, the wavelengths of first spectrum will be used


    Returns:
       TemplateSpec: MultiOrderSpectrum dictionary containing combined Template
    """

    if BaryVShift:
        if isinstance(BaryVShift,str):  # If it is a header keyword like BARYVEL,
            #Read into a list of rv shifts
            BaryVShift = [Spec.header[BaryVShift] for Spec in SpecList]
    else:
        BaryVShift = np.zeros(len(SpecList))

    if StarVShift:
        if isinstance(StarVShift,str):  # If it is a header keyword like RADVEL,
            #Read into a list of rv shifts
            StarVShift = [Spec.header[StarVShift] for Spec in SpecList]
    else:
        StarVShift = np.zeros(len(SpecList))


    if TemplateSpec is None:
        # We shall use the wavelengths of the first spectrum as Template wavelength
        # We need to interpolate to RV shifted wavel of first spectrum
        c = 299792458.  # Speed of light in m/s
        doppler_factor = (1 + StarVShift[0]/c)/(1 + BaryVShift[0]/c)  # Formula for 1+z_mes (Wright & Eastman 2014)

        TemplateSpec = MultiOrderSpectrum()
        TemplateSpec.header = fits.Header()
        TemplateSpec.header['OBJECT'] = 'Template Spectrum'
        for order in SpecList[0]:
            TemplateSpec[order] = {'wavel':SpecList[0][order]['wavel']/doppler_factor}

    if normalise:
        SpecList = [scale_spectrum(Spec) for Spec in SpecList]

    # Create a list of spectra from same orders listed together
    ListofSpectraInOrders = [[Spec[order] for Spec in SpecList] for order in SpecList[0]]
    
    # Do each order serially
    for order,SpectrumXYlist in zip(SpecList[0],ListofSpectraInOrders):
        TemplateSpec[order] = _CreateTemplateFromSpectra_singleorder(SpectrumXYlist,
                                                                     TemplateSpec[order], 
                                                                     BaryVShift=BaryVShift,
                                                                     StarVShift=StarVShift,
                                                                     cmethod=cmethod,
                                                                     interpolator=interpolator)
    # Do each order in parallel # To be implemented
    return TemplateSpec


def _CreateTemplateFromSpectra_singleorder(SpectrumXYlist,TemplateSpecorder, BaryVShift=False,
                                           StarVShift=False, cmethod='median',interpolator=None):
    """ See the doc of CreateTemplateFromSpectra for documentation on arguments.
        This is a sub function to create template for one individial order in Spectra """

    if not BaryVShift:
        BaryVShift = np.zeros(len(SpectrumXYlist))
    if not StarVShift:
        StarVShift = np.zeros(len(SpectrumXYlist))
    SpecFluxList = []
    SpecFluxVarList = []
    for SpectrumXY,rvb,rvs in zip(SpectrumXYlist,BaryVShift,StarVShift):
        TranfSpec = SpectralRVtransformer(SpectrumXY,V_bary=rvb,V_star=rvs,interpolator=interpolator)
        # Remove the bary centric velocity and stellar velocity (0) from the spectrum
        SpecFluxList.append(TranfSpec.apply_rv_redshift(TemplateSpecorder['wavel'],0,remove_rv=True))
        SpecFluxVarList.append(TranfSpec.apply_rv_redshift(TemplateSpecorder['wavel'],0,fluxkey='fluxVar',remove_rv=True))
    if cmethod =='median':
        TemplateSpecorder['flux'] = np.median(SpecFluxList,axis=0)
    elif cmethod =='average':
        TemplateSpecorder['flux'] = np.average(SpecFluxList,axis=0,
                                               weights=1.0/np.array(SpecFluxVarList))

    TemplateSpecorder['fluxVar'] = 1/np.sum(1.0/np.array(SpecFluxVarList),axis=0) 
    return TemplateSpecorder
        
        
        
