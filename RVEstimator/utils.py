#!/usr/bin/env python
""" This contains miscellaneous utility functions for the module """
import os
import numpy as np
import copy
from scipy import signal

import logging
from multiprocessing.pool import Pool
from functools32 import wraps, partial
from .interpolators import BandLimitedInterpolator, BSplineInterpolator

def unwrap_args_forfunction(func):
    """ This decorato is for unwrapping a tuple of inputs for function
    useful for wrapping multi-argument functions into Pool.map """
    @wraps(func)
    def unwrapped_args_func(argstuple):
        return func(*argstuple)
    return unwrapped_args_func

def NearestIndex(Array,value):
    """ Returns the index of element in numpy 1d Array nearest to value """
    #return np.searchsorted(Array,value)  # Works for sorted array only
    return np.abs(Array-value).argmin()

class MultiOrderSpectrum(dict):
    """ Simple Dictionary object to store multi order spectrum """
    # This is just a dictionary with header value 
    def __init__(self,*arg,**kw):
        super(MultiOrderSpectrum, self).__init__(*arg, **kw)
        # Add an attrubute called header to store contents of fits header
        self.header = None


def LoadSpectraFromFilelist(filenameslist, fileloaderfunc):
    """ Returns a list of loaded spectrum from the filenames in input filenameslist.
    filenamelist: It could be a directory name continaing fits files, 
                  A text file continaing filenames
                  A single fits filename, it will be loaded into a single element list.
                  A python list of filenames
    fileloaderfunc : function which loads the fits file and returns SpecDic
    
    This function skips filenames commented out with # in filenameslist if it is a text file
    """
    ListOfSpec = []
    if isinstance(filenameslist,list):
        # Load all the spectrum in the list
        for fname in filenameslist:
            ListOfSpec.append(fileloaderfunc(fname))
    elif os.path.splitext(filenameslist)[-1] == '.fits':
        # Return a list with single fits file
        ListOfSpec.append(fileloaderfunc(filenameslist))
    elif os.path.isdir(filenameslist):
        # Return a list of all .fits spectrum in the directory
        for fname in sorted(os.listdir(filenameslist)):
            if os.path.isfile(os.path.join(filenameslist, fname)) and \
               os.path.splitext(fname)[-1] == '.fits':
                ListOfSpec.append(fileloaderfunc(fname))
    else:
        # It must be a text file. Load the list from the file
        with open(filenameslist) as filelist:
            for fitsfile in filelist:
                if fitsfile[0] == '#' : continue
                fitsfile = fitsfile.rstrip()
                NewSpec = fileloaderfunc(fitsfile)
                ListOfSpec.append(NewSpec)
                
    return ListOfSpec


def CleanNegativeValues(SpecDic,minval=0.5,method='lift'):
    """ Cleans all negative values in the data with positive value minval.
    method: clip - clips all < minval values with minval
            lift - lifts the spectrum by adding a constant so that spectrum is above minval
    """
    
    for order in SpecDic:
        NegativeIndices = np.where(SpecDic[order]['flux'] < minval)[0]
        if len(NegativeIndices) > 0:
            logging.info('Negative data in {0}: order #{1}'.format(SpecDic.header['FITSFILE'],order))
            logging.debug('Bad pixels: {0}'.format(NegativeIndices))
            if method == 'clip':
                logging.info('Replacing: {0} with {1}'.format(SpecDic[order]['flux'][NegativeIndices],
                                                              minval))
                SpecDic[order]['flux'][NegativeIndices] = minval
            elif method == 'lift':
                ToAdd = minval - np.min(SpecDic[order]['flux'][NegativeIndices])
                logging.info('Adding: {0} with {1}'.format(SpecDic[order]['flux'][NegativeIndices],
                                                           ToAdd))
                SpecDic[order]['flux'] += ToAdd

            # Also update the fluxVar
            Gain = SpecDic.header['GAIN'] # e/adu 
            SpecDic[order]['fluxVar'] = SpecDic[order]['flux']*Gain/Gain**2


def ApplyFilter(SpecDic,Filter,newcopy=True):
    """ Applies input filter in all orders of Spectrum in the SpecDic """

    if newcopy:
        SpecDicFiltered = copy.deepcopy(SpecDic) 
    else:
        SpecDicFiltered = SpecDic

    SigalConvFunc = partial(signal.convolve, in2=Filter, mode='same')

    pool = Pool()
    FluxList = ( SpecDic[order]['flux'] for order in SpecDic )
        
    FilteredFluxList = pool.map(SigalConvFunc,FluxList)
    pool.close()

    for order,FiltFlux in zip(SpecDicFiltered,FilteredFluxList):
        SpecDicFiltered[order]['flux'] = FiltFlux

    return SpecDicFiltered


def scale_spectrum(SpecDic,scalefunc = None, ignoreTmask=True, newcopy=True):
    """ Scales the flux in spectrum dictionary by dividing with scalefunc(flux)
    Input: 
        SpecDic : The MultiOrderSpectrum dictionary continaing spectrum
        scalefunc: function which takes in flux array and outputs the scaling value (default 1/np.median)
        ignoreTmask: (default True) Include all regions and ignore Tmask while calculating scalefunc
        newcopy: Whether to create a new copy of the spectrum (True, default), or update the input spectrum (False)
    Output:
        ScaledSpec : Scale output spectrum. ie. Input Spectrum Flux * scalefunc(Flux)
    """
    if newcopy:
        ScaledSpec = copy.deepcopy(SpecDic) 
    else:
        ScaledSpec = SpecDic

    if scalefunc is None:
        scalefunc = lambda x: 1.0/np.median(x)

    for order in ScaledSpec:
        if ignoreTmask:
            scalevalue = scalefunc(ScaledSpec[order]['flux'])
        else:
            TM = ~ ScaledSpec[order]['Tmask']
            scalevalue = scalefunc(ScaledSpec[order]['flux'][TM])

        ScaledSpec[order]['flux'] = ScaledSpec[order]['flux'] * scalevalue
        ScaledSpec[order]['fluxVar'] = ScaledSpec[order]['fluxVar'] * scalevalue**2
        try:
            ScaledSpec[order]['scale'] *= scalevalue
        except KeyError:
            ScaledSpec[order]['scale'] = scalevalue

    return ScaledSpec


class SpectralRVtransformer(object):
    """ Object which has methods to transform input spectrum by radial velocities 
    or polynomial conitnuum
    Methods inside are useful for scipy.optimise.curv_fit as well. """
    def __init__(self,SpectrumXY,V_bary=0,V_star=0,interpolator=None):
        """
        SpectrumXY : Spectrum to fit transform
        V_bary : Barycentric Radial Velocity to be added to RV fit
        V_star : Zeroth order stellar radial velocity to be added to RV fit
        interpolator: interpolator to use to interpolating input Spectrum to dopler shifted output spectrum
        """
        self.c = 299792458.  # Speed of light in m/s
        self.SpectrumXY = SpectrumXY
        self.V_bary = V_bary
        self.V_star = V_star
        if interpolator is None:
            self.interp = BandLimitedInterpolator(kaiserB=13)
        else:
            self.interp = interpolator
        
        self.wranges = [] # wavelength ranges to fit RV seperately
        self.polycoeffs = [1,0] # polynomial coefficients of fitted continuum correction

    def dopplerfactor(self,starvelocity=None,baryvelocity=None):
        """ Returns the doppler factor for input velocity """
        if baryvelocity is None:
            baryvelocity = self.V_bary
        if starvelocity is None:
            starvelocity = self.V_star
        return (1 + starvelocity/self.c)/(1 + baryvelocity/self.c) # Formula for 1+z_mes (Wright & Eastman 2014)

    def multiply_poly_continuum(self,X,*params,**kwargs):
        """ Returns the Ydata for input Xdata and params after multiplying with the polynomial
        defined by the input params.
        Allowed kargs:
            fluxkey: the keyword in SpectrumXY dictionary to interpolate (default: flux)
        """
        fluxkey = kwargs.get('fluxkey', 'flux') # Default is 'flux'
        # print('Fitting only continuum with polynomial order ={0}'.format(len(params)))
        doppler_factor = self.dopplerfactor()
        PolynomialScale = np.polynomial.polynomial.polyval(X, params)
        SpectrumIntpFlux = self.interp.interpolate(X,
                                                   self.SpectrumXY['wavel'] * doppler_factor, 
                                                   self.SpectrumXY[fluxkey])
        # self.polycoeffs = params  # update the internal variable with latest call
        return SpectrumIntpFlux*PolynomialScale
    def apply_rv_redshift(self,X,*params,**kwargs):
        """ Returns the Ydata for input Xdata and params after applying the stellar radial velocity redshift.
        If the input params are a list of rv, user should make sure self.wranges 
            are correspondingly defined to apply different rv to corrseponding wavelength ranges.
        Allowed kargs:
            fluxkey: the keyword in SpectrumXY dictionary to interpolate (default: flux)
            remove_rv : (default False) True will remove rv and baryrv from the spectrum instead of applying them.
                        Note: This is not same as -ve velocity in relativity correciton
        """
        fluxkey = kwargs.get('fluxkey', 'flux') # Default is 'flux'
        remove_rv = kwargs.get('remove_rv', False) # Default is False

        doppler_factors = [self.dopplerfactor(starvelocity=self.V_star+v) for v in params]
        segmentedX = []
        if len(self.wranges) != 0:
            for ws,we in zip(self.wranges[:-1],self.wranges[1:]):
                segmentedX.append(X[NearestIndex(X,ws):NearestIndex(X,we)])
            # Last index need to be +1 to get the full array
            segmentedX[-1] = X[NearestIndex(X,ws):NearestIndex(X,we)+1]
        else:
            segmentedX = [X]

        if len(doppler_factors) != len(segmentedX):
            logging.warning('No: of wavelgnth ranges ({0}) does not match '
                            'no: of rvs ({1})'.format(len(segmentedX),len(doppler_factors)))

        OutputY = []
        for segX,dpf in zip(segmentedX,doppler_factors):
            PolynomialScale = np.polynomial.polynomial.polyval(segX, self.polycoeffs)
            if not remove_rv:
                SpectrumIntpFlux = self.interp.interpolate(segX,
                                                           self.SpectrumXY['wavel'] * dpf, 
                                                           self.SpectrumXY[fluxkey])
            else: #Remove th rv instead by dividing the doppler factor
                SpectrumIntpFlux = self.interp.interpolate(segX,
                                                           self.SpectrumXY['wavel'] / dpf, 
                                                           self.SpectrumXY[fluxkey])
                
            OutputY.append(SpectrumIntpFlux*PolynomialScale)
        return np.concatenate(OutputY)
            
