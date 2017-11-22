#!/usr/bin/env python
""" This contains miscellaneous utility functions for the module """
import os
import numpy as np
import copy
from astropy.io import fits 
from scipy import signal

import logging
from multiprocessing.pool import Pool
from functools32 import wraps, partial

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

def read_HARPS_spectrum(fitsfile,order_list=None):
    """ Loads HARPS 1D spectrum into a dictionary
    Input: 
          fitsfile: Fits file name of the HARPS 1D spectrum
          order_list: list of orders to return from the file. (default is all orders)
    Output:
          SpecDic = MultiOrderSpectrum Dictionary object containing fits header in .header
                    and each order data in sub dictionaries 
                    of the format- order : {'Obsflux':[flux array], 'Obswavel':[wavelength array],
                                            'flux':[flux array], 'wavel':[wavelength array]} 
    """

    SpecDic = MultiOrderSpectrum()
    logging.info('Loading HARPS spectrum: {0}'.format(fitsfile))
    with fits.open(fitsfile) as hdulist:
        SpecDic.header = hdulist[0].header
        SpecDic.header['FITSFILE'] =(fitsfile,'Raw Fits filename')
        SpecDic.header['GAIN'] =(1.0 ,'Gain e/ADU TO BE FIXED')
        # Now read out the order data
        if order_list is None:
            order_list = range(SpecDic.header['NAXIS2']) # all orders in the input fits file
        
        for order in order_list:
            SpecDic[order] = {'Rawflux':hdulist[0].data[order,:]}
            # Now, calculate the wavelength array
            polydeg = SpecDic.header['HIERARCH ESO DRS CAL TH DEG LL']
            c = [SpecDic.header['HIERARCH ESO DRS CAL TH COEFF LL{0}'.format((polydeg+1)*order +i)] for i in range(polydeg+1)]
            SpecDic[order]['wavel'] = np.polynomial.polynomial.polyval(range(len(SpecDic[order]['Rawflux'])), c)
            # Also, Initialise flux and wavel key with the same Observed raw values
            SpecDic[order]['flux'] = copy.deepcopy(SpecDic[order]['Rawflux'])
            # Initialise a flux Varience error by assuming poisson error 
            #IMP: Scale with Gain to obtain real noise
            Gain = SpecDic.header['GAIN'] # e/adu   # TO BE FIXED later for HARPS
            SpecDic[order]['fluxVar'] = SpecDic[order]['Rawflux']*Gain/Gain**2

    return SpecDic


def read_HPF_spectrum(fitsfile,order_list=None):
    """ Loads HPF 1D spectrum into a dictionary
    Input: 
          fitsfile: Fits file name of the HPF 1D spectrum
          order_list: list of orders to return from the file. (default is all orders)
    Output:
          SpecDic = MultiOrderSpectrum Dictionary object containing fits header in .header
                    and each order data in sub dictionaries 
                    of the format- order : {'Obsflux':[flux array], 'Obswavel':[wavelength array],
                                            'flux':[flux array], 'wavel':[wavelength array]} 
    """

    SpecDic = MultiOrderSpectrum()
    logging.info('Loading HPF spectrum: {0}'.format(fitsfile))
    with fits.open(fitsfile) as hdulist:
        SpecDic.header = hdulist[0].header
        SpecDic.header['FITSFILE'] =(fitsfile,'Raw Fits filename')
        SpecDic.header['GAIN'] =(1.0 ,'Gain e/ADU')
        # Now read out the order data
        if order_list is None:
            order_list = range(SpecDic.header['NAXIS2']) # all orders in the input fits file
        
        for order in order_list:
            SpecDic[order] = {'Rawflux':hdulist[0].data[order,:]}
            # Now, Load the wavelength array from third extension
            SpecDic[order]['wavel'] = hdulist[2].data[order,:]
            # Also, Initialise flux and wavel key with the same Observed raw values
            SpecDic[order]['flux'] = copy.deepcopy(SpecDic[order]['Rawflux'])
            # Load the Varience estimate from second extension 
            SpecDic[order]['fluxVar'] = hdulist[1].data[order,:]

    return SpecDic



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
