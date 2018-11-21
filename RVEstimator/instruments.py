#!/usr/bin/env python
""" This contains functions related to specific instruments"""
import logging
from astropy.io import fits, ascii
from astropy.stats import biweight_location
import numpy as np
import copy
from .utils import MultiOrderSpectrum
from .interpolators import BSplineInterpolator, BandLimitedInterpolator, remove_nans
import scipy.interpolate as interp
import cPickle as pickle

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


def read_HPF_spectrum(fitsfile,order_list=None,fiber='sci'):
    """ Load HPF wavelength calibrated extracted spectrum
    Input: 
          fitsfile: Fits file name of the HPF e2ds 1D spectrum
          order_list: list of orders to return from the file. (default is all orders)
          fiber: (sci|sky|cal) [default: sci] The fiber to load
    Output:
          SpecDic = MultiOrderSpectrum Dictionary object containing fits header in .header
                    and each order data in sub dictionaries 
                    of the format- order : {'Obsflux':[flux array], 'Obswavel':[wavelength array],
                                            'flux':[flux array], 'wavel':[wavelength array]} 
"""
    ext_dic = {'sci':1,'sky':2,'cal':3,'varsci':4,'varsky':5,'varcal':6,'wavel':7}  # Fits file extention dictionary
    SpecDic = MultiOrderSpectrum()
    logging.info('Loading HPF spectrum: {0}'.format(fitsfile))
    with fits.open(fitsfile) as hdulist:
        SpecDic.header = hdulist[0].header
        SpecDic.header['FITSFILE'] =(fitsfile,'Raw Fits filename')
        SpecDic.header['GAIN'] =(1.0 ,'Gain e/ADU')
        if fiber == 'cal':
            SpecDic.header['BRYV'] = (0.0, 'Zero Barycor vel for cal fiber')
        # Now read out the order data
        
        if order_list is None:
            order_list = range(hdulist[ext_dic[fiber]].header['NAXIS2']) # all orders in the input fits file
        
        for order in order_list:
            SpecDic[order] = {'Rawflux':hdulist[ext_dic[fiber]].data[order,4:-4]}
            # Now, Load the wavelength array from third extension
            SpecDic[order]['wavel'] = hdulist[ext_dic['wavel']].data[order,4:-4]
            # Also, Initialise flux and wavel key with the same Observed raw values
            SpecDic[order]['flux'] = copy.deepcopy(SpecDic[order]['Rawflux'])
            # Load the Varience estimate from second extension 
            SpecDic[order]['fluxVar'] = hdulist[ext_dic['var'+fiber]].data[order,4:-4]

    return SpecDic


# def dropnans(narray):
#     return remove_nans(narray)[0]


