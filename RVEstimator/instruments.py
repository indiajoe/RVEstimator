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


# def dropnans(narray):
#     return remove_nans(narray)[0]

def hpf_sky_model(skyfiberdata,orderlist,scifiberdataToScale=None):
    """ Returns the transformed sky fiber data for subtracting from the sci fiber. If scifiberdataToScale is provided the model is scaled to minimise residue from subtraction"""
    ### TODO: Write an improved profile shifting code taking care of sampling issue for wavelength offset
    ### This code is valid only if the instrumental drift is negligible for sky correction
    # Used only for shifting sky spectrum to Sci fiber spectrum                        
    WavlArrayHR_sky = fits.getdata('/home/joe/Downloads/LFC_wavecal_A_v1.fits')[orderlist,4:-4]
    WavlArrayHR_sci = fits.getdata('/home/joe/Downloads/LFC_wavecal_B_v1.fits')[orderlist,4:-4]
    Interpolator = BSplineInterpolator() # BandLimitedInterpolator(kaiserB=13) Slow
    skyFlux = np.vstack([Interpolator.interpolate(WavlArrayHR_sciord, # New X coords 
                                                  WavlArrayHR_skyord, # Old X coords 
                                                  skyfiberdataord) for WavlArrayHR_sciord,WavlArrayHR_skyord,skyfiberdataord in zip(WavlArrayHR_sci,WavlArrayHR_sky,skyfiberdata)])
    skymodeltosubtract = skyFlux

    if scifiberdataToScale is not None:
        SkyMask = np.loadtxt('/data/joe/joe_home/joe/Downloads/HPF_SkyEmmissionLineWavlMask_broadened_11111_Compressed.txt')
        SkyMaskFunction = interp.interp1d(SkyMask[:,0],SkyMask[:,1],kind='nearest',fill_value='extrapolate')
        WavlArrayHR_sci_SkyMask = SkyMaskFunction(WavlArrayHR_sci) > 0.5
        # Also Mask all the nan in the image
        WavlArrayHR_sci_SkyMask = WavlArrayHR_sci_SkyMask | np.isnan(scifiberdataToScale)
        # # See derivation in the google slide https://docs.google.com/presentation/d/1khM-tik6beQMdp5Og5zO-slFpEAG9KuW2HRLf1gMrhQ/edit?usp=sharing
        # InterpSciminusSky = np.vstack([interp.interp1d(WavlArrayHR_sciord[~WavlArrayHR_sci_SkyMaskord],
        #                                                (scifiberdataToScaleord-skyFluxord)[~WavlArrayHR_sci_SkyMaskord],
        #                                                kind='linear',
        #                                                fill_value="extrapolate")(WavlArrayHR_sciord) for scifiberdataToScaleord,
        #                                skyFluxord,WavlArrayHR_sciord,
        #                                WavlArrayHR_sci_SkyMaskord in zip(scifiberdataToScale,skyFlux,WavlArrayHR_sci,
        #                                                                  WavlArrayHR_sci_SkyMask)])

        InterpSky = np.vstack([interp.interp1d(WavlArrayHR_sciord[~WavlArrayHR_sci_SkyMaskord],
                                               skyFluxord[~WavlArrayHR_sci_SkyMaskord],
                                               kind='linear',bounds_error=False,
                                               fill_value=(np.nanmedian(skyFluxord[~WavlArrayHR_sci_SkyMaskord][:20]),
                                                           np.nanmedian(skyFluxord[~WavlArrayHR_sci_SkyMaskord][:-20])))(WavlArrayHR_sciord) for WavlArrayHR_sciord,
                               WavlArrayHR_sci_SkyMaskord,skyFluxord in zip(WavlArrayHR_sci,
                                                                            WavlArrayHR_sci_SkyMask,skyFlux)])
        # Commented out since it is unstable for any data with star light in it
        # # Calculate the weights for robust average estimation # biweight_location doesnot work since most of the data is too noisy
        # cleanratio, removedmask = remove_nans((((scifiberdataToScale-skyFlux)-InterpSciminusSky)/(skyFlux-InterpSky))[WavlArrayHR_sci_SkyMask])
        # # Lets use weights based on the sky line flux
        # weights = ((skyFlux-InterpSky)[WavlArrayHR_sci_SkyMask])[~removedmask]
        # # For robustness, lets set less than 10 percentil data weights to zero
        # weights[weights<np.percentile(weights,50)] = 0
        # weights[weights>np.percentile(weights,99.99)] = np.percentile(weights,99.99)  # Also prevent any single bright lines from dominating
        # # invscale = 1 + biweight_location(dropnans((((scifiberdataToScale-skyFlux)-InterpSciminusSky)/(skyFlux-InterpSky))[WavlArrayHR_sci_SkyMask]))
        # invscale = 1 + np.average(cleanratio,weights=weights)
        # logging.info('Sky scaling calculated: Sci = Sky/{0}'.format(1/invscale))
        # invscale = 0.9516
        ############ Load the scale from the twilight ratio data
        PickledRatioFile = '/data/joe/HPFdata/SkyRatiowavTCKDic_Twilight_Slope-20181005T003953_R01.optimal.fits.pkl'
        SkyRatiowavTCKDic = pickle.load(open(PickledRatioFile, 'rb'))
        invscale = np.vstack([interp.splev(w,SkyRatiowavTCKDic[o]) for w,o in zip(WavlArrayHR_sci,orderlist)])
        # Scale the sky flux
        skymodeltosubtract = InterpSky + (skyFlux -InterpSky)*invscale

    return skymodeltosubtract

