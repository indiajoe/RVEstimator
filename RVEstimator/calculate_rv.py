#!/usr/bin/env python
""" This tool is for calculating the radial velocity of input spectrum """
import sys
import pickle
import logging
import argparse
import ConfigParser
import numpy as np
import pandas as pd
from functools32 import wraps, partial
from multiprocessing.pool import Pool
import traceback
from astropy.stats import mad_std, median_absolute_deviation
from .interpolators import BSplineInterpolator, BandLimitedInterpolator
from .rve_methods import FitRVTemplateAdaptively, CreateTemplateFromSpectra
from .utils import LoadSpectraFromFilelist, scale_spectrum, CleanNegativeValues, CleanNanValues
# from .utils import unwrap_args_forfunction
###############################################
def pack_traceback_to_errormsg(func):
    """Decorator which packes any raised error traceback to its msg 
    This is useful to pack a child process function call while using multiprocess """
    @wraps(func)
    def wrappedFunc(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = "{}\nOriginal {}".format(e, traceback.format_exc())
            raise type(e)(msg)
    return wrappedFunc

def log_all_uncaughtexceptions_handler(exp_type, exp_value, exp_traceback):
    """ This handler is to override sys.excepthook to log uncaught exceptions """
    logging.error("Uncaught exception", exc_info=(exp_type, exp_value, exp_traceback))
    # call the original sys.excepthook
    sys.__excepthook__(exp_type, exp_value, exp_traceback)

#####


@pack_traceback_to_errormsg
def FitRVTemplateAdaptively_tupleargs(argtuple):
    return FitRVTemplateAdaptively(*argtuple)

def CalculateRV_bySegLSQ(SpecDic,Template,Config,noCPUs=1):
    """ Calculates RV of the spectrum by segemented LSQ """
    # Cleaning data of any negative values
    CleanNegativeValues(SpecDic,minval=Config['MinValue'],method=Config['NegativeValues'])
    # Cleaning data of any nan values
    CleanNanValues(SpecDic)

    #Normalise the spectrum (since Template is normalised)
    NSpectrum = scale_spectrum(SpecDic,
                               scalefunc = lambda x: 1/np.median(x),
                               ignoreTmask=True)

    if Config['Interpolation'] == 'BSpline': # Fast
        interpolator = BSplineInterpolator()
    elif Config['Interpolation'] == 'BandLimited': #Slow
        interpolator = BandLimitedInterpolator(kaiserB=13)

    OrdersToAnalyse = []
    # Parse string of format 25:60,63,64,70:90
    for strindx in Config['OrdersToUse'].split(','):
        if ':' in strindx:
            OrdersToAnalyse.extend(range(int(strindx.split(':')[0]),
                                         int(strindx.split(':')[1]) + 1))
        else:
            OrdersToAnalyse.append(int(strindx))

    logging.info('Orders being analysed: {0}'.format(OrdersToAnalyse))

    # Barycentric velocity from header for each order 
    # If there is no format string in the header key, by default same velocity will be used for all orders
    BaryRV = {}
    BJDdic = {}
    for order in OrdersToAnalyse:
        BaryRV[order] = NSpectrum.header[Config['BaryRVHeaderKey'].format(order)]
        BJDdic[order] = NSpectrum.header[Config['TimeHeaderKey'].format(order)]

    InputArgsTuples = ((Template[order],NSpectrum[order],
                        BaryRV[order],0,interpolator,
                        Config['TrimSize'],Config['minsize']) for order in OrdersToAnalyse)
    
    # Run sequentially useful for debugging
    if noCPUs == 1:
        Results = map(FitRVTemplateAdaptively_tupleargs,InputArgsTuples)
    else: # Run parallel
        pool = Pool(processes=noCPUs)
        Results = pool.map(FitRVTemplateAdaptively_tupleargs,InputArgsTuples)
        pool.close()
    # Sigma clip each order and save to a long list
    RVlist = []
    RVerrorlist = []
    Wavlist = []
    for rvopt,rvcov,wranges in Results:
        rvstd = mad_std(rvopt)
        rvmed = np.median(rvopt)
        GoodDataMask = np.abs(np.array(rvopt) - rvmed) < Config['SigmaClipRV']*rvstd
        RVlist.extend(np.array(rvopt)[GoodDataMask])
        RVerrorlist.extend(np.sqrt(np.diag(rvcov))[GoodDataMask]) # sigma
        Wavlist.extend(np.mean([wranges[:-1],wranges[1:]],axis=0)[GoodDataMask]) # mean central wavelength
                
    inverse_varience_weight = 1/np.array(RVerrorlist)**2
    AverageRV = np.average(RVlist,weights=inverse_varience_weight)
    AverageRV_err = np.sqrt(1/np.sum(inverse_varience_weight))
    BJD = np.average(BJDdic.values())  #SpecDic.header[Config['TimeHeaderKey']]
    return AverageRV, AverageRV_err, BJD


###############################################

def parse_str_to_types(string):
    """ Converts string to different object types they represent """
    if string == 'True':
        return True
    elif string == 'False':
        return False
    elif string == 'None':
        return None
    elif string.lstrip('-+ ').isdigit():
        return int(string)
    else:
        try:
            return float(string)
        except ValueError:
            return string

def create_configdict_from_file(configfilename):
    """ Returns a configuration object by loading the config file """
    Configloader = ConfigParser.SafeConfigParser()
    Configloader.optionxform = str  # preserve the Case sensitivity of keys
    Configloader.read(configfilename)
    # Create a Config Dictionary
    Config = {}
    for key,value in Configloader.items('rv_settings'):
        Config[key] = parse_str_to_types(value)
    return Config

def parse_args():
    """ Parses the command line input arguments for the RV calculation tool"""
    parser = argparse.ArgumentParser(description="Script to Calculate RV values")
    parser.add_argument('InputSpectra', type=str,
                        help="Input Spectra to calculate RV (fits file or list file)")
    parser.add_argument('TemplateToMatch', type=str,
                help="Template Spectrum or Mask to match Input Spectrum. \nIf the file does not exist it is created by combining all the spectra in InputSpectra for Least Square methods.")
    parser.add_argument('ConfigFile', type=str,
                help="Configuration File which contains settings for RV calculation")
    parser.add_argument('OutputTable', type=str,
                help="Output Table File for RV values")
    parser.add_argument('--SecularAcc', type=float, default=0.0,
                help="Secular acceleration of star in m/sec/year")
    parser.add_argument('--noCPUs', type=int, default=1,
                help="Number of parallel CPUs to be used to calculate RV in parallel")
    parser.add_argument('--logfile', type=str, default=None,
                help="Log Filename to write logs during the run")
    parser.add_argument("--loglevel", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help="Set the logging level")

    args = parser.parse_args()
    return args

def main():
    """ Standalone Script to calculate RV of an input spectrum"""
    # Override the default exception hook with our custom handler
    sys.excepthook = log_all_uncaughtexceptions_handler

    args = parse_args()
    if args.logfile is None:
        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                        level=logging.getLevelName(args.loglevel))
    else:
        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                            level=logging.getLevelName(args.loglevel),
                            filename=args.logfile, filemode='a')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout)) # Sent info to the stdout as well            

        logging.info('Analysing {0}'.format(args.InputSpectra))

    Config = create_configdict_from_file(args.ConfigFile)
    
    logging.info('RV Configuration: {0}'.format(Config))
    
    if Config['instrument'] == 'HARPS':
        from .instruments import read_HARPS_spectrum as fileloaderfunc
    elif Config['instrument'] == 'HPF':
        from .instruments import read_HPF_spectrum as fileloaderfunc
        
    ListOfSpec = LoadSpectraFromFilelist(args.InputSpectra, fileloaderfunc)

    logging.info('No of Spectra: {0}'.format(len(ListOfSpec)))

    if Config['method'] == 'SegmentedLSQ':
        logging.info('Calculating RV by Segmented LSQ method')
        try:
            Template = pickle.load(open(args.TemplateToMatch,'rb'))
        except IOError:
            logging.info('Creating Template from input spectra')
            map(CleanNanValues,ListOfSpec) # Clean any Nan values
            Template = CreateTemplateFromSpectra(ListOfSpec,BaryVShift=Config['BaryRVHeaderKey'],
                                                 interpolator = BSplineInterpolator(order_k=1),
                                                 cmethod='median')
            logging.info('Pickling New Template to {0}'.format(args.TemplateToMatch))
            pickle.dump(Template,open(args.TemplateToMatch,'wb'))

        CalculateRV = partial(CalculateRV_bySegLSQ,
                              Template=Template,Config=Config,noCPUs=args.noCPUs)
    else:
        logging.critical('Method :{0} Not Implemented yet'.format(Config['method']))
        raise NotImplementedError('Method :{0} Not Implemented yet'.format(Config['method']))

    #################################################################
    # We shall analyse each spectrum sequentially to keep logs simple
    # orders can be anlaysied in parallel inside the function
    OutputTable = pd.DataFrame(columns=('BJD','RV','RVerr'),index=np.arange(0, len(ListOfSpec)))
    for i,SpecDic in enumerate(ListOfSpec):
        logging.info('Analysing {0}'.format(SpecDic.header['FITSFILE']))
        # write the RV to pandas table 
        rv,rverror,bjd = CalculateRV(SpecDic)
        OutputTable.loc[i] = [bjd, rv, rverror]
    # write pandas table as output
    logging.info('Writing RV results to file://{0}'.format(args.OutputTable))
    OutputTable.to_csv(args.OutputTable,index=False)


if __name__ == '__main__':
    main()
