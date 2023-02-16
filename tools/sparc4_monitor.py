# -*- coding: iso-8859-1 -*-
"""
    Created on Jul 25 2022
    
    Description: Tool to monitor photometry in the 4 channels of SPARC4
    
    @author: Eder Martioli <martioli@lna.br>
    
    Laboratório Nacional de Astrofísica - LNA/MCTI
    
    Simple usage example:

    python /Volumes/Samsung_T5/sparc4-pipeline/tools/sparc4_monitor.py --datadir=


    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import os,sys
from optparse import OptionParser

import astropy.io.fits as fits
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation

#sparc4_pipeline_dir = os.path.dirname(__file__)

parser = OptionParser()
parser.add_option("-d", "--datadir", dest="datadir", help="data directory",type='string',default="./")
parser.add_option("-i", "--pattern", dest="pattern", help="data pattern",type='string',default="")
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h sparc4_monitor.py")
    sys.exit(1)

data_dir = options.datadir

filters = ["B","V","R","I"]

object_suffix = 'aumic'

# Set OPD geographic coordinates
longitude = -(45 + (34 + (57/60))/60) #hdr['OBSLONG']
latitude = -(22 + (32 + (4/60))/60) #hdr['OBSLAT']
altitude = 1864*u.m #hdr['OBSALT']
opd_location = EarthLocation.from_geodetic(lat=latitude, lon=longitude, height=altitude)

inputdata = {}
maxflux, time = {}, {}
for i in range(len(filters)) :
    if options.pattern == "":
        inputdata[filters[i]] = glob.glob("{}/*{}_{}.fits".format(data_dir,object_suffix,filters[i]))
    else :
            inputdata[filters[i]] = glob.glob("{}_{}.fits".format(options.pattern,filters[i]))

    maxflux[filters[i]] = np.array([])
    time[filters[i]] = np.array([])
    
    for j in range(len(inputdata[filters[i]])) :
        try :
            hdu = fits.open(inputdata[filters[i]][j])
            max = np.nanmax(hdu[0].data[0])
            date = hdu[0].header["DATE"]
            obstime = Time(date, format='isot', scale='utc', location=opd_location)
            jd = obstime.jd
        except :
            continue
        time[filters[i]] = np.append(time[filters[i]],obstime.datetime)
        maxflux[filters[i]] = np.append(maxflux[filters[i]],max)
        
    plt.plot(time[filters[i]],maxflux[filters[i]],"o", label="Filter {}".format(filters[i]))

plt.xlabel("Time")
plt.ylabel("Max Flux (ADU)")
plt.legend()
plt.show()
