# -*- coding: iso-8859-1 -*-
"""
    Created on Jun 7 2022

    Description: Library for plotting SPARC4 pipeline products
    
    @author: Eder Martioli <martioli@iap.fr>
    
    Laboratório Nacional de Astrofísica - LNA/MCTI
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import Angle, SkyCoord
from regions import CircleSkyRegion

import astropy.io.fits as fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy import units as u

import warnings
from copy import deepcopy

from astropop.math.physical import QFloat
from astropop.polarimetry import quarterwave_model, halfwave_model

from astropy.stats import sigma_clip


def plot_cal_frame(filename, output="", percentile=99.5, xcut=512, ycut=512, combine_rows=False, combine_cols=False, combine_method="mean") :

    """ Plot calibration (bias, flat) frame
    
    Parameters
    ----------
    filename : str
        string for fits file path
            
    output : str, optional
        The output plot file name to save graphic to file. If empty, it won't be saved.

    Returns
    -------
    None
    """

    hdul = fits.open(filename)

    img_data = hdul["PRIMARY"].data[0]
    err_data = hdul["PRIMARY"].data[1]
    #mask_data = hdul["PRIMARY"].data[2]

    img_mean = QFloat(np.mean(img_data),np.std(img_data))
    noise_mean = QFloat(np.mean(err_data),np.std(err_data))

    # plot best polarimetry results
    fig, axes = plt.subplots(3, 2, figsize=(16, 8), sharex=False, sharey=False, gridspec_kw={'hspace': 0.5, 'height_ratios': [4, 1, 1]})

    axes[0,0].set_title("image: mean:{}".format(img_mean))
    axes[0,0].imshow(img_data, vmin=np.percentile(img_data,100 - percentile), vmax=np.percentile(img_data, percentile), origin='lower')
    axes[0,0].set_xlabel("columns (pixel)", fontsize=16)
    axes[0,0].set_ylabel("rows (pixel)", fontsize=16)
    
    xsize, ysize = np.shape(img_data)
    x, y = np.arange(xsize), np.arange(ysize)
    
    if combine_rows :
        crow = None
        if combine_method == "mean" :
            crow = np.nanmean(img_data,axis=0)
        elif combine_method == "median" :
            crow = np.nanmedian(img_data,axis=0)
        else :
            print("ERROR: combine method must be mean or median, exiting ...")
            exit()
        axes[1,0].plot(x, crow)
        axes[1,0].set_ylabel("{} flux".format(combine_method), fontsize=16)
    else :
        axes[1,0].plot(x, img_data[ycut,:])
        axes[1,0].set_ylabel("flux".format(ycut), fontsize=16)
    axes[1,0].set_xlabel("columns (pixel)", fontsize=16)
    
    if combine_cols :
        ccol = None
        if combine_method == "mean" :
            ccol = np.nanmean(img_data,axis=1)
        elif combine_method == "median" :
            ccol = np.nanmedian(img_data,axis=1)
        else :
            print("ERROR: combine method must be mean or median, exiting ...")
            exit()
        axes[2,0].plot(y, ccol)
        axes[2,0].set_ylabel("{} flux".format(combine_method), fontsize=16)
    else :
        axes[2,0].plot(y, img_data[:,xcut])
        axes[2,0].set_ylabel("flux".format(xcut), fontsize=16)
    axes[2,0].set_xlabel("rows (pixel)", fontsize=16)

    axes[0,1].set_title("noise: mean:{}".format(noise_mean))
    axes[0,1].imshow(err_data, vmin=np.percentile(err_data,100 - percentile), vmax=np.percentile(err_data, percentile), origin='lower')
    axes[0,1].set_xlabel("columns (pixel)", fontsize=16)
    axes[0,1].set_ylabel("rows (pixel)", fontsize=16)

    axes[1,1].plot(x, err_data[ycut,:])
    axes[1,1].set_ylabel(r"$\sigma$".format(ycut), fontsize=16)
    axes[1,1].set_xlabel("columns (pixel)", fontsize=16)
    
    axes[2,1].plot(y, err_data[:,xcut])
    axes[2,1].set_ylabel(r"$\sigma$".format(xcut), fontsize=16)
    axes[2,1].set_xlabel("rows (pixel)", fontsize=16)

    plt.show()



def plot_sci_frame(filename, cat_ext=9, nstars=5, output="", percentile=98, use_sky_coords=False) :

    """ Plot science frame
    
    Parameters
    ----------
    filename : str
        string for fits file path
            
    output : str, optional
        The output plot file name to save graphic to file. If empty, it won't be saved.

    Returns
    -------
    None
    """

    hdul = fits.open(filename)
    img_data = hdul["PRIMARY"].data
    #img_data = hdul["PRIMARY"].data[0]
    #err_data = hdul["PRIMARY"].data[1]
    #mask_data = hdul["PRIMARY"].data[2]

    x, y = hdul[cat_ext].data['x'], hdul[cat_ext].data['y']
    mean_aper = np.mean(hdul[cat_ext].data['APER'])

    if nstars > len(x) :
        nstars = len(x)
        
    fig = plt.figure(figsize=(10, 10))

    if use_sky_coords :
        # load WCS from image header
        wcs_obj = WCS(hdul[0].header,naxis=2)
        
        # calculate pixel scale
        pixel_scale = proj_plane_pixel_scales(wcs_obj)

        # assume  the N-S component of the pixel scale
        pixel_scale = (pixel_scale[1] * 3600)
        
        ax = plt.subplot(projection=wcs_obj)
        ax.imshow(img_data, vmin=np.percentile(img_data, 100. - percentile), vmax=np.percentile(img_data, percentile), origin='lower', cmap='cividis', aspect='equal')

        world = wcs_obj.pixel_to_world(x, y)
        wx, wy = world.ra.degree, world.dec.degree

        for i in range(len(wx)) :
            sky_center = SkyCoord(wx[i], wy[i], unit='deg')
            sky_radius = Angle(mean_aper*pixel_scale, 'arcsec')
            sky_region = CircleSkyRegion(sky_center, sky_radius)
            pixel_region = sky_region.to_pixel(wcs_obj)
            pixel_region.plot(ax=ax, color='white', lw=2.0)
            if i < nstars :
                text_offset = 0.004
                plt.text(wx[i]+text_offset, wy[i], '{}'.format(i), c='darkred', weight='bold', fontsize=18, transform=ax.get_transform('icrs'))

        plt.xlabel(r'RA')
        plt.ylabel(r'Dec')
        overlay = ax.get_coords_overlay('icrs')
        overlay.grid(color='white', ls='dotted')
    
    else :
    
        plt.plot(x, y, 'wo', ms=mean_aper, fillstyle='none', lw=1.5, alpha=0.7)
        plt.plot(x[:nstars], y[:nstars], 'k+', ms=2*mean_aper/3, lw=1.0, alpha=0.7)
        for i in range(nstars) :
            plt.text(x[i]+1.1*mean_aper, y[i], '{}'.format(i), c='darkred', weight='bold', fontsize=18)
        plt.imshow(img_data, vmin=np.percentile(img_data, 100. - percentile), vmax=np.percentile(img_data, percentile), origin='lower')
        plt.xlabel("columns (pixel)", fontsize=16)
        plt.ylabel("rows (pixel)", fontsize=16)
    
    
    plt.show()



def plot_sci_polar_frame(filename, percentile=99.5) :
    """ Plot science polar frame
    
    Parameters
    ----------
    filename : str
        string for fits file path
            
    output : str, optional
        The output plot file name to save graphic to file. If empty, it won't be saved.

    Returns
    -------
    None
    """
    hdul = fits.open(filename)
    img_data = hdul["PRIMARY"].data
    #img_data = hdul["PRIMARY"].data[0]
    #err_data = hdul["PRIMARY"].data[1]
    #mask_data = hdul["PRIMARY"].data[2]

    x_o, y_o = hdul["CATALOG_POL_N_AP010"].data['x'], hdul["CATALOG_POL_N_AP010"].data['y']
    x_e, y_e = hdul["CATALOG_POL_S_AP010"].data['x'], hdul["CATALOG_POL_S_AP010"].data['y']
    
    mean_aper = np.mean(hdul["CATALOG_POL_N_AP010"].data['APER'])

    plt.figure(figsize=(10, 10))

    plt.plot(x_o, y_o, 'wo', ms=mean_aper, fillstyle='none', lw=1.5, alpha=0.7)
    plt.plot(x_e, y_e, 'wo', ms=mean_aper, fillstyle='none', lw=1.5, alpha=0.7)

    plt.imshow(img_data, vmin=np.percentile(img_data, 100-percentile), vmax=np.percentile(img_data, percentile), origin='lower')
    
    for i in range(len(x_o)):
        x = [x_o[i], x_e[i]]
        y = [y_o[i], y_e[i]]
        plt.plot(x, y, 'w-o', alpha=0.5)
        plt.annotate(f"{i}", [np.mean(x)-25, np.mean(y)+25], color='w')
        
    plt.xlabel("columns (pixel)", fontsize=16)
    plt.ylabel("rows (pixel)", fontsize=16)
    
    plt.show()


def plot_diff_light_curve(filename, target=0, comps=[], output="", nsig=100, plot_sum=True, plot_comps=False) :
        
    """ Plot light curve
    
    Parameters
    ----------
    filename : str
        string for fits file path
            
    output : str, optional
        The output plot file name to save graphic to file. If empty, it won't be saved.

    Returns
    -------
    None
    """

    hdul = fits.open(filename)
    # Below is a hack to avoid very high values of time in some bad data
    
    time = hdul["DIFFPHOTOMETRY"].data['TIME']
    
    mintime, maxtime = np.min(time), np.max(time)
    
    data = hdul["DIFFPHOTOMETRY"].data
    
    nstars = int((len(data.columns) - 1) / 2)
    
    warnings.filterwarnings('ignore')
    
    offset = 0.
    for i in range(nstars) :
        lc = -data['DMAG{:06d}'.format(i)][keeplowtimes]
        elc = data['EDMAG{:06d}'.format(i)][keeplowtimes]
        
        mlc = np.nanmedian(lc)
        rms = np.nanmedian(np.abs(lc-mlc)) / 0.67449
        #rms = np.nanstd(lc-mlc)
        
        keep = elc < nsig*rms
        
        if i == 0 :
            if plot_sum :
                comp_label = "SUM"
                plt.errorbar(time[keep], lc[keep], yerr=elc[keep], fmt='k.', alpha=0.8, label=r"{} $\Delta$mag={:.3f} $\sigma$={:.2f} mmag".format(comp_label, mlc, rms*1000))
                #plt.plot(time[keep], lc[keep], "k-", lw=0.5)
            
            offset = np.nanpercentile(lc[keep], 1.0) - 4.0*rms
            
            plt.hlines(offset, mintime, maxtime, colors='k', linestyle='-', lw=0.5)
            plt.hlines(offset-rms, mintime, maxtime, colors='k', linestyle='--', lw=0.5)
            plt.hlines(offset+rms, mintime, maxtime, colors='k', linestyle='--', lw=0.5)
        
        else :
            if plot_comps :
                comp_label = "C{:03d}".format(comps[i-1])
                plt.errorbar(time[keep], (lc[keep]-mlc)+offset, yerr=elc[keep], fmt='.', alpha=0.5, label=r"{} $\Delta$mag={:.3f} $\sigma$={:.2f} mmag".format(comp_label, mlc, rms*1000))
            
        #print(i, mlc, rms)

    plt.xlabel(r"time (BJD)", fontsize=16)
    plt.ylabel(r"$\Delta$mag", fontsize=16)
    plt.legend(fontsize=10)
    plt.show()


def plot_light_curve(filename, target=0, comps=[], output="", nsig=10, plot_coords=True, plot_rawmags=True, plot_sum=True, plot_comps=True, catalog_name="CATALOG_PHOT_AP008") :
        
    """ Plot light curve
    
    Parameters
    ----------
    filename : str
        string for fits file path
            
    output : str, optional
        The output plot file name to save graphic to file. If empty, it won't be saved.

    Returns
    -------
    None
    """

    hdul = fits.open(filename)
    # Below is a hack to avoid very high values of time in some bad data
    
    time = hdul["TIME_COORDS"].data['TIME']
    mintime, maxtime = np.min(time), np.max(time)

    use_sky_coords = True
    platescale, unit = 1.0, 'pix'
    if use_sky_coords :
        platescale, unit = 0.335, 'arcsec'

    pixoffset = 7*platescale

    x = hdul["TIME_COORDS"].data['X{:08d}'.format(target)]
    y = hdul["TIME_COORDS"].data['Y{:08d}'.format(target)]
    fwhm = hdul["TIME_COORDS"].data['FWHM{:08d}'.format(target)]
    if plot_coords :
        #fig, axs = plt.subplots(4, 1, figsize=(12, 6), sharex=True, sharey=False, gridspec_kw={'hspace': 0, 'height_ratios': [1, 1, 1, 1]})
        plt.plot(time, (x-np.nanmedian(x))*platescale+pixoffset, '.', color='darkblue', label='x-offset')
        plt.plot(time, (y-np.nanmedian(y))*platescale-pixoffset, '.', color='brown', label='y-offset')
        mfhwm =  np.nanmedian(fwhm)
        plt.plot(time, (fwhm-mfhwm)*platescale, color='darkgreen', label='FWHM - median={:.1f} {}'.format(mfhwm*platescale,unit))
        plt.xlabel(r"time (BJD)", fontsize=16)
        plt.ylabel(r"$\Delta$ {}".format(unit), fontsize=16)
        plt.legend(fontsize=10)
        plt.show()


    mag_offset=0.1
    time = hdul[catalog_name].data['TIME']
    mintime, maxtime = np.min(time), np.max(time)
    m = hdul[catalog_name].data['MAG{:08d}'.format(target)]
    em = hdul[catalog_name].data['EMAG{:08d}'.format(target)]
    skym = hdul[catalog_name].data['SKYMAG{:08d}'.format(target)]
    eskym = hdul[catalog_name].data['ESKYMAG{:08d}'.format(target)]
    mmag = np.nanmedian(m)
    mskym = np.nanmedian(skym)
    if plot_rawmags :
        plt.errorbar(time, mmag - m - mag_offset, yerr=em, label='raw obj dmag, mean={:.4f}'.format(mmag))
        plt.errorbar(time, mskym - skym + mag_offset, yerr=eskym, label='raw sky dmag, mean={:.4f}'.format(mskym))
        plt.xlabel(r"time (BJD)", fontsize=16)
        plt.ylabel(r"$\Delta$mag", fontsize=16)
        plt.legend(fontsize=10)
        plt.show()

    cm, ecm = [], []
    sumag = np.zeros_like(m)
    for i in range(len(comps)) :
        cmag = hdul[catalog_name].data['MAG{:08d}'.format(comps[i])]
        ecmag = hdul[catalog_name].data['EMAG{:08d}'.format(comps[i])]
        cm.append(cmag)
        ecm.append(ecmag)
        
        sumag += 10**(-0.4*cmag)
        
        mdm = np.nanmedian(cmag - m)
        dm = (cmag - m) - mdm
        rms = np.nanmedian(np.abs(dm)) / 0.67449
        edm = np.sqrt(ecmag**2 + em**2)
        keep = np.isfinite(dm)
        keep &= np.abs(dm) < nsig*rms

        if plot_comps :
            plt.errorbar(time[keep], dm[keep], yerr=edm[keep], fmt='.', alpha=0.3, label=r"C{} $\Delta$mag={:.3f} $\sigma$={:.2f} mmag".format(comps[i], mdm, rms*1000))
 
    sumag = -2.5*np.log10(sumag)
    mdm = np.nanmedian(sumag - m)
    dm = (sumag - m) - mdm
    rms = np.nanmedian(np.abs(dm)) / 0.67449
    keep = np.isfinite(dm)
    keep &= np.abs(dm) < nsig*rms
    if plot_sum :
        plt.errorbar(time[keep], dm[keep], yerr=em[keep], fmt='k.', label=r"SUM $\Delta$mag={:.3f} $\sigma$={:.2f} mmag".format(mdm, rms*1000))
        
    if plot_comps or plot_sum :
        plt.xlabel(r"time (BJD)", fontsize=16)
        plt.ylabel(r"$\Delta$mag", fontsize=16)
        plt.legend(fontsize=10)
        plt.show()


def plot_polarimetry_results(loc, pos_model_sampling=1, title_label="", wave_plate='halfwave') :

    """ Pipeline module to plot half-wave polarimetry data
    
    Parameters
    ----------
    loc : dict
        container with polarimetry data results
    pos_model_sampling : int
        step size for sampling of the position angle (deg) in the polarization model
    title_label : str
        plot title
    wave_plate : str
        wave plate mode

    Returns
    -------

    """
    
    waveplate_angles = loc["WAVEPLATE_ANGLES"]
    zi = loc["ZI"]
    qpol = loc["Q"]
    upol = loc["U"]
    vpol = loc["V"]
    ppol = loc["P"]
    theta = loc["THETA"]
    kcte = loc["K"]
    zero = loc["ZERO"]
    
    qlab = "q: {:.2f}+-{:.2f} %".format(100*qpol.nominal,100*qpol.std_dev)
    ulab = "u: {:.2f}+-{:.2f} %".format(100*upol.nominal,100*upol.std_dev)
    vlab = "v: {:.2f}+-{:.2f} %".format(100*vpol.nominal,100*vpol.std_dev)
    plab = "p: {:.2f}+-{:.2f} %".format(100*ppol.nominal,100*ppol.std_dev)
    thetalab = r"$\theta$: {:.2f}+-{:.2f} deg".format(theta.nominal,theta.std_dev)
    title_label += "\n"+qlab+"  "+ulab
    if wave_plate == 'quarterwave' :
        title_label += "  "+vlab
    title_label += "  "+plab+"  "+thetalab

    # plot best polarimetry results
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=False, gridspec_kw={'hspace': 0, 'height_ratios': [2, 1]})

    if title_label != "":
        axes[0].set_title(title_label)
                
    # define grid of position angle points for model
    pos_model = np.arange(0, 360, pos_model_sampling)
    
    # Plot the model
    best_fit_model = np.full_like(pos_model,np.nan)
    if wave_plate == 'halfwave' :
        best_fit_model = halfwave_model(pos_model, qpol.nominal, upol.nominal)
    elif wave_plate == 'quarterwave' :
        best_fit_model = quarterwave_model(pos_model, qpol.nominal, upol.nominal, vpol.nominal, zero=zero.nominal)

    axes[0].plot(pos_model, best_fit_model,'r:', alpha=0.8, label='Best fit model')
    #axes[0].fill_between(pos_model, pred_mean+pred_std, pred_mean-pred_std, color=color, alpha=0.3, edgecolor="none")
 
    # Plot data
    axes[0].errorbar(waveplate_angles, zi.nominal, yerr=zi.std_dev, fmt='ko', ms=2, capsize=2, lw=0.5, alpha=0.9, label='data')
    axes[0].set_ylabel(r"$Z(\phi) = \frac{f_\parallel(\phi)-f_\perp(\phi)}{f_\parallel(\phi)+f_\perp(\phi)}$", fontsize=16)
    axes[0].legend(fontsize=16)
    axes[0].tick_params(axis='x', labelsize=14)
    axes[0].tick_params(axis='y', labelsize=14)
    
    # Print q, u, p and theta values
    ylims = axes[0].get_ylim()
    #axes[0].text(-10, ylims[1]-0.06,'{}\n{}\n{}\n{}'.format(qlab, ulab, plab, thetalab), size=12)
    
    # Plot residuals
    observed_model = np.full_like(waveplate_angles,np.nan)
    if wave_plate == 'halfwave' :
        observed_model = halfwave_model(waveplate_angles, qpol.nominal, upol.nominal)
    elif wave_plate == 'quarterwave' :
        observed_model = quarterwave_model(waveplate_angles, qpol.nominal, upol.nominal, vpol.nominal, zero=zero.nominal)

    resids = observed_model - zi.nominal
    sig_res = np.nanstd(resids)
    
    axes[1].errorbar(waveplate_angles, resids, yerr=zi.std_dev, fmt='ko', alpha=0.5, label='residuals')
    axes[1].set_xlabel(r"waveplate position angle, $\phi$ [deg]", fontsize=16)
    axes[1].hlines(0., 0, 360, color="k", linestyles=":", lw=0.6)
    axes[1].set_ylim(-5*sig_res,+5*sig_res)
    axes[1].set_ylabel(r"residuals", fontsize=16)
    axes[1].tick_params(axis='x', labelsize=14)
    axes[1].tick_params(axis='y', labelsize=14)
    
    plt.show()



def plot_2d(x, y, z, LIM=None, LAB=None, z_lim=None, use_index_in_y=False, title="", pfilename="", cmap="gist_heat"):
    """
    Use pcolor to display sequence of spectra
    
    Inputs:
    - x:        x array of the 2D map (if x is 1D vector, then meshgrid; else: creation of Y)
    - y:        y 1D vector of the map
    - z:        2D array (sequence of spectra; shape: (len(x),len(y)))
    - LIM:      list containing: [[lim_inf(x),lim_sup(x)],[lim_inf(y),lim_sup(y)],[lim_inf(z),lim_sup(z)]]
    - LAB:      list containing: [label(x),label(y),label(z)] - label(z) -> colorbar
    - title:    title of the map
    - **kwargs: **kwargs of the matplolib function pcolor
    
    Outputs:
    - Display 2D map of the sequence of spectra z
    
    """
    
    if use_index_in_y :
        y = np.arange(len(y))
    
    if len(np.shape(x))==1:
        X,Y  = np.meshgrid(x,y)
    else:
        X = x
        Y = []
        for n in range(len(x)):
            Y.append(y[n] * np.ones(len(x[n])))
        Y = np.array(Y,dtype=float)
    Z = z

    if LIM == None :
        x_lim = [np.min(X),np.max(X)] #Limits of x axis
        y_lim = [np.min(Y),np.max(Y)] #Limits of y axis
        if z_lim == None :
            z_lim = [np.min(Z),np.max(Z)]
        LIM   = [x_lim,y_lim,z_lim]

    if LAB == None :
        ### Labels of the map
        x_lab = r"$Velocity$ [km/s]"     #Wavelength axis
        y_lab = r"Time [BJD]"         #Time axis
        z_lab = r"CCF"     #Intensity (exposures)
        LAB   = [x_lab,y_lab,z_lab]

    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (10,7)
    ax = plt.subplot(111)

    cc = ax.pcolor(X, Y, Z, vmin=LIM[2][0], vmax=LIM[2][1], cmap=cmap)
    cb = plt.colorbar(cc,ax=ax)
    
    ax.set_xlim(LIM[0][0],LIM[0][1])
    ax.set_ylim(LIM[1][0],LIM[1][1])
    
    ax.set_xlabel(LAB[0], fontsize=20)
    ax.set_ylabel(LAB[1],labelpad=15, fontsize=20)
    cb.set_label(LAB[2],rotation=270,labelpad=30, fontsize=20)

    ax.set_title(title,pad=15, fontsize=20)

    if pfilename=="" :
        plt.show()
    else :
        plt.savefig(pfilename, format='png')
    plt.clf()
    plt.close()


def plot_polar_time_series(filename, target=0) :
    pass
