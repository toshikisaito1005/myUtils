"""
Standalone routines that are used for re-sampling maps using CASA.

contents:
    hexbin_sampling

history:
2021-07-21   created by TS
2022-07-26   minor bug fix: import casa task issue
Toshiki Saito@NAOJ
"""

import os, sys, glob, copy, inspect
import numpy as np
import pyfits
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.ioff()

#from mycasa_tasks import check_first,imval_all
from mycasa_tasks import *

#

def hexbin_sampling(
    imagename,
    ra,
    dec,
    beam=0.8,
    gridsize=70,
    err=False,
    stats="mean",
    ):
    """
    Run hexagonal re-sampling with CASA. Sampling area is gridsize*beam x
    gridsize*beam [arcsec^2].

    Parameters
    ----------
    imagename : str
        2D FITS image or CASA image.
    ra : float [degree]
        image center R.A. (e.g., AGN position).
    dec : float [degree]
        image center Decl. (e.g., AGN position).
    beam : float [arcsec]
        diameter of hexagons.
        (nearly) independent sampling when hexagon size > synthesized beam size.
    gridsize : integer
        number of hex in x/y direction.
    stats : str
        which statistics hexbin should do, i.e., mean or sum.

    CASA tasks
    ----------
    imhead
    imval

    References
    ----------
    stackoverflow.com/questions/67017660
    """

    taskname = sys._getframe().f_code.co_name
    check_first(imagename, taskname)

    # FITS/CASA image to numpy array
    data,_ = imval_all(imagename)
    x    = (data["coords"][:,:,0] * 180/np.pi - ra) * 3600 # arcsec
    y    = (data["coords"][:,:,1] * 180/np.pi - dec) * 3600 # arcsec
    c    = np.nan_to_num(data["data"]) # K.km/s
    if err==True:
        c = c**2

    X    = x.reshape(-1)
    Y    = y.reshape(-1)
    C    = c.reshape(-1)

    # determine sampling grid
    size   = 0.5 * gridsize * beam
    extent = [size, -size, -size, size]

    # pixel per hex
    s      = np.sqrt(3.0)/2.0 * beam**2.0
    cdelt1 = imhead(imagename,mode="list")["cdelt1"]
    pix    = abs(float(cdelt1)) * (3600.0*180.0)/np.pi
    n      = s / pix**2.0

    # hex sampling
    fig = plt.figure(figsize=(9,9))
    gs  = gridspec.GridSpec(nrows=1,ncols=1)
    ax  = plt.subplot(gs[0:1,0:1])
    
    hexdata = ax.hexbin(X, Y, C=C, gridsize=gridsize, extent=extent, mincnt=int(n/2.))
    hexc    = np.array(hexdata.get_array())
    hexdata = ax.hexbin(X, Y, C=X, gridsize=gridsize, extent=extent, mincnt=int(n/2.))
    hexx    = np.array(hexdata.get_array())
    hexdata = ax.hexbin(X, Y, C=Y, gridsize=gridsize, extent=extent, mincnt=int(n/2.))
    hexy    = np.array(hexdata.get_array())
    hexdata = ax.hexbin(X, Y, gridsize=gridsize, extent=extent, mincnt=int(n/2.))

    # apply pixel per beam
    barea_pix = beam_area(imagename)
    if err==True:
        hexc = hexc / np.sqrt(barea_pix)
    else:
        hexc = hexc / barea_pix

    # error case
    if err==True:
        hexc = np.sqrt(hexc)*(1.0/np.sqrt(n))

    plt.clf()

    # choose stats
    if stats=="mean":
        pass
    elif stats=="sum":
        hexc = hexc * n

    return hexx, hexy, hexc
