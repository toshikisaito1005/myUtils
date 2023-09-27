"""
Standalone plotting routines that use CASA.

contents:
    myax_set
    myax_cbar
    myax_hists
    myax_fig2png_ann

    myfig_hex_map
    myfig_hex_radial
    myfig_hex_scatter
    myfig_hex_rcorner

    myfig_fits2png

    get_hists
    get_reldist_pc
    binning_scatter

history:
2021-07-21   created by TS
2021-07-28   new functions added
2021-08-16   refactored myfig_fits2png
2021-09-03   bug fix in _get_extent
2021-12-20   mycasa_plot.py accidentally deleted. curl from https://github.com/toshikisaito1005/myUtils/blob/7328a0b0943e547490b1e050a21759686d930e80/mycasa_plots.py
Toshiki Saito@Nichidai/NAOJ
"""

import os, sys, glob, copy, math, inspect
import numpy as np
import pyfits
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.ioff()

from scipy.stats import gaussian_kde
import numpy.linalg as LA
from matplotlib.colors import Normalize

from mycasa_tasks import *

#

################################
# common matplotlib parameters #
################################

fig_dpi         = 200
text_back_alpha = 0.9

############
# myax_set #
############

def myax_set(
    ax,
    grid=None,
    xlim=None,
    ylim=None,
    title=None,
    xlabel=None,
    ylabel=None,
    aspect=None,
    adjust=[0.10,0.99,0.10,0.95],
    lw_grid=1.0,
    lw_ticks=2.5,
    lw_outline=2.5,
    fsize=22,
    fsize_legend=20,
    labelbottom=True,
    labeltop=False,
    labelleft=True,
    labelright=False,
    ):
    
    # adjust edge space
    if adjust!=False:
        plt.subplots_adjust(
            left=adjust[0],
            right=adjust[1],
            bottom=adjust[2],
            top=adjust[3],
            )

    # font
    plt.rcParams["font.size"] = fsize
    plt.rcParams["legend.fontsize"] = fsize_legend

    # tick width
    ax.xaxis.set_tick_params(width=lw_ticks)
    ax.yaxis.set_tick_params(width=lw_ticks)

    # labels
    ax.tick_params(
        labelbottom=labelbottom,
        labeltop=labeltop,
        labelleft=labelleft,
        labelright=labelright,
        )

    # outline width
    axis = ["top", "bottom", "left", "right"]
    lw_outlines = [lw_outline, lw_outline, lw_outline, lw_outline]
    for a,w in zip(axis, lw_outlines):
        ax.spines[a].set_linewidth(w)

    # aspect
    if aspect is not None:
        ax.set_aspect(aspect)

    # grid
    if grid is not None:
        ax.grid(axis=grid, lw=lw_grid)

    # xylims
    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    # xylabels
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # title
    if title is not None:
        ax.set_title(title, x=0.5, y=1.015, weight="bold")

##############
# _myax_cbar #
##############

def _myax_cbar(
    fig,
    ax,
    data,
    label=None,
    clim=None,
    colorbarticks=None,
    colorbarticktexts=None,
    extend=None,
    ):

    if colorbarticks is not None:
        if extend!=None:
            cb = fig.colorbar(data, ax=ax, extend=extend, ticks=colorbarticks)
        else:
            cb = fig.colorbar(data, ax=ax, ticks=colorbarticks)
        cb.ax.set_yticklabels(colorbarticktexts)
    else:
        if extend!=None:
            cb = fig.colorbar(data, ax=ax, extend=extend)
        else:
            cb = fig.colorbar(data, ax=ax)
    
    if label is not None:
        cb.set_label(label)
    
    if clim is not None:
        cb.set_clim(clim)

    cb.outline.set_linewidth(2.5)
    cb.ax.tick_params(width=2.5)

#################
# myfig_hex_map #
#################

def myfig_hex_map(
    X,
    Y,
    C,
    output,
    beam=0.8,
    gridsize=70,
    cmap="rainbow",
    cblabel="(K km s$^{-1}$)",
    zerocbar=False,
    factor=1,
    ):
    """
    Parameters
    ----------
    X,Y,C : np.array
        hex positions and fluxes
    output : str
        output png name with an abs/rel path

    References
    ----------
    stackoverflow.com/questions/67017660
    """
    
    # determine sampling grid
    size   = 0.5 * gridsize * beam
    extent = [size, -size, -size, size]
    
    # prepare for plot
    xlim   = [extent[1]/float(factor),extent[0]/float(factor)]
    ylim   = [extent[2]/float(factor),extent[3]/float(factor)]
    title  = output.split("/")[-1].replace(".png","")

    # plot
    fig = plt.figure(figsize=(10,9))
    gs  = gridspec.GridSpec(nrows=30, ncols=30)
    ax  = plt.subplot(gs[0:30,0:30])
    myax_set(ax,xlim=xlim,ylim=ylim,title=title,aspect=1.0)

    if zerocbar==True:
        clim = [-np.max(abs(C)),np.max(abs(C))]
        hexdata = ax.hexbin(
            X, Y, C=C,
            cmap=cmap,
            gridsize=gridsize,
            extent=extent,
            vmin=clim[0],
            vmax=clim[1],
            )
        _myax_cbar(fig, ax, hexdata, label=cblabel, clim=clim)

    else:
        hexdata = ax.hexbin(
            X, Y, C=C,
            cmap=cmap,
            gridsize=gridsize,
            extent=extent,
            )
        _myax_cbar(fig, ax, hexdata, label=cblabel)
    
    print("# output = " + output.split("/")[-1])
    fig.savefig(output, dpi=fig_dpi)

####################
# myfig_hex_radial #
####################

def myfig_hex_radial(
    dist,
    flux,
    err,
    output,
    snr=3.0,
    ylog=True,
    ylabel="Integrated Intensiry (K km s$^{-1}$)",
    ):

    dist = dist[~np.isnan(flux)]
    err  = err[~np.isnan(flux)]
    flux = flux[~np.isnan(flux)]
    dist = dist[~np.isinf(flux)]
    err  = err[~np.isinf(flux)]
    flux = flux[~np.isinf(flux)]

    # get detections
    cut  = np.where(flux>abs(err*snr))
    X    = dist[cut]
    Y    = flux[cut]
    Yerr = err[cut]
    
    # get upper limits
    cut2 = np.where(flux<abs(err*snr))
    if len(cut2)>0:
        x    = dist[cut2]
        y    = err[cut2] * snr
    else:
        x,y = 0,0

    if len(Y)==0:
        print("# skip "+output.split("/")[-1]+" because # = 0.")
        return None
    
    # binned distribution
    nbins = int(np.ceil((np.log(len(x)) + 1) * 1.5))
    bx, by, byerr = binning_scatter(X, Y, nbins)
    
    # prepare for plot
    title  = output.split("/")[-1]
    xlabel = "Distance (kpc)"

    # plot
    fig = plt.figure(figsize=(10,5))
    gs  = gridspec.GridSpec(nrows=30, ncols=30)
    ax  = plt.subplot(gs[0:30,0:30])

    if ylog==True:
        xlim = [0, np.max(dist)*1.1]
        ylim = [np.min(Y)*0.5, np.max(Y)*1.5]
        ax.set_yscale("log", nonposy="clip")
    else:
        xlim = [0, np.max(dist)*1.1]
        ylim = [0, np.max(flux)*1.2]

    myax_set(ax,xlim=xlim,ylim=ylim,title=title,xlabel=xlabel,ylabel=ylabel,
        adjust=[0.13,0.97,0.15,0.90])
    
    ax.errorbar(X,Y,yerr=Yerr,c="tomato",marker="o",markersize=5,
        markeredgewidth=0,ls="none",zorder=0)
    ax.plot(x, y, c="grey", marker=".",markersize=5,
        markeredgewidth=0,ls="none",zorder=0)
    ax.errorbar(bx,by,yerr=byerr,c="black",capsize=0,
        lw=3,zorder=1e10)

    print("# output = " + output.split("/")[-1])
    fig.savefig(output, dpi=fig_dpi)

###################
# binning_scatter #
###################

def binning_scatter(x, y, nbins):
    """ References
        ----------
        stackoverflow.com/questions/15556930
    """
    n, _   = np.histogram(x, bins=nbins)
    sy, _  = np.histogram(x, bins=nbins, weights=y)
    sy2, _ = np.histogram(x, bins=nbins, weights=y*y)
    
    bin_x    = (_[1:] + _[:-1])/2
    bin_mean = sy / n
    bin_std  = np.sqrt(sy2/n - bin_mean*bin_mean)

    return bin_x, bin_mean, bin_std

#####################
# myfig_hex_scatter #
#####################

def myfig_hex_scatter(
    dist,
    xflux,
    xerr,
    yflux,
    yerr,
    output_xy,
    xname,
    yname,
    snr=3.0,
    vmax=2.0,
    plot_lines=True,
    ):
    """
    """
    
    # sn cut
    cut = np.where((xflux>=abs(xerr*snr)) & (yflux>=abs(yerr*snr)))
    x   = np.log10(xflux[cut])
    y   = np.log10(yflux[cut])
    c   = dist[cut]

    # delete nan and inf
    cut = np.where((~np.isnan(x)) & (~np.isnan(y)) & (~np.isinf(x)) & (~np.isinf(y)))
    x   = np.array(x[cut])
    y   = np.array(y[cut])
    c   = np.array(c[cut])

    # check number of data
    if np.sum(x)<=1 or np.sum(y)<=1:
        print("skip myax_hex_scatter because #<2.")
        return None

    ## plot x vs. y
    # corr coeff
    coeff = str(np.round(np.corrcoef(x,y)[0,1],2))
    
    # prepare for plot
    width   = np.max([np.max(x)-np.min(x), np.max(y)-np.min(y)]) / 2. + 0.2
    xlim    = [np.mean(x)-width, np.mean(x)+width]
    ylim    = [np.mean(y)-width, np.mean(y)+width]
    lim     = [np.min([xlim[0],ylim[0]]), np.max([xlim[1],ylim[1]])]
    title   = xname + " vs " + yname + "\n(r = " + coeff + ")"
    xlabel  = "log " + xname + " (K km s$^{-1}$)"
    ylabel  = "log " + yname
    cmap    = "rainbow_r"
    cblabel = "Distance (kpc)"

    # plot
    fig = plt.figure(figsize=(10,9))
    gs  = gridspec.GridSpec(nrows=30, ncols=30)
    ax  = plt.subplot(gs[0:30,0:30])
    myax_set(ax,xlim=xlim,ylim=ylim,title=title,xlabel=xlabel,ylabel=ylabel,
        aspect=1.0,adjust=[0.15,0.99,0.10,0.90])

    cax = ax.scatter(x,y,lw=0,s=70,c=c,cmap=cmap,norm=Normalize(vmin=0, vmax=vmax))
    
    _myax_cbar(fig, ax, cax, label=cblabel)
    
    if plot_lines==True:
        ax.plot(lim, lim, "k--", lw=3)
        ax.plot(lim, [lim[0]+1.0, lim[1]+1.0], "k--", lw=2)
        ax.plot(lim, [lim[0]-1.0, lim[1]-1.0], "k--", lw=2)
        ylabel = ylabel + " (K km s$^{-1}$)"
    
    print("# output = " + output_xy.split("/")[-1])
    fig.savefig(output_xy, dpi=fig_dpi)

#####################
# myfig_hex_rcorner #
#####################

def myfig_hex_rcorner(
    dist,
    list_flux,
    list_err,
    list_name,
    output,
    snr=3.0,
    numlimit=10,
    ):
    """
    """
    
    # prepare
    l           = range(len(list_name))
    array_coeff = np.zeros([len(list_name), len(list_name)])
    array_coeff = np.where(array_coeff==0, np.nan, array_coeff)

    for i in itertools.combinations(l, 2):
        # get x data
        x     = np.array(list_flux[:,i[0]])
        xerr  = np.array(list_err[:,i[0]])
        
        # get y data
        y     = np.array(list_flux[:,i[1]])
        yerr  = np.array(list_err[:,i[1]])

        # sn cut
        cut = np.where((x>abs(xerr*snr)) & (y>abs(yerr*snr)))
        x   = np.log10(x[cut])
        y   = np.log10(y[cut])
        r   = np.log10(dist[cut])

        # get corr coeff when enough data points
        if len(x)>=numlimit:
            array_coeff[i[1],i[0]] = np.round(np.corrcoef(x,y)[0,1], 2)

    # rename
    list_name = _rename(list_name)
    
    # prepare for plot
    title = "Pearson's $r$ (# > "+str(numlimit)+")"

    # plot
    fig = plt.figure(figsize=(10,9))
    gs  = gridspec.GridSpec(nrows=30, ncols=30)
    ax  = plt.subplot(gs[0:30,0:30])
    myax_set(ax,title=title,aspect=1.0,adjust=[0.20,0.99,0.20,0.95])

    im = ax.imshow(array_coeff, interpolation="none", vmin=-1.0, vmax=1.0, cmap="rainbow")
    
    _myax_cbar(fig, ax, im, clim=[-1,1])

    ax.set_xticks(range(len(list_name)))
    ax.set_xticklabels(list_name,rotation=90)
    ax.set_yticks(range(len(list_name)))
    ax.set_yticklabels(list_name)

    print("# output = " + output)
    fig.savefig(output, dpi=fig_dpi)

###########
# _rename #
###########

def _rename(lname):
    lname = [s.replace("13","$^{13}$").replace("17","$^{17}$") for s in lname]
    lname = [s.replace("18","$^{18}$").replace("p","$^{+}$") for s in lname]
    lname = [s.replace("c","C").replace("o","O").replace("h","H") for s in lname]
    lname = [s.replace("n","N").replace("s","S").replace("1110","(11-10)") for s in lname]
    lname = [s.replace("H3","H$_3$").replace("C3","C$_3$").replace("N2","N$_2$").replace("1211","(12-11)") for s in lname]
    lname = [s.replace("10H","10h").replace("CyC","$c$-").replace("109","(10-9)") for s in lname]
    lname = [s.replace("21","(2-1)").replace("10","(1-0)").replace("C15","C$^{15}$") for s in lname]
    lname = [s.replace("H2","H$_2$").replace("54","(5-4)") for s in lname]
    lname = [s.replace("32","(3-2)").replace("43","(4-3)") for s in lname]
    lname = [s.replace("(1-0)-9","10-9").replace("-(1-0)","-10") for s in lname]
    
    return lname

##############
# myax_hists #
##############

def myax_hists(ax, hist, pctls, hnorm=None, bar_offset=1.4, color="black"):

    if hist!=None:
        hx, hy, hnormy = hist[:,0], hist[:,1], hnorm[:,1]

        # normalizarion
        if hnorm!=None:
            hy = hy / np.sum(hnormy)

        # plor histogram
        width = abs(hx[1] - hx[0])
        ax.bar(hx, hy, lw=0, color=color, alpha=0.2, width=width, align="center")
    
        # plot percentile bars
        y = np.max( hnormy / np.sum(hnormy) ) * bar_offset
        ax.plot([pctls[0],pctls[2]], [y,y], color=color, lw=3)
        ax.plot(pctls[1], y, "o", color=color, markersize=10, markeredgewidth=0)

#############
# get_hists #
#############

def get_hists(data, bins, hrange, weights):
    hist0 = np.histogram(data, bins=bins, range=hrange, weights=None)
    histw = np.histogram(data, bins=bins, range=hrange, weights=weights)
    
    # get histograms
    x     = np.delete(hist0[1],-1)
    y0    = hist0[0]
    yw    = histw[0]
    hist0 = np.c_[x,y0]
    histw = np.c_[x,yw]
    
    # get three percentiles
    pctls0 = _get_pctls(data,None)
    pctlsw = _get_pctls(data,weights)

    return hist0, pctls0, histw, pctlsw

##############
# _get_pctls #
##############

def _get_pctls(x, weights=None):
    x[np.isnan(x)] = 0
    x[np.isinf(x)] = 0
    x = x[x!=0]
    if len(x)>2:
        p16 = _w_pctl(x, 16, weights)
        p50 = _w_pctl(x, 50, weights)
        p84 = _w_pctl(x, 84, weights)
    else:
        p16,p50,p84 = None,None,None

    return np.array([p16,p50,p84])

###########
# _w_pctl #
###########

def _w_pctl(data, percents, weights=None):
    ''' stackoverflow.com/questions/21844024
        percents in units of 1%
        weights specifies the frequency (count) of data.
    '''
    if weights is None:
        return np.percentile(data, percents)
    ind=np.argsort(data)
    d=data[ind]
    w=weights[ind]
    p=1.*w.cumsum()/w.sum()*100
    y=np.interp(percents, p, d)
    return y

##################
# get_reldist_pc #
##################

def get_reldist_pc(
    ra_deg,
    dec_deg,
    ra_cnt,
    dec_cnt,
    scale,
    pa,
    incl,
    ):
    sin_pa   = math.sin(math.radians(pa))
    cos_pa   = math.cos(math.radians(pa))
    cos_incl = math.cos(math.radians(incl))
    
    ra_rel_deg  = (ra_deg - ra_cnt)
    dec_rel_deg = (dec_deg - dec_cnt)

    ra_rel_deproj_deg  = (ra_rel_deg*cos_pa - dec_rel_deg*sin_pa)
    deg_rel_deproj_deg = (ra_rel_deg*sin_pa + dec_rel_deg*cos_pa) / cos_incl

    dist_pc   = np.sqrt(ra_rel_deproj_deg**2 + deg_rel_deproj_deg**2) * 3600 * scale
    theta_deg = np.degrees(np.arctan2(ra_rel_deproj_deg, deg_rel_deproj_deg))

    return dist_pc, theta_deg

##################
# myfig_fits2png #
##################

def myfig_fits2png(
    # general
    imcolor,
    outfile,
    imcontour1=None,
    imcontour2=None,
    imcontour3=None,
    imsize_as=50,
    ra_cnt=None,
    dec_cnt=None,
    # contour 1
    unit_cont1=None,
    levels_cont1=[0.01,0.02,0.04,0.08,0.16,0.32,0.64,0.96],
    width_cont1=[1.0],
    color_cont1="black",
    # contour 2
    unit_cont2=None,
    levels_cont2=[0.01,0.02,0.04,0.08,0.16,0.32,0.64,0.96],
    width_cont2=[1.0],
    color_cont2="red",
    # contour 3
    unit_cont3=None,
    levels_cont3=[0.01,0.02,0.04,0.08,0.16,0.32,0.64,0.96],
    width_cont3=[1.0],
    color_cont3="red",
    # imshow
    fig_dpi=200,
    set_grid="both",
    set_title=None,
    colorlog=False,
    set_bg_color="white",
    set_cmap="rainbow",
    showzero=False,
    showbeam=True,
    color_beam="black",
    scalebar=None,
    label_scalebar=None,
    color_scalebar="black",
    extend=None,
    comment=None,
    comment_color="black",
    # imshow colorbar
    set_cbar=True,
    clim=None,
    label_cbar=None,
    lw_cbar=1.0,
    # annotation
    numann=None,
    textann=True,
    txtfiles=None,
    colorbarticks=None,
    colorbarticktexts=None,
    ):
    """
    Parameters
    ----------
    unit_cont1 : float (int) or None
        if None, unit_cont1 is set to the image peak value.

    levels_cont1 : float list
        output contour levels = unit_cont1 * levels_cont1.
        default is 1% to 96% of the image peak value.

    ra_cnt and dec_cnt : str
        user-defined image center in degree units.
    """

    print("# run fits2png")
    print("# imcolor = " + imcolor)
    os.system("rm -rf " + outfile)

    #################
    ### preparation #
    #################

    # make sure fits format
    if imcolor[-5:]!=".fits":
        run_exportfits(
            imcolor,
            imcolor + ".fits",
            dropstokes = True,
            dropdeg    = True,
            )
        imcolor += ".fits"

    if imcontour1!=None:
        if imcontour1[-5:]!=".fits":
            run_exportfits(
                imcontour1,
                imcontour1 + ".fits",
                dropstokes = True,
                dropdeg    = True,
                )
            imcontour1 += ".fits"

    if imcontour2!=None:
        if imcontour2[-5:]!=".fits":
            run_exportfits(
                imcontour2,
                imcontour2 + ".fits",
                dropstokes = True,
                dropdeg    = True,
                )
            imcontour2 += ".fits"

    if imcontour3!=None:
        if imcontour3[-5:]!=".fits":
            run_exportfits(
                imcontour3,
                imcontour3 + ".fits",
                dropstokes = True,
                dropdeg    = True,
                )
            imcontour3 += ".fits"

    # read imcolor
    hdu        = pyfits.open(imcolor)
    image_data = hdu[0].data[:,:]
    image_data = np.where(~np.isnan(image_data), image_data, 0)
    datamax    = np.max(image_data)

    pix_ra_as  = hdu[0].header["CDELT1"] * 3600
    ra_im_deg  = hdu[0].header["CRVAL1"]
    ra_im_pix  = hdu[0].header["CRPIX1"]
    ra_size    = hdu[0].header["NAXIS1"]

    pix_dec_as = hdu[0].header["CDELT2"] * 3600
    dec_im_deg = hdu[0].header["CRVAL2"]
    dec_im_pix = hdu[0].header["CRPIX2"]
    dec_size   = hdu[0].header["NAXIS2"]

    # get centers if None
    if ra_cnt==None:
        ra_cnt = str(ra_im_deg) + "deg"

    if dec_cnt==None:
        dec_cnt = str(dec_im_deg) + "deg"

    # determine contour levels
    if imcontour1!=None:
        contour_data1, levels_cont1 = \
            _get_contour_levels(imcontour1,unit_cont1,levels_cont1)

    if imcontour2!=None:
        contour_data2, levels_cont2 = \
            _get_contour_levels(imcontour2,unit_cont2,levels_cont2)

    if imcontour3!=None:
        contour_data3, levels_cont3 = \
            _get_contour_levels(imcontour3,unit_cont3,levels_cont3)

    # define imaging extent
    extent = _get_extent(ra_cnt,dec_cnt,ra_im_deg,dec_im_deg,
        ra_im_pix,dec_im_pix,pix_ra_as,pix_dec_as,ra_size,dec_size)

    # set lim
    xlim = [imsize_as/2.0, -imsize_as/2.0]
    if float(dec_cnt.replace("deg",""))>0:
        ylim = [-imsize_as/2.0, imsize_as/2.0]
    else:
        ylim = [-imsize_as/2.0, imsize_as/2.0]

    # set colorlog
    if colorlog==True:
        norm = LogNorm(vmin=0.02*datamax, vmax=datamax)
    else:
        norm = None

    # set None pixels
    image_data = np.where(~np.isinf(image_data),image_data,0)
    image_data = np.where(~np.isnan(image_data),image_data,0)
    if showzero==False:
        image_data[np.where(image_data==0)] = None

    ##########
    # imshow #
    ##########

    # plot
    plt.figure(figsize=(13,10))
    gs = gridspec.GridSpec(nrows=10, ncols=10)
    ax = plt.subplot(gs[0:10,0:10])

    # set ax parameter
    xl, yl = "R.A. Offset (arcsec)", "Decl. Offset (arcsec)"
    ad = [0.19,0.99,0.10,0.90]
    myax_set(ax,grid=set_grid,xlim=xlim,ylim=ylim,title=set_title,
        xlabel=xl,ylabel=yl,adjust=ad)

    cim = ax.imshow(image_data,cmap=set_cmap,norm=norm,
        extent=extent,interpolation="none")

    if imcontour1!=None:
        ax.contour(contour_data1,levels=levels_cont1,extent=extent,
            colors=color_cont1,linewidths=width_cont1,origin="upper")

    if imcontour2!=None:
        ax.contour(contour_data2,levels=levels_cont2,extent=extent,
            colors=color_cont2,linewidths=width_cont2,origin="upper")

    if imcontour3!=None:
        ax.contour(contour_data3,levels=levels_cont3,extent=extent,
            colors=color_cont3,linewidths=width_cont3,origin="upper")

    if set_bg_color!=None:
        ax.axvspan(xlim[0],xlim[1],ylim[0],ylim[1],color=set_bg_color,zorder=0)

    # colorbar
    cim.set_clim(clim)
    if set_cbar==True:
        _myax_cbar(
            plt,
            ax,
            cim,
            label=label_cbar,
            clim=clim,
            colorbarticks=colorbarticks,
            colorbarticktexts=colorbarticktexts,
            extend=extend,
            )

    # add beam size
    if showbeam==True:
        _myax_showbeam(ax,imcolor,ra_cnt,xlim,ylim,color_beam)

    # add scalebar
    if scalebar!=None:
        _myax_scalebar(ax,ra_cnt,xlim,ylim,label_scalebar,scalebar,color_scalebar)

    # add comment
    if comment!=None:
        _myax_comment(ax,dec_cnt,xlim,ylim,comment,comment_color)

    # annotation
    if numann!=None:
        myax_fig2png_ann(ax,numann,ra_cnt,dec_cnt,textann,txtfiles)

    # save
    plt.savefig(outfile, dpi=fig_dpi)

def myax_fig2png_ann(ax,number,ra_cnt,dec_cnt,add_text=True,txtfiles=None):
    """
    This is annotation sets for specific figures. For example,
    number==1 is used for Figure 1 of the NGC 1068 CI outflow paper.
    number==2 is used for Figure 1 of the NGC 3110 CO paper.
    """
    if number=="n6240_cont":
        ax.plot(3.9,9.2,marker="+",markeredgewidth=1.5,markersize=20,color="tomato")
        ax.plot(5.5,8.2,marker="+",markeredgewidth=1.5,markersize=20,color="tomato")
        arc = patches.Arc(xy=(0,0), width=30,
            height=12, angle=0, theta1=110, theta2=250,
            fill=False, edgecolor="tomato",
            alpha=2.0, lw=1.5, ls="dashed")
        ax.add_patch(arc)

    #####################
    # Figures of LST WP #
    #####################

    if number=="lst_n1097sim":
        width = 333*1.21203420e-06*3600*180/np.pi
        height = 579*1.21203420e-06*3600*180/np.pi
        rec = patches.Rectangle(xy=(-width/2.,-height/2.), width=width,
            height=height, fill=False, edgecolor="white",
            alpha=1.0, lw=3.5, ls="dashed")
        ax.add_patch(rec)

    ####################################################
    # Figures of the n1068 13co roration diagram paper #
    ####################################################

    if number=="13co":
        diameter_cnd = 6.0 # = 432 pc
        efov1 = patches.Ellipse(xy=(-0,0), width=diameter_cnd,
            height=diameter_cnd, angle=0, fill=False, edgecolor="black",
            alpha=1.0, lw=3.5)
        ax.add_patch(efov1)

        diameter_sbr = 23.0 # = 1440 pc
        efov2 = patches.Ellipse(xy=(-0,0), width=diameter_sbr,
            height=diameter_sbr, angle=0, fill=False, edgecolor="black",
            alpha=1.0, lw=3.5)
        ax.add_patch(efov2)

    ######################################
    # Figures of CI-GMC NGC 1068 project #
    ######################################

    if number=="ci-gmc":
        x     = (txtfiles["XCTR_DEG"] - float(ra_cnt.split("deg")[0])) * 3600.
        y     = (txtfiles["YCTR_DEG"] - float(dec_cnt.split("deg")[0])) * 3600.
        pos   = txtfiles["POSANG"] * 180 / np.pi
        s2n   = txtfiles["S2N"]
        major = txtfiles["RAD_NOEX"] / 72.
        minor = txtfiles["MOMMINPIX"] / txtfiles["MOMMAJPIX"] * txtfiles["RAD_NOEX"] / 72.

        for i in range(len(x)):
            if s2n[i]>=5.0:
                this_x   = x[i]
                this_y   = y[i]
                this_pos = pos[i]
                this_w   = major[i]
                this_h   = minor[i]

                ell = patches.Ellipse(
                    xy=(this_x,this_y),
                    width=this_w,
                    height=this_h,
                    angle=this_pos,
                    fill=False,
                    edgecolor="red",
                    facecolor="red",
                    alpha=1.0,
                    lw=1.0)

                ax.add_patch(ell)

        theta1      = -10.0 # degree
        theta2      = 70.0 # degree
        fov_diamter = 32#16.5 # arcsec (12m+7m Band 8)

        efov1 = patches.Ellipse(xy=(-0,0), width=fov_diamter,
            height=fov_diamter, angle=0, fill=False, edgecolor="black",
            alpha=1.0, lw=3.5)

        ax.add_patch(efov1)

        # add annotation comment
        if add_text==True:
            ax.plot([0,-5], [0,7.5], lw=3, c="black")
            ax.text(-8.5, 8.0, "AGN position",
                horizontalalignment="right", verticalalignment="center", weight="bold")

            ax.plot([7,7.5], [2.5,7.0], "--", lw=3, c="black")
            ax.plot([-1,4.8], [7.0,7.5], "--", lw=3, c="black")
            ax.text(8.5, 8.0, "Bicone",
                horizontalalignment="left", verticalalignment="center", weight="bold")
            ax.text(8.5, 7.3, "boundary",
                horizontalalignment="left", verticalalignment="center", weight="bold")

            # plot NGC 1068 AGN and outflow geometry
            x1 = fov_diamter/2.0 * np.cos(np.radians(-1*theta1+90))
            y1 = fov_diamter/2.0 * np.sin(np.radians(-1*theta1+90))
            ax.plot([x1, -x1], [y1, -y1], "--", c="black", lw=3.5)
            x2 = fov_diamter/2.0 * np.cos(np.radians(-1*theta2+90))
            y2 = fov_diamter/2.0 * np.sin(np.radians(-1*theta2+90))
            ax.plot([x2, -x2], [y2, -y2], "--", c="black", lw=3.5)

    if number=="ci-gmc2":
        txtfile1 = txtfiles[0]
        x     = (txtfile1["XCTR_DEG"] - float(ra_cnt.split("deg")[0])) * 3600.
        y     = (txtfile1["YCTR_DEG"] - float(dec_cnt.split("deg")[0])) * 3600.
        pos   = txtfile1["POSANG"] * 180 / np.pi
        s2n   = txtfile1["S2N"]
        major = txtfile1["RAD_NOEX"] / 72.
        minor = txtfile1["MOMMINPIX"] / txtfile1["MOMMAJPIX"] * txtfile1["RAD_NOEX"] / 72.

        for i in range(len(x)):
            if s2n[i]>=7.0:
                this_x   = x[i]
                this_y   = y[i]
                this_pos = pos[i]
                this_w   = major[i]
                this_h   = minor[i]

                ell = patches.Ellipse(
                    xy=(this_x,this_y),
                    width=this_w,
                    height=this_h,
                    angle=this_pos,
                    fill=True,
                    edgecolor="deepskyblue",
                    facecolor="deepskyblue",
                    alpha=0.5,
                    lw=1.0)

                ax.add_patch(ell)

        txtfile1 = txtfiles[1]
        x     = (txtfile1["XCTR_DEG"] - float(ra_cnt.split("deg")[0])) * 3600.
        y     = (txtfile1["YCTR_DEG"] - float(dec_cnt.split("deg")[0])) * 3600.
        pos   = txtfile1["POSANG"] * 180 / np.pi
        s2n   = txtfile1["S2N"]
        major = txtfile1["RAD_NOEX"] / 72.
        minor = txtfile1["MOMMINPIX"] / txtfile1["MOMMAJPIX"] * txtfile1["RAD_NOEX"] / 72.

        for i in range(len(x)):
            if s2n[i]>=7.0:
                this_x   = x[i]
                this_y   = y[i]
                this_pos = pos[i]
                this_w   = major[i]
                this_h   = minor[i]

                ell = patches.Ellipse(
                    xy=(this_x,this_y),
                    width=this_w,
                    height=this_h,
                    angle=this_pos,
                    fill=True,
                    edgecolor="tomato",
                    facecolor="tomato",
                    alpha=0.5,
                    lw=1.0)

                ax.add_patch(ell)

        theta1      = -10.0 # degree
        theta2      = 70.0 # degree
        fov_diamter = 16.5 # arcsec (12m+7m Band 8)

        fov_diamter = 16.5
        efov1 = patches.Ellipse(xy=(-0,0), width=fov_diamter,
            height=fov_diamter, angle=0, fill=False, edgecolor="black",
            alpha=1.0, lw=3.5)

        ax.add_patch(efov1)

        # add annotation comment
        if add_text==True:
            ax.plot([0,-5], [0,7.5], lw=3, c="black")
            ax.text(-8.5, 8.0, "AGN position",
                horizontalalignment="right", verticalalignment="center", weight="bold")

            ax.plot([7,7.5], [2.5,7.0], "--", lw=3, c="black")
            ax.plot([-1,4.8], [7.0,7.5], "--", lw=3, c="black")
            ax.text(8.5, 8.0, "Bicone",
                horizontalalignment="left", verticalalignment="center", weight="bold")
            ax.text(8.5, 7.3, "boundary",
                horizontalalignment="left", verticalalignment="center", weight="bold")

            # plot NGC 1068 AGN and outflow geometry
            x1 = fov_diamter/2.0 * np.cos(np.radians(-1*theta1+90))
            y1 = fov_diamter/2.0 * np.sin(np.radians(-1*theta1+90))
            ax.plot([x1, -x1], [y1, -y1], "--", c="black", lw=3.5)
            x2 = fov_diamter/2.0 * np.cos(np.radians(-1*theta2+90))
            y2 = fov_diamter/2.0 * np.sin(np.radians(-1*theta2+90))
            ax.plot([x2, -x2], [y2, -y2], "--", c="black", lw=3.5)

    #######################################
    # Figure 1 of the NGC 1068 CI outflow #
    #######################################

    if number==1:
        theta1      = -10.0 # degree
        theta2      = 70.0 # degree
        fov_diamter = 16.5 # arcsec (12m+7m Band 8)

        efov1 = patches.Ellipse(xy=(-0,0), width=fov_diamter,
            height=fov_diamter, angle=0, fill=False, edgecolor="black",
            alpha=1.0, lw=3.5)

        ax.add_patch(efov1)

        # plot NGC 1068 AGN and outflow geometry
        x1 = fov_diamter/2.0 * np.cos(np.radians(-1*theta1+90))
        y1 = fov_diamter/2.0 * np.sin(np.radians(-1*theta1+90))
        ax.plot([x1, -x1], [y1, -y1], "--", c="black", lw=3.5)
        x2 = fov_diamter/2.0 * np.cos(np.radians(-1*theta2+90))
        y2 = fov_diamter/2.0 * np.sin(np.radians(-1*theta2+90))
        ax.plot([x2, -x2], [y2, -y2], "--", c="black", lw=3.5)

        # add annotation comment
        if add_text==True:
            ax.plot([0,-5], [0,7.5], lw=3, c="black")
            ax.text(-8.5, 8.0, "AGN position",
                horizontalalignment="right", verticalalignment="center", weight="bold")

            ax.plot([7,7.5], [2.5,7.0], "--", lw=3, c="black")
            ax.plot([-1,4.8], [7.0,7.5], "--", lw=3, c="black")
            ax.text(8.5, 8.0, "Bicone",
                horizontalalignment="left", verticalalignment="center", weight="bold")
            ax.text(8.5, 7.3, "boundary",
                horizontalalignment="left", verticalalignment="center", weight="bold")

    ####################################
    # Figures of the NGC 3110 CO paper #
    ####################################

    if number=="n3110_irac":
        # add annotation comment
        if add_text==True:
            ax.text(-15, 10, "NGC 3110", color="white",
                horizontalalignment="right", verticalalignment="center", weight="bold")
            ax.text(-60, -30, "MCG-01-26-013", color="white",
                horizontalalignment="right", verticalalignment="center", weight="bold")

    if number=="n3110_co_moms":
        # highlight speak
        this_e = patches.Arc(xy=(3,-14), width=20, height=40, angle=-10,
            theta1=-55, theta2=35,
            fill=False, edgecolor="black", alpha=1.0, lw=4)

        ax.add_patch(this_e)

        ax.scatter(-8,4,marker="+",c="black",s=700,lw=4)

        if add_text==True:
            ax.text(15, -29, "thin southern arm", color="black", rotation=90,
                horizontalalignment="center", verticalalignment="bottom", weight="bold")

            ax.plot([-8,-19],[4,8],"-",c="black",lw=2)
            ax.text(-18, 9, "northern\nblob", color="black",
                horizontalalignment="left", verticalalignment="bottom", weight="bold")

    ###########################################
    # Figure 2 of C8.5 spectral scan proposal #
    ###########################################

    if number==3:
        # plot CND outer radius
        cnd_radius = 7.0 # racsec
        e_cnd = patches.Ellipse(xy=(-0,0), width=cnd_radius,
            height=cnd_radius, angle=0, fill=False, edgecolor="black",
            alpha=1.0, lw=2.5, ls="dashed")

        ax.add_patch(e_cnd)

        sbr_radius = 34.0 # racsec
        e_sbr = patches.Ellipse(xy=(-0,0), width=sbr_radius,
            height=sbr_radius, angle=0, fill=False, edgecolor="black",
            alpha=1.0, lw=3.5, ls="dashed")

        ax.add_patch(e_sbr)

        if add_text==True:
            t = ax.text(0, 4.0, "CND", color="black",
                horizontalalignment="center", verticalalignment="bottom", weight="bold")
            t.set_bbox(dict(facecolor="white", alpha=0.8, lw=0))

            t = ax.text(0, 18.0, "Starburst Ring", color="black",
                horizontalalignment="center", verticalalignment="bottom", weight="bold")
            t.set_bbox(dict(facecolor="white", alpha=0.8, lw=0))

    if number==4:
        if txtfiles!=None:
            # b3 fov
            f = open(txtfiles[0],"r")
            b3_fov = f.readlines()[2:]
            f.close()
            b3_fov = [s.split(",")[0:2] for s in b3_fov]
            b3_size = 35.0 * 300 / 97.99845

            # b6 fov
            f = open(txtfiles[1],"r")
            b6_fov = f.readlines()[2:]
            f.close()
            b6_fov = [s.split(",")[0:2] for s in b6_fov]
            b6_size = 35.0 * 300 / 224.844215

            # plot B3 FoV
            for this_fov in b3_fov:
                x = this_fov[0].replace(":","h",1).replace(":","m",1)+"s"
                y = this_fov[1].replace(":","d",1).replace(":","m",1)+"s"
                c = SkyCoord(x, y)
                ra_dgr = c.ra.degree
                dec_dgr = c.dec.degree

                thisx = (float(ra_cnt.split("deg")[0]) - ra_dgr) * 3600.
                thisy = (float(dec_cnt.split("deg")[0]) - dec_dgr) * 3600.

                this_e = patches.Ellipse(xy=(-thisx,thisy), width=b3_size,
                    height=b3_size, angle=0, fill=False, edgecolor="grey",
                    alpha=1.0, lw=1.0)

                ax.add_patch(this_e)

            # plot B3 FoV
            for this_fov in b6_fov:
                x = this_fov[0].replace(":","h",1).replace(":","m",1)+"s"
                y = this_fov[1].replace(":","d",1).replace(":","m",1)+"s"
                c = SkyCoord(x, y)
                ra_dgr = c.ra.degree
                dec_dgr = c.dec.degree

                thisx = (float(ra_cnt.split("deg")[0]) - ra_dgr) * 3600.
                thisy = (float(dec_cnt.split("deg")[0]) - dec_dgr) * 3600.

                this_e = patches.Ellipse(xy=(-thisx,-thisy), width=b6_size,
                    height=b6_size, angle=0, fill=False, edgecolor="black",
                    alpha=1.0, lw=1.0, ls="dashed")

                ax.add_patch(this_e)

        if add_text==True:
            t = ax.text(-15, 17, "Proposed Band 6 FoV", color="black", rotation=-32,
                horizontalalignment="center", verticalalignment="center", weight="bold")
            #t.set_bbox(dict(facecolor="white", alpha=0.8, lw=0))

            t = ax.text(0, 50, "Proposed Band 3 FoV", color="black",
                horizontalalignment="center", verticalalignment="bottom", weight="bold")
            t.set_bbox(dict(facecolor="white", alpha=0.8, lw=0))

    #####################################
    # Figure 1 of C8.5 catom21 proposal #
    #####################################

    if number==5:
        for this_txt in txtfiles:
            # 10 fov
            f = open(this_txt,"r")
            b10_fov = f.readlines()[2:]
            f.close()
            b10_fov = [s.split(",")[0:2] for s in b10_fov]
            b10_size = 12.381 # 35.0 * 300 / 97.99845

            # plot B10 FoV
            for this_fov in b10_fov:
                x = this_fov[0].replace(":","h",1).replace(":","m",1)+"s"
                y = this_fov[1].replace(":","d",1).replace(":","m",1)+"s"
                c = SkyCoord(x, y)
                ra_dgr = c.ra.degree
                dec_dgr = c.dec.degree

                thisx = (float(ra_cnt.split("deg")[0]) - ra_dgr) * 3600.
                thisy = (float(dec_cnt.split("deg")[0]) - dec_dgr) * 3600.

                this_e = patches.Ellipse(xy=(-thisx,-thisy), width=b10_size,
                    height=b10_size, angle=0, fill=False, edgecolor="white",
                    alpha=1.0, lw=1.5, zorder=1e9)

                ax.add_patch(this_e)

        if add_text==True:
            t = ax.text(-6, 20, "Proposed Band 10", color="white", rotation=0,
                horizontalalignment="center", verticalalignment="center", weight="bold")
            t = ax.text(-6, 18, "mosaic-1", color="white", rotation=0,
                horizontalalignment="center", verticalalignment="center", weight="bold")

            t = ax.text(12, -5, "mosaic-2", color="white", rotation=0,
                horizontalalignment="center", verticalalignment="center", weight="bold")

def _myax_comment(ax,dec_cnt,xlim,ylim,comment,comment_color):
    if float(dec_cnt.replace("deg",""))>0:
        t = ax.text(min(xlim)*-0.9, max(ylim)*0.9,
            comment, horizontalalignment="left", verticalalignment="top",
            color=comment_color, weight="bold")

    else:
        t = ax.text(min(xlim)*-0.9, max(ylim)*0.9,
            comment, horizontalalignment="left", verticalalignment="top",
            color=comment_color, weight="bold")

    t.set_bbox(dict(facecolor="white", alpha=0.2, lw=0))

def _myax_scalebar(ax,ra_cnt,xlim,ylim,label_scalebar,scalebar,color_scalebar):

    if float(ra_cnt.replace("deg",""))>0:
        ax.text(min(xlim)*0.8, max(ylim)*-0.9,
            label_scalebar, horizontalalignment="right", color=color_scalebar)

        e2 = patches.Rectangle(xy = ( min(xlim)*0.8, max(ylim)*-0.8 ),
            width=scalebar, height=0.1, linewidth=4, edgecolor=color_scalebar)

    else:
        ax.text(min(xlim)*0.8, max(ylim)*-0.9,
            label_scalebar, horizontalalignment="right", color=color_scalebar)

        e2 = patches.Rectangle(xy = ( min(xlim)*0.8, max(ylim)*-0.8 ),
            width=scalebar, height=0.1, linewidth=4, edgecolor=color_scalebar)

    ax.add_patch(e2)

def _myax_showbeam(ax,fitsimage,ra_cnt,xlim,ylim,color_beam):

    header = imhead(fitsimage,mode="list")

    if "beammajor" in header.keys():
        bmaj = header["beammajor"]["value"]
        bmin = header["beamminor"]["value"]
        bpa  = header["beampa"]["value"]
    else:
        bmaj,bmin,bpa = 0.01,0.01,0.0

    if float(ra_cnt.replace("deg",""))>0:
        ax.text(min(xlim)*-0.8, max(ylim)*-0.9,
            "beam", horizontalalignment="left", color=color_beam)

        e1 = patches.Ellipse(xy = ( -min(xlim)*0.8-bmin/2.0, -max(ylim)*0.8 ),
            width=bmin, height=bmaj, angle=-bpa, fc=color_beam)

    else:
        ax.text(min(xlim)*-0.8, max(ylim)*-0.9,
            "beam", horizontalalignment="left", color=color_beam)

        e1 = patches.Ellipse(xy = ( -min(xlim)*0.8-bmin/2.0, -max(ylim)*0.8 ),
            width=bmin, height=bmaj, angle=-bpa, fc=color_beam)

    ax.add_patch(e1)

def _get_extent(
    ra_cnt,
    dec_cnt,
    ra_im_deg,
    dec_im_deg,
    ra_im_pix,
    dec_im_pix,
    pix_ra_as,
    pix_dec_as,
    ra_size,
    dec_size,
    ):

    if ra_cnt==None:
        offset_ra_deg = 0
    else:
        offset_ra_deg = ra_im_deg - float(ra_cnt.replace("deg",""))

    offset_ra_pix = offset_ra_deg * 3600 / float(pix_ra_as)
    this_ra = ra_im_pix - offset_ra_pix

    if dec_cnt==None:
        offset_dec_deg = 0
    else:
        offset_dec_deg = dec_im_deg - float(dec_cnt.replace("deg",""))

    offset_dec_pix = offset_dec_deg * 3600 / float(pix_dec_as)
    this_dec = dec_im_pix - offset_dec_pix

    xext_min = float(pix_ra_as) * (0.5 - this_ra)
    xext_max = float(pix_ra_as) * (0.5 - this_ra + ra_size)

    if float(dec_cnt.replace("deg",""))>0:
        yext_min = float(pix_dec_as) * (0.5 - this_dec)
        yext_max = float(pix_dec_as) * (0.5 - this_dec + dec_size)
    else:
        yext_min = float(pix_dec_as) * (0.5 - this_dec)
        yext_max = float(pix_dec_as) * (0.5 - this_dec + dec_size)

    extent = [xext_min, xext_max, yext_max, yext_min]

    return extent

def _get_contour_levels(fitsimage,unit_contour,levels_contour):

    hdu = pyfits.open(fitsimage)
    contour_data = hdu[0].data[:,:]

    if unit_contour==None:
        unit_contour = imhead(fitsimage, "list")["datamax"]

    output_contours = map(lambda x: x * unit_contour, levels_contour)

    return contour_data, output_contours

################
# image magick #
################

def immagick_crop(
    infile,
    outfile,
    box,
    delin=False,
    convert="/usr/bin/convert ",
    ):
    print("# run immagick_crop")
    os.system(convert + " -crop " + box + " " + infile + " " + outfile)

    if delin==True:
        os.system("rm -rf " + infile)

def immagick_append(
    infile1,
    infile2,
    outfile,
    axis="row",
    delin=False,
    convert="/usr/bin/convert ",
    ):
    print("# run immagick_append")
    
    if axis=="row":
        axis="+"
    elif axis=="column":
        axis="-"
    
    os.system(convert + " " + axis + "append -border 0x0 " + infile1 + " " + infile2 + " " + outfile)
    
    if delin==True:
        os.system("rm -rf " + infile1)
        os.system("rm -rf " + infile2)

def immagick_append_three(
    infile1,
    infile2,
    infile3,
    outfile,
    axis="row",
    delin=False,
    convert="/usr/bin/convert ",
    ):
    print("# run immagick_append_three")

    if axis=="row":
        axis="+"
    elif axis=="column":
        axis="-"

    os.system(convert + " " + axis + "append -border 0x0 " + infile1 + " " + infile2 + " " + infile3 + " " + outfile)

    if delin==True:
        os.system("rm -rf " + infile1)
        os.system("rm -rf " + infile2)
        os.system("rm -rf " + infile3)

def combine_two_png(
    infile1,
    infile2,
    outfile,
    box1,
    box2,
    axis="row",
    delin=False,
    ):
    print("# run combine_two_png")
    done1 = glob.glob(infile1)
    done2 = glob.glob(infile2)
    if done1:
        if done2:
            os.system("rm -rf " + outfile)
            immagick_crop(infile1, infile1+"_tmp1.png", box=box1, delin=delin)
            immagick_crop(infile2, infile2+"_tmp1.png", box=box2, delin=delin)
            immagick_append(infile1+"_tmp1.png", infile2+"_tmp1.png", outfile, axis=axis, delin=delin)

    os.system("rm -rf " + infile1 + "_tmp1.png")
    os.system("rm -rf " + infile2 + "_tmp1.png")
    
    if delin==True:
        os.system("rm -rf " + infile1)
        os.system("rm -rf " + infile2)

def combine_three_png(
    infile1,
    infile2,
    infile3,
    outfile,
    box1,
    box2,
    box3,
    axis="row",
    delin=False,
    ):
    print("# run combine_two_png")
    done1 = glob.glob(infile1)
    done2 = glob.glob(infile2)
    done3 = glob.glob(infile3)
    if done1:
        if done2:
            if done3:
                immagick_crop(infile1, infile1+"_tmp1.png", box=box1, delin=delin)
                immagick_crop(infile2, infile2+"_tmp1.png", box=box2, delin=delin)
                immagick_crop(infile3, infile3+"_tmp1.png", box=box3, delin=delin)
                immagick_append_three(infile1+"_tmp1.png", infile2+"_tmp1.png", infile3+"_tmp1.png", outfile, axis=axis, delin=delin)

    os.system("rm -rf " + infile1 + "_tmp1.png")
    os.system("rm -rf " + infile2 + "_tmp1.png")
    os.system("rm -rf " + infile3 + "_tmp1.png")

    if delin==True:
        os.system("rm -rf " + infile1)
        os.system("rm -rf " + infile2)
        os.system("rm -rf " + infile3)

#######
# end #
#######