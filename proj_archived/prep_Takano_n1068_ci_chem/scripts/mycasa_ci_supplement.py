import os, re, sys, glob, math, scipy, pyfits
import numpy as np
from scipy.optimize import curve_fit

### matplotlib
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
plt.ioff()

### import CASA tasks
from taskinit import *
import analysisUtils as aU
from impv import impv
from imval import imval
from immath import immath
from imhead import imhead
from imstat import imstat
from impbcor import impbcor
from imrebin import imrebin
from imregrid import imregrid
from imsmooth import imsmooth
from makemask import makemask
from immoments import immoments
from specsmooth import specsmooth
from imsubimage import imsubimage
from importfits import importfits
from exportfits import exportfits

# import modules in the same directory
sys.path.append("./scripts/")
import mycasa_ci_supplement as ci_modules
reload(ci_modules)
exec(open("./scripts/define_param.py").read())
exec(open("./scripts/define_figure.py").read())

################
# CASA related #
################
def eazy_imregrid(
    fitsimage,
    template,
    outputfits,
    factor=1.0, # multiply this factor for pixel values; e.g., when convert from Jy to mJy
    smooth_to=None,
    ):
    """
    """

    print("#####################")
    print("# eazy_imregrid: " + fitsimage.split("/")[-1])
    print("#####################")

    # import template FITS to CASA
    this_template = "template.image"
    os.system("rm -rf " + this_template)
    importfits(
        fitsimage = template,
        imagename = this_template,
        )

    # smooth
    if smooth_to!=None:
        # imsmooth
        imsmooth(
            imagename = fitsimage,
            targetres = True,
            major     = str(smooth_to) + "arcsec",
            minor     = str(smooth_to) + "arcsec",
            pa        = "0deg",
            outfile   = outputfits + "_tmp1",
            )
        # imregrid
        os.system("rm -rf " + outputfits + "_tmp2")
        imregrid(
            imagename = outputfits + "_tmp1",
            template  = this_template,
            output    = outputfits + "_tmp2",
            axes      = [0,1], # regrid only in xy directions, i.e., no channel regrid if cube
            )
        # rename
        os.system("rm -rf " + outputfits + "_tmp1")
        os.system("mv " + outputfits + "_tmp2 " + outputfits + "_tmp1")
    else:
        # regrid
        os.system("rm -rf " + outputfits + "_tmp1")
        imregrid(
            imagename = fitsimage,
            template  = this_template,
            output    = outputfits + "_tmp1",
            axes      = [0,1],
            )

    # immath
    os.system("rm -rf " + outputfits + "_tmp2")
    immath(
        imagename = outputfits + "_tmp1",
        expr      = "IM0*" + str(factor),
        outfile   = outputfits + "_tmp2",
        )

    # export CASA map to FITS
    os.system("rm -rf " + outputfits)
    exportfits(
        imagename = outputfits + "_tmp2",
        fitsimage = outputfits,
        velocity  = True, # frequency axis in km/s unit
        history   = False, # discard history header
        )

    # delete intermediates
    os.system("rm -rf " + this_template)
    os.system("rm -rf " + outputfits + "_tmp1")
    os.system("rm -rf " + outputfits + "_tmp2")

def eazy_imval(
    imagename,
    region=None,
    ):

    # imval
    if region==None:
        shape = imhead(imagename,mode="list")["shape"]
        box   = "0,0," + str(shape[0]-1) + "," + str(shape[1]-1)
        data  = imval(imagename,box=box)
        return data, box

    else:
        data  = imval(imagename,region=region)
        return data

def get_dist_theta(
    ra_deg,
    dec_deg,
    ra_cnt,
    dec_cnt,
    pa=0,
    incl=0,
    ):
    """
    """

    sin_pa   = math.sin(math.radians(pa))
    cos_pa   = math.cos(math.radians(pa))
    cos_incl = math.cos(math.radians(incl))
    
    ra_rel_deg  = (ra_deg - ra_cnt)
    dec_rel_deg = (dec_deg - dec_cnt)

    ra_rel_deproj_deg  = (ra_rel_deg*cos_pa - dec_rel_deg*sin_pa)
    deg_rel_deproj_deg = (ra_rel_deg*sin_pa + dec_rel_deg*cos_pa) / cos_incl

    dist_as   = np.sqrt(ra_rel_deproj_deg**2 + deg_rel_deproj_deg**2) * 3600
    theta_deg = np.degrees(np.arctan2(ra_rel_deproj_deg, deg_rel_deproj_deg))

    return dist_as, theta_deg

################
# plot spectra #
################
def plot_spec(
    template,
    cube_ci10,
    cube_co10,
    ecube_ci10,
    ecube_co10,
    this_ra,
    this_dec,
    output,
    this_ylim,
    ):

    # prepare
    this_ra  = float(this_ra.replace("deg",""))
    this_dec = float(this_dec.replace("deg",""))
    xlim     = [800,1400]

    # get xy grid
    l,_ = eazy_imval(template)
    ra  = l["coords"][:,:,0] * 180/np.pi
    dec = l["coords"][:,:,1] * 180/np.pi

    # search position
    ra_pos  = np.nanargmin( abs(ra[:,0] - this_ra) )
    dec_pos = np.nanargmin( abs(dec[0] - this_dec) )
    box = str(ra_pos)+","+str(dec_pos)+","+str(ra_pos+1)+","+str(dec_pos)
    print("# spectra at " + box)

    # get spectra
    vel_ci   = (freq_ci10-imval(cube_ci10, box=box)["coords"][0][:,2]) / freq_ci10 * 299792.458
    spec_ci  = imval(cube_ci10, box=box)["data"][0]
    espec_ci = imval(ecube_ci10, box=box)["data"][0]
    spec_ci  = np.nan_to_num(spec_ci)
    espec_ci = np.nan_to_num(espec_ci)

    vel_co   = (freq_co10-imval(cube_co10, box=box)["coords"][0][:,2]) / freq_co10 * 299792.458
    spec_co  = imval(cube_co10, box=box)["data"][0]
    espec_co = imval(ecube_co10, box=box)["data"][0]
    spec_co  = np.nan_to_num(spec_co)
    espec_co = np.nan_to_num(espec_co)

    ########
    # plot #
    ########
    fig = plt.figure(figsize=(13,10))
    gs  = gridspec.GridSpec(nrows=10, ncols=10)
    ax1 = plt.subplot(gs[0:10,0:10])
    ad  = [0.215,0.83,0.10,0.90]

    ci_modules.myax_set(ax1,grid="both",xlim=xlim,ylim=this_ylim,title=None,xlabel=None,ylabel=None,adjust=ad)

    ax1.fill_between(vel_co,spec_co-espec_co,spec_co+espec_co,lw=0,color="lightgreen")
    ax1.plot(vel_co,spec_co,lw=1,color="green")
    ax1.fill_between(vel_ci,spec_ci-espec_ci,spec_ci+espec_ci,lw=0,color="tomato")
    ax1.plot(vel_ci,spec_ci,lw=1,color="red")

    # ann
    ax1.plot(xlim,[0,0],"--",color="grey",lw=1)

    # save
    os.system("rm -rf " + output)
    plt.savefig(output, dpi=200)

##################
# plot histogram #
##################
def create_hist(
    data,
    weights=None,
    histarea=None,
    bins=30,
    hrange=[-2.5,0.5],
    ):
    # calculate histogram
    histy,histx = np.histogram(data,bins=bins,range=hrange,weights=weights)
    hist_width  = abs(histx[1]-histx[0]) / 2.

    # area
    if histarea==None:
        histarea = float(np.sum(histy))

    # reshape
    histx = histx[:-1] + hist_width
    histy = histy / histarea

    return histx, histy, histarea, hist_width*2

######################
# plot circular slit #
######################
def get_circ_slit_three(
    ci10,
    co10,
    cont,
    slitlength, # in arcsec units
    slitra,     # center, deg, string; e.g., "40.669625deg"
    slitdec,    # center, deg, string
    kernel=1./42.25, # LOWESS width
    ):
    """
    """

    # slice ci10
    dist_ci, slit_ci, scatter_ci = get_circ_slit(
        ci10,
        slitlength,
        slitra,
        slitdec,
        )
    max_ci     = np.nanmax(slit_ci)
    slit_ci    = slit_ci / max_ci
    scatter_ci = scatter_ci / max_ci
    dist_ci    = dist_ci

    # slice co10
    dist_co, slit_co, scatter_co = get_circ_slit(
        co10,
        slitlength,
        slitra,
        slitdec,
        )
    max_co     = np.nanmax(slit_co)
    slit_co    = slit_co / max_co
    scatter_co = scatter_co / max_co
    dist_co    = dist_co

    # slice cont
    dist_cont, slit_cont, scatter_cont = get_circ_slit(
        cont,
        slitlength,
        slitra,
        slitdec,
        )
    max_cont     = np.nanmax(slit_cont)
    slit_cont    = slit_cont / max_cont
    scatter_cont = scatter_cont / max_cont
    dist_cont    = dist_cont

    # summarize
    list_ci   = [dist_ci, slit_ci, scatter_ci, max_ci]
    list_co   = [dist_co, slit_co, scatter_co, max_co]
    list_cont = [dist_cont, slit_cont, scatter_cont, max_cont]

    return list_ci, list_co, list_cont

def get_circ_slit(
    fitsimage,
    slitwidth,
    slitra,    # center, deg, string; e.g., "40.669625deg"
    slitdec,   # center, deg, string
    kernel=1./30., # LOWESS width
    ):
    """
    """

    # get values
    data, box = eazy_imval(fitsimage)
    data      = data["data"] * data["mask"]
    data      = data.flatten()

    # get ra and dec
    data_coords = imval(fitsimage,box=box)["coords"]
    ra_deg      = data_coords[:,:,0] * 180/np.pi
    ra_deg      = ra_deg.flatten()
    dec_deg     = data_coords[:,:,1] * 180/np.pi
    dec_deg     = dec_deg.flatten()

    # calc radius and theta
    slitra  = float(slitra.replace("deg",""))
    slitdec = float(slitdec.replace("deg",""))
    dist_as, theta = get_dist_theta(ra_deg, dec_deg, slitra, slitdec)
    #theta = theta + 90
    #theta = np.where(theta>180,theta-360,theta)

    # get data along the defined circular slit
    cut  = np.where( (dist_as<=slitwidth) & (~np.isnan(data)) & (~np.isnan(ra_deg)) & (~np.isnan(dec_deg)) )
    x, y = lowess(np.array(theta[cut]), np.array(data[cut]), f=kernel)

    return np.array(theta[cut]), x, y

#############
# plot slit #
#############
def get_slit_three(
    ci10,
    co10,
    cont,
    slitangle,  # from north to east
    slitlength, # in arcsec units
    slitwidth,  # in arcsec units
    slitra,     # center, deg, string; e.g., "40.669625deg"
    slitdec,    # center, deg, string
    kernel=1./42.25, # LOWESS width
    ):
    """
    """

    # slice ci10
    dist_ci, slit_ci, scatter_ci = get_slit(
        ci10,
        slitangle, # from north to east
        slitwidth,  # width in arcsec units
        slitra,
        slitdec,
        )
    cut = np.where(abs(dist_ci)<slitlength/2.)
    max_ci     = np.nanmax(slit_ci[cut])
    slit_ci    = slit_ci[cut] / max_ci
    scatter_ci = scatter_ci[cut] / max_ci
    dist_ci    = dist_ci[cut]

    # slice co10
    dist_co, slit_co, scatter_co = get_slit(
        co10,
        slitangle, # from north to east
        slitwidth,  # width in arcsec units
        slitra,
        slitdec,
        )
    cut = np.where(abs(dist_co)<slitlength/2.)
    max_co     = np.nanmax(slit_co[cut])
    slit_co    = slit_co[cut] / max_co
    scatter_co = scatter_co[cut] / max_co
    dist_co    = dist_co[cut]

    # slice cont
    dist_cont, slit_cont, scatter_cont = get_slit(
        cont,
        slitangle, # from north to east
        slitwidth,  # width in arcsec units
        slitra,
        slitdec,
        )
    cut = np.where(abs(dist_cont)<slitlength/2.)
    max_cont     = np.nanmax(slit_cont[cut])
    slit_cont    = slit_cont[cut] / max_cont
    scatter_cont = scatter_cont[cut] / max_cont
    dist_cont    = dist_cont[cut]

    # summarize
    list_ci   = [dist_ci, slit_ci, scatter_ci, max_ci]
    list_co   = [dist_co, slit_co, scatter_co, max_co]
    list_cont = [dist_cont, slit_cont, scatter_cont, max_cont]

    return list_ci, list_co, list_cont

def get_slit(
    fitsimage,
    slitangle, # from north to east
    slitwidth, # width in arcsec units
    slitra,    # center, deg, string; e.g., "40.669625deg"
    slitdec,   # center, deg, string
    kernel=1./42.25, # LOWESS width
    ):
    """
    """

    slitangle = slitangle + 90

    # get values
    data, box = eazy_imval(fitsimage)
    data      = data["data"] * data["mask"]
    data      = data.flatten()

    # get ra and dec
    data_coords = imval(fitsimage,box=box)["coords"]
    ra_deg      = data_coords[:,:,0] * 180/np.pi
    ra_deg      = ra_deg.flatten()
    dec_deg     = data_coords[:,:,1] * 180/np.pi
    dec_deg     = dec_deg.flatten()

    # calc radius and theta
    slitra  = float(slitra.replace("deg",""))
    slitdec = float(slitdec.replace("deg",""))
    dist_as, theta = get_dist_theta(ra_deg, dec_deg, slitra, slitdec)
    theta   = np.where(theta<0,theta+360,theta)

    # get data along the defined slit
    boundary_right = dist_as * np.cos(np.radians(theta-slitangle))
    boundary_left  = -1 * dist_as * np.cos(np.radians(theta-slitangle))
    cut            = np.where((boundary_right<=slitwidth/2.0) & (boundary_left<=slitwidth/2.0) & (~np.isnan(data)) & (~np.isnan(ra_deg)) & (~np.isnan(dec_deg)))

    dist2 = dist_as * np.sin(np.radians(theta-slitangle))
    x, y  = lowess(np.array(dist2[cut]), np.array(data[cut]), f=kernel)

    return np.array(dist2[cut]), x, y

def lowess(x, y, f=1./3.):
    """
    Basic LOWESS smoother with uncertainty.
    Note:
        - Not robust (so no iteration) and only normally distributed errors. 
        - No higher order polynomials d=1 so linear smoother.

    Reference:
    https://james-brennan.github.io/posts/lowess_conf/
    """

    # get some paras
    xwidth = f*(x.max()-x.min()) # effective width after reduction factor

    N = len(x) # number of obs
    # Don't assume the data is sorted
    order = np.argsort(x)

    # storage
    y_sm = np.zeros_like(y)
    y_stderr = np.zeros_like(y)

    # define the weigthing function -- clipping too!
    tricube = lambda d : np.clip((1- np.abs(d)**3)**3, 0, 1)

    # run the regression for each observation i
    for i in range(N):
        dist = np.abs((x[order][i]-x[order]))/xwidth
        w = tricube(dist)
        # form linear system with the weights
        A = np.stack([w, x[order]*w]).T
        b = w * y[order]
        ATA = A.T.dot(A)
        ATb = A.T.dot(b)
        # solve the syste
        sol = np.linalg.solve(ATA, ATb)
        # predict for the observation only
        yest = A[i].dot(sol)# equiv of A.dot(yest) just for k
        place = order[i]
        y_sm[place]=yest 
        sigma2 = (np.sum((A.dot(sol) -y [order])**2)/N )
        # Calculate the standard error
        y_stderr[place] = np.sqrt(sigma2 * 
                                A[i].dot(np.linalg.inv(ATA)
                                                    ).dot(A[i]))
    
    return y_sm, y_stderr

############
# fit slit #
############
def f_gauss(x, a1, a2, a3, b1, b3, c1, c3):
    """
    """
    beam_as    = 0.8
    beam_sigma = beam_as / (2*np.sqrt(2*np.log(2)))

    f1 = a1 * np.exp( -(x-b1)**2 / (2*c1**2) )
    f2 = a2 * np.exp( -(x-0)**2 / (2*beam_sigma**2) )
    f3 = a3 * np.exp( -(x-b3)**2 / (2*c3**2) )
    
    return f1 + f2 + f3

###############
# plot 2D map #
###############

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
        ylim = [imsize_as/2.0, -imsize_as/2.0]
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
    """

    ##############
    # numann = 0 #
    ##############

    if number==0:
        fov_diamter = 16.5 # arcsec (12m+7m Band 8)

        efov1 = patches.Ellipse(xy=(-0,0), width=fov_diamter,
            height=fov_diamter, angle=0, fill=False, edgecolor="black",
            alpha=1.0, lw=3.5)
        ax.add_patch(efov1)

        efov2 = patches.Ellipse(xy=(-6.1992,-13.599972), width=fov_diamter,
            height=fov_diamter, angle=0, fill=False, edgecolor="black",
            alpha=1.0, lw=3.5)
        ax.add_patch(efov2)

        efov3 = patches.Ellipse(xy=(19.1016,1.400004), width=fov_diamter,
            height=fov_diamter, angle=0, fill=False, edgecolor="black",
            alpha=1.0, lw=3.5)
        ax.add_patch(efov3)

    ##################
    # numann = 1,2,3 #
    ##################

    if number==1:
        fov_diamter = 16.5 # arcsec (12m+7m Band 8)

        # FoV
        efov1 = patches.Ellipse(xy=(0,0), width=fov_diamter,
            height=fov_diamter, angle=0, fill=False, edgecolor="black",
            alpha=1.0, lw=3.5)
        ax.add_patch(efov1)

        # slit through cnd
        ax.plot([10,-10],[10*0.176,-10*0.176],"--",color="black",lw=2.5)
        ax.plot([5,-5],[5*np.sqrt(3),-5*np.sqrt(3)],"--",color="black",lw=2.5)

        ax.plot( 0.0,  0.0, "+", markersize=20, markeredgewidth=4, color="black")
        ax.plot( (40.66987917-40.66962133)*3600, (-0.01328889+0.01331803)*3600, "+", markersize=20, markeredgewidth=4, color="black")
        ax.plot( (40.66929167-40.66962133)*3600, (-0.01328889+0.01331803)*3600, "+", markersize=20, markeredgewidth=4, color="black")

        """
        # circular slit
        radius1 = 0.5
        radius2 = 2.0

        circ1 = patches.Ellipse(xy=(0,0), width=radius1*2, ls="dashed",
            height=radius1*1, angle=0, fill=False, edgecolor="black",
            alpha=1.0, lw=2.5)
        ax.add_patch(circ1)
        circ2 = patches.Ellipse(xy=(0,0), width=radius2*2, ls="dashed",
            height=radius2*1, angle=0, fill=False, edgecolor="black",
            alpha=1.0, lw=2.5)
        ax.add_patch(circ2)
        """

        if add_text==True:
            ax.text(1+0.2,1*np.sqrt(3)-0.2,"\"Outflow\" slit", color="black", weight="bold", rotation=-60, ha="right", va="bottom")
            ax.text(2.5+0.5,2.5*0.176-0.5,"\"CND\" slit", color="black", weight="bold", rotation=-10, ha="right", va="bottom")

    if number==2:
        fov_diamter = 16.5 # arcsec (12m+7m Band 8)

        efov2 = patches.Ellipse(xy=(0,0), width=fov_diamter,
            height=fov_diamter, angle=0, fill=False, edgecolor="black",
            alpha=1.0, lw=3.5)
        ax.add_patch(efov2)

        ax.plot( 0.0,  0.0, "+", markersize=20, markeredgewidth=4, color="black")

    if number==3:
        fov_diamter = 16.5 # arcsec (12m+7m Band 8)

        efov3 = patches.Ellipse(xy=(0,0), width=fov_diamter,
            height=fov_diamter, angle=0, fill=False, edgecolor="black",
            alpha=1.0, lw=3.5)
        ax.add_patch(efov3)

        ax.plot( 0.0,  0.0, "+", markersize=20, markeredgewidth=4, color="black")

def _myax_comment(ax,dec_cnt,xlim,ylim,comment,comment_color):
    if float(dec_cnt.replace("deg",""))>0:
        t = ax.text(min(xlim)*-0.9, max(ylim)*-0.9,
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

#######
# end #
#######