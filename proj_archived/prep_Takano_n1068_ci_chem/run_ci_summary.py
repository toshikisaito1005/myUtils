import os, re, sys, glob, math, scipy, pyfits
import importlib
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
plt.ioff() # turn off interactive X window

# import modules in the same directory
sys.path.append("./scripts/")
import mycasa_ci_supplement as ci_modules
reload(ci_modules)
exec(open("./scripts/define_param.py").read())
exec(open("./scripts/define_figure.py").read())

"""
a program which analyse ngc1068's ci10, co10, and band 8 dust continuum data.

- data reference:
2017.1.00586.S (CI eP1-3P0; PI: Takano Shuro, 12m+7m, 3-pointing)
2018.1.01684.S (CO 1-0; PI: Tosaki Tomoka, 12m+7m)

- imaging script:
phangs-alma imaging pipeline v2 (Leroy et al. 2021, ApJS, 255, 19)

- usage:
1. check hard-coded parameters in script_define_param.py
2. put this script in the directory which has the data_raw/ and scripts/ directories
3. run by execfile("script_ci_summary.py") in CASA5

- history:
2022-03-03   create this script, mycasa_plots.py, and script_define_param.py
2022-03-07   create plot_angle_cnd
2022-03-11   create plot_scatter
2022-03-13   create plot_spectra
Toshiki Saito@Nichidai/NAOJ
"""

###########
# toggles #
###########
# preparation for figure creation (running once is enough)
refresh_analysis     = False # step 0: delete dir_analysis (see the definition in script_define_param.py)
do_alingment         = False # step 1: regrid all the input FITS files into the grid of mom0_ci10 (see the definition in script_define_param.py)

# plot figures ready for paper
plot_whole_map       = False # fig_whole_galaxy_ci_co.png
plot_each_fov_map    = False # fig_fov*.png
plot_slit_outflow    = False # fig_slit_outflow.png
plot_slit_cnd        = False # fig_slit_cnd.png
plot_angle_cnd       = False # fig_circ_slit_cnd.png
plot_histogram_fov1  = False # fig_hist_fov1.png
plot_histogram_fov2  = False # fig_hist_fov2.png
plot_histogram_fov3  = False # fig_hist_fov3.png
plot_scatter_co_ci   = False # fig_scatter_co_ci.png
plot_scatter_cont_ci = False # fig_scatter_cont_ci.png
plot_spectra_agn     = True # outpng_spectra_*.png

##############################
# main part starts from here #
##############################

# refresh_analysis
if refresh_analysis==True:
    os.system("rm -rf " + dir_analysis) # delete analysis directory
    os.mkdir(dir_analysis)              # create it again
    os.system("rm -rf " + dir_figures)  # delete figure directory
    os.mkdir(dir_figures)               # create it again

# do_alingment
if do_alingment==True:
    tmpl = input_mom0_co10 # define template image (= co10 moment map)
    #
    ci_modules.eazy_imregrid(input_cube_co10,tmpl,cube_co10) # see mycasa_ci_supplement.py
    ci_modules.eazy_imregrid(input_mom0_co10,tmpl,mom0_co10)
    ci_modules.eazy_imregrid(input_mom1_co10,tmpl,mom1_co10)
    ci_modules.eazy_imregrid(input_mom2_co10,tmpl,mom2_co10)
    #
    ci_modules.eazy_imregrid(input_ecube_co10,tmpl,ecube_co10)
    ci_modules.eazy_imregrid(input_emom0_co10,tmpl,emom0_co10)
    ci_modules.eazy_imregrid(input_emom1_co10,tmpl,emom1_co10)
    ci_modules.eazy_imregrid(input_emom2_co10,tmpl,emom2_co10)
    #
    ci_modules.eazy_imregrid(input_cube_ci10,tmpl,cube_ci10)
    ci_modules.eazy_imregrid(input_mom0_ci10,tmpl,mom0_ci10)
    ci_modules.eazy_imregrid(input_mom1_ci10,tmpl,mom1_ci10)
    ci_modules.eazy_imregrid(input_mom2_ci10,tmpl,mom2_ci10)
    #
    ci_modules.eazy_imregrid(input_ecube_ci10,tmpl,ecube_ci10)
    ci_modules.eazy_imregrid(input_emom0_ci10,tmpl,emom0_ci10)
    ci_modules.eazy_imregrid(input_emom1_ci10,tmpl,emom1_ci10)
    ci_modules.eazy_imregrid(input_emom2_ci10,tmpl,emom2_ci10)
    #
    ci_modules.eazy_imregrid(input_map_cont_fov1,tmpl,map_cont_fov1,factor=1000,smooth_to=0.8)
    ci_modules.eazy_imregrid(input_map_cont_fov2,tmpl,map_cont_fov2,factor=1000,smooth_to=0.8)
    ci_modules.eazy_imregrid(input_map_cont_fov3,tmpl,map_cont_fov3,factor=1000,smooth_to=0.8)

# plot_whole_map
if plot_whole_map==True:
    ci_modules.myfig_fits2png(
        # general
        mom0_ci10,
        outpng_whole,
        imcontour1        = mom0_co10,
        imsize_as         = imsize_whole,
        ra_cnt            = ra_agn,
        dec_cnt           = dec_agn,
        # contour
        unit_cont1        = unit_cont_whole,
        levels_cont1      = levels_cont_whole,
        width_cont1       = [1.0],
        color_cont1       = color_cont_whole,
        # imshow
        set_title         = title_whole,
        colorlog          = colorlog_whole,
        set_bg_color      = "white",
        set_cmap          = set_cmap_whole,
        color_beam        = "black",
        scalebar          = scalebar_500pc,
        label_scalebar    = label_scalebar_500pc,
        color_scalebar    = "black",
        # imshow colorbar
        set_cbar          = True,
        clim              = None, # colorbar range, e.g., [0,100]
        label_cbar        = "(K km s$^{-1}$)",
        # annotation
        numann            = 0, # see mycasa_ci_supplement.py
        textann           = False, # see mycasa_ci_supplement.py
        colorbarticks     = cbarticks_whole,
        colorbarticktexts = cbarticktexts_whole,
        )

# plot_each_fov_map
if plot_each_fov_map==True:
    imcontour1 = mom0_co10

    #########
    # FoV-1 #
    #########
    # ci mom0
    ci_modules.myfig_fits2png(
        # general
        mom0_ci10,
        outpng_fov1_ci,
        imcontour1        = imcontour1,
        imsize_as         = imsize_fov1_ci,
        ra_cnt            = ra_fov1,
        dec_cnt           = dec_fov1,
        # contour
        unit_cont1        = unit_cont_fov1_ci,
        levels_cont1      = levels_cont_fov1_ci,
        width_cont1       = [1.0],
        color_cont1       = color_cont_fov1_ci,
        # imshow
        set_grid          = "None",
        set_title         = title_fov1_ci,
        colorlog          = colorlog_fov1_ci,
        set_bg_color      = "white",
        set_cmap          = set_cmap_fov1_ci,
        color_beam        = "black",
        scalebar          = scalebar_100pc,
        label_scalebar    = label_scalebar_100pc,
        color_scalebar    = "black",
        # imshow colorbar
        set_cbar          = True,
        clim              = None,
        label_cbar        = "(K km s$^{-1}$)",
        # annotation
        numann            = 1,
        textann           = True,
        colorbarticks     = cbarticks_fov1_ci,
        colorbarticktexts = cbarticktexts_fov1_ci,
        )

    # ci mom1
    ci_modules.myfig_fits2png(
        # general
        mom1_ci10,
        outpng_fov1_ci_mom1,
        imcontour1        = imcontour1,
        imsize_as         = imsize_fov1_ci,
        ra_cnt            = ra_fov1,
        dec_cnt           = dec_fov1,
        # contour
        unit_cont1        = unit_cont_fov1_ci,
        levels_cont1      = levels_cont_fov1_ci,
        width_cont1       = [1.0],
        color_cont1       = color_cont_fov1_ci,
        # imshow
        set_grid          = "None",
        set_title         = title_fov1_ci_mom1,
        colorlog          = False,
        set_bg_color      = "white",
        set_cmap          = set_cmap_fov1_ci,
        color_beam        = "black",
        scalebar          = scalebar_100pc,
        label_scalebar    = label_scalebar_100pc,
        color_scalebar    = "black",
        # imshow colorbar
        set_cbar          = True,
        clim              = clim_mom1,
        label_cbar        = "(km s$^{-1}$)",
        # annotation
        numann            = 1,
        textann           = True,
        colorbarticks     = None,
        colorbarticktexts = None,
        )

    # cont
    ci_modules.myfig_fits2png(
        # general
        map_cont_fov1,
        outpng_fov1_cont,
        imcontour1        = imcontour1,
        imsize_as         = imsize_fov1_cont,
        ra_cnt            = ra_fov1,
        dec_cnt           = dec_fov1,
        # contour
        unit_cont1        = unit_cont_fov1_cont,
        levels_cont1      = levels_cont_fov1_cont,
        width_cont1       = [1.0],
        color_cont1       = color_cont_fov1_cont,
        # imshow
        set_grid          = "None",
        set_title         = title_fov1_cont,
        colorlog          = colorlog_fov1_cont,
        set_bg_color      = "white",
        set_cmap          = set_cmap_fov1_cont,
        color_beam        = "black",
        scalebar          = scalebar_100pc,
        label_scalebar    = label_scalebar_100pc,
        color_scalebar    = "black",
        # imshow colorbar
        set_cbar          = True,
        clim              = clim_fov1_cont,
        label_cbar        = "(mJy beam$^{-1}$)",
        # annotation
        numann            = 1,
        textann           = True,
        colorbarticks     = cbarticks_fov1_cont,
        colorbarticktexts = cbarticktexts_fov1_cont,
        )

    #########
    # FoV-2 #
    #########
    # ci mom0
    ci_modules.myfig_fits2png(
        # general
        mom0_ci10,
        outpng_fov2_ci,
        imcontour1        = imcontour1,
        imsize_as         = imsize_fov2_ci,
        ra_cnt            = ra_fov2,
        dec_cnt           = dec_fov2,
        # contour
        unit_cont1        = unit_cont_fov2_ci,
        levels_cont1      = levels_cont_fov2_ci,
        width_cont1       = [1.0],
        color_cont1       = color_cont_fov2_ci,
        # imshow
        set_grid          = "None",
        set_title         = title_fov2_ci,
        colorlog          = colorlog_fov2_ci,
        set_bg_color      = "white",
        set_cmap          = set_cmap_fov2_ci,
        color_beam        = "black",
        scalebar          = scalebar_100pc,
        label_scalebar    = label_scalebar_100pc,
        color_scalebar    = "black",
        # imshow colorbar
        set_cbar          = True,
        clim              = None,
        label_cbar        = "(K km s$^{-1}$)",
        # annotation
        numann            = 2,
        textann           = True,
        colorbarticks     = cbarticks_fov2_ci,
        colorbarticktexts = cbarticktexts_fov2_ci,
        )

    # ci mom1
    ci_modules.myfig_fits2png(
        # general
        mom1_ci10,
        outpng_fov2_ci_mom1,
        imcontour1        = imcontour1,
        imsize_as         = imsize_fov2_ci,
        ra_cnt            = ra_fov2,
        dec_cnt           = dec_fov2,
        # contour
        unit_cont1        = unit_cont_fov2_ci,
        levels_cont1      = levels_cont_fov2_ci,
        width_cont1       = [1.0],
        color_cont1       = color_cont_fov2_ci,
        # imshow
        set_grid          = "None",
        set_title         = title_fov2_ci_mom1,
        colorlog          = False,
        set_bg_color      = "white",
        set_cmap          = set_cmap_fov2_ci,
        color_beam        = "black",
        scalebar          = scalebar_100pc,
        label_scalebar    = label_scalebar_100pc,
        color_scalebar    = "black",
        # imshow colorbar
        set_cbar          = True,
        clim              = clim_mom1,
        label_cbar        = "(km s$^{-1}$)",
        # annotation
        numann            = 2,
        textann           = True,
        colorbarticks     = None,
        colorbarticktexts = None,
        )

    # cont
    ci_modules.myfig_fits2png(
        # general
        map_cont_fov2,
        outpng_fov2_cont,
        imcontour1        = imcontour1,
        imsize_as         = imsize_fov2_cont,
        ra_cnt            = ra_fov2,
        dec_cnt           = dec_fov2,
        # contour
        unit_cont1        = unit_cont_fov2_cont,
        levels_cont1      = levels_cont_fov2_cont,
        width_cont1       = [1.0],
        color_cont1       = color_cont_fov2_cont,
        # imshow
        set_grid          = "None",
        set_title         = title_fov2_cont,
        colorlog          = colorlog_fov2_cont,
        set_bg_color      = "white",
        set_cmap          = set_cmap_fov2_cont,
        color_beam        = "black",
        scalebar          = scalebar_100pc,
        label_scalebar    = label_scalebar_100pc,
        color_scalebar    = "black",
        # imshow colorbar
        set_cbar          = True,
        clim              = clim_fov2_cont,
        label_cbar        = "(mJy beam$^{-1}$)",
        # annotation
        numann            = 2,
        textann           = True,
        colorbarticks     = cbarticks_fov2_cont,
        colorbarticktexts = cbarticktexts_fov2_cont,
        )

    #########
    # FoV-3 #
    #########
    # ci mom0
    ci_modules.myfig_fits2png(
        # general
        mom0_ci10,
        outpng_fov3_ci,
        imcontour1        = imcontour1,
        imsize_as         = imsize_fov3_ci,
        ra_cnt            = ra_fov3,
        dec_cnt           = dec_fov3,
        # contour
        unit_cont1        = unit_cont_fov3_ci,
        levels_cont1      = levels_cont_fov3_ci,
        width_cont1       = [1.0],
        color_cont1       = color_cont_fov3_ci,
        # imshow
        set_grid          = "None",
        set_title         = title_fov3_ci,
        colorlog          = colorlog_fov3_ci,
        set_bg_color      = "white",
        set_cmap          = set_cmap_fov3_ci,
        color_beam        = "black",
        scalebar          = scalebar_100pc,
        label_scalebar    = label_scalebar_100pc,
        color_scalebar    = "black",
        # imshow colorbar
        set_cbar          = True,
        clim              = None,
        label_cbar        = "(K km s$^{-1}$)",
        # annotation
        numann            = 3,
        textann           = True,
        colorbarticks     = cbarticks_fov3_ci,
        colorbarticktexts = cbarticktexts_fov3_ci,
        )

    # ci mom1
    ci_modules.myfig_fits2png(
        # general
        mom1_ci10,
        outpng_fov3_ci_mom1,
        imcontour1        = imcontour1,
        imsize_as         = imsize_fov3_ci,
        ra_cnt            = ra_fov3,
        dec_cnt           = dec_fov3,
        # contour
        unit_cont1        = unit_cont_fov3_ci,
        levels_cont1      = levels_cont_fov3_ci,
        width_cont1       = [1.0],
        color_cont1       = color_cont_fov3_ci,
        # imshow
        set_grid          = "None",
        set_title         = title_fov3_ci_mom1,
        colorlog          = False,
        set_bg_color      = "white",
        set_cmap          = set_cmap_fov3_ci,
        color_beam        = "black",
        scalebar          = scalebar_100pc,
        label_scalebar    = label_scalebar_100pc,
        color_scalebar    = "black",
        # imshow colorbar
        set_cbar          = True,
        clim              = clim_mom1,
        label_cbar        = "(km s$^{-1}$)",
        # annotation
        numann            = 3,
        textann           = True,
        colorbarticks     = None,
        colorbarticktexts = None,
        )
    
    # cont
    ci_modules.myfig_fits2png(
        # general
        map_cont_fov3,
        outpng_fov3_cont,
        imcontour1        = imcontour1,
        imsize_as         = imsize_fov3_cont,
        ra_cnt            = ra_fov3,
        dec_cnt           = dec_fov3,
        # contour
        unit_cont1        = unit_cont_fov3_cont,
        levels_cont1      = levels_cont_fov3_cont,
        width_cont1       = [1.0],
        color_cont1       = color_cont_fov3_cont,
        # imshow
        set_grid          = "None",
        set_title         = title_fov3_cont,
        colorlog          = colorlog_fov3_cont,
        set_bg_color      = "white",
        set_cmap          = set_cmap_fov3_cont,
        color_beam        = "black",
        scalebar          = scalebar_100pc,
        label_scalebar    = label_scalebar_100pc,
        color_scalebar    = "black",
        # imshow colorbar
        set_cbar          = True,
        clim              = clim_fov3_cont,
        label_cbar        = "(mJy beam$^{-1}$)",
        # annotation
        numann            = 3,
        textann           = True,
        colorbarticks     = cbarticks_fov3_cont,
        colorbarticktexts = cbarticktexts_fov3_cont,
        )

# plot_slit_outflow
if plot_slit_outflow==True:
    ###############
    # preparation #
    ###############
    list_ci, list_co, list_cont = ci_modules.get_slit_three(
        mom0_ci10,
        mom0_co10,
        map_cont_fov1,
        slit_angle_outflow,
        slit_length_outflow,
        slit_width_outflow,
        ra_agn,
        dec_agn,
        )

    ################
    # Gaussian fit #
    ################
    os.system("rm -rf " + outtxt_slit_outflow)
    f = open(outtxt_slit_outflow,"w")

    # fit ci10
    p0         = [0.8,0.1,1.0,-1.0,2.0,1.0,1.0] # initial guess
    cut        = np.where((list_ci[0]>=-2.0) & (list_ci[0]<=2.5))
    popt,pcov  = curve_fit(ci_modules.f_gauss, list_ci[0][cut], list_ci[1][cut], p0=p0, maxfev=10000)
    perr       = np.sqrt(np.diag(pcov))
    best_ci    = ci_modules.f_gauss(list_ci[0], *popt)

    f.write("#############################################\n")
    f.write("# ci10 Gauss 1:\n")
    f.write("# peak     = " + str( np.round(popt[0],3) ) + "+/-" + str( np.round(perr[0],3) ) + " [K.km/s]\n")
    f.write("# position = " + str( np.round(popt[3],2) ) + "+/-" + str( np.round(perr[3],3) ) + " [arcsec]\n")
    f.write("# FWHM     = " + str( np.round(popt[5]*2*np.sqrt(2*np.log(2)),2) ) + "+/-" + str( np.round(perr[5]*2*np.sqrt(2*np.log(2)),2) ) + " [arcsec]\n")
    f.write("#\n")
    f.write("# ci10 Gauss 2:\n")
    f.write("# peak     = " + str( np.round(popt[1],3) ) + "+/-" + str( np.round(perr[1],3) ) + " [K.km/s]\n")
    f.write("# position = 0 [arcsec] (fixed)\n")
    f.write("# FWHM     = beam * (2*np.sqrt(2*np.log(2))) [arcsec] (fixed)\n")
    f.write("#\n")
    f.write("# ci10 Gauss 3:\n")
    f.write("# peak     = " + str( np.round(popt[2],3) ) + "+/-" + str( np.round(perr[2],3) ) + " [K.km/s]\n")
    f.write("# position = " + str( np.round(popt[4],2) ) + "+/-" + str( np.round(perr[4],3) ) + " [arcsec]\n")
    f.write("# FWHM     = " + str( np.round(popt[6]*2*np.sqrt(2*np.log(2)),2) ) + "+/-" + str( np.round(perr[6]*2*np.sqrt(2*np.log(2)),2) ) + " [arcsec]\n")
    f.write("#############################################\n\n")

    # fit co10
    p0         = [1.0,0.1,0.3,-1.0,2.0,1.0,1.0] # initial guess
    cut        = np.where((list_co[0]>=-2.0) & (list_co[0]<=2.5))
    popt,pcov  = curve_fit(ci_modules.f_gauss, list_co[0][cut], list_co[1][cut], p0=p0, maxfev=10000)
    perr       = np.sqrt(np.diag(pcov))
    best_co    = ci_modules.f_gauss(list_co[0], *popt)

    f.write("#############################################\n")
    f.write("# co10 Gauss 1:\n")
    f.write("# peak     = " + str( np.round(popt[0],3) ) + "+/-" + str( np.round(perr[0],3) ) + " [K.km/s]\n")
    f.write("# position = " + str( np.round(popt[3],2) ) + "+/-" + str( np.round(perr[3],3) ) + " [arcsec]\n")
    f.write("# FWHM     = " + str( np.round(popt[5]*2*np.sqrt(2*np.log(2)),2) ) + "+/-" + str( np.round(perr[5]*2*np.sqrt(2*np.log(2)),2) ) + " [arcsec]\n")
    f.write("#\n")
    f.write("# co10 Gauss 2:\n")
    f.write("# peak     = " + str( np.round(popt[1],3) ) + "+/-" + str( np.round(perr[1],3) ) + " [K.km/s]\n")
    f.write("# position = 0 [arcsec] (fixed)\n")
    f.write("# FWHM     = beam * (2*np.sqrt(2*np.log(2))) [arcsec] (fixed)\n")
    f.write("#\n")
    f.write("# co10 Gauss 3:\n")
    f.write("# peak     = " + str( np.round(popt[2],3) ) + "+/-" + str( np.round(perr[2],3) ) + " [K.km/s]\n")
    f.write("# position = " + str( np.round(popt[4],2) ) + "+/-" + str( np.round(perr[4],3) ) + " [arcsec]\n")
    f.write("# FWHM     = " + str( np.round(popt[6]*2*np.sqrt(2*np.log(2)),2) ) + "+/-" + str( np.round(perr[6]*2*np.sqrt(2*np.log(2)),2) ) + " [arcsec]\n")
    f.write("#############################################\n\n")

    # fit cont
    p0         = [0.2,0.8,0.3,-0.5,1.5,1.0,0.5] # initial guess
    cut        = np.where((list_cont[0]>=-2.0) & (list_cont[0]<=2.5))
    popt,pcov  = curve_fit(ci_modules.f_gauss, list_cont[0][cut], list_cont[1][cut], p0=p0, maxfev=10000)
    perr       = np.sqrt(np.diag(pcov))
    best_cont  = ci_modules.f_gauss(list_cont[0], *popt)

    f.write("#############################################\n")
    f.write("# cont Gauss 1:\n")
    f.write("# peak     = " + str( np.round(popt[0],3) ) + "+/-" + str( np.round(perr[0],3) ) + " [K.km/s]\n")
    f.write("# position = " + str( np.round(popt[3],2) ) + "+/-" + str( np.round(perr[3],3) ) + " [arcsec]\n")
    f.write("# FWHM     = " + str( np.round(popt[5]*2*np.sqrt(2*np.log(2)),2) ) + "+/-" + str( np.round(perr[5]*2*np.sqrt(2*np.log(2)),2) ) + " [arcsec]\n")
    f.write("#\n")
    f.write("# cont Gauss 2:\n")
    f.write("# peak     = " + str( np.round(popt[1],3) ) + "+/-" + str( np.round(perr[1],3) ) + " [K.km/s]\n")
    f.write("# position = 0 [arcsec] (fixed)\n")
    f.write("# FWHM     = beam * (2*np.sqrt(2*np.log(2))) [arcsec] (fixed)\n")
    f.write("#\n")
    f.write("# cont Gauss 3:\n")
    f.write("# peak     = " + str( np.round(popt[2],3) ) + "+/-" + str( np.round(perr[2],3) ) + " [K.km/s]\n")
    f.write("# position = " + str( np.round(popt[4],2) ) + "+/-" + str( np.round(perr[4],3) ) + " [arcsec]\n")
    f.write("# FWHM     = " + str( np.round(popt[6]*2*np.sqrt(2*np.log(2)),2) ) + "+/-" + str( np.round(perr[6]*2*np.sqrt(2*np.log(2)),2) ) + " [arcsec]\n")
    f.write("#############################################\n")

    f.close()

    ########
    # plot #
    ########
    fig = plt.figure(figsize=(13,10))
    gs  = gridspec.GridSpec(nrows=10, ncols=10)
    ax1 = plt.subplot(gs[0:10,0:10])
    ad  = [0.215,0.83,0.10,0.90]
    ci_modules.myax_set(
        ax1,
        grid="both",
        xlim=[-slit_length_outflow/2.,slit_length_outflow/2.],
        ylim=[-0.1,1.1],title="\"Outflow\" slit",
        xlabel="Slit length (arcsec)",
        ylabel="Normalized intensity",
        adjust=ad,
        )

    ax1.scatter(
        list_ci[0],
        list_ci[1],
        marker="o",
        s=20,
        lw=0,
        alpha=1.0,
        color="red",
        zorder=1e9,
        )
    ax1.errorbar(
        list_ci[0],
        list_ci[1],
        yerr=list_ci[2],
        fmt=".",
        markersize=0,
        lw=2.5,
        color="tomato",
        capsize=0,
        zorder=1e8,
        )

    ax1.scatter(
        list_co[0],
        list_co[1],
        marker="o",
        s=20,
        lw=0,
        alpha=1.0,
        color="green",
        zorder=1e7,
        )
    ax1.errorbar(
        list_co[0],
        list_co[1],
        yerr=list_co[2],
        fmt=".",
        markersize=0,
        lw=2.5,
        color="lightgreen",
        capsize=0,
        zorder=1e6,
        )

    ax1.scatter(
        list_cont[0],
        list_cont[1],
        marker="o",
        s=20,
        lw=0,
        alpha=1.0,
        color="blue",
        zorder=1e5,
        )
    ax1.errorbar(
        list_cont[0],
        list_cont[1],
        yerr=list_cont[2],
        fmt=".",
        markersize=0,
        lw=2.5,
        color="deepskyblue",
        capsize=0,
        zorder=1e4,
        )

    # best fit (use these lines if you want to plot best-fit lines)
    #ax1.plot(list_ci[0], best_ci, ".", color="black")
    #ax1.plot(list_co[0], best_co, ".", color="black")
    #ax1.plot(list_cont[0], best_cont, ".", color="black")

    # ann
    ax1.text(
        0.05,
        0.90,
        "[CI]",
        color="tomato",
        weight="bold",
        transform=ax1.transAxes,
        )
    ax1.text(
        0.05,
        0.85,
        "CO",
        color="lightgreen",
        weight="bold",
        transform=ax1.transAxes,
        )
    ax1.text(
        0.05,
        0.80,
        "Continuum",
        color="deepskyblue",
        weight="bold",
        transform=ax1.transAxes,
        )

    # save
    os.system("rm -rf " + outpng_slit_outflow)
    plt.savefig(outpng_slit_outflow, dpi=200)

# plot_slit_cnd
if plot_slit_cnd==True:
    list_ci, list_co, list_cont = ci_modules.get_slit_three(
        mom0_ci10,
        mom0_co10,
        map_cont_fov1,
        slit_angle_cnd,
        slit_length_cnd,
        slit_width_cnd,
        ra_agn,
        dec_agn,
        )

    ################
    # Gaussian fit #
    ################
    os.system("rm -rf " + outtxt_slit_cnd)
    f = open(outtxt_slit_cnd,"w")

    # fit ci10
    p0         = [0.8,0.1,1.0,-1.0,2.0,1.0,1.0] # initial guess
    cut        = np.where((list_ci[0]>=-2.0) & (list_ci[0]<=2.5))
    popt,pcov  = curve_fit(ci_modules.f_gauss, list_ci[0][cut], list_ci[1][cut], p0=p0, maxfev=10000)
    perr       = np.sqrt(np.diag(pcov))
    best_ci    = ci_modules.f_gauss(list_ci[0], *popt)

    f.write("#############################################\n")
    f.write("# ci10 Gauss 1:\n")
    f.write("# peak     = " + str( np.round(popt[0],3) ) + "+/-" + str( np.round(perr[0],3) ) + " [K.km/s]\n")
    f.write("# position = " + str( np.round(popt[3],2) ) + "+/-" + str( np.round(perr[3],3) ) + " [arcsec]\n")
    f.write("# FWHM     = " + str( np.round(popt[5]*2*np.sqrt(2*np.log(2)),2) ) + "+/-" + str( np.round(perr[5]*2*np.sqrt(2*np.log(2)),2) ) + " [arcsec]\n")
    f.write("#\n")
    f.write("# ci10 Gauss 2:\n")
    f.write("# peak     = " + str( np.round(popt[1],3) ) + "+/-" + str( np.round(perr[1],3) ) + " [K.km/s]\n")
    f.write("# position = 0 [arcsec] (fixed)\n")
    f.write("# FWHM     = beam * (2*np.sqrt(2*np.log(2))) [arcsec] (fixed)\n")
    f.write("#\n")
    f.write("# ci10 Gauss 3:\n")
    f.write("# peak     = " + str( np.round(popt[2],3) ) + "+/-" + str( np.round(perr[2],3) ) + " [K.km/s]\n")
    f.write("# position = " + str( np.round(popt[4],2) ) + "+/-" + str( np.round(perr[4],3) ) + " [arcsec]\n")
    f.write("# FWHM     = " + str( np.round(popt[6]*2*np.sqrt(2*np.log(2)),2) ) + "+/-" + str( np.round(perr[6]*2*np.sqrt(2*np.log(2)),2) ) + " [arcsec]\n")
    f.write("#############################################\n\n")

    # fit co10
    p0         = [1.0,0.1,0.3,-1.0,2.0,1.0,1.0] # initial guess
    cut        = np.where((list_co[0]>=-2.0) & (list_co[0]<=2.5))
    popt,pcov  = curve_fit(ci_modules.f_gauss, list_co[0][cut], list_co[1][cut], p0=p0, maxfev=10000)
    perr       = np.sqrt(np.diag(pcov))
    best_co    = ci_modules.f_gauss(list_co[0], *popt)

    f.write("#############################################\n")
    f.write("# co10 Gauss 1:\n")
    f.write("# peak     = " + str( np.round(popt[0],3) ) + "+/-" + str( np.round(perr[0],3) ) + " [K.km/s]\n")
    f.write("# position = " + str( np.round(popt[3],2) ) + "+/-" + str( np.round(perr[3],3) ) + " [arcsec]\n")
    f.write("# FWHM     = " + str( np.round(popt[5]*2*np.sqrt(2*np.log(2)),2) ) + "+/-" + str( np.round(perr[5]*2*np.sqrt(2*np.log(2)),2) ) + " [arcsec]\n")
    f.write("#\n")
    f.write("# co10 Gauss 2:\n")
    f.write("# peak     = " + str( np.round(popt[1],3) ) + "+/-" + str( np.round(perr[1],3) ) + " [K.km/s]\n")
    f.write("# position = 0 [arcsec] (fixed)\n")
    f.write("# FWHM     = beam * (2*np.sqrt(2*np.log(2))) [arcsec] (fixed)\n")
    f.write("#\n")
    f.write("# co10 Gauss 3:\n")
    f.write("# peak     = " + str( np.round(popt[2],3) ) + "+/-" + str( np.round(perr[2],3) ) + " [K.km/s]\n")
    f.write("# position = " + str( np.round(popt[4],2) ) + "+/-" + str( np.round(perr[4],3) ) + " [arcsec]\n")
    f.write("# FWHM     = " + str( np.round(popt[6]*2*np.sqrt(2*np.log(2)),2) ) + "+/-" + str( np.round(perr[6]*2*np.sqrt(2*np.log(2)),2) ) + " [arcsec]\n")
    f.write("#############################################\n\n")

    # fit cont
    p0         = [0.2,0.8,0.3,-0.5,1.5,1.0,0.5] # initial guess
    cut        = np.where((list_cont[0]>=-2.0) & (list_cont[0]<=2.5))
    popt,pcov  = curve_fit(ci_modules.f_gauss, list_cont[0][cut], list_cont[1][cut], p0=p0, maxfev=10000)
    perr       = np.sqrt(np.diag(pcov))
    best_cont  = ci_modules.f_gauss(list_cont[0], *popt)

    f.write("#############################################\n")
    f.write("# cont Gauss 1:\n")
    f.write("# peak     = " + str( np.round(popt[0],3) ) + "+/-" + str( np.round(perr[0],3) ) + " [K.km/s]\n")
    f.write("# position = " + str( np.round(popt[3],2) ) + "+/-" + str( np.round(perr[3],3) ) + " [arcsec]\n")
    f.write("# FWHM     = " + str( np.round(popt[5]*2*np.sqrt(2*np.log(2)),2) ) + "+/-" + str( np.round(perr[5]*2*np.sqrt(2*np.log(2)),2) ) + " [arcsec]\n")
    f.write("#\n")
    f.write("# cont Gauss 2:\n")
    f.write("# peak     = " + str( np.round(popt[1],3) ) + "+/-" + str( np.round(perr[1],3) ) + " [K.km/s]\n")
    f.write("# position = 0 [arcsec] (fixed)\n")
    f.write("# FWHM     = beam * (2*np.sqrt(2*np.log(2))) [arcsec] (fixed)\n")
    f.write("#\n")
    f.write("# cont Gauss 3:\n")
    f.write("# peak     = " + str( np.round(popt[2],3) ) + "+/-" + str( np.round(perr[2],3) ) + " [K.km/s]\n")
    f.write("# position = " + str( np.round(popt[4],2) ) + "+/-" + str( np.round(perr[4],3) ) + " [arcsec]\n")
    f.write("# FWHM     = " + str( np.round(popt[6]*2*np.sqrt(2*np.log(2)),2) ) + "+/-" + str( np.round(perr[6]*2*np.sqrt(2*np.log(2)),2) ) + " [arcsec]\n")
    f.write("#############################################\n\n")

    f.close()

    ########
    # plot #
    ########
    fig = plt.figure(figsize=(13,10))
    gs  = gridspec.GridSpec(nrows=10, ncols=10)
    ax1 = plt.subplot(gs[0:10,0:10])
    ad  = [0.215,0.83,0.10,0.90]
    ci_modules.myax_set(
        ax1,
        grid="both",
        xlim=[-slit_length_cnd/2.,slit_length_cnd/2.],
        ylim=[-0.1,1.1],title="\"CND\" slit",
        xlabel="Slit length (arcsec)",
        ylabel="Normalized intensity",
        adjust=ad,
        )

    ax1.scatter(
        list_ci[0],
        list_ci[1],
        marker="o",
        s=20,
        lw=0,
        alpha=1.0,
        color="red",
        zorder=1e9,
        )
    ax1.errorbar(
        list_ci[0],
        list_ci[1],
        yerr=list_ci[2],
        fmt=".",
        markersize=0,
        lw=2.5,
        color="tomato",
        capsize=0,
        zorder=1e8,
        )

    ax1.scatter(
        list_co[0],
        list_co[1],
        marker="o",
        s=20,
        lw=0,
        alpha=1.0,
        color="green",
        zorder=1e7,
        )
    ax1.errorbar(
        list_co[0],
        list_co[1],
        yerr=list_co[2],
        fmt=".",
        markersize=0,
        lw=2.5,
        color="lightgreen",
        capsize=0,
        zorder=1e6,
        )

    ax1.scatter(
        list_cont[0],
        list_cont[1],
        marker="o",
        s=20,
        lw=0,
        alpha=1.0,
        color="blue",
        zorder=1e5,
        )
    ax1.errorbar(
        list_cont[0],
        list_cont[1],
        yerr=list_cont[2],
        fmt=".",
        markersize=0,
        lw=2.5,
        color="deepskyblue",
        capsize=0,
        zorder=1e4,
        )

    # best fit
    #ax1.plot(list_ci[0], best_ci, ".", color="grey")
    #ax1.plot(list_co[0], best_co, ".", color="grey")
    #ax1.plot(list_cont[0], best_cont, ".", color="grey")

    # save
    os.system("rm -rf " + outpng_slit_cnd)
    plt.savefig(outpng_slit_cnd, dpi=200)

# plot_angle_cnd
if plot_angle_cnd==True:
    list_ci, list_co, list_cont = ci_modules.get_circ_slit_three(
        mom0_ci10,
        mom0_co10,
        map_cont_fov1,
        circ_slit_outerr_radius,
        ra_agn,
        dec_agn,
        )

    ########
    # plot #
    ########
    fig = plt.figure(figsize=(13,10))
    gs  = gridspec.GridSpec(nrows=10, ncols=10)
    ax1 = plt.subplot(gs[0:10,0:10])
    ad  = [0.215,0.83,0.10,0.90]
    ci_modules.myax_set(
        ax1,
        grid="both",
        xlim=[180,-180],
        ylim=[-0.1,1.2],title="\"CND\" circular slit",
        xlabel="Slit angle (degree)",
        ylabel="Normalized intensity",
        adjust=ad,
        )

    ax1.scatter(
        list_ci[0],
        list_ci[1],
        marker="o",
        s=20,
        lw=0,
        alpha=1.0,
        color="red",
        zorder=1e9,
        )
    ax1.errorbar(
        list_ci[0],
        list_ci[1],
        yerr=list_ci[2],
        fmt=".",
        markersize=0,
        lw=8.0,
        color="tomato",
        capsize=0,
        zorder=1e8,
        )

    ax1.scatter(
        list_co[0],
        list_co[1],
        marker="o",
        s=20,
        lw=0,
        alpha=1.0,
        color="green",
        zorder=1e7,
        )
    ax1.errorbar(
        list_co[0],
        list_co[1],
        yerr=list_co[2],
        fmt=".",
        markersize=0,
        lw=8.0,
        color="lightgreen",
        capsize=0,
        zorder=1e6,
        )

    ax1.scatter(
        list_cont[0],
        list_cont[1],
        marker="o",
        s=20,
        lw=0,
        alpha=1.0,
        color="blue",
        zorder=1e5,
        )
    ax1.errorbar(
        list_cont[0],
        list_cont[1],
        yerr=list_cont[2],
        fmt=".",
        markersize=0,
        lw=8.0,
        color="deepskyblue",
        capsize=0,
        zorder=1e4,
        )

    # save
    os.system("rm -rf " + outpng_circ_slit_cnd)
    plt.savefig(outpng_circ_slit_cnd, dpi=200)

# plot_histogram
if plot_histogram_fov1==True:

    l,_       = ci_modules.eazy_imval(map_cont_fov1)
    l         = l["data"] * l["mask"]
    data_cont = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(mom0_ci10)
    l         = l["data"] * l["mask"]
    data_ci10 = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(emom0_ci10)
    l         = l["data"] * l["mask"]
    err_ci10  = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(mom0_co10)
    l         = l["data"] * l["mask"]
    data_co10 = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(emom0_co10)
    l         = l["data"] * l["mask"]
    err_co10  = np.array(l.flatten())

    cut = np.where((data_cont>np.log10(0.6*3)) & (data_ci10>err_ci10*snr_mom0) & (data_co10>err_co10*snr_mom0))
    data_cont = np.log10(data_cont[cut])
    data_ci10 = np.log10(data_ci10[cut])
    data_co10 = np.log10(data_co10[cut])

    # get area-weighted histograms
    x_cont, y_cont, _, width_cont = ci_modules.create_hist(
        data_cont,
        weights=None,
        histarea=None,
        bins=histgram_nbins,
        hrange=range_cont,
        )
    x0_cont, y0_cont = [], []
    for i in range(len(x_cont)):
        x0_cont.append(x_cont[i])
        x0_cont.append(x_cont[i]+width_cont)
        y0_cont.append(y_cont[i])
        y0_cont.append(y_cont[i])

    x_co10, y_co10, _, width_co10 = ci_modules.create_hist(
        data_co10,
        weights=None,
        histarea=None,
        bins=histgram_nbins,
        hrange=range_co10,
        )
    x0_co10, y0_co10 = [], []
    for i in range(len(x_co10)):
        x0_co10.append(x_co10[i])
        x0_co10.append(x_co10[i]+width_co10)
        y0_co10.append(y_co10[i])
        y0_co10.append(y_co10[i])

    x_ci10, y_ci10, _, width_ci10 = ci_modules.create_hist(
        data_ci10,
        weights=None,
        histarea=None,
        bins=histgram_nbins,
        hrange=range_ci10,
        )
    x0_ci10, y0_ci10 = [], []
    for i in range(len(x_ci10)):
        x0_ci10.append(x_ci10[i])
        x0_ci10.append(x_ci10[i]+width_ci10)
        y0_ci10.append(y_ci10[i])
        y0_ci10.append(y_ci10[i])

    # get intensity-weighted histograms
    xw_cont, yw_cont, _, widthw_cont = ci_modules.create_hist(
        data_cont,
        weights=10**data_cont,
        histarea=None,
        bins=histgram_nbins,
        hrange=range_cont,
        )
    xw_co10, yw_co10, _, widthw_co10 = ci_modules.create_hist(
        data_co10,
        weights=10**data_co10,
        histarea=None,
        bins=histgram_nbins,
        hrange=range_co10,
        )
    xw_ci10, yw_ci10, _, widthw_ci10 = ci_modules.create_hist(
        data_ci10,
        weights=10**data_ci10,
        histarea=None,
        bins=histgram_nbins,
        hrange=range_ci10,
        )

    ########
    # plot #
    ########
    fig = plt.figure(figsize=(13,10))
    gs  = gridspec.GridSpec(nrows=12, ncols=10)
    ax1 = plt.subplot(gs[0:3,0:10])
    ax2 = plt.subplot(gs[4:7,0:10])
    ax3 = plt.subplot(gs[8:11,0:10])
    ad  = [0.215,0.83,0.10,0.90]

    ci_modules.myax_set(ax1,grid="x",xlim=range_cont,ylim=ylim_cont,title="FoV-1: intensity histograms",xlabel=None,ylabel="Density",adjust=ad)
    ci_modules.myax_set(ax2,grid="x",xlim=range_co10,ylim=ylim_co10,title=None,xlabel=None,ylabel="Density",adjust=ad)
    ci_modules.myax_set(ax3,grid="x",xlim=range_ci10,ylim=ylim_ci10,title=None,xlabel="log Intensity",ylabel="Density",adjust=ad)

    # area-weighted histograms
    ax1.plot(x0_cont,y0_cont,lw=2,color="grey")
    ax2.plot(x0_co10,y0_co10,lw=2,color="grey")
    ax3.plot(x0_ci10,y0_ci10,lw=2,color="grey")

    # intensity-weighted histograms
    ax1.bar(xw_cont,yw_cont,width=widthw_cont,lw=0,alpha=1.0,color="deepskyblue")
    ax2.bar(xw_co10,yw_co10,width=widthw_co10,lw=0,alpha=1.0,color="lightgreen")
    ax3.bar(xw_ci10,yw_ci10,width=widthw_ci10,lw=0,alpha=1.0,color="tomato")

    # ann
    ax1.text(0.05,0.80,"Continuum",color="deepskyblue",weight="bold",transform=ax1.transAxes)
    ax1.text(0.95,0.80,"Intensity-weighted",color="deepskyblue",transform=ax1.transAxes,ha="right")
    ax1.text(0.95,0.65,"Area-weighted",color="grey",transform=ax1.transAxes,ha="right")
    ax2.text(0.05,0.80,"CO",color="lightgreen",weight="bold",transform=ax2.transAxes)
    ax2.text(0.95,0.80,"Intensity-weighted",color="lightgreen",transform=ax2.transAxes,ha="right")
    ax2.text(0.95,0.65,"Area-weighted",color="grey",transform=ax2.transAxes,ha="right")
    ax3.text(0.05,0.80,"[CI]",color="tomato",weight="bold",transform=ax3.transAxes)
    ax3.text(0.95,0.80,"Intensity-weighted",color="tomato",transform=ax3.transAxes,ha="right")
    ax3.text(0.95,0.65,"Area-weighted",color="grey",transform=ax3.transAxes,ha="right")

    # save
    os.system("rm -rf " + outpng_hist_fov1)
    plt.savefig(outpng_hist_fov1, dpi=200)

if plot_histogram_fov2==True:

    l,_       = ci_modules.eazy_imval(map_cont_fov2)
    l         = l["data"] * l["mask"]
    data_cont = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(mom0_ci10)
    l         = l["data"] * l["mask"]
    data_ci10 = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(emom0_ci10)
    l         = l["data"] * l["mask"]
    err_ci10  = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(mom0_co10)
    l         = l["data"] * l["mask"]
    data_co10 = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(emom0_co10)
    l         = l["data"] * l["mask"]
    err_co10  = np.array(l.flatten())

    cut = np.where((data_cont>np.log10(0.6*3)) & (data_ci10>err_ci10*snr_mom0) & (data_co10>err_co10*snr_mom0))
    data_cont = np.log10(data_cont[cut])
    data_ci10 = np.log10(data_ci10[cut])
    data_co10 = np.log10(data_co10[cut])

    # get area-weighted histograms
    x_cont, y_cont, _, width_cont = ci_modules.create_hist(
        data_cont,
        weights=None,
        histarea=None,
        bins=histgram_nbins,
        hrange=range_cont,
        )
    x0_cont, y0_cont = [], []
    for i in range(len(x_cont)):
        x0_cont.append(x_cont[i])
        x0_cont.append(x_cont[i]+width_cont)
        y0_cont.append(y_cont[i])
        y0_cont.append(y_cont[i])

    x_co10, y_co10, _, width_co10 = ci_modules.create_hist(
        data_co10,
        weights=None,
        histarea=None,
        bins=histgram_nbins,
        hrange=range_co10,
        )
    x0_co10, y0_co10 = [], []
    for i in range(len(x_co10)):
        x0_co10.append(x_co10[i])
        x0_co10.append(x_co10[i]+width_co10)
        y0_co10.append(y_co10[i])
        y0_co10.append(y_co10[i])

    x_ci10, y_ci10, _, width_ci10 = ci_modules.create_hist(
        data_ci10,
        weights=None,
        histarea=None,
        bins=histgram_nbins,
        hrange=range_ci10,
        )
    x0_ci10, y0_ci10 = [], []
    for i in range(len(x_ci10)):
        x0_ci10.append(x_ci10[i])
        x0_ci10.append(x_ci10[i]+width_ci10)
        y0_ci10.append(y_ci10[i])
        y0_ci10.append(y_ci10[i])

    # get intensity-weighted histograms
    xw_cont, yw_cont, _, widthw_cont = ci_modules.create_hist(
        data_cont,
        weights=10**data_cont,
        histarea=None,
        bins=histgram_nbins,
        hrange=range_cont,
        )
    xw_co10, yw_co10, _, widthw_co10 = ci_modules.create_hist(
        data_co10,
        weights=10**data_co10,
        histarea=None,
        bins=histgram_nbins,
        hrange=range_co10,
        )
    xw_ci10, yw_ci10, _, widthw_ci10 = ci_modules.create_hist(
        data_ci10,
        weights=10**data_ci10,
        histarea=None,
        bins=histgram_nbins,
        hrange=range_ci10,
        )

    # plot
    fig = plt.figure(figsize=(13,10))
    gs  = gridspec.GridSpec(nrows=12, ncols=10)
    ax1 = plt.subplot(gs[0:3,0:10])
    ax2 = plt.subplot(gs[4:7,0:10])
    ax3 = plt.subplot(gs[8:11,0:10])
    ad  = [0.215,0.83,0.10,0.90]
    ci_modules.myax_set(ax1,grid="x",xlim=range_cont,ylim=ylim_cont,title="FoV-2: intensity histogram",xlabel=None,ylabel="Density",adjust=ad)
    ci_modules.myax_set(ax2,grid="x",xlim=range_co10,ylim=ylim_co10,title=None,xlabel=None,ylabel="Density",adjust=ad)
    ci_modules.myax_set(ax3,grid="x",xlim=range_ci10,ylim=ylim_ci10,title=None,xlabel="log Intensity",ylabel="Density",adjust=ad)

    # area-weighted histograms
    ax1.plot(x0_cont,y0_cont,lw=2,color="grey")
    ax2.plot(x0_co10,y0_co10,lw=2,color="grey")
    ax3.plot(x0_ci10,y0_ci10,lw=2,color="grey")

    # intensity-weighted histograms
    ax1.bar(xw_cont,yw_cont,width=widthw_cont,lw=0,alpha=1.0,color="deepskyblue")
    ax2.bar(xw_co10,yw_co10,width=widthw_co10,lw=0,alpha=1.0,color="lightgreen")
    ax3.bar(xw_ci10,yw_ci10,width=widthw_ci10,lw=0,alpha=1.0,color="tomato")

    # save
    os.system("rm -rf " + outpng_hist_fov2)
    plt.savefig(outpng_hist_fov2, dpi=200)

if plot_histogram_fov3==True:

    l,_       = ci_modules.eazy_imval(map_cont_fov3)
    l         = l["data"] * l["mask"]
    data_cont = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(mom0_ci10)
    l         = l["data"] * l["mask"]
    data_ci10 = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(emom0_ci10)
    l         = l["data"] * l["mask"]
    err_ci10  = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(mom0_co10)
    l         = l["data"] * l["mask"]
    data_co10 = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(emom0_co10)
    l         = l["data"] * l["mask"]
    err_co10  = np.array(l.flatten())

    cut = np.where((data_cont>np.log10(0.6*3)) & (data_ci10>err_ci10*snr_mom0) & (data_co10>err_co10*snr_mom0))
    data_cont = np.log10(data_cont[cut])
    data_ci10 = np.log10(data_ci10[cut])
    data_co10 = np.log10(data_co10[cut])

    # get area-weighted histograms
    x_cont, y_cont, _, width_cont = ci_modules.create_hist(
        data_cont,
        weights=None,
        histarea=None,
        bins=histgram_nbins,
        hrange=range_cont,
        )
    x0_cont, y0_cont = [], []
    for i in range(len(x_cont)):
        x0_cont.append(x_cont[i])
        x0_cont.append(x_cont[i]+width_cont)
        y0_cont.append(y_cont[i])
        y0_cont.append(y_cont[i])

    x_co10, y_co10, _, width_co10 = ci_modules.create_hist(
        data_co10,
        weights=None,
        histarea=None,
        bins=histgram_nbins,
        hrange=range_co10,
        )
    x0_co10, y0_co10 = [], []
    for i in range(len(x_co10)):
        x0_co10.append(x_co10[i])
        x0_co10.append(x_co10[i]+width_co10)
        y0_co10.append(y_co10[i])
        y0_co10.append(y_co10[i])

    x_ci10, y_ci10, _, width_ci10 = ci_modules.create_hist(
        data_ci10,
        weights=None,
        histarea=None,
        bins=histgram_nbins,
        hrange=range_ci10,
        )
    x0_ci10, y0_ci10 = [], []
    for i in range(len(x_ci10)):
        x0_ci10.append(x_ci10[i])
        x0_ci10.append(x_ci10[i]+width_ci10)
        y0_ci10.append(y_ci10[i])
        y0_ci10.append(y_ci10[i])

    # get intensity-weighted histograms
    xw_cont, yw_cont, _, widthw_cont = ci_modules.create_hist(
        data_cont,
        weights=10**data_cont,
        histarea=None,
        bins=histgram_nbins,
        hrange=range_cont,
        )
    xw_co10, yw_co10, _, widthw_co10 = ci_modules.create_hist(
        data_co10,
        weights=10**data_co10,
        histarea=None,
        bins=histgram_nbins,
        hrange=range_co10,
        )
    xw_ci10, yw_ci10, _, widthw_ci10 = ci_modules.create_hist(
        data_ci10,
        weights=10**data_ci10,
        histarea=None,
        bins=histgram_nbins,
        hrange=range_ci10,
        )

    # plot
    fig = plt.figure(figsize=(13,10))
    gs  = gridspec.GridSpec(nrows=12, ncols=10)
    ax1 = plt.subplot(gs[0:3,0:10])
    ax2 = plt.subplot(gs[4:7,0:10])
    ax3 = plt.subplot(gs[8:11,0:10])
    ad  = [0.215,0.83,0.10,0.90]
    ci_modules.myax_set(ax1,grid="x",xlim=range_cont,ylim=ylim_cont,title="FoV-3: Intensity histogram",xlabel=None,ylabel="Density",adjust=ad)
    ci_modules.myax_set(ax2,grid="x",xlim=range_co10,ylim=ylim_co10,title=None,xlabel=None,ylabel="Density",adjust=ad)
    ci_modules.myax_set(ax3,grid="x",xlim=range_ci10,ylim=ylim_ci10,title=None,xlabel="log Intensity",ylabel="Density",adjust=ad)

    # area-weighted histograms
    ax1.plot(x0_cont,y0_cont,lw=2,color="grey")
    ax2.plot(x0_co10,y0_co10,lw=2,color="grey")
    ax3.plot(x0_ci10,y0_ci10,lw=2,color="grey")

    # intensity-weighted histograms
    ax1.bar(xw_cont,yw_cont,width=widthw_cont,lw=0,alpha=1.0,color="deepskyblue")
    ax2.bar(xw_co10,yw_co10,width=widthw_co10,lw=0,alpha=1.0,color="lightgreen")
    ax3.bar(xw_ci10,yw_ci10,width=widthw_ci10,lw=0,alpha=1.0,color="tomato")

    # save
    os.system("rm -rf " + outpng_hist_fov3)
    plt.savefig(outpng_hist_fov3, dpi=200)

if plot_scatter_co_ci==True:

    #################
    # get data fov1 #
    #################
    l,_       = ci_modules.eazy_imval(map_cont_fov1)
    l         = l["data"] * l["mask"]
    data_cont = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(mom0_ci10)
    l         = l["data"] * l["mask"]
    data_ci10 = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(emom0_ci10)
    l         = l["data"] * l["mask"]
    err_ci10  = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(mom0_co10)
    l         = l["data"] * l["mask"]
    data_co10 = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(emom0_co10)
    l         = l["data"] * l["mask"]
    err_co10  = np.array(l.flatten())

    cut = np.where((data_cont>np.log10(0.6*3)) & (data_ci10>err_ci10*snr_mom0) & (data_co10>err_co10*snr_mom0))
    data_cont_fov1 = np.log10(data_cont[cut])
    data_ci10_fov1 = np.log10(data_ci10[cut])
    data_co10_fov1 = np.log10(data_co10[cut])

    #################
    # get data fov2 #
    #################
    l,_       = ci_modules.eazy_imval(map_cont_fov2)
    l         = l["data"] * l["mask"]
    data_cont = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(mom0_ci10)
    l         = l["data"] * l["mask"]
    data_ci10 = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(emom0_ci10)
    l         = l["data"] * l["mask"]
    err_ci10  = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(mom0_co10)
    l         = l["data"] * l["mask"]
    data_co10 = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(emom0_co10)
    l         = l["data"] * l["mask"]
    err_co10  = np.array(l.flatten())

    cut = np.where((data_cont>np.log10(0.6*3)) & (data_ci10>err_ci10*snr_mom0) & (data_co10>err_co10*snr_mom0))
    data_cont_fov2 = np.log10(data_cont[cut])
    data_ci10_fov2 = np.log10(data_ci10[cut])
    data_co10_fov2 = np.log10(data_co10[cut])

    #################
    # get data fov3 #
    #################
    l,_       = ci_modules.eazy_imval(map_cont_fov3)
    l         = l["data"] * l["mask"]
    data_cont = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(mom0_ci10)
    l         = l["data"] * l["mask"]
    data_ci10 = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(emom0_ci10)
    l         = l["data"] * l["mask"]
    err_ci10  = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(mom0_co10)
    l         = l["data"] * l["mask"]
    data_co10 = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(emom0_co10)
    l         = l["data"] * l["mask"]
    err_co10  = np.array(l.flatten())

    cut = np.where((data_cont>np.log10(0.6*3)) & (data_ci10>err_ci10*snr_mom0) & (data_co10>err_co10*snr_mom0))
    data_cont_fov3 = np.log10(data_cont[cut])
    data_ci10_fov3 = np.log10(data_ci10[cut])
    data_co10_fov3 = np.log10(data_co10[cut])

    ########
    # plot #
    ########
    fig = plt.figure(figsize=(13,10))
    gs  = gridspec.GridSpec(nrows=10, ncols=10)
    ax1 = plt.subplot(gs[0:10,0:10])
    ad  = [0.215,0.83,0.10,0.90]
    ci_modules.myax_set(ax1,grid="both",xlim=xlim_co_ci,ylim=ylim_co_ci,title=None,xlabel=None,ylabel=None,adjust=ad)

    ax1.scatter(data_co10_fov1,data_ci10_fov1,s=30,color="tomato")
    ax1.scatter(data_co10_fov2,data_ci10_fov2,s=30,color="lightgreen")
    ax1.scatter(data_co10_fov3,data_ci10_fov3,s=30,color="deepskyblue")

    # line
    ax1.plot(xlim_co_ci,ylim_co_ci,"--",color="black",lw=1)
    ax1.plot(xlim_co_ci,[ylim_co_ci[0]-1.0,ylim_co_ci[1]-1.0],"--",color="black",lw=1)

    # ann
    ax1.text(0.05,0.95,"FoV-1",color="tomato",weight="bold",transform=ax1.transAxes)
    ax1.text(0.05,0.90,"FoV-2",color="lightgreen",weight="bold",transform=ax1.transAxes)
    ax1.text(0.05,0.85,"FoV-3",color="deepskyblue",weight="bold",transform=ax1.transAxes)

    # save
    os.system("rm -rf " + outpng_scatter_co_ci)
    plt.savefig(outpng_scatter_co_ci, dpi=200)

if plot_scatter_cont_ci==True:

    #################
    # get data fov1 #
    #################
    l,_       = ci_modules.eazy_imval(map_cont_fov1)
    l         = l["data"] * l["mask"]
    data_cont = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(mom0_ci10)
    l         = l["data"] * l["mask"]
    data_ci10 = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(emom0_ci10)
    l         = l["data"] * l["mask"]
    err_ci10  = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(mom0_co10)
    l         = l["data"] * l["mask"]
    data_co10 = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(emom0_co10)
    l         = l["data"] * l["mask"]
    err_co10  = np.array(l.flatten())

    cut = np.where((data_cont>np.log10(0.6*3)) & (data_ci10>err_ci10*snr_mom0) & (data_co10>err_co10*snr_mom0))
    data_cont_fov1 = np.log10(data_cont[cut])
    data_ci10_fov1 = np.log10(data_ci10[cut])
    data_co10_fov1 = np.log10(data_co10[cut])

    #################
    # get data fov2 #
    #################
    l,_       = ci_modules.eazy_imval(map_cont_fov2)
    l         = l["data"] * l["mask"]
    data_cont = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(mom0_ci10)
    l         = l["data"] * l["mask"]
    data_ci10 = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(emom0_ci10)
    l         = l["data"] * l["mask"]
    err_ci10  = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(mom0_co10)
    l         = l["data"] * l["mask"]
    data_co10 = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(emom0_co10)
    l         = l["data"] * l["mask"]
    err_co10  = np.array(l.flatten())

    cut = np.where((data_cont>np.log10(0.6*3)) & (data_ci10>err_ci10*snr_mom0) & (data_co10>err_co10*snr_mom0))
    data_cont_fov2 = np.log10(data_cont[cut])
    data_ci10_fov2 = np.log10(data_ci10[cut])
    data_co10_fov2 = np.log10(data_co10[cut])

    #################
    # get data fov3 #
    #################
    l,_       = ci_modules.eazy_imval(map_cont_fov3)
    l         = l["data"] * l["mask"]
    data_cont = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(mom0_ci10)
    l         = l["data"] * l["mask"]
    data_ci10 = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(emom0_ci10)
    l         = l["data"] * l["mask"]
    err_ci10  = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(mom0_co10)
    l         = l["data"] * l["mask"]
    data_co10 = np.array(l.flatten())

    l,_       = ci_modules.eazy_imval(emom0_co10)
    l         = l["data"] * l["mask"]
    err_co10  = np.array(l.flatten())

    cut = np.where((data_cont>np.log10(0.6*3)) & (data_ci10>err_ci10*snr_mom0) & (data_co10>err_co10*snr_mom0))
    data_cont_fov3 = np.log10(data_cont[cut])
    data_ci10_fov3 = np.log10(data_ci10[cut])
    data_co10_fov3 = np.log10(data_co10[cut])

    ########
    # plot #
    ########
    fig = plt.figure(figsize=(13,10))
    gs  = gridspec.GridSpec(nrows=10, ncols=10)
    ax1 = plt.subplot(gs[0:10,0:10])
    ad  = [0.215,0.83,0.10,0.90]
    ci_modules.myax_set(ax1,grid="both",xlim=xlim_cont_ci,ylim=ylim_cont_ci,title=None,xlabel=None,ylabel=None,adjust=ad)

    ax1.scatter(data_cont_fov1,data_ci10_fov1,s=30,color="tomato")
    ax1.scatter(data_cont_fov2,data_ci10_fov2,s=30,color="lightgreen")
    ax1.scatter(data_cont_fov3,data_ci10_fov3,s=30,color="deepskyblue")

    # line
    #ax1.plot(xlim_cont_ci,ylim_co_ci,"--",color="black",lw=1)
    #ax1.plot(xlim_cont_ci,[ylim_co_ci[0]-1.0,ylim_co_ci[1]-1.0],"--",color="black",lw=1)

    # ann
    ax1.text(0.05,0.95,"FoV-1",color="tomato",weight="bold",transform=ax1.transAxes)
    ax1.text(0.05,0.90,"FoV-2",color="lightgreen",weight="bold",transform=ax1.transAxes)
    ax1.text(0.05,0.85,"FoV-3",color="deepskyblue",weight="bold",transform=ax1.transAxes)

    # save
    os.system("rm -rf " + outpng_scatter_cont_ci)
    plt.savefig(outpng_scatter_cont_ci, dpi=200)

# plot_spectra
if plot_spectra_agn==True:
    # FoV-1 = AGN
    ci_modules.plot_spec(
        mom0_co10,
        cube_ci10,
        cube_co10,
        ecube_ci10,
        ecube_co10,
        ra_agn,
        dec_agn,
        outpng_spectra_agn,
        [-0.9,2.0],
        )

    # FoV-2
    ci_modules.plot_spec(
        mom0_co10,
        cube_ci10,
        cube_co10,
        ecube_ci10,
        ecube_co10,
        ra_fov2,
        dec_fov2,
        outpng_spectra_fov2,
        None, # [-0.9,2.0],
        )

    # FoV-3
    ci_modules.plot_spec(
        mom0_co10,
        cube_ci10,
        cube_co10,
        ecube_ci10,
        ecube_co10,
        ra_fov3,
        dec_fov3,
        outpng_spectra_fov3,
        None, # [-0.9,2.0],
        )

    # E-knot
    ci_modules.plot_spec(
        mom0_co10,
        cube_ci10,
        cube_co10,
        ecube_ci10,
        ecube_co10,
        ra_eknot,
        dec_eknot,
        outpng_spectra_eknot,
        None, # [-0.9,2.0],
        )

    # W-knot
    ci_modules.plot_spec(
        mom0_co10,
        cube_ci10,
        cube_co10,
        ecube_ci10,
        ecube_co10,
        ra_wknot,
        dec_wknot,
        outpng_spectra_wknot,
        None, # [-0.9,2.0],
        )

#######
# end #
#######