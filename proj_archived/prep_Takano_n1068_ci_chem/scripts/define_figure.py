import os
import sys
import glob

"""
hard-code...
- tbe
"""

################
# ngc1068 info #
################
clim_mom1 = [966,1266]
freq_ci10 = 492.16065100e9
freq_co10 = 115.27120180e9

# AGN position = FoV-1 (Gallimore et al. 2004)
# SkyCoord('02h42m40.70912s', '-00d00m47.9449s')
ra_agn   = "40.66962133deg"
dec_agn  = "-0.01331803deg"
ra_fov1  = ra_agn
dec_fov1 = dec_agn

# SBring SW = FoV-2 (Nakajima et al. 2015)
# SkyCoord('02h42m40.298s', '-00d01m01.638s')
ra_fov2  = "40.66790833deg"
dec_fov2 = "-0.01712167deg"

# SBring CH3OH peak = FoV-3 (obtained from Tosaki fits data, measured with CASA viewer (2016.4.14))
# SkyCoord('02h42m41.993s','-00d00m46.46s')
ra_fov3  = "40.67497083deg"
dec_fov3 = "-0.01290556deg"

# E-knot (Table 2 of Viti et al. 2004)
# SkyCoord('02h42m40.771s','-00d00m47.84s')
ra_eknot  = "40.66987917deg"
dec_eknot = "-0.01328889deg"

# W-knot (Table 2 of Viti et al. 2004)
# SkyCoord('02h42m40.630s','-00d00m47.84s')
ra_wknot  = "40.66929167deg"
dec_wknot = "-0.01328889deg"

####################
# scale at ngc1068 # scale bar for figures
####################
scale_pc = 72 # 1 arcsec
scalebar_100pc = 100. / scale_pc
label_scalebar_100pc = "100 pc"
scalebar_500pc = 500. / scale_pc
label_scalebar_500pc = "500 pc"

##################
# plot_whole_map #
##################
outpng_whole      = dir_figures + "fig_whole_galaxy_ci_co.png"
imsize_whole      = 57
title_whole       = "[CI] $^3P_1$-$^3P_0$ (color) + CO(1-0) (contour)"
unit_cont_whole   = None # if None, set ot np.max
levels_cont_whole = [0.02,0.04,0.08,0.16,0.32,0.64,0.96]
color_cont_whole  = "black"
colorlog_whole    = True
set_cmap_whole    = "rainbow" # google "matpltolib cmap"
cbarticks_whole   = [10**1.5,10**2,10**2.5,10**2.85]
cbarticktexts_whole = ["10$^{1.5}$","10$^{2.0}$","10$^{2.5}$","10$^{2.85}$"]

#####################
# plot_each_fov_map #
#####################
# fov-1 ci10 mom0
outpng_fov1_ci      = dir_figures + "fig_fov1_ci10_mom0.png"
imsize_fov1_ci      = 17
title_fov1_ci       = "FoV-1: [CI] $^3P_1$-$^3P_0$"
unit_cont_fov1_ci   = None # if None, set ot np.max
levels_cont_fov1_ci = [0.02,0.04,0.08,0.16,0.32,0.64,0.96]
color_cont_fov1_ci  = "black"
colorlog_fov1_ci    = True
set_cmap_fov1_ci    = "rainbow"
cbarticks_fov1_ci   = None
cbarticktexts_fov1_ci = None

# fov-1 ci10 mom1
outpng_fov1_ci_mom1 = dir_figures + "fig_fov1_ci10_mom1.png"
title_fov1_ci_mom1  = "FoV-1: [CI] velocity"

# fov-1 cont
outpng_fov1_cont      = dir_figures + "fig_fov1_cont.png"
imsize_fov1_cont      = 17
title_fov1_cont       = "FoV-1: 610 um continuum"
unit_cont_fov1_cont   = None # if None, set ot np.max
levels_cont_fov1_cont = [0.02,0.04,0.08,0.16,0.32,0.64,0.96]
color_cont_fov1_cont  = "black"
colorlog_fov1_cont    = False
set_cmap_fov1_cont    = "rainbow"
clim_fov1_cont        = [-1.5,33]
cbarticks_fov1_cont   = None
cbarticktexts_fov1_cont = None

# fov-2 ci10
outpng_fov2_ci      = dir_figures + "fig_fov2_ci10_mom0.png"
imsize_fov2_ci      = 17
title_fov2_ci       = "FoV-2: [CI] $^3P_1$-$^3P_0$"
unit_cont_fov2_ci   = None
levels_cont_fov2_ci = [0.02,0.04,0.08,0.16,0.32,0.64,0.96]
color_cont_fov2_ci  = "black"
colorlog_fov2_ci    = True
set_cmap_fov2_ci    = "rainbow"
cbarticks_fov2_ci   = None
cbarticktexts_fov2_ci = None

# fov-2 ci10 mom1
outpng_fov2_ci_mom1 = dir_figures + "fig_fov2_ci10_mom1.png"
title_fov2_ci_mom1  = "FoV-2: [CI] velocity"

# fov-2 cont
outpng_fov2_cont      = dir_figures + "fig_fov2_cont.png"
imsize_fov2_cont      = 17
title_fov2_cont       = "FoV-2: 610 um continuum"
unit_cont_fov2_cont   = None
levels_cont_fov2_cont = [0.02,0.04,0.08,0.16,0.32,0.64,0.96]
color_cont_fov2_cont  = "black"
colorlog_fov2_cont    = False
set_cmap_fov2_cont    = "rainbow"
clim_fov2_cont        = [-1.5,24]
cbarticks_fov2_cont   = None
cbarticktexts_fov2_cont = None

# fov-3 ci10
outpng_fov3_ci      = dir_figures + "fig_fov3_ci10_mom0.png"
imsize_fov3_ci      = 17
title_fov3_ci       = "FoV-3: [CI] $^3P_1$-$^3P_0$"
unit_cont_fov3_ci   = None
levels_cont_fov3_ci = [0.02,0.04,0.08,0.16,0.32,0.64,0.96]
color_cont_fov3_ci  = "black"
colorlog_fov3_ci    = True
set_cmap_fov3_ci    = "rainbow"
cbarticks_fov3_ci   = None
cbarticktexts_fov3_ci = None

# fov-3 ci10 mom1
outpng_fov3_ci_mom1 = dir_figures + "fig_fov3_ci10_mom1.png"
title_fov3_ci_mom1  = "FoV-3: [CI] velocity"

# fov-3 cont
outpng_fov3_cont      = dir_figures + "fig_fov3_cont.png"
imsize_fov3_cont      = 17
title_fov3_cont       = "FoV-3: 610 um continuum"
unit_cont_fov3_cont   = None
levels_cont_fov3_cont = [0.02,0.04,0.08,0.16,0.32,0.64,0.96]
color_cont_fov3_cont  = "black"
colorlog_fov3_cont    = False
set_cmap_fov3_cont    = "rainbow"
clim_fov3_cont        = [-1.5,4.8]
cbarticks_fov3_cont   = None 
cbarticktexts_fov3_cont = None

#####################
# plot_slit_outflow #
#####################
outpng_slit_outflow = dir_figures + "fig_slit_outflow.png"
outtxt_slit_outflow = dir_figures + "fig_slit_outflow.txt"
slit_angle_outflow  = 30.0
slit_width_outflow  = 0.8
slit_length_outflow = 16.5

#################
# plot_slit_cnd #
#################
outpng_slit_cnd = dir_figures + "fig_slit_cnd.png"
outtxt_slit_cnd = dir_figures + "fig_slit_cnd.txt"
slit_angle_cnd  = 80.0
slit_width_cnd  = 0.8
slit_length_cnd = 16.5

##################
# plot_angle_cnd #
##################
outpng_circ_slit_cnd    = dir_figures + "fig_circ_slit_cnd.png"
circ_slit_outerr_radius = 2.0 # arcsec

##################
# plot_histogram #
##################
outpng_hist_fov1 = dir_figures + "fig_hist_fov1.png"
outpng_hist_fov2 = dir_figures + "fig_hist_fov2.png"
outpng_hist_fov3 = dir_figures + "fig_hist_fov3.png"
snr_mom0         = 3.0
range_cont       = [-0.7,1.7] # range of continuum histogram
range_co10       = [0.6,3.4]  # range of co10 histogram
range_ci10       = [0.1,3.0]  # range of ci10 histogram
histgram_nbins   = 50
ylim_cont        = [0,0.10]
ylim_co10        = [0,0.18]
ylim_ci10        = [0,0.16]

################
# plot_scatter #
################
outpng_scatter_co_ci   = dir_figures + "fig_scatter_co_ci.png"
outpng_scatter_cont_ci = dir_figures + "fig_scatter_cont_ci.png"
xlim_co_ci             = [-0.5,3.5]
ylim_co_ci             = [-0.5,3.5]
xlim_cont_ci           = [-1.5,2.5]
ylim_cont_ci           = [-0.5,3.5]

################
# plot_spectra #
################
outpng_spectra_agn   = dir_figures + "fig_spectra_agn.png"
outpng_spectra_fov2  = dir_figures + "fig_spectra_fov2.png"
outpng_spectra_fov3  = dir_figures + "fig_spectra_fov3.png"
outpng_spectra_eknot = dir_figures + "fig_spectra_e_knot.png"
outpng_spectra_wknot = dir_figures + "fig_spectra_w_knot.png"

#######
# end #
#######