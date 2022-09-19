"""
Python class for the NGC 1068 CI-GMC project.

requirements:
CASA Version 5.4.0-70, ananlysisUtils, astropy

data:

usage:
> import os
> from scripts_n1068_ci_gmc import ToolsCIGMC as tools
>
> # key
> tl = tools(
>     refresh     = False,
>     keyfile_gal = "/home02/saitots/myUtils/keys_n1068_ci_gmc/key_ngc1068.txt",
>     keyfile_fig = "/home02/saitots/myUtils/keys_n1068_ci_gmc/key_figures.txt",
>     )
>
> # main
> tl.run_ngc1068_cigmc(
>     # analysis
>     do_prepare             = True,
>     # plot
>     # supplement
>     )
>
> os.system("rm -rf *.last")

paper drafts:
Date         Filename                To

references:
https://github.com/PhangsTeam/pycprops
https://qiita.com/Shinji_Fujita/items/7463038f70401040aedc

history:
2021-12-05   created (in Sinkansen!)
Toshiki Saito@Nichidai/NAOJ
"""

import os, sys, glob
import numpy as np

from mycasa_tasks import *
from mycasa_plots import *

############
# ToolsPCA #
############
class ToolsCIGMC():
    """
    Class for the NGC 1068 CI-GMC project.
    """

    ############
    # __init__ #
    ############

    def __init__(
        self,
        keyfile_fig  = None,
        keyfile_gal  = None,
        refresh      = False,
        delete_inter = True,
        ):
        # initialize keys
        self.keyfile_gal = keyfile_gal
        self.keyfile_fig = keyfile_fig

        # intialize task
        self.refresh      = refresh
        self.delete_inter = delete_inter
        self.taskname     = None

        # initialize directories
        self.dir_raw      = None
        self.dir_ready    = None
        self.dir_other    = None
        self.dir_products = None
        self.fig_dpi      = 200

        # import parameters
        if keyfile_fig is not None:
            self.modname = "ToolsCIGMC."
            self._set_dir()            # directories
            self._set_input_fits()     # input maps
            self._set_output_fits()    # output maps
            self._set_input_param()    # input parameters
            self._set_output_txt_png() # output txt and png

    def _set_dir(self):
        """
        """

        self.dir_proj     = self._read_key("dir_proj")
        self.dir_raw      = self.dir_proj + self._read_key("dir_raw")
        self.dir_ready    = self.dir_proj + self._read_key("dir_ready")
        self.dir_other    = self.dir_proj + self._read_key("dir_other")
        self.dir_products = self.dir_proj + self._read_key("dir_products")
        self.dir_final    = self.dir_proj + self._read_key("dir_final")

        self._create_dir(self.dir_ready)
        self._create_dir(self.dir_products)
        self._create_dir(self.dir_final)

    def _set_input_fits(self):
        """
        """

        self.cube_co10   = self.dir_raw + self._read_key("cube_co10")
        self.cube_ci10   = self.dir_raw + self._read_key("cube_ci10")

        self.ncube_co10  = self.dir_raw + self._read_key("ncube_co10")
        self.ncube_ci10  = self.dir_raw + self._read_key("ncube_ci10")

        self.mask_co10   = self.dir_raw + self._read_key("mask_co10")
        self.mask_ci10   = self.dir_raw + self._read_key("mask_ci10")

        self.mom0_co10   = self.dir_raw + self._read_key("mom0_co10")
        self.mom0_ci10   = self.dir_raw + self._read_key("mom0_ci10")

        self.emom0_co10  = self.dir_raw + self._read_key("emom0_co10")
        self.emom0_ci10  = self.dir_raw + self._read_key("emom0_ci10")

        self.cprops_co10 = self.dir_raw + self._read_key("cprops_co10")
        self.cprops_ci10 = self.dir_raw + self._read_key("cprops_ci10")

    def _set_output_fits(self):
        """
        """

        self.outfits_mom0_co10 = self.dir_ready + self._read_key("outfits_mom0_co10")
        self.outfits_mom0_ci10 = self.dir_ready + self._read_key("outfits_mom0_ci10")

    def _set_input_param(self):
        """
        """

        # ngc1068 properties
        self.ra_agn    = float(self._read_key("ra_agn", "gal").split("deg")[0])
        self.dec_agn   = float(self._read_key("dec_agn", "gal").split("deg")[0])
        self.ra_fov2   = 40.6679
        self.dec_fov2  = -0.0171116
        self.ra_fov3   = 40.675
        self.dec_fov3  = -0.012926
        self.scale_pc  = float(self._read_key("scale", "gal"))
        self.scale_kpc = self.scale_pc / 1000.

        self.beam      = 0.8
        self.snr_mom   = 4.0
        self.r_cnd     = 3.0 * self.scale_pc / 1000. # kpc
        self.r_cnd_as  = 3.0
        self.r_sbr     = 10.0 * self.scale_pc / 1000. # kpc
        self.r_sbr_as  = 10.0

        self.theta1      = -1 * -10.0 + 90.0 # 100
        self.theta2      = -1 * 70.0  + 90.0 # 20
        self.fov_diamter = 16.5

        self.snr_cprops = 7.0
        self.alpha_ci   = 15.0

    def _set_output_txt_png(self):
        """
        """

        # output png
        self.outpng_cprops_co10_agn  = self.dir_products + self._read_key("outpng_cprops_co10_agn")
        self.outpng_cprops_ci10_agn  = self.dir_products + self._read_key("outpng_cprops_ci10_agn")

        self.outpng_cprops_co10_fov2 = self.dir_products + self._read_key("outpng_cprops_co10_fov2")
        self.outpng_cprops_ci10_fov2 = self.dir_products + self._read_key("outpng_cprops_ci10_fov2")

        self.outpng_cprops_co10_fov3 = self.dir_products + self._read_key("outpng_cprops_co10_fov3")
        self.outpng_cprops_ci10_fov3 = self.dir_products + self._read_key("outpng_cprops_ci10_fov3")

        self.outpng_hist_rad  = self.dir_products + self._read_key("outpng_hist_rad")
        self.outpng_hist_sigv = self.dir_products + self._read_key("outpng_hist_sigv")
        self.outpng_hist_mvir = self.dir_products + self._read_key("outpng_hist_mvir")

        self.outpng_larson_1st = self.dir_products + self._read_key("outpng_larson_1st")
        self.outpng_larson_2nd = self.dir_products + self._read_key("outpng_larson_2nd")
        self.outpng_larson_3rd = self.dir_products + self._read_key("outpng_larson_3rd")

        # final
        print("TBE.")

    #####################
    # run_ngc1068_cigmc #
    #####################

    def run_ngc1068_cigmc(
        self,
        # analysis
        do_prepare   = False,
        print_cprops = False,
        # plot figures in paper
        map_cprops   = False,
        plot_cprops  = False,
        plot_larson  = False,
        # supplement
        ):
        """
        This method runs all the methods which will create figures in the paper.
        """

        # analysis
        if do_prepare==True:
            self.do_align()

        if print_cprops==True:
            print("### conda activate cprops")
            print("### python mypython_cprops.py")
            print("")
            print("import os, sys, pycprops")
            print("import astropy.units as u")
            print("from astropy.io import fits")
            print("import numpy as np")
            print("import matplotlib.pyplot as plt")
            print("")
            print("# https://qiita.com/Shinji_Fujita/items/7463038f70401040aedc")
            print("")
            print("# input")
            print("cube_cn10h  = '../data_ready/ngc1068_b3_12m_cn10h_0p8as.regrid.fits'")
            print("cube_hcop10 = '../data_ready/ngc1068_b3_12m_hcop10_0p8as.regrid.fits'")
            print("cube_hcn10  = '../data_ready/ngc1068_b3_12m_hcn10_0p8as.regrid.fits'")
            print("cube_co10   = '../data_ready/ngc1068_b3_12m+7m_co10_0p8as.regrid.fits'")
            print("cube_ci10   = '../data_ready/ngc1068_b8_12m+7m_ci10.fits'")
            print("d           = 13.97 * 1e6 * u.pc")
            print("cubes       = [cube_cn10h,cube_hcop10,cube_hcn10,cube_co10,cube_ci10]")
            print("")
            print("# output")
            print("cprops_cn10h  = '../data_ready/ngc1068_cn10h_cprops.fits'")
            print("cprops_hcop10 = '../data_ready/ngc1068_hcop10_cprops.fits'")
            print("cprops_hcn10  = '../data_ready/ngc1068_hcn10_cprops.fits'")
            print("cprops_co10   = '../data_ready/ngc1068_co10_cprops.fits'")
            print("cprops_ci10   = '../data_ready/ngc1068_ci10_cprops.fits'")
            print("outfiles = [cprops_cn10h,cprops_hcop10,cprops_hcn10,cprops_co10,cprops_ci10]")
            print("")
            print("# run (repeat by lines)")
            print("cubefile = cube_hcn10")
            print("outfile  = cprops_hcn10")
            print("mask     = cubefile")
            print("")
            print("# run")
            print("for i in range(len(cubes)):")
            print("    cubefile = cubes[i]")
            print("    outfile  = outfiles[i]")
            print("    mask     = cubefile")
            print("    pycprops.fits2props(")
            print("        cubefile,")
            print("        mask_file=mask,")
            print("        distance=d,")
            print("        asgnname=cubefile[:-5]+'.asgn.fits',")
            print("        propsname=cubefile[:-5]+'.props.fits',")
            print("        )")
            print("    os.system('mv ' + cubefile[:-5]+'.props.fits' + ' ' + outfile)")

        if map_cprops==True:
            self.map_cprops()

        if plot_cprops==True:
            self.plot_cprops()

        if plot_larson==True:
            self.plot_larson()

    ####################
    # immagick_figures #
    ####################

    def immagick_figures(
        self,
        delin=False,
        ):
        """
        """

        """
        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outpng_pca_hexmap.replace("???","1"),taskname)

        print("#########################")
        print("# create final_pca_mom0 #")
        print("#########################")

        combine_three_png(
            self.outpng_pca_scatter,
            self.outpng_pca_hexmap.replace("???","1"),
            self.outpng_pca_hexmap.replace("???","2"),
            self.final_pca_mom0,
            self.box_map,
            self.box_map,
            self.box_map,
            delin=delin,
            )
        """

    ###############
    # plot_larson #
    ###############

    def plot_larson(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.cprops_ci10,taskname)

        x_cone, y_cone, radius_cone, sigv_cone, mvir_cone, tpeak_cone, mci_cone, \
            x_nocone, y_nocone, radius_nocone, sigv_nocone, mvir_nocone, tpeak_nocone, mci_nocone, \
            x_sbr, y_sbr, radius_sbr, sigv_sbr, mvir_sbr, tpeak_sbr, mci_sbr = self._import_cprops_table(self.cprops_ci10)

        ####################
        # plot: larson 1st #
        ####################
        xlim   = None #[0.4*72-10,2.0*72+10]
        ylim   = None
        title  = "Larson's 1st law"
        xlabel = "log Diameter (pc)"
        ylabel = "log velocity dispersion (km s$^{-1}$)"

        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        ad  = [0.215,0.83,0.10,0.90]
        myax_set(ax1, "both", xlim, ylim, title, xlabel, ylabel, adjust=ad)

        ax1.scatter(np.log10(radius_cone), np.log10(sigv_cone), lw=0, s=160, color="red", alpha=0.5)
        ax1.scatter(np.log10(radius_nocone), np.log10(sigv_nocone), lw=0, s=160, color="blue", alpha=0.5)
        ax1.scatter(np.log10(radius_sbr), np.log10(sigv_sbr), lw=0, s=160, color="grey", alpha=0.5)

        # save
        os.system("rm -rf " + self.outpng_larson_1st)
        plt.savefig(self.outpng_larson_1st, dpi=self.fig_dpi)

        ####################
        # plot: larson 2nd #
        ####################
        xlim   = None #[0.4*72-10,2.0*72+10]
        ylim   = None
        title  = "Larson's 2nd law"
        xlabel = "log M(H$_2$) ($M_{\odot}$)"
        ylabel = "log velocity dispersion (km s$^{-1}$)"

        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        ad  = [0.215,0.83,0.10,0.90]
        myax_set(ax1, "both", xlim, ylim, title, xlabel, ylabel, adjust=ad)

        ax1.scatter(np.log10(mci_cone*self.alpha_ci), np.log10(sigv_cone), lw=0, s=160, color="red", alpha=0.5)
        ax1.scatter(np.log10(mci_nocone*self.alpha_ci), np.log10(sigv_nocone), lw=0, s=160, color="blue", alpha=0.5)
        ax1.scatter(np.log10(mci_sbr*self.alpha_ci), np.log10(sigv_sbr), lw=0, s=160, color="grey", alpha=0.5)

        # save
        os.system("rm -rf " + self.outpng_larson_2nd)
        plt.savefig(self.outpng_larson_2nd, dpi=self.fig_dpi)

    ###############
    # plot_cprops #
    ###############

    def plot_cprops(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.cprops_ci10,taskname)

        x_cone, y_cone, radius_cone, sigv_cone, mvir_cone, tpeak_cone, mci_cone, \
            x_nocone, y_nocone, radius_nocone, sigv_nocone, mvir_nocone, tpeak_nocone, mci_nocone, \
            x_sbr, y_sbr, radius_sbr, sigv_sbr, mvir_sbr, tpeak_sbr, mci_sbr = self._import_cprops_table(self.cprops_ci10)

        ################
        # plot: radius #
        ################
        xlim   = [0.4*72-10,2.0*72+10]
        ylim   = None
        title  = "Cloud radius"
        xlabel = "Radius (pc)"
        ylabel = "Count density"

        h = np.histogram(radius_cone, bins=10, range=xlim)
        x_rad_cone, y_rad_cone = h[1][:-1], h[0]/float(np.sum(h[0]))
        h = np.histogram(radius_nocone, bins=10, range=xlim)
        x_rad_nocone, y_rad_nocone = h[1][:-1], h[0]/float(np.sum(h[0]))
        h = np.histogram(radius_sbr, bins=10, range=xlim)
        x_rad_sbr, y_rad_sbr = h[1][:-1], h[0]/float(np.sum(h[0]))

        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        ad  = [0.215,0.83,0.10,0.90]
        myax_set(ax1, "x", xlim, ylim, title, xlabel, ylabel, adjust=ad)

        ax1.bar(x_rad_cone, y_rad_cone, lw=0, color="red", width=x_rad_cone[1]-x_rad_cone[0], alpha=0.5)
        ax1.bar(x_rad_nocone, y_rad_nocone, lw=0, color="blue", width=x_rad_nocone[1]-x_rad_nocone[0], alpha=0.5)
        ax1.bar(x_rad_sbr, y_rad_sbr, lw=0, color="grey", width=x_rad_sbr[1]-x_rad_sbr[0], alpha=0.5)

        # save
        os.system("rm -rf " + self.outpng_hist_rad)
        plt.savefig(self.outpng_hist_rad, dpi=self.fig_dpi)

        ###############
        # plot: sigma #
        ###############
        xlim   = [0,35]
        ylim   = None
        title  = "Cloud velocity dispersion"
        xlabel = "Velocity dispersion (km s$^{-1}$)"
        ylabel = "Count density"

        h = np.histogram(sigv_cone, bins=10, range=xlim)
        x_sigv_cone, y_sigv_cone = h[1][:-1], h[0]/float(np.sum(h[0]))
        h = np.histogram(sigv_nocone, bins=10, range=xlim)
        x_sigv_nocone, y_sigv_nocone = h[1][:-1], h[0]/float(np.sum(h[0]))
        h = np.histogram(sigv_sbr, bins=10, range=xlim)
        x_sigv_sbr, y_sigv_sbr = h[1][:-1], h[0]/float(np.sum(h[0]))

        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        ad  = [0.215,0.83,0.10,0.90]
        myax_set(ax1, "x", xlim, ylim, title, xlabel, ylabel, adjust=ad)

        ax1.bar(x_sigv_cone, y_sigv_cone, lw=0, color="red", width=x_sigv_cone[1]-x_sigv_cone[0], alpha=0.5)
        ax1.bar(x_sigv_nocone, y_sigv_nocone, lw=0, color="blue", width=x_sigv_nocone[1]-x_sigv_nocone[0], alpha=0.5)
        ax1.bar(x_sigv_sbr, y_sigv_sbr, lw=0, color="grey", width=x_sigv_sbr[1]-x_sigv_sbr[0], alpha=0.5)

        # save
        os.system("rm -rf " + self.outpng_hist_sigv)
        plt.savefig(self.outpng_hist_sigv, dpi=self.fig_dpi)

        ##############
        # plot: mvir #
        ##############
        xlim   = [5.4,8.0]
        ylim   = None
        title  = "Cloud virial mass"
        xlabel = "Virial mass ($M_{\odot}$)"
        ylabel = "Count density"

        h = np.histogram(mvir_cone, bins=10, range=xlim)
        x_mvir_cone, y_mvir_cone = h[1][:-1], h[0]/float(np.sum(h[0]))
        h = np.histogram(mvir_nocone, bins=10, range=xlim)
        x_mvir_nocone, y_mvir_nocone = h[1][:-1], h[0]/float(np.sum(h[0]))
        h = np.histogram(mvir_sbr, bins=10, range=xlim)
        x_mvir_sbr, y_mvir_sbr = h[1][:-1], h[0]/float(np.sum(h[0]))

        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        ad  = [0.215,0.83,0.10,0.90]
        myax_set(ax1, "x", xlim, ylim, title, xlabel, ylabel, adjust=ad)

        ax1.bar(x_mvir_cone, y_mvir_cone, lw=0, color="red", width=x_mvir_cone[1]-x_mvir_cone[0], alpha=0.5)
        ax1.bar(x_mvir_nocone, y_mvir_nocone, lw=0, color="blue", width=x_mvir_nocone[1]-x_mvir_nocone[0], alpha=0.5)
        ax1.bar(x_mvir_sbr, y_mvir_sbr, lw=0, color="grey", width=x_mvir_sbr[1]-x_mvir_sbr[0], alpha=0.5)

        # save
        os.system("rm -rf " + self.outpng_hist_mvir)
        plt.savefig(self.outpng_hist_mvir, dpi=self.fig_dpi)

    ########################
    # _import_cprops_table #
    ########################

    def _import_cprops_table(
        self,
        table,
        ):
        """
        """

        # import cprops table
        f = pyfits.open(table)
        tb = f[1].data

        # extract parameters
        x      = (tb["XCTR_DEG"] - self.ra_agn) * -3600.
        y      = (tb["YCTR_DEG"] - self.dec_agn) * 3600.
        s2n    = tb["S2N"]
        radius = tb["RAD_NODC_NOEX"]
        sigv   = tb["SIGV_NODC_NOEX"]
        mvir   = tb["MVIR_MSUN"]
        tpeak  = tb["TMAX_K"]
        mci    = tb["MLUM_MSUN"]

        cut    = np.where(~np.isnan(radius) & ~np.isnan(sigv) & ~np.isnan(mvir) & ~np.isnan(tpeak))
        x      = x[cut]
        y      = y[cut]
        s2n    = s2n[cut]
        radius = radius[cut]
        sigv   = sigv[cut]
        mvir   = np.log10(mvir[cut])
        tpeak  = tpeak[cut]
        mci    = mci[cut]

        # bicone definition
        r          = np.sqrt(x**2 + y**2)
        theta      = np.degrees(np.arctan2(x, y)) + 90
        theta      = np.where(theta>0,theta,theta+360)
        cut_cone   = np.where((s2n>=self.snr_cprops) & (r<self.fov_diamter/2.0) & (theta>=self.theta2) & (theta<self.theta1) | (s2n>=self.snr_cprops) & (r<self.fov_diamter/2.0) & (theta>=self.theta2+180) & (theta<self.theta1+180))
        cut_nocone = np.where((s2n>=self.snr_cprops) & (r<self.fov_diamter/2.0) & (theta>=self.theta1) & (theta<self.theta2+180) | (s2n>=self.snr_cprops) & (r<self.fov_diamter/2.0) & (theta>=self.theta1+180) & (theta<self.theta2+360) | (s2n>=self.snr_cprops) & (r<self.fov_diamter/2.0) & (theta<self.theta1+180) & (theta<self.theta2))
        cut_sbr    = np.where((s2n>=self.snr_cprops) & (r>=self.fov_diamter/2.0))

        # data
        x_cone      = x[cut_cone]
        y_cone      = y[cut_cone]
        radius_cone = radius[cut_cone]
        sigv_cone   = sigv[cut_cone]
        mvir_cone   = mvir[cut_cone]
        tpeak_cone  = tpeak[cut_cone]
        mci_cone    = mci[cut_cone]

        x_nocone      = x[cut_nocone]
        y_nocone      = y[cut_nocone]
        radius_nocone = radius[cut_nocone]
        sigv_nocone   = sigv[cut_nocone]
        mvir_nocone   = mvir[cut_nocone]
        tpeak_nocone  = tpeak[cut_nocone]
        mci_nocone    = mci[cut_nocone]

        x_sbr      = x[cut_sbr]
        y_sbr      = y[cut_sbr]
        radius_sbr = radius[cut_sbr]
        sigv_sbr   = sigv[cut_sbr]
        mvir_sbr   = mvir[cut_sbr]
        tpeak_sbr  = tpeak[cut_sbr]
        mci_sbr    = mci[cut_sbr]

        return x_cone, y_cone, radius_cone, sigv_cone, mvir_cone, tpeak_cone, x_nocone, mci_cone, y_nocone, radius_nocone, sigv_nocone, mvir_nocone, tpeak_nocone, mci_nocone, x_sbr, y_sbr, radius_sbr, sigv_sbr, mvir_sbr, tpeak_sbr, mci_sbr

    ##############
    # map_cprops #
    ##############

    def map_cprops(
        self,
        delin=False,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.cprops_co10,taskname)

        # import fits table
        f = pyfits.open(self.cprops_co10)
        tb_co10 = f[1].data

        f = pyfits.open(self.cprops_ci10)
        tb_ci10 = f[1].data

        # extract tag
        self._plot_cprops_map(
            self.outfits_mom0_co10,
            tb_co10,
            "CO(1-0)",
            self.outpng_cprops_co10_agn,
            self.outpng_cprops_co10_fov2,
            self.outpng_cprops_co10_fov3,
            )
        self._plot_cprops_map(
            self.outfits_mom0_ci10,
            tb_ci10,
            "[CI](1-0)",
            self.outpng_cprops_ci10_agn,
            self.outpng_cprops_ci10_fov2,
            self.outpng_cprops_ci10_fov3,
            )

    ####################
    # _plot_cprops_map #
    ####################

    def _plot_cprops_map(
        self,
        imagename,
        this_tb,
        linename,
        outpng_agn,
        outpng_fov2,
        outpng_fov3,
        ):
        """
        # CLOUDNUM
        # XCTR_DEG
        # YCTR_DEG
        # VCTR_KMS
        # RAD_PC
        # SIGV_KMS
        # FLUX_KKMS_PC2
        # MVIR_MSUN
        # S2N
        """

        scalebar = 100. / self.scale_pc
        label_scalebar = "100 pc"

        myfig_fits2png(
            imagename,
            outpng_agn,
            imsize_as = 18.0,
            ra_cnt    = str(self.ra_agn) + "deg",
            dec_cnt   = str(self.dec_agn) + "deg",
            numann    = "ci-gmc",
            txtfiles  = this_tb,
            set_title = linename + " Cloud Catalog",
            scalebar  = scalebar,
            label_scalebar = label_scalebar,
            colorlog  = True,
            set_cmap  = "Greys",
            )

        myfig_fits2png(
            imagename,
            outpng_fov2,
            imsize_as = 18.0,
            ra_cnt    = str(self.ra_fov2) + "deg",
            dec_cnt   = str(self.dec_fov2) + "deg",
            numann    = "ci-gmc",
            txtfiles  = this_tb,
            set_title = linename + " Cloud Catalog",
            scalebar  = scalebar,
            label_scalebar = label_scalebar,
            colorlog  = True,
            set_cmap  = "Greys",
            )

        myfig_fits2png(
            imagename,
            outpng_fov3,
            imsize_as = 18.0,
            ra_cnt    = str(self.ra_fov3) + "deg",
            dec_cnt   = str(self.dec_fov3) + "deg",
            numann    = "ci-gmc",
            txtfiles  = this_tb,
            set_title = linename + " Cloud Catalog",
            scalebar  = scalebar,
            label_scalebar = label_scalebar,
            colorlog  = True,
            set_cmap  = "Greys",
            )

    ############
    # do_align #
    ############

    def do_align(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.mom0_co10,taskname)

        # read template
        template = "template.image"
        run_importfits(self.mom0_ci10,template)

        # regrid co10 to ci10
        run_imregrid(self.mom0_co10,template,self.mom0_co10+"_tmp1")
        run_imregrid(self.emom0_co10,template,self.emom0_co10+"_tmp1")
        os.system("rm -rf " + template)

        # clip
        expr = "iif(IM0/IM1>"+str(self.snr_mom)+",IM0,0)"
        run_immath_two(self.mom0_ci10,self.emom0_ci10,self.outfits_mom0_ci10+"_tmp2",expr)
        run_immath_two(self.mom0_co10+"_tmp1",self.emom0_co10+"_tmp1",self.outfits_mom0_co10+"_tmp2",expr,delin=True)

        # exportfits
        run_exportfits(self.outfits_mom0_ci10+"_tmp2",self.outfits_mom0_ci10,delin=True)
        run_exportfits(self.outfits_mom0_co10+"_tmp2",self.outfits_mom0_co10,delin=True)

    ###############
    # _create_dir #
    ###############

    def _create_dir(self, this_dir):

        if self.refresh==True:
            print("## refresh " + this_dir)
            os.system("rm -rf " + this_dir)

        if not glob.glob(this_dir):
            print("## create " + this_dir)
            os.mkdir(this_dir)

        else:
            print("## not refresh " + this_dir)

    #############
    # _read_key #
    #############

    def _read_key(self, key, keyfile="fig", delimiter=",,,"):

        if keyfile=="gal":
            keyfile = self.keyfile_gal
        elif keyfile=="fig":
            keyfile = self.keyfile_fig

        keydata  = np.loadtxt(keyfile,dtype="str",delimiter=delimiter)
        keywords =\
             np.array([s.replace(" ","") for s in keydata[:,0]])
        values   = keydata[:,1]
        value    = values[np.where(keywords==key)[0][0]]

        return value

    #########################
    # will be decomissioned #
    #########################

    def do_align_old(
        self,
        ):
        """
        add ncube alignment
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.cube_ci10,taskname)

        self.cn10h_ready  = self.dir_ready + self._read_key("cube_cn10h")[:-5] + ".regrid.fits"
        self.hcop10_ready = self.dir_ready + self._read_key("cube_hcop10")[:-5] + ".regrid.fits"
        self.hcn10_ready  = self.dir_ready + self._read_key("cube_hcn10")[:-5] + ".regrid.fits"
        self.co10_ready   = self.dir_ready + self._read_key("cube_co10")[:-5] + ".regrid.fits"
        self.ci10_ready   = self.dir_ready + self._read_key("cube_ci10")
        template          = "template.image"

        self.cn10h_nready  = self.dir_ready + self._read_key("ncube_cn10h")[:-5] + ".regrid.fits"
        self.hcop10_nready = self.dir_ready + self._read_key("ncube_hcop10")[:-5] + ".regrid.fits"
        self.hcn10_nready  = self.dir_ready + self._read_key("ncube_hcn10")[:-5] + ".regrid.fits"
        self.co10_nready   = self.dir_ready + self._read_key("ncube_co10")[:-5] + ".regrid.fits"
        self.ci10_nready   = self.dir_ready + self._read_key("ncube_ci10")

        # get restfreq
        restf_cn10h  = imhead(self.cube_cn10h,mode="list")["restfreq"][0]
        restf_hcop10 = imhead(self.cube_hcop10,mode="list")["restfreq"][0]
        restf_hcn10  = imhead(self.cube_hcn10,mode="list")["restfreq"][0]
        restf_co10   = imhead(self.cube_co10,mode="list")["restfreq"][0]
        restf_ci10   = imhead(self.cube_ci10,mode="list")["restfreq"][0]

        # regrid to ci10 cube 98,71,179,151
        run_importfits(self.cube_ci10,template+"2")
        imsubimage(template+"2",template,box="98,71,179,151")
        run_imregrid(self.cube_cn10h,template,self.cn10h_ready+".image",axes=[0,1])
        run_imregrid(self.cube_hcop10,template,self.hcop10_ready+".image",axes=[0,1])
        run_imregrid(self.cube_hcn10,template,self.hcn10_ready+".image",axes=[0,1])
        run_imregrid(self.cube_co10,template,self.co10_ready+".image",axes=[0,1])

        run_imregrid(self.ncube_cn10h,template,self.cn10h_nready+".image",axes=[0,1])
        run_imregrid(self.ncube_hcop10,template,self.hcop10_nready+".image",axes=[0,1])
        run_imregrid(self.ncube_hcn10,template,self.hcn10_nready+".image",axes=[0,1])
        run_imregrid(self.ncube_co10,template,self.co10_nready+".image",axes=[0,1])
        os.system("rm -rf " + template + " " + template + "2")

        # to fits
        run_exportfits(self.cn10h_nready+".image",self.cn10h_nready+"2",delin=True,velocity=True)
        run_exportfits(self.hcop10_nready+".image",self.hcop10_nready+"2",delin=True,velocity=True)
        run_exportfits(self.hcn10_nready+".image",self.hcn10_nready+"2",delin=True,velocity=True)
        run_exportfits(self.co10_nready+".image",self.co10_nready+"2",delin=True,velocity=True)
        os.system("cp -r " + self.ncube_ci10 + " " + self.ci10_nready + "2")

        run_exportfits(self.cn10h_ready+".image",self.cn10h_ready+"2",delin=True,velocity=True)
        run_exportfits(self.hcop10_ready+".image",self.hcop10_ready+"2",delin=True,velocity=True)
        run_exportfits(self.hcn10_ready+".image",self.hcn10_ready+"2",delin=True,velocity=True)
        run_exportfits(self.co10_ready+".image",self.co10_ready+"2",delin=True,velocity=True)
        os.system("cp -r " + self.cube_ci10 + " " + self.ci10_ready + "2")

        # change header to CPROPS format
        hdu = fits.open(self.cn10h_ready+"2")[0]
        d, h = hdu.data, hdu.header
        h["CTYPE3"] = "VELOCITY"
        h["RESTFREQ"] = restf_hcn10
        fits.PrimaryHDU(d, h).writeto(self.cn10h_ready, overwrite=True)
        os.system("rm -rf " + self.cn10h_ready + "2")
        
        hdu = fits.open(self.hcop10_ready+"2")[0]
        d, h = hdu.data, hdu.header
        h["CTYPE3"] = "VELOCITY"
        h["RESTFREQ"] = restf_hcn10
        fits.PrimaryHDU(d, h).writeto(self.hcop10_ready, overwrite=True)
        os.system("rm -rf " + self.hcop10_ready + "2")

        hdu = fits.open(self.hcn10_ready+"2")[0]
        d, h = hdu.data, hdu.header
        h["CTYPE3"] = "VELOCITY"
        h["RESTFREQ"] = restf_hcn10
        fits.PrimaryHDU(d, h).writeto(self.hcn10_ready, overwrite=True)
        os.system("rm -rf " + self.hcn10_ready + "2")

        hdu = fits.open(self.co10_ready+"2")[0]
        d, h = hdu.data, hdu.header
        h["CTYPE3"] = "VELOCITY"
        h["RESTFREQ"] = restf_co10
        fits.PrimaryHDU(d, h).writeto(self.co10_ready, overwrite=True)
        os.system("rm -rf " + self.co10_ready + "2")

        hdu = fits.open(self.ci10_ready+"2")[0]
        d, h = hdu.data, hdu.header
        h["CTYPE3"] = "VELOCITY"
        h["RESTFREQ"] = restf_ci10
        fits.PrimaryHDU(d, h).writeto(self.ci10_ready, overwrite=True)
        os.system("rm -rf " + self.ci10_ready + "2")

        #
        hdu = fits.open(self.cn10h_nready+"2")[0]
        d, h = hdu.data, hdu.header
        h["CTYPE3"] = "VELOCITY"
        h["RESTFREQ"] = restf_hcn10
        fits.PrimaryHDU(d, h).writeto(self.cn10h_nready, overwrite=True)
        os.system("rm -rf " + self.cn10h_nready + "2")
        
        hdu = fits.open(self.hcop10_nready+"2")[0]
        d, h = hdu.data, hdu.header
        h["CTYPE3"] = "VELOCITY"
        h["RESTFREQ"] = restf_hcn10
        fits.PrimaryHDU(d, h).writeto(self.hcop10_nready, overwrite=True)
        os.system("rm -rf " + self.hcop10_nready + "2")

        hdu = fits.open(self.hcn10_nready+"2")[0]
        d, h = hdu.data, hdu.header
        h["CTYPE3"] = "VELOCITY"
        h["RESTFREQ"] = restf_hcn10
        fits.PrimaryHDU(d, h).writeto(self.hcn10_nready, overwrite=True)
        os.system("rm -rf " + self.hcn10_nready + "2")

        hdu = fits.open(self.co10_nready+"2")[0]
        d, h = hdu.data, hdu.header
        h["CTYPE3"] = "VELOCITY"
        h["RESTFREQ"] = restf_co10
        fits.PrimaryHDU(d, h).writeto(self.co10_nready, overwrite=True)
        os.system("rm -rf " + self.co10_nready + "2")

        hdu = fits.open(self.ci10_nready+"2")[0]
        d, h = hdu.data, hdu.header
        h["CTYPE3"] = "VELOCITY"
        h["RESTFREQ"] = restf_ci10
        fits.PrimaryHDU(d, h).writeto(self.ci10_nready, overwrite=True)
        os.system("rm -rf " + self.ci10_nready + "2")

    def _plot_all_param(
        self,
        this_tb,
        linename,
        outpng_header,
        snr=4,
        ):
        """
        # CLOUDNUM
        # XCTR_DEG
        # YCTR_DEG
        # VCTR_KMS
        # RAD_PC
        # SIGV_KMS
        # FLUX_KKMS_PC2
        # MVIR_MSUN
        # S2N
        """

        radlimit = 72.*0.8/2.

        # FLUX_KKMS_PC2
        params = ["FLUX_KKMS_PC2","RAD_PC","SIGV_KMS"]
        footers = ["flux","radius","disp"]
        for i in range(len(params)):
            this_param  = params[i]
            this_footer = footers[i]
            cut         = np.where((this_tb["S2N"]>=snr) & (this_tb["RAD_PC"]>=radlimit) & (this_tb["RAD_PC"]<=radlimit*10))
            this_x      = (this_tb["XCTR_DEG"][cut] - self.ra_agn) * 3600.  
            this_y      = (this_tb["YCTR_DEG"][cut] - self.dec_agn) * 3600.  
            this_c      = this_tb[this_param][cut]
            this_outpng = outpng_header.replace(".png","_"+this_footer+".png")

            if this_param=="FLUX_KKMS_PC2":
                this_c = np.log(this_c)

            self._plot_cpropsmap(
                this_outpng,
                this_x,
                this_y,
                this_c,
                linename + " (" + this_param + ")",
                title_cbar="(K km s$^{-1}$)",
                cmap="rainbow",
                add_text=False,
                label="",
                )

    def _plot_cpropsmap(
        self,
        outpng,
        x,y,c,
        title,
        title_cbar="(K km s$^{-1}$)",
        cmap="rainbow",
        plot_cbar=True,
        ann=True,
        lim=13.0,
        size=100,
        add_text=False,
        label="",
        ):
        """
        """

        # set plt, ax
        fig = plt.figure(figsize=(13,10))
        plt.rcParams["font.size"] = 16
        gs = gridspec.GridSpec(nrows=10, ncols=10)
        ax = plt.subplot(gs[0:10,0:10])

        # set ax parameter
        myax_set(
        ax,
        grid=None,
        xlim=[lim, -lim],
        ylim=[-lim, lim],
        xlabel="R.A. offset (arcsec)",
        ylabel="Decl. offset (arcsec)",
        adjust=[0.10,0.99,0.10,0.93],
        )
        ax.set_aspect('equal', adjustable='box')

        # plot
        im = ax.scatter(x, y, s=size, c=c, cmap=cmap, marker="o", linewidths=0)

        # cbar
        cbar = plt.colorbar(im)
        if plot_cbar==True:
            cax  = fig.add_axes([0.19, 0.12, 0.025, 0.35])
            fig.colorbar(im, cax=cax).set_label(label)

        # scale bar
        bar = 100 / self.scale_pc
        ax.plot([-10,-10+bar],[-10,-10],"-",color="black",lw=4)
        ax.text(-10, -10.5, "100 pc",
                horizontalalignment="right", verticalalignment="top")

        # text
        ax.text(0.03, 0.93, title, color="black", transform=ax.transAxes, weight="bold", fontsize=24)

        # ann
        if ann==True:
            theta1      = -10.0 # degree
            theta2      = 70.0 # degree
            fov_diamter = 16.5 # arcsec (12m+7m Band 8)

            fov_diamter = 16.5
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
            ax.plot([0,-7], [0,10], lw=3, c="black")
            ax.text(-10.5, 10.5, "AGN position",
                horizontalalignment="right", verticalalignment="center", weight="bold")

        # save
        os.system("rm -rf " + outpng)
        plt.savefig(outpng, dpi=300)

#####################
# end of ToolsCIGMC #
#####################
