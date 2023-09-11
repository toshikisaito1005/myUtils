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
from scipy import stats

from mycasa_tasks import *
from mycasa_plots import *

def density_estimation(m1, m2, xlim, ylim):
    X, Y = np.mgrid[xlim[0]:xlim[1]:100j, ylim[0]:ylim[1]:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    return X, Y, Z

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
        self.legend_fontsize = 20

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

        self.tpeak_co10  = self.dir_raw + self._read_key("tpeak_co10")
        self.tpeak_ci10  = self.dir_raw + self._read_key("tpeak_ci10")

        self.cprops_co10 = self.dir_raw + self._read_key("cprops_co10")
        self.cprops_ci10 = self.dir_raw + self._read_key("cprops_ci10")

    def _set_output_fits(self):
        """
        """

        self.outfits_mom0_co10  = self.dir_ready + self._read_key("outfits_mom0_co10")
        self.outfits_mom0_ci10  = self.dir_ready + self._read_key("outfits_mom0_ci10")
        self.outfits_mom0_ratio = self.dir_ready + self._read_key("outfits_mom0_ratio")

    def _set_input_param(self):
        """
        """

        # ngc1068 properties
        self.ra_agn    = float(self._read_key("ra_agn", "gal").split("deg")[0])
        self.dec_agn   = float(self._read_key("dec_agn", "gal").split("deg")[0])
        self.ra_fov2   = float(self._read_key("ra_fov2", "gal").split("deg")[0])
        self.dec_fov2  = float(self._read_key("dec_fov2", "gal").split("deg")[0])
        self.ra_fov3   = float(self._read_key("ra_fov3", "gal").split("deg")[0])
        self.dec_fov3  = float(self._read_key("dec_fov3", "gal").split("deg")[0])
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

        self.snr_cprops = 5.0
        self.alpha_ci   = 12.0
        self.alpha_co   = 1.0

        self.imsize_as  = 18

        self.xlim_larson_1st = [1.9,2.65]
        self.ylim_larson_1st = [0.4,1.6]
        self.xlim_larson_2nd = [6.5,9.0]
        self.ylim_larson_2nd = [0.4,1.6]
        self.xlim_larson_3rd = [1.9,2.65]
        self.ylim_larson_3rd = [0.2,2.3]

    def _set_output_txt_png(self):
        """
        """

        self.outpng_hist_co10_pix = self.dir_products + self._read_key("outpng_hist_co10_pix")
        self.outpng_hist_ci10_pix = self.dir_products + self._read_key("outpng_hist_ci10_pix")

        self.outpng_hist_co10_snr = self.dir_products + self._read_key("outpng_hist_co10_snr")
        self.outpng_hist_ci10_snr = self.dir_products + self._read_key("outpng_hist_ci10_snr")

        # output png
        self.outpng_cprops_co10_agn       = self.dir_products + self._read_key("outpng_cprops_co10_agn")
        self.outpng_cprops_ci10_agn       = self.dir_products + self._read_key("outpng_cprops_ci10_agn")
        self.outpng_cprops_ci10_co10_agn  = self.dir_products + self._read_key("outpng_cprops_ci10_co10_agn")

        self.outpng_cprops_co10_fov2      = self.dir_products + self._read_key("outpng_cprops_co10_fov2")
        self.outpng_cprops_ci10_fov2      = self.dir_products + self._read_key("outpng_cprops_ci10_fov2")
        self.outpng_cprops_ci10_co10_fov2 = self.dir_products + self._read_key("outpng_cprops_ci10_co10_fov2")

        self.outpng_cprops_co10_fov3      = self.dir_products + self._read_key("outpng_cprops_co10_fov3")
        self.outpng_cprops_ci10_fov3      = self.dir_products + self._read_key("outpng_cprops_ci10_fov3")
        self.outpng_cprops_ci10_co10_fov3 = self.dir_products + self._read_key("outpng_cprops_ci10_co10_fov3")

        self.outpng_ci_hist_rad           = self.dir_products + self._read_key("outpng_ci_hist_rad")
        self.outpng_ci_hist_sigv          = self.dir_products + self._read_key("outpng_ci_hist_sigv")
        self.outpng_ci_hist_mvir          = self.dir_products + self._read_key("outpng_ci_hist_mvir")
        self.outpng_ci_larson_1st         = self.dir_products + self._read_key("outpng_ci_larson_1st")
        self.outpng_ci_larson_2nd         = self.dir_products + self._read_key("outpng_ci_larson_2nd")
        self.outpng_ci_larson_3rd         = self.dir_products + self._read_key("outpng_ci_larson_3rd")

        self.outpng_co_hist_rad           = self.dir_products + self._read_key("outpng_co_hist_rad")
        self.outpng_co_hist_sigv          = self.dir_products + self._read_key("outpng_co_hist_sigv")
        self.outpng_co_hist_mvir          = self.dir_products + self._read_key("outpng_co_hist_mvir")
        self.outpng_co_larson_1st         = self.dir_products + self._read_key("outpng_co_larson_1st")
        self.outpng_co_larson_2nd         = self.dir_products + self._read_key("outpng_co_larson_2nd")
        self.outpng_co_larson_3rd         = self.dir_products + self._read_key("outpng_co_larson_3rd")

        self.outpng_cico_larson_1st       = self.dir_products + self._read_key("outpng_cico_larson_1st")
        self.outpng_cico_larson_2nd       = self.dir_products + self._read_key("outpng_cico_larson_2nd")
        self.outpng_cico_larson_3rd       = self.dir_products + self._read_key("outpng_cico_larson_3rd")

        self.outpng_map_ratio             = self.dir_products + self._read_key("outpng_map_ratio")

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
        data_stats   = False,
        # plot figures in paper
        # supplement
        do_stack     = False,
        map_cprops   = False,
        plot_cprops  = False,
        plot_larson  = False,
        plot_map     = False,
        ):
        """
        This method runs all the methods which will create figures in the paper.
        """

        # analysis
        if do_prepare==True:
            self.do_align()
            self.do_align_cube()

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

        if data_stats==True:
            self.data_stats()

        """
        if do_stack==True:
            self.do_stack()

        if map_cprops==True:
            self.map_cprops()

        if plot_cprops==True:
            self.plot_ci_cprops()
            self.plot_co_cprops()

        if plot_larson==True:
            self.plot_ci_larson()
            self.plot_co_larson()
            self.plot_cico_larson()

        if plot_map==True:
            self.plot_map()
        """

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

    ##############
    # data_stats #
    ##############

    def data_stats(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.cprops_co10,taskname)

        #c_ci,_  = imval_all(self.cube_ci10)
        #nc_ci,_ = imval_all(self.ncube_ci10)
        #c_co,_  = imval_all(self.cube_co10.replace(".fits","_aligned.fits"))
        #nc_co,_ = imval_all(self.ncube_co10.replace(".fits","_aligned.fits"))

        ###########
        # prepare #
        ###########

        data,_  = imval_all(self.cube_ci10)
        data    = data["data"].flatten()
        ndata,_ = imval_all(self.ncube_ci10)
        ndata   = ndata["data"].flatten()
        data_co10 = data[data/ndata>-10000]

        histx, histy, histrange, peak, rms, x_bestfit, y_bestfit, _ = self._gaussfit_noise(data_co10,bins=1000)

        xlim     = [0, 10*rms]
        ylim     = [0, np.max(histy)*1.05]
        title    = "CO(1-0) Cube"
        xlabel   = "Absolute voxel value (K)"
        ylabel   = "Count"
        binwidth = (histrange[1]-histrange[0]) / 1000.
        c_pos    = "tomato"
        c_neg    = "deepskyblue"

        ########
        # plot #
        ########

        plt.figure(figsize=(13,10))
        gs = gridspec.GridSpec(nrows=10, ncols=10)
        ax = plt.subplot(gs[0:10,0:10])

        ad = [0.215,0.83,0.10,0.90]
        myax_set(ax, None, xlim, ylim, title, xlabel, ylabel, adjust=ad)
        #ax.set_yticks(np.linspace(0,20000,3)[1:])

        # plot hists
        ax.bar(histx, histy, width=binwidth, align="center", lw=0, color=c_pos, alpha=1.0)
        ax.bar(-1*histx, histy, width=binwidth, align="center", lw=0, color=c_neg, alpha=1.0)

        # plot bestfit
        ax.plot(x_bestfit, y_bestfit, "-", c="black", lw=5)

        # # plot 1 sigma and 2.5sigma dashed vertical lines
        ax.plot([rms, rms], ylim, "--", color='black', lw=2)
        ax.plot([rms*2.5, rms*2.5], ylim, "--", color='black', lw=2)

        # legend
        ax.text(0.95, 0.93, "positive voxel histogram", color=c_pos, horizontalalignment="right", transform=ax.transAxes, size=self.legend_fontsize, fontweight="bold")
        ax.text(0.95, 0.88, "negative voxel histogram", color=c_neg, horizontalalignment="right", transform=ax.transAxes, size=self.legend_fontsize, fontweight="bold")
        ax.text(0.95, 0.83, "best-fit Gaussian", color="black", horizontalalignment="right", transform=ax.transAxes, size=self.legend_fontsize, fontweight="bold")
        #
        x    = rms / (xlim[1]-xlim[0]) + 0.01
        text = r"1$\sigma$ = "+str(rms).ljust(5, "0") + " K"
        ax.text(x, 0.96, text, color="black", horizontalalignment="left", verticalalignment="top", transform=ax.transAxes, rotation=90)
        #
        x    = rms*2.5 / (xlim[1]-xlim[0]) + 0.01
        text = str(2.5)+r"$\sigma$ = "+str(np.round(rms*2.5,3)).ljust(5, "0") + " K"
        ax.text(x, 0.96, text, color="black", horizontalalignment="left", verticalalignment="top", transform=ax.transAxes, rotation=90)

        plt.savefig(self.outpng_hist_co10_pix, dpi=self.fig_dpi)

        ###########
        # prepare #
        ###########

        snr_co10 = data[data/ndata>-10000] / ndata[data/ndata>-10000]

        histx, histy, histrange, peak, rms, x_bestfit, y_bestfit, _ = self._gaussfit_noise(snr_co10,bins=1000)

        xlim     = [0, 10*rms]
        ylim     = [0, np.max(histy)*1.05]
        title    = "CO(1-0) SNR Cube"
        xlabel   = "Absolute voxel SNR"
        ylabel   = "Count"
        binwidth = (histrange[1]-histrange[0]) / 1000.
        c_pos    = "tomato"
        c_neg    = "deepskyblue"

        ########
        # plot #
        ########

        plt.figure(figsize=(13,10))
        gs = gridspec.GridSpec(nrows=10, ncols=10)
        ax = plt.subplot(gs[0:10,0:10])

        ad = [0.215,0.83,0.10,0.90]
        myax_set(ax, None, xlim, ylim, title, xlabel, ylabel, adjust=ad)
        #ax.set_yticks(np.linspace(0,20000,3)[1:])

        # plot hists
        ax.bar(histx, histy, width=binwidth, align="center", lw=0, color=c_pos, alpha=1.0)
        ax.bar(-1*histx, histy, width=binwidth, align="center", lw=0, color=c_neg, alpha=1.0)

        # plot bestfit
        ax.plot(x_bestfit, y_bestfit, "-", c="black", lw=5)

        # # plot 1 sigma and 2.5sigma dashed vertical lines
        ax.plot([rms, rms], ylim, "--", color='black', lw=2)
        ax.plot([rms*2.5, rms*2.5], ylim, "--", color='black', lw=2)

        # legend
        ax.text(0.95, 0.93, "positive voxel histogram", color=c_pos, horizontalalignment="right", transform=ax.transAxes, size=self.legend_fontsize, fontweight="bold")
        ax.text(0.95, 0.88, "negative voxel histogram", color=c_neg, horizontalalignment="right", transform=ax.transAxes, size=self.legend_fontsize, fontweight="bold")
        ax.text(0.95, 0.83, "best-fit Gaussian", color="black", horizontalalignment="right", transform=ax.transAxes, size=self.legend_fontsize, fontweight="bold")
        #
        x    = rms / (xlim[1]-xlim[0]) + 0.01
        text = r"1$\sigma$ = "+str(rms).ljust(5, "0")
        ax.text(x, 0.96, text, color="black", horizontalalignment="left", verticalalignment="top", transform=ax.transAxes, rotation=90)
        #
        x    = rms*2.5 / (xlim[1]-xlim[0]) + 0.01
        text = str(2.5)+r"$\sigma$ = "+str(np.round(rms*2.5,3)).ljust(5, "0")
        ax.text(x, 0.96, text, color="black", horizontalalignment="left", verticalalignment="top", transform=ax.transAxes, rotation=90)

        plt.savefig(self.outpng_hist_co10_snr, dpi=self.fig_dpi)

    ###################
    # _gaussfit_noise #
    ###################

    def _gaussfit_noise(
        self,
        data,
        bins=500,
        snr=0.5,
        ):
        """
        plot_noise
        """

        data[np.isnan(data)] = 0
        data[np.isinf(data)] = 0
        data = data[data!=0]

        # data
        histrange    = [data.min(), data.max()]
        p84_data     = np.percentile(data, 16) * -1  # 84th percentile of the inversed histogram
        histogram    = np.histogram(data, bins=bins, range=histrange)
        histx, histy = histogram[1][:-1], histogram[0]
        histx4fit    = histx[histx<p84_data*snr]
        histy4fit    = histy[histx<p84_data*snr]

        # fit
        x_bestfit    = np.linspace(histrange[0], histrange[1], bins)
        popt,_       = curve_fit(self._func1, histx4fit, histy4fit, p0=[np.max(histy4fit),p84_data], maxfev=10000)
        peak         = popt[0]
        rms          = abs(np.round(popt[1], 5))
        y_bestfit    = self._func1(x_bestfit, peak, rms)

        return histx, histy, histrange, peak, rms, x_bestfit, y_bestfit, p84_data

    def _func1(self, x, a, c):
        """
        """
        return a*np.exp(-(x)**2/(2*c**2))

    ############
    # do_stack #
    ############

    def do_stack(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.cprops_co10,taskname)

        x_co_cone, y_co_cone, v_co_cone, radius_co_cone, sigv_co_cone, \
        x_co_nocone, y_co_nocone, v_co_nocone, radius_co_nocone, sigv_co_nocone, \
        x_co_sbr, y_co_sbr, v_co_sbr, radius_co_sbr, sigv_co_sbr \
            = self._import_cprops_table(self.cprops_co10,addv=True)

        x_ci_cone, y_ci_cone, v_ci_cone, radius_ci_cone, sigv_ci_cone, \
        x_ci_nocone, y_ci_nocone, v_ci_nocone, radius_ci_nocone, sigv_ci_nocone, \
        x_ci_sbr, y_ci_sbr, v_ci_sbr, radius_ci_sbr, sigv_ci_sbr \
            = self._import_cprops_table(self.cprops_ci10,addv=True)

        shape   = imhead(self.cube_co10,mode="list")["shape"]
        box     = "0,0," + str(shape[0]-1) + "," + str(shape[1]-1)
        data    = imval(self.cube_co10,box=box)
        coords  = data["coords"]
        co_data = data["data"]
        co_x    = coords[:,0,0,0]
        co_y    = coords[0,:,0,1]
        co_freq = coords[0,0,:,2]

        """
        shape   = imhead(self.cube_ci10,mode="list")["shape"]
        box     = "0,0," + str(shape[0]-1) + "," + str(shape[1]-1)
        data    = imval(self.cube_ci10,box=box)
        coords  = data["coords"]
        ci_data = data["data"]
        ci_x    = coords[:,0,0,0]
        ci_y    = coords[0,:,0,1]
        ci_freq = coords[0,0,:,2]
        """

        #
        for i in range(len(x_co_cone)):
            this_x_co_cone = x_co_cone[i]
            this_y_co_cone = y_co_cone[i]
            this_v_co_cone = v_co_cone[i]
            this_r_co_cone = int(radius_co_cone[i] / self.scale_pc)
            this_s_co_cone = sigv_co_cone[i]

            x_center = np.argmin(np.abs((co_x-this_x_co_cone)))
            x_left   = x_center - this_r_co_cone*2
            x_right  = x_center + this_r_co_cone*2 + 2
            y_center = np.argmin(np.abs((co_y-this_y_co_cone)))
            y_left   = y_center - this_r_co_cone*2
            y_right  = y_center + this_r_co_cone*2 + 2

            print(this_x_co_cone)
            print(this_y_co_cone)
            print(x_center)
            print(y_center)
            print([x_left,x_right,y_left,y_right])

    ############
    # plot_map #
    ############

    def plot_map(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outfits_mom0_ci10,taskname)

        #
        myfig_fits2png(
            imcolor=self.outfits_mom0_ratio,
            outfile=self.outpng_map_ratio,
            imcontour1=self.outfits_mom0_co10,
            imsize_as=self.imsize_as,
            ra_cnt=str(self.ra_agn)+"deg",
            dec_cnt=str(self.dec_agn)+"deg",
            #unit_cont1=rms_vla,
            #levels_cont1=[-3,3,6,12,24,48],
            width_cont1=[1.0],
            set_title="ratio",
            colorlog=True,
            #scalebar=scalebar,
            #label_scalebar=label_scalebar,
            set_cbar=True,
            label_cbar="",
            clim=[0.1,1],
            #set_bg_color=set_bg_color,
            )

    ##################
    # plot_co_larson #
    ##################

    def plot_co_larson(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.cprops_co10,taskname)

        x_cone, y_cone, radius_cone, sigv_cone, mvir_cone, tpeak_cone, lci_cone, \
            x_nocone, y_nocone, radius_nocone, sigv_nocone, mvir_nocone, tpeak_nocone, lci_nocone, \
            x_sbr, y_sbr, radius_sbr, sigv_sbr, mvir_sbr, tpeak_sbr, lci_sbr = self._import_cprops_table(self.cprops_co10)

        mvir_cone   = 10**mvir_cone
        mvir_nocone = 10**mvir_nocone
        mvir_sbr    = 10**mvir_sbr

        ####################
        # plot: larson 1st #
        ####################
        xlim   = self.xlim_larson_1st
        ylim   = self.ylim_larson_1st
        title  = "Larson's 1st law"
        xlabel = "log Diameter (pc)"
        ylabel = "log velocity dispersion (km s$^{-1}$)"

        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        ad  = [0.215,0.83,0.10,0.90]
        myax_set(ax1, "both", xlim, ylim, title, xlabel, ylabel, adjust=ad)

        ax1.scatter(np.log10(radius_cone*2.0), np.log10(sigv_cone), lw=0, s=160, color="red", alpha=0.5)
        ax1.scatter(np.log10(radius_nocone*2.0), np.log10(sigv_nocone), lw=0, s=160, color="blue", alpha=0.5)
        ax1.scatter(np.log10(radius_sbr*2.0), np.log10(sigv_sbr), lw=0, s=160, color="grey", alpha=0.5)

        # save
        os.system("rm -rf " + self.outpng_co_larson_1st)
        plt.savefig(self.outpng_co_larson_1st, dpi=self.fig_dpi)

        ####################
        # plot: larson 2nd #
        ####################
        xlim   = self.xlim_larson_2nd
        ylim   = self.ylim_larson_2nd
        title  = "Larson's 2nd law"
        xlabel = "log M(H$_2$) ($M_{\odot}$)"
        ylabel = "log velocity dispersion (km s$^{-1}$)"

        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        ad  = [0.215,0.83,0.10,0.90]
        myax_set(ax1, "both", xlim, ylim, title, xlabel, ylabel, adjust=ad)

        ax1.scatter(np.log10(lci_cone*self.alpha_co), np.log10(sigv_cone), lw=0, s=160, color="red", alpha=0.5)
        ax1.scatter(np.log10(lci_nocone*self.alpha_co), np.log10(sigv_nocone), lw=0, s=160, color="blue", alpha=0.5)
        ax1.scatter(np.log10(lci_sbr*self.alpha_co), np.log10(sigv_sbr), lw=0, s=160, color="grey", alpha=0.5)

        # save
        os.system("rm -rf " + self.outpng_co_larson_2nd)
        plt.savefig(self.outpng_co_larson_2nd, dpi=self.fig_dpi)

        ####################
        # plot: larson 3rd #
        ####################
        density_cone   = lci_cone*self.alpha_co/(4./3.*np.pi*radius_cone**3)
        density_nocone = lci_nocone*self.alpha_co/(4./3.*np.pi*radius_nocone**3)
        density_sbr    = lci_sbr*self.alpha_co/(4./3.*np.pi*radius_sbr**3)

        rvir_cone   = mvir_cone / (lci_cone*self.alpha_co)
        rvir_nocone = mvir_nocone / (lci_nocone*self.alpha_co)
        rvir_sbr    = mvir_sbr / (lci_sbr*self.alpha_co)

        xlim   = self.xlim_larson_3rd
        ylim   = self.ylim_larson_3rd
        title  = "Larson's 3rd law"
        xlabel = "log Diameter (pc)"
        ylabel = "log Volume density (cm$^{-3}$)"

        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        ad  = [0.215,0.83,0.10,0.90]
        myax_set(ax1, "both", xlim, ylim, title, xlabel, ylabel, adjust=ad)

        ax1.scatter(np.log10(radius_cone*2.0)[rvir_cone>=np.median(rvir_cone)], np.log10(density_cone)[rvir_cone>=np.median(rvir_cone)], lw=0, s=40, color="red", alpha=0.5)
        ax1.scatter(np.log10(radius_nocone*2.0)[rvir_nocone>=np.median(rvir_nocone)], np.log10(density_nocone)[rvir_nocone>=np.median(rvir_nocone)], lw=0, s=40, color="blue", alpha=0.5)
        ax1.scatter(np.log10(radius_sbr*2.0)[rvir_sbr>=np.median(rvir_sbr)], np.log10(density_sbr)[rvir_sbr>=np.median(rvir_sbr)], lw=0, s=40, color="grey", alpha=0.5)

        ax1.scatter(np.log10(radius_cone*2.0)[rvir_cone<np.median(rvir_cone)], np.log10(density_cone)[rvir_cone<np.median(rvir_cone)], lw=0, s=160, color="red", alpha=0.5)
        ax1.scatter(np.log10(radius_nocone*2.0)[rvir_nocone<np.median(rvir_nocone)], np.log10(density_nocone)[rvir_nocone<np.median(rvir_nocone)], lw=0, s=160, color="blue", alpha=0.5)
        ax1.scatter(np.log10(radius_sbr*2.0)[rvir_sbr<np.median(rvir_sbr)], np.log10(density_sbr)[rvir_sbr<np.median(rvir_sbr)], lw=0, s=160, color="grey", alpha=0.5)

        # save
        os.system("rm -rf " + self.outpng_co_larson_3rd)
        plt.savefig(self.outpng_co_larson_3rd, dpi=self.fig_dpi)

    ####################
    # plot_cico_larson #
    ####################

    def plot_cico_larson(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.cprops_ci10,taskname)

        x_ci_cone, y_ci_cone, radius_ci_cone, sigv_ci_cone, mvir_ci_cone, tpeak_ci_cone, lci_ci_cone, \
            x_ci_nocone, y_ci_nocone, radius_ci_nocone, sigv_ci_nocone, mvir_ci_nocone, tpeak_ci_nocone, lci_ci_nocone, \
            x_ci_sbr, y_ci_sbr, radius_ci_sbr, sigv_ci_sbr, mvir_ci_sbr, tpeak_ci_sbr, lci_ci_sbr = self._import_cprops_table(self.cprops_ci10)
        x_co_cone, y_co_cone, radius_co_cone, sigv_co_cone, mvir_co_cone, tpeak_co_cone, lci_co_cone, \
            x_co_nocone, y_co_nocone, radius_co_nocone, sigv_co_nocone, mvir_co_nocone, tpeak_co_nocone, lci_co_nocone, \
            x_co_sbr, y_co_sbr, radius_co_sbr, sigv_co_sbr, mvir_co_sbr, tpeak_co_sbr, lci_co_sbr = self._import_cprops_table(self.cprops_co10)

        ####################
        # plot: larson 1st #
        ####################
        xlim   = self.xlim_larson_1st
        ylim   = self.ylim_larson_1st
        title  = "Larson's 1st law"
        xlabel = "log Diameter (pc)"
        ylabel = "log velocity dispersion (km s$^{-1}$)"

        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        ad  = [0.215,0.83,0.10,0.90]
        myax_set(ax1, "both", xlim, ylim, title, xlabel, ylabel, adjust=ad)

        alpha=1.0#0.3
        size=200#100
        ax1.scatter(np.log10(radius_co_cone*2.0), np.log10(sigv_co_cone), lw=0, s=200, color="deepskyblue", alpha=1.0)
        ax1.scatter(np.log10(radius_co_nocone*2.0), np.log10(sigv_co_nocone), lw=0, s=size, color="deepskyblue", alpha=alpha)
        ax1.scatter(np.log10(radius_co_sbr*2.0), np.log10(sigv_co_sbr), lw=0, s=size, color="deepskyblue", alpha=alpha)

        X, Y, Z = density_estimation(
            np.r_[np.log10(radius_co_cone*2.0),np.log10(radius_co_nocone*2.0),np.log10(radius_co_sbr*2.0)],
            np.r_[np.log10(sigv_co_cone),np.log10(sigv_co_nocone),np.log10(sigv_co_sbr)],
            xlim, ylim)
        ax1.contour(X, Y, Z, colors="blue")

        ax1.scatter(np.log10(radius_ci_cone*2.0), np.log10(sigv_ci_cone), lw=0, s=200, color="tomato", alpha=1.0)
        ax1.scatter(np.log10(radius_ci_nocone*2.0), np.log10(sigv_ci_nocone), lw=0, s=size, color="tomato", alpha=alpha)
        ax1.scatter(np.log10(radius_ci_sbr*2.0), np.log10(sigv_ci_sbr), lw=0, s=size, color="tomato", alpha=alpha)

        X, Y, Z = density_estimation(
            np.r_[np.log10(radius_ci_cone*2.0),np.log10(radius_ci_nocone*2.0),np.log10(radius_ci_sbr*2.0)],
            np.r_[np.log10(sigv_ci_cone),np.log10(sigv_ci_nocone),np.log10(sigv_ci_sbr)],
            xlim, ylim)
        ax1.contour(X, Y, Z, colors="red")

        # plot larson
        x1 = xlim[0]
        x2 = xlim[1]
        y1 = np.log10(1.10 * (10**x1)**0.38)
        y2 = np.log10(1.10 * (10**x2)**0.38)
        ax1.plot([x1,x2],[y1,y2],"--",lw=3,color="grey")

        # text
        ax1.text(0.03, 0.93, "[CI] (outflow)", color="tomato", transform=ax1.transAxes, weight="bold", fontsize=24)
        ax1.text(0.03, 0.88, "[CI] (non-outflow)", color="tomato", transform=ax1.transAxes, fontsize=24)
        ax1.text(0.03, 0.83, "CO (outflow)", color="deepskyblue", transform=ax1.transAxes, weight="bold", fontsize=24)
        ax1.text(0.03, 0.78, "CO (non-outflow)", color="deepskyblue", transform=ax1.transAxes, fontsize=24)

        # fill
        ax1.axvspan(xlim[0], np.log10(55*2), color='grey', alpha=.5, lw=0)
        ax1.axvspan(np.log10(55*2), xlim[1], 0, (np.log10(2.6*2)-ylim[0]) / (ylim[1]-ylim[0]), color='grey', alpha=.5, lw=0)

        # save
        os.system("rm -rf " + self.outpng_cico_larson_1st)
        plt.savefig(self.outpng_cico_larson_1st, dpi=self.fig_dpi)

        ####################
        # plot: larson 2nd #
        ####################
        xlim   = self.xlim_larson_2nd
        ylim   = self.ylim_larson_2nd
        title  = "Larson's 2nd law"
        xlabel = "log M(H$_2$) ($M_{\odot}$)"
        ylabel = "log velocity dispersion (km s$^{-1}$)"

        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        ad  = [0.215,0.83,0.10,0.90]
        myax_set(ax1, "both", xlim, ylim, title, xlabel, ylabel, adjust=ad)

        ax1.scatter(np.log10(lci_co_cone*self.alpha_co), np.log10(sigv_co_cone), lw=0, s=200, color="deepskyblue", alpha=1.0)
        ax1.scatter(np.log10(lci_co_nocone*self.alpha_co), np.log10(sigv_co_nocone), lw=0, s=100, color="deepskyblue", alpha=0.3)
        ax1.scatter(np.log10(lci_co_sbr*self.alpha_co), np.log10(sigv_co_sbr), lw=0, s=100, color="deepskyblue", alpha=0.3)

        ax1.scatter(np.log10(lci_ci_cone*self.alpha_ci), np.log10(sigv_ci_cone), lw=0, s=200, color="tomato", alpha=1.0)
        ax1.scatter(np.log10(lci_ci_nocone*self.alpha_ci), np.log10(sigv_ci_nocone), lw=0, s=100, color="tomato", alpha=0.3)
        ax1.scatter(np.log10(lci_ci_sbr*self.alpha_ci), np.log10(sigv_ci_sbr), lw=0, s=100, color="tomato", alpha=0.3)

        # save
        os.system("rm -rf " + self.outpng_cico_larson_2nd)
        plt.savefig(self.outpng_cico_larson_2nd, dpi=self.fig_dpi)

        ####################
        # plot: larson 3rd #
        ####################
        density_co_cone   = lci_co_cone*self.alpha_co/(4./3.*np.pi*radius_co_cone**3)
        density_co_nocone = lci_co_nocone*self.alpha_co/(4./3.*np.pi*radius_co_nocone**3)
        density_co_sbr    = lci_co_sbr*self.alpha_co/(4./3.*np.pi*radius_co_sbr**3)
        density_ci_cone   = lci_ci_cone*self.alpha_ci/(4./3.*np.pi*radius_ci_cone**3)
        density_ci_nocone = lci_ci_nocone*self.alpha_ci/(4./3.*np.pi*radius_ci_nocone**3)
        density_ci_sbr    = lci_ci_sbr*self.alpha_ci/(4./3.*np.pi*radius_ci_sbr**3)

        xlim   = self.xlim_larson_3rd
        ylim   = self.ylim_larson_3rd
        title  = "Larson's 3rd law"
        xlabel = "log Diameter (pc)"
        ylabel = "log Volume density (cm$^{-3}$)"

        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        ad  = [0.215,0.83,0.10,0.90]
        myax_set(ax1, "both", xlim, ylim, title, xlabel, ylabel, adjust=ad)

        ax1.scatter(np.log10(radius_co_cone*2.0), np.log10(density_co_cone), lw=0, s=200, color="deepskyblue", alpha=1.0)
        ax1.scatter(np.log10(radius_co_nocone*2.0), np.log10(density_co_nocone), lw=0, s=100, color="deepskyblue", alpha=0.3)
        ax1.scatter(np.log10(radius_co_sbr*2.0), np.log10(density_co_sbr), lw=0, s=100, color="deepskyblue", alpha=0.3)

        ax1.scatter(np.log10(radius_ci_cone*2.0), np.log10(density_ci_cone), lw=0, s=200, color="tomato", alpha=1.0)
        ax1.scatter(np.log10(radius_ci_nocone*2.0), np.log10(density_ci_nocone), lw=0, s=100, color="tomato", alpha=0.3)
        ax1.scatter(np.log10(radius_ci_sbr*2.0), np.log10(density_ci_sbr), lw=0, s=100, color="tomato", alpha=0.3)

        # save
        os.system("rm -rf " + self.outpng_cico_larson_3rd)
        plt.savefig(self.outpng_cico_larson_3rd, dpi=self.fig_dpi)

    ##################
    # plot_ci_larson #
    ##################

    def plot_ci_larson(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.cprops_ci10,taskname)

        x_cone, y_cone, radius_cone, sigv_cone, mvir_cone, tpeak_cone, lci_cone, \
            x_nocone, y_nocone, radius_nocone, sigv_nocone, mvir_nocone, tpeak_nocone, lci_nocone, \
            x_sbr, y_sbr, radius_sbr, sigv_sbr, mvir_sbr, tpeak_sbr, lci_sbr = self._import_cprops_table(self.cprops_ci10)

        mvir_cone   = 10**mvir_cone
        mvir_nocone = 10**mvir_nocone
        mvir_sbr    = 10**mvir_sbr

        ####################
        # plot: larson 1st #
        ####################
        xlim   = self.xlim_larson_1st
        ylim   = self.ylim_larson_1st
        title  = "Larson's 1st law"
        xlabel = "log Diameter (pc)"
        ylabel = "log velocity dispersion (km s$^{-1}$)"

        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        ad  = [0.215,0.83,0.10,0.90]
        myax_set(ax1, "both", xlim, ylim, title, xlabel, ylabel, adjust=ad)

        ax1.scatter(np.log10(radius_cone*2.0), np.log10(sigv_cone), lw=0, s=160, color="red", alpha=0.5)
        ax1.scatter(np.log10(radius_nocone*2.0), np.log10(sigv_nocone), lw=0, s=160, color="blue", alpha=0.5)
        ax1.scatter(np.log10(radius_sbr*2.0), np.log10(sigv_sbr), lw=0, s=160, color="grey", alpha=0.5)

        # save
        os.system("rm -rf " + self.outpng_ci_larson_1st)
        plt.savefig(self.outpng_ci_larson_1st, dpi=self.fig_dpi)

        ####################
        # plot: larson 2nd #
        ####################
        xlim   = self.xlim_larson_2nd
        ylim   = self.ylim_larson_2nd
        title  = "Larson's 2nd law"
        xlabel = "log M(H$_2$) ($M_{\odot}$)"
        ylabel = "log velocity dispersion (km s$^{-1}$)"

        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        ad  = [0.215,0.83,0.10,0.90]
        myax_set(ax1, "both", xlim, ylim, title, xlabel, ylabel, adjust=ad)

        ax1.scatter(np.log10(lci_cone*self.alpha_ci), np.log10(sigv_cone), lw=0, s=160, color="red", alpha=0.5)
        ax1.scatter(np.log10(lci_nocone*self.alpha_ci), np.log10(sigv_nocone), lw=0, s=160, color="blue", alpha=0.5)
        ax1.scatter(np.log10(lci_sbr*self.alpha_ci), np.log10(sigv_sbr), lw=0, s=160, color="grey", alpha=0.5)

        # save
        os.system("rm -rf " + self.outpng_ci_larson_2nd)
        plt.savefig(self.outpng_ci_larson_2nd, dpi=self.fig_dpi)

        ####################
        # plot: larson 3rd #
        ####################
        density_cone   = lci_cone*self.alpha_ci/(4./3.*np.pi*radius_cone**3)
        density_nocone = lci_nocone*self.alpha_ci/(4./3.*np.pi*radius_nocone**3)
        density_sbr    = lci_sbr*self.alpha_ci/(4./3.*np.pi*radius_sbr**3)

        rvir_cone   = mvir_cone / (lci_cone*self.alpha_ci)
        rvir_nocone = mvir_nocone / (lci_nocone*self.alpha_ci)
        rvir_sbr    = mvir_sbr / (lci_sbr*self.alpha_ci)

        xlim   = self.xlim_larson_3rd
        ylim   = self.ylim_larson_3rd
        title  = "Larson's 3rd law"
        xlabel = "log Diameter (pc)"
        ylabel = "log Volume density (cm$^{-3}$)"

        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        ad  = [0.215,0.83,0.10,0.90]
        myax_set(ax1, "both", xlim, ylim, title, xlabel, ylabel, adjust=ad)

        ax1.scatter(np.log10(radius_cone*2.0)[rvir_cone>=np.median(rvir_cone)], np.log10(density_cone)[rvir_cone>=np.median(rvir_cone)], lw=0, s=40, color="red", alpha=0.5)
        ax1.scatter(np.log10(radius_nocone*2.0)[rvir_nocone>=np.median(rvir_nocone)], np.log10(density_nocone)[rvir_nocone>=np.median(rvir_nocone)], lw=0, s=40, color="blue", alpha=0.5)
        ax1.scatter(np.log10(radius_sbr*2.0)[rvir_sbr>=np.median(rvir_sbr)], np.log10(density_sbr)[rvir_sbr>=np.median(rvir_sbr)], lw=0, s=40, color="grey", alpha=0.5)

        ax1.scatter(np.log10(radius_cone*2.0)[rvir_cone<np.median(rvir_cone)], np.log10(density_cone)[rvir_cone<np.median(rvir_cone)], lw=0, s=160, color="red", alpha=0.5)
        ax1.scatter(np.log10(radius_nocone*2.0)[rvir_nocone<np.median(rvir_nocone)], np.log10(density_nocone)[rvir_nocone<np.median(rvir_nocone)], lw=0, s=160, color="blue", alpha=0.5)
        ax1.scatter(np.log10(radius_sbr*2.0)[rvir_sbr<np.median(rvir_sbr)], np.log10(density_sbr)[rvir_sbr<np.median(rvir_sbr)], lw=0, s=160, color="grey", alpha=0.5)

        # save
        os.system("rm -rf " + self.outpng_ci_larson_3rd)
        plt.savefig(self.outpng_ci_larson_3rd, dpi=self.fig_dpi)

    ##################
    # plot_co_cprops #
    ##################

    def plot_co_cprops(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.cprops_co10,taskname)

        x_cone, y_cone, radius_cone, sigv_cone, mvir_cone, tpeak_cone, mci_cone, \
            x_nocone, y_nocone, radius_nocone, sigv_nocone, mvir_nocone, tpeak_nocone, mci_nocone, \
            x_sbr, y_sbr, radius_sbr, sigv_sbr, mvir_sbr, tpeak_sbr, mci_sbr = self._import_cprops_table(self.cprops_co10)

        ################
        # plot: radius #
        ################
        xlim   = [60,200]
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
        os.system("rm -rf " + self.outpng_co_hist_rad)
        plt.savefig(self.outpng_co_hist_rad, dpi=self.fig_dpi)

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
        os.system("rm -rf " + self.outpng_co_hist_sigv)
        plt.savefig(self.outpng_co_hist_sigv, dpi=self.fig_dpi)

        ##############
        # plot: mvir #
        ##############
        xlim   = [6.0,9.0]
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
        os.system("rm -rf " + self.outpng_co_hist_mvir)
        plt.savefig(self.outpng_co_hist_mvir, dpi=self.fig_dpi)

    ##################
    # plot_ci_cprops #
    ##################

    def plot_ci_cprops(
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
        os.system("rm -rf " + self.outpng_ci_hist_rad)
        plt.savefig(self.outpng_ci_hist_rad, dpi=self.fig_dpi)

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
        os.system("rm -rf " + self.outpng_ci_hist_sigv)
        plt.savefig(self.outpng_ci_hist_sigv, dpi=self.fig_dpi)

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
        os.system("rm -rf " + self.outpng_ci_hist_mvir)
        plt.savefig(self.outpng_ci_hist_mvir, dpi=self.fig_dpi)

    ########################
    # _import_cprops_table #
    ########################

    def _import_cprops_table(
        self,
        table,
        addv=False,
        ):
        """
        """

        # import cprops table
        f = pyfits.open(table)
        tb = f[1].data

        # extract parameters
        x      = (tb["XCTR_DEG"] - self.ra_agn) * -3600.
        y      = (tb["YCTR_DEG"] - self.dec_agn) * 3600.
        v      = tb["VCTR_KMS"]
        s2n    = tb["S2N"]
        radius = tb["RAD_NODC_NOEX"]
        sigv   = tb["SIGV_NODC_NOEX"]
        mvir   = tb["MVIR_MSUN"]
        tpeak  = tb["TMAX_K"]
        mci    = tb["MLUM_MSUN"]
        xdeg   = tb["XCTR_DEG"]
        ydeg   = tb["YCTR_DEG"]
        x_fov2 = (tb["XCTR_DEG"] - self.ra_fov2) * -3600.
        y_fov2 = (tb["YCTR_DEG"] - self.dec_fov2) * -3600.
        x_fov3 = (tb["XCTR_DEG"] - self.ra_fov3) * -3600.
        y_fov3 = (tb["YCTR_DEG"] - self.dec_fov3) * -3600.

        cut    = np.where(~np.isnan(radius) & ~np.isnan(sigv) & ~np.isnan(mvir) & ~np.isnan(tpeak))
        x      = x[cut]
        y      = y[cut]
        s2n    = s2n[cut]
        radius = radius[cut]
        sigv   = sigv[cut]
        mvir   = np.log10(mvir[cut])
        tpeak  = tpeak[cut]
        mci    = mci[cut]
        r_fov2 = np.sqrt(x_fov2[cut]**2 + y_fov2[cut]**2)
        r_fov3 = np.sqrt(x_fov3[cut]**2 + y_fov3[cut]**2)

        # bicone definition
        r          = np.sqrt(x**2 + y**2)
        theta      = np.degrees(np.arctan2(x, y)) + 90
        theta      = np.where(theta>0,theta,theta+360)
        cut_cone   = np.where((s2n>=self.snr_cprops) & (r<self.fov_diamter/2.0) & (theta>=self.theta2) & (theta<self.theta1) | (s2n>=self.snr_cprops) & (r<self.fov_diamter/2.0) & (theta>=self.theta2+180) & (theta<self.theta1+180))
        cut_nocone = np.where((s2n>=self.snr_cprops) & (r<self.fov_diamter/2.0) & (theta>=self.theta1) & (theta<self.theta2+180) | (s2n>=self.snr_cprops) & (r<self.fov_diamter/2.0) & (theta>=self.theta1+180) & (theta<self.theta2+360) | (s2n>=self.snr_cprops) & (r<self.fov_diamter/2.0) & (theta<self.theta1+180) & (theta<self.theta2))
        cut_sbr    = np.where((s2n>=self.snr_cprops) & (r>=self.fov_diamter/2.0) & (r_fov2<=self.fov_diamter/2.0) | (s2n>=self.snr_cprops) & (r>=self.fov_diamter/2.0) & (r_fov3<=self.fov_diamter/2.0))

        # data
        x_cone      = x[cut_cone]
        y_cone      = y[cut_cone]
        v_cone      = v[cut_cone]
        radius_cone = radius[cut_cone]
        sigv_cone   = sigv[cut_cone]
        mvir_cone   = mvir[cut_cone]
        tpeak_cone  = tpeak[cut_cone]
        mci_cone    = mci[cut_cone]
        xdeg_cone   = xdeg[cut_cone]
        ydeg_cone   = ydeg[cut_cone]

        x_nocone      = x[cut_nocone]
        y_nocone      = y[cut_nocone]
        v_nocone      = v[cut_nocone]
        radius_nocone = radius[cut_nocone]
        sigv_nocone   = sigv[cut_nocone]
        mvir_nocone   = mvir[cut_nocone]
        tpeak_nocone  = tpeak[cut_nocone]
        mci_nocone    = mci[cut_nocone]
        xdeg_nocone   = xdeg[cut_nocone]
        ydeg_nocone   = ydeg[cut_nocone]

        x_sbr      = x[cut_sbr]
        y_sbr      = y[cut_sbr]
        v_sbr      = v[cut_sbr]
        radius_sbr = radius[cut_sbr]
        sigv_sbr   = sigv[cut_sbr]
        mvir_sbr   = mvir[cut_sbr]
        tpeak_sbr  = tpeak[cut_sbr]
        mci_sbr    = mci[cut_sbr]
        xdeg_sbr   = xdeg[cut_sbr]
        ydeg_sbr   = ydeg[cut_sbr]

        if addv==False:
            return x_cone, y_cone, radius_cone, sigv_cone, mvir_cone, tpeak_cone, mci_cone, x_nocone, y_nocone, radius_nocone, sigv_nocone, mvir_nocone, tpeak_nocone, mci_nocone, x_sbr, y_sbr, radius_sbr, sigv_sbr, mvir_sbr, tpeak_sbr, mci_sbr
        else:
            return xdeg_cone, ydeg_cone, v_cone, radius_cone, sigv_cone, xdeg_nocone, ydeg_nocone, v_nocone, radius_nocone, sigv_nocone, xdeg_sbr, ydeg_sbr, v_sbr, radius_sbr, sigv_sbr

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
            self.tpeak_co10,#self.outfits_mom0_co10,
            tb_co10,
            "CO(1-0)",
            self.outpng_cprops_co10_agn,
            self.outpng_cprops_co10_fov2,
            self.outpng_cprops_co10_fov3,
            )
        self._plot_cprops_map(
            self.tpeak_ci10,#self.outfits_mom0_ci10,
            tb_ci10,
            "[CI](1-0)",
            self.outpng_cprops_ci10_agn,
            self.outpng_cprops_ci10_fov2,
            self.outpng_cprops_ci10_fov3,
            )
        self._plot_cprops_map2(
            self.tpeak_ci10,#self.outfits_mom0_ci10,
            tb_co10,
            tb_ci10,
            "[CI](1-0)+CO(1-0)",
            self.outpng_cprops_ci10_co10_agn,
            self.outpng_cprops_ci10_co10_fov2,
            self.outpng_cprops_ci10_co10_fov3,
            )

    #####################
    # _plot_cprops_map2 #
    #####################

    def _plot_cprops_map2(
        self,
        imagename,
        this_tb1,
        this_tb2,
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
            numann    = "ci-gmc2",
            txtfiles  = [this_tb1,this_tb2],
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
            numann    = "ci-gmc2",
            textann   = False,
            txtfiles  = [this_tb1,this_tb2],
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
            numann    = "ci-gmc2",
            textann   = False,
            txtfiles  = [this_tb1,this_tb2],
            set_title = linename + " Cloud Catalog",
            scalebar  = scalebar,
            label_scalebar = label_scalebar,
            colorlog  = True,
            set_cmap  = "Greys",
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
            textann   = False,
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
            textann   = False,
            txtfiles  = this_tb,
            set_title = linename + " Cloud Catalog",
            scalebar  = scalebar,
            label_scalebar = label_scalebar,
            colorlog  = True,
            set_cmap  = "Greys",
            )

    #################
    # do_align_cube #
    #################

    def do_align_cube(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.cube_co10,taskname)

        # read template
        template = "template.image"
        run_importfits(self.cube_ci10,template,defaultaxes=True)

        # regrid co10 to ci10
        run_imregrid(self.cube_co10,template,self.cube_co10+"_tmp1")
        run_imregrid(self.ncube_co10,template,self.ncube_co10+"_tmp1")

        # mask
        os.system("rm -rf mask.image")
        makemask(mode="copy", inpimage=template, inpmask=template+":mask0", output="mask.image:mask0", overwrite=False)
        run_exportfits("mask.image","mask.fits",dropstokes=True,delin=True)
        run_importfits("mask.fits","mask.image",delin=True)
        run_immath_two(self.cube_co10+"_tmp1","mask.image",self.cube_co10+"_tmp2","iif(IM1>-10000,IM0,IM1)")
        run_immath_two(self.ncube_co10+"_tmp1","mask.image",self.ncube_co10+"_tmp2","iif(IM1>-10000,IM0,IM1)")
        os.system("rm -rf " + template)
        os.system("rm -rf mask.image")
        os.system("rm -rf " + self.cube_co10 + "_tmp1")
        os.system("rm -rf " + self.ncube_co10 + "_tmp1")

        # exportfits
        run_exportfits(self.cube_co10+"_tmp2",self.cube_co10.replace(".fits","_aligned.fits"),delin=True,velocity=True)
        run_exportfits(self.ncube_co10+"_tmp2",self.ncube_co10.replace(".fits","_aligned.fits"),delin=True,velocity=True)

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
        run_immath_two(self.outfits_mom0_ci10+"_tmp2",self.outfits_mom0_co10+"_tmp2",self.outfits_mom0_ratio+"_tmp2","iif(IM1>0,IM0/IM1,0)")

        run_immath_two(self.outfits_mom0_ci10+"_tmp2",self.outfits_mom0_ratio+"_tmp2",self.outfits_mom0_ratio+"_tmp3","iif(IM1>0,0,1)")
        run_immath_two(self.outfits_mom0_ratio+"_tmp2",self.outfits_mom0_ratio+"_tmp3",self.outfits_mom0_ratio+"_tmp4","IM0+IM1")
        os.system("rm -rf " + self.outfits_mom0_ratio+"_tmp2")
        os.system("rm -rf " + self.outfits_mom0_ratio+"_tmp3")

        # exportfits
        run_exportfits(self.outfits_mom0_ci10+"_tmp2",self.outfits_mom0_ci10,delin=True)
        run_exportfits(self.outfits_mom0_co10+"_tmp2",self.outfits_mom0_co10,delin=True)
        run_exportfits(self.outfits_mom0_ratio+"_tmp4",self.outfits_mom0_ratio,delin=True)

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
