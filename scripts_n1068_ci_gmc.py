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
2023-09-27   gave up procpros because of inconsistency between CI and CO catalog
2023-09-28   started beamwise measurement
Toshiki Saito@Nichidai/NAOJ
"""

import os, sys, glob
import numpy as np
from scipy import stats
import matplotlib.patheffects as PathEffects

from mycasa_tasks import *
from mycasa_plots import *
from mycasa_sampling import *

def density_estimation(m1, m2, xlim, ylim):
    X, Y = np.mgrid[xlim[0]:xlim[1]:40j, ylim[0]:ylim[1]:40j]
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

        self.mom0_co10  = self.dir_raw + self._read_key("mom0_co10")
        self.mom0_ci10  = self.dir_raw + self._read_key("mom0_ci10")
        self.emom0_co10 = self.dir_raw + self._read_key("emom0_co10")
        self.emom0_ci10 = self.dir_raw + self._read_key("emom0_ci10")
        self.mom2_co10  = self.dir_raw + self._read_key("mom2_co10")
        self.mom2_ci10  = self.dir_raw + self._read_key("mom2_ci10")
        self.emom2_co10 = self.dir_raw + self._read_key("emom2_co10")
        self.emom2_ci10 = self.dir_raw + self._read_key("emom2_ci10")
        #
        self.fits_vla   = self.dir_raw + self._read_key("vla")
        self.fits_paa   = self.dir_raw + self._read_key("paa")

        #
        self.cube_co10   = self.dir_raw + self._read_key("cube_co10")
        self.cube_ci10   = self.dir_raw + self._read_key("cube_ci10")

        self.ncube_co10  = self.dir_raw + self._read_key("ncube_co10")
        self.ncube_ci10  = self.dir_raw + self._read_key("ncube_ci10")

        self.mask_co10   = self.dir_raw + self._read_key("mask_co10")
        self.mask_ci10   = self.dir_raw + self._read_key("mask_ci10")

        self.tpeak_co10  = self.dir_raw + self._read_key("tpeak_co10")
        self.tpeak_ci10  = self.dir_raw + self._read_key("tpeak_ci10")

        self.cprops_co10 = self.dir_raw + self._read_key("cprops_co10")
        self.cprops_ci10 = self.dir_raw + self._read_key("cprops_ci10")

        self.asgn_co10 = self.dir_raw + self._read_key("asgn_co10")
        self.asgn_ci10 = self.dir_raw + self._read_key("asgn_ci10")

    def _set_output_fits(self):
        """
        """

        self.outfits_mom0_co10  = self.dir_ready + self._read_key("outfits_mom0_co10")
        self.outfits_mom0_ci10  = self.dir_ready + self._read_key("outfits_mom0_ci10")
        self.outfits_mom0_ratio = self.dir_ready + self._read_key("outfits_mom0_ratio")

    def _set_input_param(self):
        """
        """

        #
        self.ra_agn  = float(self._read_key("ra_agn", "gal").split("deg")[0])
        self.dec_agn = float(self._read_key("dec_agn", "gal").split("deg")[0])
        self.snr_mom = 4.0

        ### old
        self.ra_fov2   = float(self._read_key("ra_fov2", "gal").split("deg")[0])
        self.dec_fov2  = float(self._read_key("dec_fov2", "gal").split("deg")[0])
        self.ra_fov3   = float(self._read_key("ra_fov3", "gal").split("deg")[0])
        self.dec_fov3  = float(self._read_key("dec_fov3", "gal").split("deg")[0])
        self.scale_pc  = float(self._read_key("scale", "gal"))
        self.scale_kpc = self.scale_pc / 1000.

        self.beam      = 0.8
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

        self.xlim_larson_1st = [1.1,2.7]
        self.ylim_larson_1st = [0.35,1.95]
        self.xlim_larson_2nd = [6.5,9.0]
        self.ylim_larson_2nd = [0.4,1.6]
        self.xlim_larson_3rd = [1.9,2.65]
        self.ylim_larson_3rd = [0.2,2.3]

    def _set_output_txt_png(self):
        """
        """

        self.outtxt_hexcat_co10 = self.dir_products + self._read_key("outtxt_hexcat_co10")
        self.outtxt_hexcat_ci10 = self.dir_products + self._read_key("outtxt_hexcat_ci10")
        self.outpng_r_vs_disp   = self.dir_products + self._read_key("outpng_r_vs_disp")
        self.outpng_map_ci_mom0 = self.dir_products + self._read_key("outpng_map_ci_mom0")
        self.outpng_map_ci_mom2 = self.dir_products + self._read_key("outpng_map_ci_mom2")
        self.outpng_map_co_mom0 = self.dir_products + self._read_key("outpng_map_co_mom0")
        self.outpng_map_co_mom2 = self.dir_products + self._read_key("outpng_map_co_mom2")
        self.outpng_map_vla     = self.dir_products + self._read_key("outpng_map_vla")
        self.outpng_map_paa     = self.dir_products + self._read_key("outpng_map_paa")
        self.outpng_map_ratio_m0 = self.dir_products + self._read_key("outpng_map_ratio_m0")
        self.outpng_map_ratio_m2 = self.dir_products + self._read_key("outpng_map_ratio_m2")

        # supplement
        self.outpng_radial_disp = self.dir_products + self._read_key("outpng_radial_disp")
        self.outpng_radial_mom0 = self.dir_products + self._read_key("outpng_radial_mom0")

        ### old
        self.outpng_hist_co10_pix  = self.dir_products + self._read_key("outpng_hist_co10_pix")
        self.outpng_hist_ci10_pix  = self.dir_products + self._read_key("outpng_hist_ci10_pix")
        self.outpng_hist_ratio_pix = self.dir_products + self._read_key("outpng_hist_ratio_pix")

        self.outpng_hist_co10_snr  = self.dir_products + self._read_key("outpng_hist_co10_snr")
        self.outpng_hist_ci10_snr  = self.dir_products + self._read_key("outpng_hist_ci10_snr")
        self.outpng_hist_ratio_snr = self.dir_products + self._read_key("outpng_hist_ratio_snr")

        self.outpng_hist_snr       = self.dir_products + self._read_key("outpng_hist_snr")
        self.outpng_hist_sigv      = self.dir_products + self._read_key("outpng_hist_sigv")
        self.outpng_hist_rad       = self.dir_products + self._read_key("outpng_hist_rad")
        self.outpng_hist_tpeak     = self.dir_products + self._read_key("outpng_hist_tpeak")

        self.outpng_cico_larson_1st = self.dir_products + self._read_key("outpng_cico_larson_1st")
        self.outpng_cico_dyn        = self.dir_products + self._read_key("outpng_cico_dyn")

        self.outtxt_catalog_ci = self.dir_products + self._read_key("outtxt_catalog_ci")
        self.outtxt_catalog_co = self.dir_products + self._read_key("outtxt_catalog_co")

        self.outpng_ci_sigv_v_ratio  = self.dir_products + self._read_key("outpng_ci_sigv_v_ratio")
        self.outpng_ci_coeff_v_ratio = self.dir_products + self._read_key("outpng_ci_coeff_v_ratio")

        # supplement
        self.outpng_ci_hist_rad           = self.dir_products + self._read_key("outpng_ci_hist_rad")
        self.outpng_ci_hist_sigv          = self.dir_products + self._read_key("outpng_ci_hist_sigv")
        self.outpng_ci_hist_mvir          = self.dir_products + self._read_key("outpng_ci_hist_mvir")
        self.outpng_ci_hist_snr           = self.dir_products + self._read_key("outpng_ci_hist_snr")

        self.outpng_co_hist_rad           = self.dir_products + self._read_key("outpng_co_hist_rad")
        self.outpng_co_hist_sigv          = self.dir_products + self._read_key("outpng_co_hist_sigv")
        self.outpng_co_hist_mvir          = self.dir_products + self._read_key("outpng_co_hist_mvir")
        self.outpng_co_hist_snr           = self.dir_products + self._read_key("outpng_co_hist_snr")

        self.outpng_cprops_co10_agn       = self.dir_products + self._read_key("outpng_cprops_co10_agn")
        self.outpng_cprops_ci10_agn       = self.dir_products + self._read_key("outpng_cprops_ci10_agn")
        self.outpng_cprops_ci10_co10_agn  = self.dir_products + self._read_key("outpng_cprops_ci10_co10_agn")

        self.outpng_cprops_co10_fov2      = self.dir_products + self._read_key("outpng_cprops_co10_fov2")
        self.outpng_cprops_ci10_fov2      = self.dir_products + self._read_key("outpng_cprops_ci10_fov2")
        self.outpng_cprops_ci10_co10_fov2 = self.dir_products + self._read_key("outpng_cprops_ci10_co10_fov2")

        self.outpng_cprops_co10_fov3      = self.dir_products + self._read_key("outpng_cprops_co10_fov3")
        self.outpng_cprops_ci10_fov3      = self.dir_products + self._read_key("outpng_cprops_ci10_fov3")
        self.outpng_cprops_ci10_co10_fov3 = self.dir_products + self._read_key("outpng_cprops_ci10_co10_fov3")

        self.outpng_ci_larson_1st         = self.dir_products + self._read_key("outpng_ci_larson_1st")
        self.outpng_ci_larson_2nd         = self.dir_products + self._read_key("outpng_ci_larson_2nd")
        self.outpng_ci_larson_3rd         = self.dir_products + self._read_key("outpng_ci_larson_3rd")

        self.outpng_co_larson_1st         = self.dir_products + self._read_key("outpng_co_larson_1st")
        self.outpng_co_larson_2nd         = self.dir_products + self._read_key("outpng_co_larson_2nd")
        self.outpng_co_larson_3rd         = self.dir_products + self._read_key("outpng_co_larson_3rd")

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
        do_sampling  = False,
        # plot
        plot_scatter = False,
        plot_hexmap  = False,
        # supplement
        plot_radial  = False,
        ):
        """
        This method runs all the methods which will create figures in the paper.
        """

        # analysis
        if do_sampling==True:
            self.do_sampling() # 2023-09-28: created

        # plot
        if plot_scatter==True:
            self.plot_scatter() # 2023-09-28: created

        if plot_hexmap==True:
            self.plot_hexmap() # 2023-10-04: created

        # supplement
        if plot_radial==True:
            self.plot_radial() # 2023-09-29: created

    # this is basically pycprops
    def run_ngc1068_cigmc_old(
        self,
        # analysis
        do_prepare   = False,
        print_cprops = False,
        data_stats   = False,
        meas_ratio   = False,
        # plot
        plot_cprops  = False,
        map_cprops   = False,
        plot_larson  = False,
        plot_ratio   = False,
        map_ratio    = False,
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

        if meas_ratio==True:
            self.meas_ratio()

        if plot_cprops==True:
            self.hist_cprops()

        if map_cprops==True:
            self.map_cprops()

        if plot_larson==True:
            self.plot_larson()

        if plot_ratio==True:
            self.plot_ratio()

        if map_ratio==True:
            self.map_ratio()

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

    ################
    # plot_hexmap #
    ################

    def plot_hexmap(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outtxt_hexcat_ci10,taskname)

        # ci10
        data_ci10 = np.loadtxt(self.outtxt_hexcat_ci10)
        self._plot_hexmap(
            self.outpng_map_ci_mom0,
            data_ci10[:,0][data_ci10[:,2]>data_ci10[:,3]*self.snr_mom],
            data_ci10[:,1][data_ci10[:,2]>data_ci10[:,3]*self.snr_mom],
            data_ci10[:,2][data_ci10[:,2]>data_ci10[:,3]*self.snr_mom],
            "[CI] Integrated Intensity",
            cmap     = "Reds",
            ann      = True,
            add_text = True,
            lim      = 9.9,
            size     = 820,
            bgcolor  = "white",
            textcolor= "black",
            label    = "(K km s$^{-1}$)",
            )
        self._plot_hexmap(
            self.outpng_map_ci_mom2,
            data_ci10[:,0][data_ci10[:,2]>data_ci10[:,3]*self.snr_mom],
            data_ci10[:,1][data_ci10[:,2]>data_ci10[:,3]*self.snr_mom],
            data_ci10[:,4][data_ci10[:,2]>data_ci10[:,3]*self.snr_mom],
            "[CI] Velocity Dispersion",
            cmap     = "Reds",
            ann      = True,
            add_text = False,
            lim      = 9.9,
            size     = 820,
            bgcolor  = "white",
            textcolor= "black",
            label    = "(km s$^{-1}$)",
            )

        # ci10
        data_co10 = np.loadtxt(self.outtxt_hexcat_co10)
        self._plot_hexmap(
            self.outpng_map_co_mom0,
            data_co10[:,0][data_co10[:,2]>data_co10[:,3]*self.snr_mom],
            data_co10[:,1][data_co10[:,2]>data_co10[:,3]*self.snr_mom],
            data_co10[:,2][data_co10[:,2]>data_co10[:,3]*self.snr_mom],
            "CO Integrated Intensity",
            cmap     = "Blues",
            ann      = True,
            add_text = False,
            lim      = 9.9,
            size     = 820,
            bgcolor  = "white",
            textcolor= "black",
            label    = "(K km s$^{-1}$)",
            )
        self._plot_hexmap(
            self.outpng_map_co_mom2,
            data_co10[:,0][data_co10[:,2]>data_co10[:,3]*self.snr_mom],
            data_co10[:,1][data_co10[:,2]>data_co10[:,3]*self.snr_mom],
            data_co10[:,4][data_co10[:,2]>data_co10[:,3]*self.snr_mom],
            "CO Velocity Dispersion",
            cmap     = "Blues",
            ann      = True,
            add_text = False,
            lim      = 9.9,
            size     = 820,
            bgcolor  = "white",
            textcolor= "black",
            label    = "(km s$^{-1}$)",
            )

        # ratio
        r = data_ci10[:,2]/data_co10[:,2]
        r[r>1] = 1
        self._plot_hexmap(
            self.outpng_map_ratio_m0,
            data_co10[:,0][r>0],
            data_co10[:,1][r>0],
            r[r>0],
            "[CI]/CO Intensity Ratio",
            cmap     = "rainbow",
            ann      = True,
            add_text = False,
            lim      = 30, #9.9,
            size     = 273, #820,
            bgcolor  = "white",
            textcolor= "black",
            label    = "(K km s$^{-1}$)",
            )
        r = data_ci10[:,4]/data_co10[:,4]
        r[r>1] = 1
        self._plot_hexmap(
            self.outpng_map_ratio_m2,
            data_co10[:,0][r>0],
            data_co10[:,1][r>0],
            r[r>0],
            "[CI]/CO Dispersion Ratio",
            cmap     = "rainbow",
            ann      = True,
            add_text = False,
            lim      = 30, #9.9,
            size     = 273, #820,
            bgcolor  = "white",
            textcolor= "black",
            label    = "(K km s$^{-1}$)",
            )

        # other
        self._plot_hexmap(
            self.outpng_map_vla,
            data_co10[:,0],
            data_co10[:,1],
            np.log10(data_co10[:,6]),
            "VLA 6 GHz Continuum",
            cmap     = "Greens",
            ann      = True,
            add_text = False,
            lim      = 9.9,
            size     = 820,
            bgcolor  = "white",
            textcolor= "black",
            label    = "",
            vmin     = -4.0,
            )
        self._plot_hexmap(
            self.outpng_map_paa,
            data_co10[:,0],
            data_co10[:,1],
            np.log10(data_co10[:,7]),
            "HST Paschen alpha",
            cmap     = "Greens",
            ann      = True,
            add_text = False,
            lim      = 9.9,
            size     = 820,
            bgcolor  = "white",
            textcolor= "black",
            label    = "",
            vmin     = 0.5,
            )

    ################
    # _plot_hexmap #
    ################

    def _plot_hexmap(
        self,
        outpng,
        x,y,c,
        title,
        cmap="rainbow",
        ann=False,
        lim=29.5,
        size=690,
        add_text=False,
        label="(K km s$^{-1}$)",
        bgcolor="white",
        scalebar="100pc",
        textcolor="black",
        vmin=0,
        ):
        """
        """

        # set plt, ax
        fig = plt.figure(figsize=(13,10))
        gs = gridspec.GridSpec(nrows=10, ncols=10)
        ax = plt.subplot(gs[0:10,0:10])
        ax.axhspan(-lim, lim, color=bgcolor, zorder=0)

        # set ax parameter
        myax_set(
        ax,
        grid=None,
        xlim=[lim, -lim],
        ylim=[-lim, lim],
        xlabel="R.A. offset (arcsec)",
        ylabel="Decl. offset (arcsec)",
        adjust=[0.10,0.99,0.10,0.93],
        fsize=25,
        )
        ax.set_aspect('equal', adjustable='box')

        c[np.isnan(c)] = -1e7
        # plot
        if vmin!=False:
            x = x[c>vmin]
            y = y[c>vmin]
            c = c[c>vmin]

        im = ax.scatter(x, y, s=size, c=c, cmap=cmap, marker="h", linewidths=0)

        # cbar
        cbar = plt.colorbar(im)
        cbar.set_label(label, color="black")

        # scale bar
        if scalebar=="100pc":
            bar = 100 / self.scale_pc
            ax.plot([-8,-8+bar],[-8,-8],"-",color=textcolor,lw=4)
            ax.text(-8, -8.5, "100 pc", color=textcolor,
                    horizontalalignment="right", verticalalignment="top")

        # text
        ax.text(0.03, 0.93, title, color=textcolor, transform=ax.transAxes, weight="bold", fontsize=32)

        # ann
        if ann==True:
            theta1      = -10.0 # degree
            theta2      = 70.0 # degree
            fov_diamter = 16.5 # arcsec (12m+7m Band 8)

            fov_diamter = 16.5
            efov1 = patches.Ellipse(xy=(-0,0), width=fov_diamter,
                height=fov_diamter, angle=0, fill=False, edgecolor=textcolor,
                alpha=1.0, lw=3.5)

            ax.add_patch(efov1)

            # plot NGC 1068 AGN and outflow geometry
            x1 = fov_diamter/2.0 * np.cos(np.radians(-1*theta1+90))
            y1 = fov_diamter/2.0 * np.sin(np.radians(-1*theta1+90))
            ax.plot([x1, -x1], [y1, -y1], "--", c=textcolor, lw=3.5)
            x2 = fov_diamter/2.0 * np.cos(np.radians(-1*theta2+90))
            y2 = fov_diamter/2.0 * np.sin(np.radians(-1*theta2+90))
            ax.plot([x2, -x2], [y2, -y2], "--", c=textcolor, lw=3.5)

        # add annotation comment
        if add_text==True:
            ax.plot([0,-7], [0,7], lw=3, c=textcolor)
            ax.text(-8.5, 7.5, "AGN", ha="right", va="center", weight="bold", color=textcolor)

        # save
        os.system("rm -rf " + outpng)
        plt.savefig(outpng, dpi=300)

    ################
    # plot_scatter #
    ################

    def plot_scatter(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outtxt_hexcat_ci10,taskname)

        # import co10
        data_co10 = np.loadtxt(self.outtxt_hexcat_co10)

        x_co10     = data_co10[:,0]
        y_co10     = data_co10[:,1]
        mom0_co10  = data_co10[:,2]
        emom0_co10 = data_co10[:,3]
        mom2_co10  = data_co10[:,4]
        emom2_co10 = data_co10[:,5]

        r_co10     = np.sqrt(x_co10**2 + y_co10**2)
        theta      = np.degrees(np.arctan2(x_co10, y_co10)) + 90
        theta_co10 = np.where(theta>0, theta, theta+360)

        # import ci10
        data_ci10 = np.loadtxt(self.outtxt_hexcat_ci10)

        x_ci10     = data_ci10[:,0]
        y_ci10     = data_ci10[:,1]
        mom0_ci10  = data_ci10[:,2]
        emom0_ci10 = data_ci10[:,3]
        mom2_ci10  = data_ci10[:,4]
        emom2_ci10 = data_ci10[:,5]

        r_ci10     = np.sqrt(x_ci10**2 + y_ci10**2)
        theta      = np.degrees(np.arctan2(x_ci10, y_ci10)) + 90
        theta_ci10 = np.where(theta>0, theta, theta+360)

        # calc
        cut = np.where((mom0_co10>emom0_co10*self.snr_mom) & (mom2_co10>emom2_co10*2))
        x_co10_all     = x_co10[cut]
        y_co10_all     = y_co10[cut]
        emom0_co10_all = emom0_co10[cut] / mom0_co10[cut] / np.log(10)
        mom0_co10_all  = np.log10(mom0_co10[cut])
        emom2_co10_all = emom2_co10[cut] / mom2_co10[cut] / np.log(10)
        mom2_co10_all  = np.log10(mom2_co10[cut])
        r_co10_all     = r_co10[cut]

        #cut = np.where((mom0_co10>emom0_co10*self.snr_mom) & (mom2_co10>emom2_co10) & (r_co10<self.fov_diamter/2.0) & (r_co10>self.r_cnd_as) & (theta_co10>=self.theta2) & (theta_co10<self.theta1) | (mom0_co10>emom0_co10*self.snr_mom) & (mom2_co10>emom2_co10*self.snr_mom) & (r_co10<self.fov_diamter/2.0) & (r_co10>self.r_cnd_as) & (theta_co10>=self.theta2+180) & (theta_co10<self.theta1+180))
        cut = np.where((mom0_ci10>mom0_co10) & (mom0_co10>emom0_co10*self.snr_mom) & (mom2_co10>emom2_co10) & (mom0_ci10>emom0_ci10*self.snr_mom) & (mom2_ci10>emom2_ci10))
        x_co10_cone     = x_co10[cut]
        y_co10_cone     = y_co10[cut]
        emom0_co10_cone = emom0_co10[cut] / mom0_co10[cut] / np.log(10)
        mom0_co10_cone  = np.log10(mom0_co10[cut])
        emom2_co10_cone = emom2_co10[cut] / mom2_co10[cut] / np.log(10)
        mom2_co10_cone  = np.log10(mom2_co10[cut])

        cut = np.where((mom0_ci10>emom0_ci10*self.snr_mom) & (mom2_ci10>emom2_ci10*2))
        x_ci10_all     = x_ci10[cut]
        y_ci10_all     = y_ci10[cut]
        emom0_ci10_all = emom0_ci10[cut] / mom0_ci10[cut] / np.log(10)
        mom0_ci10_all  = np.log10(mom0_ci10[cut])
        emom2_ci10_all = emom2_ci10[cut] / mom2_ci10[cut] / np.log(10)
        mom2_ci10_all  = np.log10(mom2_ci10[cut])
        r_ci10_all     = r_ci10[cut]

        #fig = plt.figure(figsize=(10,10))
        #plt.scatter(-1*x_co10_cone, y_co10_cone, color='grey')
        #plt.savefig("test.png", dpi=self.fig_dpi)

        #cut = np.where((mom0_ci10>emom0_ci10*self.snr_mom) & (mom2_ci10>emom2_ci10) & (r_ci10<self.fov_diamter/2.0) & (r_ci10>self.r_cnd_as) & (theta_ci10>=self.theta2) & (theta_ci10<self.theta1) | (mom0_ci10>emom0_ci10*self.snr_mom) & (mom2_ci10>emom2_ci10*self.snr_mom) & (r_ci10<self.fov_diamter/2.0) & (r_ci10>self.r_cnd_as) & (theta_ci10>=self.theta2+180) & (theta_ci10<self.theta1+180))
        cut = np.where((mom0_ci10>mom0_co10) & (mom0_co10>emom0_co10*self.snr_mom) & (mom2_co10>emom2_co10) & (mom0_ci10>emom0_ci10*self.snr_mom) & (mom2_ci10>emom2_ci10))
        x_ci10_cone     = x_ci10[cut]
        y_ci10_cone     = y_ci10[cut]
        emom0_ci10_cone = emom0_ci10[cut] / mom0_ci10[cut] / np.log(10)
        mom0_ci10_cone  = np.log10(mom0_ci10[cut])
        emom2_ci10_cone = emom2_ci10[cut] / mom2_ci10[cut] / np.log(10)
        mom2_ci10_cone  = np.log10(mom2_ci10[cut])

        ########
        # plot #
        ########
        x_co10 = mom0_co10_all
        y_co10 = mom2_co10_all
        c_co10 = r_co10_all
        x_ci10 = mom0_ci10_all
        y_ci10 = mom2_ci10_all
        c_ci10 = r_ci10_all

        xerr_co10 = emom0_co10_all
        yerr_co10 = emom2_co10_all
        xerr_ci10 = emom0_ci10_all
        yerr_ci10 = emom2_ci10_all

        x2_co10 = mom0_co10_cone
        y2_co10 = mom2_co10_cone
        x2_ci10 = mom0_ci10_cone
        y2_ci10 = mom2_ci10_cone

        xlim   = [np.min([np.nanmin(x_co10),np.nanmin(x_ci10)])-0.4,np.max([np.nanmax(x_co10),np.nanmax(x_ci10)])+0.4]
        ylim   = [np.min([np.nanmin(y_co10),np.nanmin(y_ci10)])-0.4,np.max([np.nanmax(y_co10),np.nanmax(y_ci10)])+0.4]
        vmax   = np.max(r_ci10)
        title  = "None"
        xlabel = "log$_{10}$ Integrated Intensity (K km s$^{-1}$)" # "log$_{10}$ H$_2$ Surface Density ($M_{\odot}$ pc$^{-2}$)"
        ylabel = "log$_{10}$ Velocity Dispersion (km s$^{-1}$)"
        alpha  = 1.0
        size   = 30

        # plot
        fig = plt.figure(figsize=(10,10))
        gs  = gridspec.GridSpec(nrows=200, ncols=200)
        ax1 = plt.subplot(gs[30:200,10:170])
        ax2 = plt.subplot(gs[0:30,10:170])
        ax3 = plt.subplot(gs[30:200,170:200])
        ad  = [0.215,0.83,0.10,0.90]
        myax_set(ax1, None, xlim, ylim, None, xlabel, ylabel, adjust=ad)
        myax_set(ax2, None, xlim, None, None, None, None)
        myax_set(ax3, None, None, ylim, None, None, None)

        # ax2 ticks
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.tick_params('x', length=0, which='major')
        ax2.tick_params('y', length=0, which='major')
        ax2.set_xticks([])
        ax2.set_yticks([])

        # ax3 ticks
        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.spines['left'].set_visible(False)
        ax3.tick_params('x', length=0, which='major')
        ax3.tick_params('y', length=0, which='major')
        ax3.set_xticks([])
        ax3.set_yticks([])

        # ax4
        #ax4 = plt.axes([0, 0, 1, 1])
        #position = InsetPosition(ax4, [0.6, 0.125, 0.225, 0.225])
        #ax4.set_axes_locator(position)
        #ax4.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        #ax4.set_xlim(xlim)
        #ax4.set_ylim(ylim)

        # plot co10 all
        X, Y, Z = density_estimation(x_co10, y_co10, xlim, ylim)
        ax1.contour(X, Y, Z, colors="blue", linewidths=[1], alpha=1.0, zorder=5e8)
        self._scatter_hist(x_co10, y_co10, ax1, ax2, ax3, "deepskyblue", xlim, ylim, "s")
        ax1.errorbar(x_co10, y_co10, xerr=xerr_co10, yerr=yerr_co10, fmt='.', color='grey', zorder=0, lw=1, capsize=0, markersize=0)

        # plot ci10 all
        X, Y, Z = density_estimation(x_ci10, y_ci10, xlim, ylim)
        ax1.contour(X, Y, Z, colors="red", linewidths=[1], alpha=1.0, zorder=5e8)
        self._scatter_hist(x_ci10, y_ci10, ax1, ax2, ax3, "tomato", xlim, ylim, "o", offset=0.15)
        ax1.errorbar(x_ci10, y_ci10, xerr=xerr_ci10, yerr=yerr_ci10, fmt='.', color='grey', zorder=0, lw=1, capsize=0, markersize=0)

        # plot co10 cone
        ax1.scatter(x2_co10, y2_co10, facecolor='lightgrey', edgecolor='blue', lw=2, s=70, marker="s", alpha=1.0, zorder=1e9)

        # plot ci10 cone
        ax1.scatter(x2_ci10, y2_ci10, facecolor='lightgrey', edgecolor='maroon', lw=2, s=70, marker="o", alpha=1.0, zorder=1e9)

        # pctl bar xaxis
        ypos = 0.97 * (ylim[1] - ylim[0]) + ylim[0]
        ax1.plot([np.percentile(x_ci10,16),np.percentile(x_ci10,84)], [ypos,ypos], '-', color="tomato", lw=3)
        ax1.scatter(np.percentile(x_ci10,50), ypos, marker='o', s=100, color="tomato", zorder=1e9)

        ypos = 0.95 * (ylim[1] - ylim[0]) + ylim[0]
        ax1.plot([np.percentile(x_co10,16),np.percentile(x_co10,84)], [ypos,ypos], '-', color="deepskyblue", lw=3)
        ax1.scatter(np.percentile(x_co10,50), ypos, marker='o', s=100, color="deepskyblue", zorder=1e9)

        ypos = 0.93 * (ylim[1] - ylim[0]) + ylim[0]
        ax1.plot([np.percentile(x2_ci10,16),np.percentile(x2_ci10,84)], [ypos,ypos], '-', color="maroon", lw=3)
        ax1.scatter(np.percentile(x2_ci10,50), ypos, marker='o', s=100, facecolor='lightgrey', edgecolor='maroon', lw=3, zorder=1e9)

        ypos = 0.91 * (ylim[1] - ylim[0]) + ylim[0]
        ax1.plot([np.percentile(x2_co10,16),np.percentile(x2_co10,84)], [ypos,ypos], '-', color="blue", lw=3)
        ax1.scatter(np.percentile(x2_co10,50), ypos, marker='o', s=100, facecolor='lightgrey', edgecolor='blue', lw=3, zorder=1e9)

        # pctl bar yaxis
        xpos = 0.97 * (xlim[1] - xlim[0]) + xlim[0]
        ax1.plot([xpos,xpos], [np.percentile(y_ci10,16),np.percentile(y_ci10,84)], '-', color="tomato", lw=3)
        ax1.scatter(xpos, np.percentile(y_ci10,50), marker='o', s=100, color="tomato", zorder=1e9)

        xpos = 0.95 * (xlim[1] - xlim[0]) + xlim[0]
        ax1.plot([xpos,xpos], [np.percentile(y_co10,16),np.percentile(y_co10,84)], '-', color="deepskyblue", lw=3)
        ax1.scatter(xpos, np.percentile(y_co10,50), marker='o', s=100, color="deepskyblue", zorder=1e9)

        xpos = 0.93 * (xlim[1] - xlim[0]) + xlim[0]
        ax1.plot([xpos,xpos], [np.percentile(y2_ci10,16),np.percentile(y2_ci10,84)], '-', color="maroon", lw=3)
        ax1.scatter(xpos, np.percentile(y2_ci10,50), marker='o', s=100, facecolor='lightgrey', edgecolor='maroon', lw=3, zorder=1e9)

        xpos = 0.91 * (xlim[1] - xlim[0]) + xlim[0]
        ax1.plot([xpos,xpos], [np.percentile(y2_co10,16),np.percentile(y2_co10,84)], '-', color="blue", lw=3)
        ax1.scatter(xpos, np.percentile(y2_co10,50), marker='o', s=100, facecolor='lightgrey', edgecolor='blue', lw=3, zorder=1e9)

        # plot ac4
        #ax4.scatter(x_co10, y_co10, c=c_co10, cmap='rainbow_r', lw=0, s=30, marker="s", alpha=1.0, vmin=0, vmax=vmax)
        #ax4.scatter(x_ci10, y_ci10, c=c_ci10, cmap='rainbow_r', lw=0, s=30, marker="o", alpha=1.0, vmin=0, vmax=vmax)

        txt = ax1.text(0.03, 0.90, "[CI] Clouds", color="tomato", transform=ax1.transAxes, weight="bold", fontsize=20)
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
        txt = ax1.text(0.03, 0.85, "CO Clouds", color="deepskyblue", transform=ax1.transAxes, weight="bold", fontsize=20)
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
        txt = ax1.text(0.03, 0.80, "[CI] Clouds (Outflow)", color="lightgrey", transform=ax1.transAxes, weight="bold", fontsize=20)
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='maroon')])
        txt = ax1.text(0.03, 0.75, "CO Clouds (Outflow)", color="lightgrey", transform=ax1.transAxes, weight="bold", fontsize=20)
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='blue')])

        # save
        os.system("rm -rf " + self.outpng_r_vs_disp)
        plt.savefig(self.outpng_r_vs_disp, dpi=self.fig_dpi)

    #################
    # _scatter_hist #
    #################

    def _scatter_hist(self, x, y, ax, ax_histx, ax_histy, color, xlim, ylim, marker="o", offset=0, lw=0):
        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # the scatter plot:
        ax.scatter(x, y, c=color, lw=lw, s=70, marker=marker, alpha=1.0, zorder=1e8)

        # now determine nice limits by hand:
        binwidth = 0.05
        xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        lim = (int(xymax/binwidth) + 1) * binwidth

        # hist
        #bins = np.arange(-lim, lim + binwidth, binwidth)
        #ax_histx.hist(x, bins=bins, color=color, lw=0, alpha=0.5)
        #x_histy.hist(y, bins=bins, color=color, lw=0, alpha=0.5, orientation='horizontal')

        # kde
        x_grid = np.arange(xlim[0], xlim[1], (xlim[1]-xlim[0])/70.)
        xkde = stats.gaussian_kde(x)
        x2 = xkde(x_grid)
        x2[0] = 0
        x2[-1] = 0
        ax_histx.plot(x_grid, x2, color=color, lw=1)
        ax_histx.fill_between(x_grid, 0, x2, color=color, alpha=0.5)

        y_grid = np.arange(ylim[0], ylim[1], (ylim[1]-ylim[0])/70.)
        ykde = stats.gaussian_kde(y)
        y2 = ykde(y_grid)
        y2[0] = 0
        y2[-1] = 0
        ax_histy.plot(y2, y_grid, color=color, lw=1)
        ax_histy.fill_between(y2, 0, y_grid, color=color, alpha=0.5)

        ax_histx.set_xlim(xlim)
        ax_histy.set_ylim(ylim)

        """
        # stats of xhist
        ax_histx.text(0.74, 0.85, "16$^{th}$-50$^{th}$-84$^{th}$ pctls.", color="black", transform=ax_histx.transAxes, fontsize=15)
        this_txt = str(np.round(np.percentile(x,16),2))+"-"+str(np.round(np.percentile(x,50),2))+"-"+str(np.round(np.percentile(x,84),2))
        ax_histx.text(0.74, 0.70-offset, this_txt, color=color, transform=ax_histx.transAxes, fontsize=15)

        # stats of yhist
        ax_histy.text(0.74, 0.95, "16$^{th}$-50$^{th}$-84$^{th}$ pctls.", color="black", transform=ax_histy.transAxes, fontsize=15, rotation=-90)
        this_txt = str(np.round(np.percentile(y,16),2))+"-"+str(np.round(np.percentile(y,50),2))+"-"+str(np.round(np.percentile(y,84),2))
        ax_histy.text(0.59-offset, 0.95, this_txt, color=color, transform=ax_histy.transAxes, fontsize=15, rotation=-90)
        """

    ###############
    # do_sampling #
    ###############

    def do_sampling(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.mom0_co10,taskname)

        # mom0
        hexx_co10, hexy_co10, hexc_co10_mom0 = hexbin_sampling(
            self.mom0_co10,
            self.ra_agn,
            self.dec_agn,
            beam=55/72.,
            gridsize=70,
            err=False,
            )
        _, _, hexc_co10_emom0 = hexbin_sampling(
            self.emom0_co10,
            self.ra_agn,
            self.dec_agn,
            beam=55/72.,
            gridsize=70,
            err=True,
            )

        os.system("rm -rf template.image test.image")
        importfits(
            fitsimage = self.mom0_co10,
            imagename = "template.image",
            )
        imregrid(
            imagename = self.mom0_ci10,
            template = "template.image",
            output = "test.image",
            )
        hexx_ci10, hexy_ci10, hexc_ci10_mom0 = hexbin_sampling(
            "test.image", #self.mom0_ci10,
            self.ra_agn,
            self.dec_agn,
            beam=55/72.,
            gridsize=70,
            err=False,
            )
        os.system("rm -rf test.image")
        imregrid(
            imagename = self.emom0_ci10,
            template = "template.image",
            output = "test.image",
            )
        _, _, hexc_ci10_emom0 = hexbin_sampling(
            "test.image", # self.emom0_ci10,
            self.ra_agn,
            self.dec_agn,
            beam=55/72.,
            gridsize=70,
            err=True,
            )

        # mom2
        _, _, hexc_co10_mom2 = hexbin_sampling(
            self.mom2_co10,
            self.ra_agn,
            self.dec_agn,
            beam=55/72.,
            gridsize=70,
            err=False,
            )
        _, _, hexc_co10_emom2 = hexbin_sampling(
            self.emom2_co10,
            self.ra_agn,
            self.dec_agn,
            beam=55/72.,
            gridsize=70,
            err=True,
            )

        os.system("rm -rf test.image")
        imregrid(
            imagename = self.mom2_ci10,
            template = "template.image",
            output = "test.image",
            )
        _, _, hexc_ci10_mom2 = hexbin_sampling(
            "test.image", # self.mom2_ci10,
            self.ra_agn,
            self.dec_agn,
            beam=55/72.,
            gridsize=70,
            err=False,
            )
        os.system("rm -rf test.image")
        imregrid(
            imagename = self.emom2_ci10,
            template = "template.image",
            output = "test.image",
            )
        _, _, hexc_ci10_emom2 = hexbin_sampling(
            "test.image", #self.emom2_ci10,
            self.ra_agn,
            self.dec_agn,
            beam=55/72.,
            gridsize=70,
            err=True,
            )

        # other
        os.system("rm -rf test.image test.image2")
        imsmooth(
            imagename = self.fits_vla,
            targetres = True,
            major="0.8arcsec",
            minor="0.8arcsec",
            pa="0deg",
            outfile="test.image2")
        imregrid(
            imagename = "test.image2",
            template = "template.image",
            output = "test.image",
            )
        _, _, hexc_vla = hexbin_sampling(
            "test.image",
            self.ra_agn,
            self.dec_agn,
            beam=55/72.,
            gridsize=70,
            err=False,
            )
        os.system("rm -rf test.image test.image2")
        imsmooth(
            imagename = self.fits_paa,
            targetres = False,
            major="0.8arcsec",
            minor="0.8arcsec",
            pa="0deg",
            outfile="test.image2")
        imregrid(
            imagename = "test.image2",
            template = "template.image",
            output = "test.image",
            )
        _, _, hexc_paa = hexbin_sampling(
            self.fits_paa,
            self.ra_agn,
            self.dec_agn,
            beam=55/72.,
            gridsize=70,
            err=False,
            )
        os.system("rm -rf template.image test.image test.image2")

        # combine
        data_co10 = np.c_[hexx_co10, hexy_co10, hexc_co10_mom0, hexc_co10_emom0, hexc_co10_mom2, hexc_co10_emom2, hexc_vla, hexc_paa]
        np.savetxt(self.outtxt_hexcat_co10, data_co10)
        data_ci10 = np.c_[hexx_ci10, hexy_ci10, hexc_ci10_mom0, hexc_ci10_emom0, hexc_ci10_mom2, hexc_ci10_emom2]
        np.savetxt(self.outtxt_hexcat_ci10, data_ci10)

    #########################
    #########################
    # run_ngc1068_cigmc_old #
    #########################
    #########################

    #############
    # map_ratio #
    #############

    def map_ratio(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.cprops_co10,taskname)

        # import fits table
        f = pyfits.open(self.cprops_ci10)
        tb_ci10 = f[1].data

        scalebar = 100. / self.scale_pc
        label_scalebar = "100 pc"

        chans  = '50~139'
        chans2 = '0~89'
        this_ra = 40.66900417
        this_dec = -0.01395556

        os.system('rm -rf ci10.subimage')
        imsubimage(
            imagename = self.asgn_ci10, #self.cube_ci10,
            outfile   = 'ci10.subimage',
            chans     = chans,
            )

        for i in range(90):
            print('# loop ' + str(i))
            os.system('rm -rf this_ci10.subimage')
            imsubimage(
                imagename = 'ci10.subimage',
                outfile   = 'this_ci10.subimage',
                chans     = str(i),
                )

            os.system('rm -rf channel_' + str(i).zfill(2) + '.png')
            myfig_fits2png(
                'this_ci10.subimage',
                'channel_' + str(i).zfill(2) + '.png',
                imsize_as = 5.0,
                ra_cnt    = str(this_ra) + "deg",
                dec_cnt   = str(this_dec) + "deg",
                numann    = "ci-gmc",
                txtfiles  = tb_ci10,
                scalebar  = scalebar,
                label_scalebar = label_scalebar,
                colorlog  = False, #True,
                set_cmap  = "prism", # "Greys",
                textann   = False,
                set_title = 'channel ' + str(i),
                )

        os.system('rm -rf ci10.subimage')
        os.system('rm -rf this_ci10.subimage')
        os.system('rm -rf this_ci10.subimage.fits')
        os.system('convert -delay 10 -loop 0 channel_*.png movie_ci10.gif')

        """
        #
        os.system('rm -rf co10.subimage')
        imsubimage(
            imagename = self.cube_co10,
            outfile   = 'co10.subimage',
            chans     = chans,
            )

        for i in range(90):
            print('# loop ' + str(i))
            os.system('rm -rf this_co10.subimage')
            imsubimage(
                imagename = 'co10.subimage',
                outfile   = 'this_co10.subimage',
                chans     = str(i),
                )

            os.system('rm -rf channel_' + str(i).zfill(2) + '.png')
            myfig_fits2png(
                'this_co10.subimage',
                'channel_' + str(i).zfill(2) + '.png',
                imsize_as = 5.0,
                ra_cnt    = str(this_ra) + "deg",
                dec_cnt   = str(this_dec) + "deg",
                numann    = "ci-gmc",
                txtfiles  = tb_ci10,
                scalebar  = scalebar,
                label_scalebar = label_scalebar,
                colorlog  = True,
                set_cmap  = "Greys",
                textann   = False,
                set_title = 'channel ' + str(i),
                )

        os.system('rm -rf co10.subimage')
        os.system('rm -rf this_co10.subimage')
        os.system('rm -rf this_co10.subimage.fits')
        os.system('convert -delay 10 -loop 0 channel_*.png movie_co10.gif')
        """

    ##############
    # plot_ratio #
    ##############

    def plot_ratio(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outtxt_catalog_ci,taskname)

        data  = np.loadtxt(self.outtxt_catalog_ci)
        x     = data[:,2]
        y     = data[:,4]
        z     = data[:,3]
        yerr  = data[:,5]
        s2n   = data[:,1]
        r     = data[:,6]
        theta = data[:,7]
        xpos  = data[:,8]
        ypos  = data[:,9]

        x     = np.log10(x[s2n>5])
        y     = np.log10(y[s2n>5])
        z     = np.log10(z[s2n>5])
        yerr  = np.log10(yerr[s2n>5])
        r     = r[s2n>5]
        theta = theta[s2n>5]
        xpos  = xpos[s2n>5]
        ypos  = ypos[s2n>5]

        cut   = np.where(~np.isnan(x) & ~np.isnan(y) & ~np.isnan(yerr))
        x_all = x[cut]
        y_all = y[cut]
        z_all = z[cut]

        #cut_cone = np.where(~np.isnan(x) & ~np.isnan(y) & ~np.isnan(yerr) & (r<self.fov_diamter/2.0) & (theta>=self.theta2) & (theta<self.theta1) | ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(yerr) & (r<self.fov_diamter/2.0) & (theta>=self.theta2+180) & (theta<self.theta1+180))
        cut_cone = np.where(~np.isnan(x) & ~np.isnan(y) & ~np.isnan(yerr) & (xpos>0) & (xpos<4) & (ypos<-1.5) & (ypos>-4) & (r<self.fov_diamter/2.0) & (theta>=self.theta2+180) & (theta<self.theta1+180))
        x_cone = x[cut_cone]
        y_cone = y[cut_cone]
        z_cone = z[cut_cone]

        #
        data = np.loadtxt(self.outtxt_catalog_co)
        x2    = data[:,2]
        y2    = data[:,4]
        z2    = data[:,3]
        y2err = data[:,5]
        s2n2  = data[:,1]
        r2    = data[:,6]
        theta2 = data[:,7]

        x2    = np.log10(x2[s2n2>5])
        y2    = np.log10(y2[s2n2>5])
        z2    = np.log10(z2[s2n2>5])
        y2err = np.log10(y2err[s2n2>5])
        r2    = r2[s2n2>5]
        theta2 = theta2[s2n2>5]

        cut    = np.where(~np.isnan(x2) & ~np.isnan(y2) & ~np.isnan(y2err))
        x2_all = x2[cut]
        y2_all = y2[cut]
        z2_all = z2[cut]

        cut_cone = np.where(~np.isnan(x2) & ~np.isnan(y2) & ~np.isnan(y2err) & (r2<self.fov_diamter/2.0) & (theta2>=self.theta2) & (theta2<self.theta1) | ~np.isnan(x2) & ~np.isnan(y2) & ~np.isnan(y2err) & (r2<self.fov_diamter/2.0) & (theta2>=self.theta2+180) & (theta2<self.theta1+180))
        x2_cone = x2[cut_cone]
        y2_cone = y2[cut_cone]
        z2_cone = z2[cut_cone]

        ########
        # plot #
        ########

        xlim   = None
        ylim   = None
        xlabel = "log$_{10}$ Velocity Dispersion (km s$^{-1}$)"
        ylabel = "log$_{10}$ Ratio"

        # plot
        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        ad  = [0.215,0.83,0.10,0.90]
        myax_set(ax1, None, xlim, ylim, None, xlabel, ylabel, adjust=ad)

        ax1.scatter(x_all, y_all, c="tomato", lw=0, s=100)
        ax1.scatter(x2_all, y2_all, c="deepskyblue", lw=0, s=100)

        ax1.scatter(x_cone, y_cone, c="tomato", lw=2, s=100)
        #ax1.scatter(x2_cone, y2_cone, c="deepskyblue", lw=2, s=100)

        plt.savefig(self.outpng_ci_sigv_v_ratio, dpi=self.fig_dpi)

        xlim   = None
        ylim   = None
        xlabel = "log$_{10}$ $\sigma^2$/$r$"
        ylabel = "log$_{10}$ Ratio"

        # plot
        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        ad  = [0.215,0.83,0.10,0.90]
        myax_set(ax1, None, xlim, ylim, None, xlabel, ylabel, adjust=ad)

        ax1.scatter(z_all, y_all, c="tomato", lw=0, s=100)
        ax1.scatter(z2_all, y2_all, c="deepskyblue", lw=0, s=100)

        ax1.scatter(z_cone, y_cone, c="tomato", lw=2, s=100)
        #ax1.scatter(z2_cone, y2_cone, c="deepskyblue", lw=2, s=100)

        plt.savefig(self.outpng_ci_coeff_v_ratio, dpi=self.fig_dpi)

    ##############
    # meas_ratio #
    ##############

    def meas_ratio(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.cprops_co10,taskname)

        # import cprops table
        f = pyfits.open(self.cprops_co10)
        tb = f[1].data
        cnum_co10  = tb["XCTR_PIX"]
        x_co10     = (tb["XCTR_DEG"] - self.ra_agn) * -3600.
        y_co10     = (tb["YCTR_DEG"] - self.dec_agn) * 3600.
        s2n_co10   = tb["S2N"]
        sigv_co10  = tb["SIGV_KMS"]
        dyn_co10   = tb["SIGV_KMS"] * tb["SIGV_KMS"] / tb["RAD_PC"]

        r_co10     = np.sqrt(x_co10**2 + y_co10**2)
        theta      = np.degrees(np.arctan2(x_co10, y_co10)) + 90
        theta_co10 = np.where(theta>0, theta, theta+360)

        f = pyfits.open(self.cprops_ci10)
        tb = f[1].data
        cnum_ci10  = tb["XCTR_PIX"]
        x_ci10     = (tb["XCTR_DEG"] - self.ra_agn) * -3600.
        y_ci10     = (tb["YCTR_DEG"] - self.dec_agn) * 3600.
        s2n_ci10   = tb["S2N"]
        sigv_ci10  = tb["SIGV_KMS"]
        dyn_ci10   = tb["SIGV_KMS"] * tb["SIGV_KMS"] / tb["RAD_PC"]

        r_ci10     = np.sqrt(x_ci10**2 + y_ci10**2)
        theta      = np.degrees(np.arctan2(x_ci10, y_ci10)) + 90
        theta_ci10 = np.where(theta>0, theta, theta+360)

        # import
        f,_ = imval_all(self.asgn_ci10)
        mask_ci10 = f["data"].flatten()

        f,_ = imval_all(self.asgn_co10)
        mask_co10 = f["data"].flatten()

        f,_ = imval_all(self.cube_co10.replace(".fits","_aligned.fits"))
        data_co10 = f["data"].flatten()
        f,_ = imval_all(self.ncube_co10.replace(".fits","_aligned.fits"))
        ndata_co10 = f["data"].flatten()

        f,_ = imval_all(self.cube_ci10)
        data_ci10 = f["data"].flatten()
        f,_ = imval_all(self.ncube_ci10)
        ndata_ci10 = f["data"].flatten()

        # measure line ratio for ci10 clouds
        ci_catalog_ratio = []
        for i in range(len(cnum_ci10)):
            this_co10  = np.nan_to_num(data_co10[mask_ci10==i])
            this_nco10 = np.nan_to_num(ndata_co10[mask_ci10==i])
            this_ci10  = np.nan_to_num(data_ci10[mask_ci10==i])
            this_nci10 = np.nan_to_num(ndata_ci10[mask_ci10==i])

            mask = np.where((this_co10>this_nco10*2.5) & (this_ci10>this_nci10*2.5))
            this_co10   = np.sum(this_co10[mask])
            this_nco10  = np.sqrt(np.sum((this_nco10[mask])**2))
            this_ci10   = np.sum(this_ci10[mask])
            this_nci10  = np.sqrt(np.sum((this_nci10[mask])**2))

            this_ratio  = this_ci10 / this_co10
            this_nratio = this_ratio * np.sqrt((this_nco10/this_co10)**2 + (this_nci10/this_ci10)**2)

            ci_catalog_ratio.append([i, s2n_ci10[i], sigv_ci10[i], dyn_ci10[i], this_ratio, this_nratio, r_ci10[i], theta_ci10[i], x_ci10[i], y_ci10[i]])

        ci_catalog_ratio = np.array(ci_catalog_ratio)
        np.savetxt(self.outtxt_catalog_ci, ci_catalog_ratio)

        # measure line ratio for co10 clouds
        co_catalog_ratio = []
        for i in range(len(cnum_co10)):
            this_co10  = np.nan_to_num(data_co10[mask_co10==i])
            this_nco10 = np.nan_to_num(ndata_co10[mask_co10==i])
            this_ci10  = np.nan_to_num(data_ci10[mask_co10==i])
            this_nci10 = np.nan_to_num(ndata_ci10[mask_co10==i])

            mask = np.where((this_co10>this_nco10*2.5) & (this_ci10>this_nci10*2.5))
            this_co10   = np.sum(this_co10[mask])
            this_nco10  = np.sqrt(np.sum((this_nco10[mask])**2))
            this_ci10   = np.sum(this_ci10[mask])
            this_nci10  = np.sqrt(np.sum((this_nci10[mask])**2))

            this_ratio  = this_ci10 / this_co10
            this_nratio = this_ratio * np.sqrt((this_nco10/this_co10)**2 + (this_nci10/this_ci10)**2)

            co_catalog_ratio.append([i, s2n_co10[i], sigv_co10[i], dyn_co10[i], this_ratio, this_nratio, r_co10[i], theta_co10[i]])

        co_catalog_ratio = np.array(co_catalog_ratio)
        np.savetxt(self.outtxt_catalog_co, co_catalog_ratio)

    ###############
    # plot_larson #
    ###############

    def plot_larson(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.cprops_co10,taskname)

        ####################
        # extract all data #
        ####################

        # import cprops table
        f = pyfits.open(self.cprops_co10)
        tb = f[1].data

        # extract parameters
        x_fov1_co10  = (tb["XCTR_DEG"] - self.ra_agn) * -3600.
        y_fov1_co10  = (tb["YCTR_DEG"] - self.dec_agn) * 3600.
        s2n_co10     = tb["S2N"]
        radius_co10  = tb["RAD_PC"]
        sigv_co10    = tb["SIGV_KMS"]
        mvir_co10    = tb["MVIR_MSUN"]
        density_co10 = tb["MLUM_MSUN"] / radius_co10 / radius_co10 * 4.0 / 6.7
        tpeak_co10   = tb["TMAX_K"]
        dyn_co10     = tb["SIGV_KMS"] * tb["SIGV_KMS"] / tb["RAD_PC"]

        # import cprops table
        f = pyfits.open(self.cprops_ci10)
        tb = f[1].data

        # extract parameters
        x_fov1_ci10  = (tb["XCTR_DEG"] - self.ra_agn) * -3600.
        y_fov1_ci10  = (tb["YCTR_DEG"] - self.dec_agn) * 3600.
        s2n_ci10     = tb["S2N"]
        radius_ci10  = tb["RAD_PC"]
        sigv_ci10    = tb["SIGV_KMS"]
        mvir_ci10    = tb["MVIR_MSUN"]
        density_ci10 = tb["MLUM_MSUN"] / radius_ci10 / radius_ci10 * 20. / 6.7
        tpeak_ci10   = tb["TMAX_K"]
        dyn_ci10     = tb["SIGV_KMS"] * tb["SIGV_KMS"] / tb["RAD_PC"]

        x_co10 = radius_co10[s2n_co10>self.snr_cprops]
        y_co10 = sigv_co10[s2n_co10>self.snr_cprops]
        x_co10 = np.nan_to_num(np.log10(x_co10))
        y_co10 = np.nan_to_num(np.log10(y_co10))

        x_ci10 = radius_ci10[s2n_ci10>self.snr_cprops]
        y_ci10 = sigv_ci10[s2n_ci10>self.snr_cprops]
        x_ci10 = np.nan_to_num(np.log10(x_ci10))
        y_ci10 = np.nan_to_num(np.log10(y_ci10))


        x2_co10 = density_co10[s2n_co10>self.snr_cprops]
        x2_co10 = np.log10(x2_co10)
        y2_co10 = dyn_co10[s2n_co10>self.snr_cprops]
        y2_co10 = np.log10(y2_co10)
        cut = np.where((~np.isnan(x2_co10)) & (~np.isnan(y2_co10)) & (x2_co10!=0) & (y2_co10!=0))
        x2_co10 = x2_co10[cut]
        y2_co10 = y2_co10[cut]

        x2_ci10 = density_ci10[s2n_ci10>self.snr_cprops]
        x2_ci10 = np.log10(x2_ci10)
        y2_ci10 = dyn_ci10[s2n_ci10>self.snr_cprops]
        y2_ci10 = np.log10(y2_ci10)
        cut = np.where((~np.isnan(x2_ci10)) & (~np.isnan(y2_ci10)) & (x2_ci10!=0) & (y2_ci10!=0))
        x2_ci10 = x2_ci10[cut]
        y2_ci10 = y2_ci10[cut]

        ########################
        # extract outflow data #
        ########################
        r_fov1_co10 = np.sqrt(x_fov1_co10**2 + y_fov1_co10**2)
        theta = np.degrees(np.arctan2(x_fov1_co10, y_fov1_co10)) + 90
        theta = np.where(theta>0, theta, theta+360)

        cut_cone = np.where((s2n_co10>=self.snr_cprops) & (r_fov1_co10<self.fov_diamter/2.0) & (theta>=self.theta2) & (theta<self.theta1) | (s2n_co10>=self.snr_cprops) & (r_fov1_co10<self.fov_diamter/2.0) & (theta>=self.theta2+180) & (theta<self.theta1+180))

        x_co10_cone = radius_co10[cut_cone]
        y_co10_cone = sigv_co10[cut_cone]
        x_co10_cone = np.nan_to_num(np.log10(x_co10_cone))
        y_co10_cone = np.nan_to_num(np.log10(y_co10_cone))

        x2_co10_cone = density_co10[cut_cone]
        y2_co10_cone = dyn_co10[cut_cone]
        x2_co10_cone = np.nan_to_num(np.log10(x2_co10_cone))
        y2_co10_cone = np.nan_to_num(np.log10(y2_co10_cone))

        r_fov1_ci10 = np.sqrt(x_fov1_ci10**2 + y_fov1_ci10**2)
        theta = np.degrees(np.arctan2(x_fov1_ci10, y_fov1_ci10)) + 90
        theta = np.where(theta>0, theta, theta+360)

        cut_cone = np.where((s2n_ci10>=self.snr_cprops) & (r_fov1_ci10<self.fov_diamter/2.0) & (theta>=self.theta2) & (theta<self.theta1) | (s2n_ci10>=self.snr_cprops) & (r_fov1_ci10<self.fov_diamter/2.0) & (theta>=self.theta2+180) & (theta<self.theta1+180))
        cut_cone = np.where((x_fov1_ci10>0) & (x_fov1_ci10<4) & (y_fov1_ci10<-1.5) & (y_fov1_ci10>-4) & (s2n_ci10>=self.snr_cprops) & (r_fov1_ci10<self.fov_diamter/2.0) & (theta>=self.theta2+180) & (theta<self.theta1+180))

        x_ci10_cone = radius_ci10[cut_cone]
        y_ci10_cone = sigv_ci10[cut_cone]
        x_ci10_cone = np.nan_to_num(np.log10(x_ci10_cone))
        y_ci10_cone = np.nan_to_num(np.log10(y_ci10_cone))

        x2_ci10_cone = density_ci10[cut_cone]
        y2_ci10_cone = dyn_ci10[cut_cone]
        x2_ci10_cone = np.nan_to_num(np.log10(x2_ci10_cone))
        y2_ci10_cone = np.nan_to_num(np.log10(y2_ci10_cone))

        ####################
        # plot: larson 1st #
        ####################

        xlim   = self.xlim_larson_1st
        ylim   = self.ylim_larson_1st
        title  = "Larson's 1st law"
        xlabel = "log$_{10}$ Radius (pc)"
        ylabel = "log$_{10}$ Velocity Dispersion (km s$^{-1}$)"
        alpha  = 1.0
        size   = 30

        # plot
        fig = plt.figure(figsize=(10,10))
        gs  = gridspec.GridSpec(nrows=200, ncols=200)
        ax1 = plt.subplot(gs[30:200,10:170])
        ax2 = plt.subplot(gs[0:30,10:170])
        ax3 = plt.subplot(gs[30:200,170:200])
        ad  = [0.215,0.83,0.10,0.90]
        myax_set(ax1, None, xlim, ylim, None, xlabel, ylabel, adjust=ad)
        myax_set(ax2, None, xlim, None, None, None, None)
        myax_set(ax3, None, None, ylim, None, None, None)

        # ax2 ticks
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.tick_params('x', length=0, which='major')
        ax2.tick_params('y', length=0, which='major')
        ax2.set_xticks([])
        ax2.set_yticks([])

        # ax3 ticks
        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.spines['left'].set_visible(False)
        ax3.tick_params('x', length=0, which='major')
        ax3.tick_params('y', length=0, which='major')
        ax3.set_xticks([])
        ax3.set_yticks([])

        # co10
        X, Y, Z = density_estimation(x_co10, y_co10, xlim, ylim)
        ax1.contour(X, Y, Z, colors="blue", linewidths=[2], alpha=0.2)

        # ci10
        X, Y, Z = density_estimation(x_ci10, y_ci10, xlim, ylim)
        ax1.contour(X, Y, Z, colors="red", linewidths=[2], alpha=0.2)

        # scatterhist
        self._scatter_hist(x_co10, y_co10, ax1, ax2, ax3, "deepskyblue", xlim, ylim, "s")
        self._scatter_hist(x_ci10, y_ci10, ax1, ax2, ax3, "tomato", xlim, ylim, offset=0.15)

        # scatter for outflow data
        #ax1.scatter(x_co10_cone, y_co10_cone, c="deepskyblue", lw=2, s=100, marker="s")
        ax1.scatter(x_ci10_cone, y_ci10_cone, c="tomato", lw=2, s=100)

        # text
        txt = ax1.text(0.03, 0.93, "CO(1-0) Clouds", color="deepskyblue", transform=ax1.transAxes, weight="bold", fontsize=24)
        txt.set_path_effects([PathEffects.withStroke(linewidth=4, foreground='w')])
        txt = ax1.text(0.03, 0.88, "[CI](1-0) Clouds", color="tomato", transform=ax1.transAxes, weight="bold", fontsize=24)
        txt.set_path_effects([PathEffects.withStroke(linewidth=4, foreground='w')])
        txt = ax1.text(0.03, 0.83, "Clouds with peak S/N > 5", color="black", transform=ax1.transAxes, fontsize=16)
        txt.set_path_effects([PathEffects.withStroke(linewidth=4, foreground='w')])

        # line
        ax1.plot(xlim, [np.log10(2.39),np.log10(2.39)], color='grey', alpha=.5, lw=1, zorder=0)
        ax1.plot([np.log10(55),np.log10(55)], ylim, color='grey', alpha=.5, lw=1, zorder=0)

        # const surface density lines (e.g., Leroy+15)
        G = 6.674*10**-11
        m2km = 10**-3
        kg2msun = 5.02785e-31
        pc2kms = 3.086*10**13
        Cnst = np.log10((np.pi*G/5.)**0.5*m2km**1.5*kg2msun**-0.5*pc2kms**-1*pc2kms**0.5)

        y_285   = [0.5*np.log10(285*1.)+0.5*xlim[0]+Cnst, 0.5*np.log10(285*1.)+0.5*xlim[1]+Cnst]
        y_285x5 = [0.5*np.log10(285*5.)+0.5*xlim[0]+Cnst, 0.5*np.log10(285*5.)+0.5*xlim[1]+Cnst]
        y_285w5 = [0.5*np.log10(285/5.)+0.5*xlim[0]+Cnst, 0.5*np.log10(285/5.)+0.5*xlim[1]+Cnst]

        ax1.plot(xlim, y_285x5, linestyle='dashed', color='grey', alpha=.5, lw=1, zorder=0)
        ax1.plot(xlim, y_285, linestyle='dashed', color='black', alpha=.5, lw=2, zorder=0)
        ax1.plot(xlim, y_285w5, linestyle='dashed', color='grey', alpha=.5, lw=1, zorder=0)

        # save
        os.system("rm -rf " + self.outpng_cico_larson_1st)
        plt.savefig(self.outpng_cico_larson_1st, dpi=self.fig_dpi)

        #########################
        # plot: dynamical state #
        #########################

        xlim   = [0.5,5] # self.xlim_larson_1st
        ylim   = [-1,2] # self.ylim_larson_1st
        title  = "Dynamical state"
        xlabel = "log$_{10}$ $\Sigma_{H_2}$ ($M_{\odot}$ pc$^{-2}$)"
        ylabel = "log$_{10}$ $\sigma^2$/$r$"
        alpha  = 1.0
        size   = 30

        # plot
        fig = plt.figure(figsize=(10,10))
        gs  = gridspec.GridSpec(nrows=200, ncols=200)
        ax1 = plt.subplot(gs[30:200,10:170])
        ax2 = plt.subplot(gs[0:30,10:170])
        ax3 = plt.subplot(gs[30:200,170:200])
        ad  = [0.215,0.83,0.10,0.90]
        myax_set(ax1, None, xlim, ylim, None, xlabel, ylabel, adjust=ad)
        myax_set(ax2, None, xlim, None, None, None, None)
        myax_set(ax3, None, None, ylim, None, None, None)

        # ax2 ticks
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.tick_params('x', length=0, which='major')
        ax2.tick_params('y', length=0, which='major')
        ax2.set_xticks([])
        ax2.set_yticks([])

        # ax3 ticks
        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.spines['left'].set_visible(False)
        ax3.tick_params('x', length=0, which='major')
        ax3.tick_params('y', length=0, which='major')
        ax3.set_xticks([])
        ax3.set_yticks([])

        # co10
        X, Y, Z = density_estimation(x2_co10, y2_co10, xlim, ylim)
        ax1.contour(X, Y, Z, colors="blue", linewidths=[2], alpha=0.2)

        # ci10
        X, Y, Z = density_estimation(x2_ci10, y2_ci10, xlim, ylim)
        ax1.contour(X, Y, Z, colors="red", linewidths=[2], alpha=0.2)

        # scatterhist
        self._scatter_hist(x2_co10, y2_co10, ax1, ax2, ax3, "deepskyblue", xlim, ylim, "s")
        self._scatter_hist(x2_ci10, y2_ci10, ax1, ax2, ax3, "tomato", xlim, ylim, offset=0.15)

        # scatter for outflow data
        #ax1.scatter(x2_co10_cone, y2_co10_cone, c="deepskyblue", lw=2, s=100, marker="s")
        ax1.scatter(x2_ci10_cone, y2_ci10_cone, c="tomato", lw=2, s=100)

        # text
        txt = ax1.text(0.03, 0.93, "CO(1-0) Clouds", color="deepskyblue", transform=ax1.transAxes, weight="bold", fontsize=24)
        txt.set_path_effects([PathEffects.withStroke(linewidth=4, foreground='w')])
        txt = ax1.text(0.03, 0.88, "[CI](1-0) Clouds", color="tomato", transform=ax1.transAxes, weight="bold", fontsize=24)
        txt.set_path_effects([PathEffects.withStroke(linewidth=4, foreground='w')])
        txt = ax1.text(0.03, 0.83, "Clouds with peak S/N > 5", color="black", transform=ax1.transAxes, fontsize=16)
        txt.set_path_effects([PathEffects.withStroke(linewidth=4, foreground='w')])

        # save
        os.system("rm -rf " + self.outpng_cico_dyn)
        plt.savefig(self.outpng_cico_dyn, dpi=self.fig_dpi)

    ###############
    # hist_cprops #
    ###############

    def hist_cprops(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.cprops_co10,taskname)

        # import cprops table
        f = pyfits.open(self.cprops_co10)
        tb = f[1].data

        # extract parameters
        s2n_co10    = tb["S2N"]
        radius_co10 = tb["RAD_PC"] # tb["RAD_NODC_NOEX"]
        sigv_co10   = tb["SIGV_KMS"] # tb["SIGV_NODC_NOEX"]
        mvir_co10   = tb["MVIR_MSUN"]
        tpeak_co10  = tb["TMAX_K"]

        # import cprops table
        f = pyfits.open(self.cprops_ci10)
        tb = f[1].data

        # extract parameters
        s2n_ci10    = tb["S2N"]
        radius_ci10 = tb["RAD_PC"]
        sigv_ci10   = tb["SIGV_KMS"]
        mvir_ci10   = tb["MVIR_MSUN"]
        tpeak_ci10  = tb["TMAX_K"]

        ########
        # plot #
        ########

        # SNR dist
        self._plot_hist_cprops(
            xlim      = [5,105],
            ylim      = None,
            title     = "Cloud SNR",
            xlabel    = "SNR",
            ylabel    = "Count density",
            outpng    = self.outpng_hist_snr,
            data_co10 = s2n_co10,
            data_ci10 = s2n_ci10,
            s2n_co10  = s2n_co10,
            s2n_ci10  = s2n_ci10,
            )

        # velocity dispersion dist
        self._plot_hist_cprops(
            xlim      = [0,80],
            ylim      = None,
            title     = "Cloud Dispersion",
            xlabel    = "Velocity Dispersion (km s$^{-1}$)",
            ylabel    = "Count density",
            outpng    = self.outpng_hist_sigv,
            data_co10 = sigv_co10,
            data_ci10 = sigv_ci10,
            s2n_co10  = s2n_co10,
            s2n_ci10  = s2n_ci10,
            )

        # radius dist
        self._plot_hist_cprops(
            xlim      = [0,300],
            ylim      = None,
            title     = "Cloud Radius",
            xlabel    = "Radius (pc)",
            ylabel    = "Count density",
            outpng    = self.outpng_hist_rad,
            data_co10 = radius_co10,
            data_ci10 = radius_ci10,
            s2n_co10  = s2n_co10,
            s2n_ci10  = s2n_ci10,
            )

        # tpeak dist
        self._plot_hist_cprops(
            xlim      = [0,30],
            ylim      = None,
            title     = "Cloud Peak Temperature",
            xlabel    = "Peak Temperature (K)",
            ylabel    = "Count density",
            outpng    = self.outpng_hist_tpeak,
            data_co10 = tpeak_co10,
            data_ci10 = tpeak_ci10,
            s2n_co10  = s2n_co10,
            s2n_ci10  = s2n_ci10,
            )

    #####################
    # _plot_hist_cprops #
    #####################

    def _plot_hist_cprops(
        self,
        xlim,
        ylim,
        title,
        xlabel,
        ylabel,
        outpng,
        data_co10,
        data_ci10,
        s2n_co10,
        s2n_ci10,
        ):
        """
        """

        # import co10
        this_co10 = data_co10[s2n_co10>self.snr_cprops]
        h = np.histogram(this_co10, bins=20, range=xlim)
        x_co10, y_co10 = h[1][:-1], h[0]/float(np.sum(h[0]))

        # import ci10
        this_ci10 = data_ci10[s2n_ci10>self.snr_cprops]
        h = np.histogram(this_ci10, bins=20, range=xlim)
        x_ci10, y_ci10 = h[1][:-1], h[0]/float(np.sum(h[0]))

        # plot
        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        ad  = [0.215,0.83,0.10,0.90]
        myax_set(ax1, "x", xlim, ylim, title, xlabel, ylabel, adjust=ad)

        ax1.bar(x_co10, y_co10, lw=0, color="deepskyblue", width=x_co10[1]-x_co10[0], alpha=0.5)
        ax1.bar(x_ci10, y_ci10, lw=0, color="tomato", width=x_ci10[1]-x_ci10[0], alpha=0.5)

        ax1.text(0.95, 0.93, "CO(1-0)", color="deepskyblue", horizontalalignment="right", transform=ax1.transAxes, size=self.legend_fontsize, fontweight="bold")
        ax1.text(0.95, 0.88, "[CI](1-0)", color="tomato", horizontalalignment="right", transform=ax1.transAxes, size=self.legend_fontsize, fontweight="bold")

        # save
        os.system("rm -rf " + outpng)
        plt.savefig(outpng, dpi=self.fig_dpi)

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

        ###########
        # CO(1-0) #
        ###########

        ###########
        # prepare #
        ###########

        data,_  = imval_all(self.cube_co10.replace(".fits","_aligned.fits"))
        data    = data["data"].flatten()
        ndata,_ = imval_all(self.ncube_co10.replace(".fits","_aligned.fits"))
        ndata   = ndata["data"].flatten()
        data_co10  = data[data/ndata>-10000]

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

        ###########
        # CI(1-0) #
        ###########

        ###########
        # prepare #
        ###########

        data,_  = imval_all(self.cube_ci10)
        data    = data["data"].flatten()
        ndata,_ = imval_all(self.ncube_ci10)
        ndata   = ndata["data"].flatten()
        data_ci10  = data[data/ndata>-10000]

        histx, histy, histrange, peak, rms, x_bestfit, y_bestfit, _ = self._gaussfit_noise(data_ci10,bins=1000)

        xlim     = [0, 10*rms]
        ylim     = [0, np.max(histy)*1.05]
        title    = "[CI](1-0) Cube"
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

        plt.savefig(self.outpng_hist_ci10_pix, dpi=self.fig_dpi)

        ###########
        # prepare #
        ###########

        snr_ci10 = data[data/ndata>-10000] / ndata[data/ndata>-10000]

        histx, histy, histrange, peak, rms, x_bestfit, y_bestfit, _ = self._gaussfit_noise(snr_ci10,bins=1000)

        xlim     = [0, 10*rms]
        ylim     = [0, np.max(histy)*1.05]
        title    = "[CI](1-0) SNR Cube"
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

        plt.savefig(self.outpng_hist_ci10_snr, dpi=self.fig_dpi)

        ###################
        # CI(1-0)/CO(1-0) #
        ###################

        ###########
        # prepare #
        ###########

        data,_  = imval_all(self.cube_co10.replace(".fits","_aligned.fits"))
        data    = data["data"].flatten()
        ndata,_ = imval_all(self.ncube_co10.replace(".fits","_aligned.fits"))
        ndata   = ndata["data"].flatten()

        data2,_  = imval_all(self.cube_ci10)
        data2    = data2["data"].flatten()
        ndata2,_ = imval_all(self.ncube_ci10)
        ndata2   = ndata2["data"].flatten()

        cut        = np.where((data>-100000) & (ndata>-100000) & (data2>-100000) & (ndata2>-100000) & (data2>-30*data) & (data2<30*data))
        data_co10  = data[cut]
        ndata_co10 = ndata[cut]
        data_ci10  = data2[cut]
        ndata_ci10 = ndata2[cut]

        data_ratio = data_ci10 / data_co10

        histx, histy, histrange, peak, rms, x_bestfit, y_bestfit, _ = self._gaussfit_noise(data_ratio,bins=5000)

        xlim     = [0, 10*rms]
        ylim     = [0, np.max(histy)*1.05]
        title    = "Ratio Cube"
        xlabel   = "Absolute voxel value"
        ylabel   = "Count"
        binwidth = (histrange[1]-histrange[0]) / 5000.
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

        plt.savefig(self.outpng_hist_ratio_pix, dpi=self.fig_dpi)

        ###########
        # prepare #
        ###########

        ndata_ratio = np.abs(data_ci10 / data_co10 * np.sqrt((ndata_ci10/data_ci10)**2 + (ndata_co10/data_co10)**2))
        snr_ratio   = data_ratio / ndata_ratio

        histx, histy, histrange, peak, rms, x_bestfit, y_bestfit, _ = self._gaussfit_noise(snr_ratio,bins=500)

        xlim     = [0, 10*rms]
        ylim     = [0, np.max(histy)*1.05]
        title    = "Ratio SNR Cube"
        xlabel   = "Absolute voxel SNR"
        ylabel   = "Count"
        binwidth = (histrange[1]-histrange[0]) / 500.
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

        plt.savefig(self.outpng_hist_ratio_snr, dpi=self.fig_dpi)

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
            textann   = False,
            txtfiles  = [this_tb1,this_tb2],
            set_title = linename + " Cloud Catalog",
            scalebar  = scalebar,
            label_scalebar = label_scalebar,
            colorlog  = True,
            set_cmap  = "Greys",
            set_grid  = None,
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
            set_grid  = None,
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
            set_grid  = None,
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
    #########################
    # will be decomissioned #
    #########################
    #########################

    ###############
    # plot_radial #
    ###############

    def plot_radial(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outtxt_hexcat_ci10,taskname)

        # import co10
        data_co10 = np.loadtxt(self.outtxt_hexcat_co10)

        x_co10     = data_co10[:,0]
        y_co10     = data_co10[:,1]
        mom0_co10  = data_co10[:,2]
        emom0_co10 = data_co10[:,3]
        mom2_co10  = data_co10[:,4]
        emom2_co10 = data_co10[:,5]

        r_co10     = np.sqrt(x_co10**2 + y_co10**2)
        theta      = np.degrees(np.arctan2(x_co10, y_co10)) + 90
        theta_co10 = np.where(theta>0, theta, theta+360)

        cut = np.where((mom0_co10>emom0_co10*self.snr_mom) & (mom2_co10>emom2_co10))
        x_co10_all     = x_co10[cut]
        y_co10_all     = y_co10[cut]
        emom0_co10_all = emom0_co10[cut] / mom0_co10[cut] / np.log(10)
        mom0_co10_all  = np.log10(mom0_co10[cut])
        emom2_co10_all = emom2_co10[cut] / mom2_co10[cut] / np.log(10)
        mom2_co10_all  = np.log10(mom2_co10[cut])
        r_co10_all     = r_co10[cut]

        cut = np.where((mom0_co10>emom0_co10*self.snr_mom) & (mom2_co10>emom2_co10) & (r_co10<self.fov_diamter/2.0) & (r_co10>self.r_cnd_as) & (theta_co10>=self.theta2) & (theta_co10<self.theta1) | (mom0_co10>emom0_co10*self.snr_mom) & (mom2_co10>emom2_co10*self.snr_mom) & (r_co10<self.fov_diamter/2.0) & (r_co10>self.r_cnd_as) & (theta_co10>=self.theta2+180) & (theta_co10<self.theta1+180))
        x_co10_cone     = x_co10[cut]
        y_co10_cone     = y_co10[cut]
        emom0_co10_cone = emom0_co10[cut] / mom0_co10[cut] / np.log(10)
        mom0_co10_cone  = np.log10(mom0_co10[cut])
        emom2_co10_cone = emom2_co10[cut] / mom2_co10[cut] / np.log(10)
        mom2_co10_cone  = np.log10(mom2_co10[cut])
        r_co10_cone     = np.sqrt(x_co10_cone**2 + y_co10_cone**2)

        # import ci10
        data_ci10 = np.loadtxt(self.outtxt_hexcat_ci10)

        x_ci10     = data_ci10[:,0]
        y_ci10     = data_ci10[:,1]
        mom0_ci10  = data_ci10[:,2]
        emom0_ci10 = data_ci10[:,3]
        mom2_ci10  = data_ci10[:,4]
        emom2_ci10 = data_ci10[:,5]

        r_ci10     = np.sqrt(x_ci10**2 + y_ci10**2)
        theta      = np.degrees(np.arctan2(x_ci10, y_ci10)) + 90
        theta_ci10 = np.where(theta>0, theta, theta+360)

        cut = np.where((mom0_ci10>emom0_ci10*self.snr_mom) & (mom2_ci10>emom2_ci10))
        x_ci10_all     = x_ci10[cut]
        y_ci10_all     = y_ci10[cut]
        emom0_ci10_all = emom0_ci10[cut] / mom0_ci10[cut] / np.log(10)
        mom0_ci10_all  = np.log10(mom0_ci10[cut])
        emom2_ci10_all = emom2_ci10[cut] / mom2_ci10[cut] / np.log(10)
        mom2_ci10_all  = np.log10(mom2_ci10[cut])
        r_ci10_all     = r_ci10[cut]

        cut = np.where((mom0_ci10>emom0_ci10*self.snr_mom) & (mom2_ci10>emom2_ci10) & (r_ci10<self.fov_diamter/2.0) & (r_ci10>self.r_cnd_as) & (theta_ci10>=self.theta2) & (theta_ci10<self.theta1) | (mom0_ci10>emom0_ci10*self.snr_mom) & (mom2_ci10>emom2_ci10*self.snr_mom) & (r_ci10<self.fov_diamter/2.0) & (r_ci10>self.r_cnd_as) & (theta_ci10>=self.theta2+180) & (theta_ci10<self.theta1+180))
        x_ci10_cone     = x_ci10[cut]
        y_ci10_cone     = y_ci10[cut]
        emom0_ci10_cone = emom0_ci10[cut] / mom0_ci10[cut] / np.log(10)
        mom0_ci10_cone  = np.log10(mom0_ci10[cut])
        emom2_ci10_cone = emom2_ci10[cut] / mom2_ci10[cut] / np.log(10)
        mom2_ci10_cone  = np.log10(mom2_ci10[cut])
        r_ci10_cone     = np.sqrt(x_ci10_cone**2 + y_ci10_cone**2)

        ########
        # plot #
        ########
        x_co10 = r_co10_all
        y_co10 = mom2_co10_all
        x_ci10 = r_ci10_all
        y_ci10 = mom2_ci10_all
        x2_co10 = r_co10_cone
        y2_co10 = mom2_co10_cone
        x2_ci10 = r_ci10_cone
        y2_ci10 = mom2_ci10_cone

        xlim   = [0,np.max([np.nanmax(x_co10),np.nanmax(x_ci10)])+0.3]
        ylim   = [np.min([np.nanmin(y_co10),np.nanmin(y_ci10)])-0.3,np.max([np.nanmax(y_co10),np.nanmax(y_ci10)])+0.3]
        title  = "None"
        xlabel = "Distance from center (pc)"
        ylabel = "log$_{10}$ Velocity Dispersion (km s$^{-1}$)" # "log$_{10}$ Integrated Intensity (K km s$^{-1}$)"
        alpha  = 1.0
        size   = 30

        # plot
        fig = plt.figure(figsize=(10,10))
        gs  = gridspec.GridSpec(nrows=200, ncols=200)
        ax1 = plt.subplot(gs[30:200,10:170])
        ax2 = plt.subplot(gs[0:30,10:170])
        ax3 = plt.subplot(gs[30:200,170:200])
        ad  = [0.215,0.83,0.10,0.90]
        myax_set(ax1, None, xlim, ylim, None, xlabel, ylabel, adjust=ad)
        myax_set(ax2, None, xlim, None, None, None, None)
        myax_set(ax3, None, None, ylim, None, None, None)

        # ax2 ticks
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.tick_params('x', length=0, which='major')
        ax2.tick_params('y', length=0, which='major')
        ax2.set_xticks([])
        ax2.set_yticks([])

        # ax3 ticks
        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.spines['left'].set_visible(False)
        ax3.tick_params('x', length=0, which='major')
        ax3.tick_params('y', length=0, which='major')
        ax3.set_xticks([])
        ax3.set_yticks([])

        # plot co10 all
        self._scatter_hist(x_co10, y_co10, ax1, ax2, ax3, "deepskyblue", xlim, ylim, "s")

        # plot ci10 all
        self._scatter_hist(x_ci10, y_ci10, ax1, ax2, ax3, "tomato", xlim, ylim, "o", offset=0.15)

        # plot co10 cone
        ax1.scatter(x2_co10, y2_co10, facecolor='lightgrey', edgecolor='blue', lw=2, s=70, marker="s", alpha=1.0, zorder=1e9)

        # plot ci10 cone
        ax1.scatter(x2_ci10, y2_ci10, facecolor='lightgrey', edgecolor='red', lw=2, s=70, marker="o", alpha=1.0, zorder=1e9)

        # pctl bar xaxis
        ypos = 0.97 * (ylim[1] - ylim[0]) + ylim[0]
        ax1.plot([np.percentile(x_ci10,16),np.percentile(x_ci10,84)], [ypos,ypos], '-', color="tomato", lw=2)
        ax1.scatter(np.percentile(x_ci10,50), ypos, marker='o', s=100, color="tomato", zorder=1e9)

        ypos = 0.95 * (ylim[1] - ylim[0]) + ylim[0]
        ax1.plot([np.percentile(x_co10,16),np.percentile(x_co10,84)], [ypos,ypos], '-', color="deepskyblue", lw=2)
        ax1.scatter(np.percentile(x_co10,50), ypos, marker='o', s=100, color="deepskyblue", zorder=1e9)

        ypos = 0.93 * (ylim[1] - ylim[0]) + ylim[0]
        ax1.plot([np.percentile(x2_ci10,16),np.percentile(x2_ci10,84)], [ypos,ypos], '-', color="red", lw=2)
        ax1.scatter(np.percentile(x2_ci10,50), ypos, marker='o', s=100, facecolor='lightgrey', edgecolor='red', lw=2, zorder=1e9)

        ypos = 0.91 * (ylim[1] - ylim[0]) + ylim[0]
        ax1.plot([np.percentile(x2_co10,16),np.percentile(x2_co10,84)], [ypos,ypos], '-', color="blue", lw=2)
        ax1.scatter(np.percentile(x2_co10,50), ypos, marker='o', s=100, facecolor='lightgrey', edgecolor='blue', lw=2, zorder=1e9)

        # pctl bar yaxis
        xpos = 0.97 * (xlim[1] - xlim[0]) + xlim[0]
        ax1.plot([xpos,xpos], [np.percentile(y_ci10,16),np.percentile(y_ci10,84)], '-', color="tomato", lw=2)
        ax1.scatter(xpos, np.percentile(y_ci10,50), marker='o', s=100, color="tomato", zorder=1e9)

        xpos = 0.95 * (xlim[1] - xlim[0]) + xlim[0]
        ax1.plot([xpos,xpos], [np.percentile(y_co10,16),np.percentile(y_co10,84)], '-', color="deepskyblue", lw=2)
        ax1.scatter(xpos, np.percentile(y_co10,50), marker='o', s=100, color="deepskyblue", zorder=1e9)

        xpos = 0.93 * (xlim[1] - xlim[0]) + xlim[0]
        ax1.plot([xpos,xpos], [np.percentile(y2_ci10,16),np.percentile(y2_ci10,84)], '-', color="red", lw=2)
        ax1.scatter(xpos, np.percentile(y2_ci10,50), marker='o', s=100, facecolor='lightgrey', edgecolor='red', lw=2, zorder=1e9)

        xpos = 0.91 * (xlim[1] - xlim[0]) + xlim[0]
        ax1.plot([xpos,xpos], [np.percentile(y2_co10,16),np.percentile(y2_co10,84)], '-', color="blue", lw=2)
        ax1.scatter(xpos, np.percentile(y2_co10,50), marker='o', s=100, facecolor='lightgrey', edgecolor='blue', lw=2, zorder=1e9)

        # save
        os.system("rm -rf " + self.outpng_radial_disp)
        plt.savefig(self.outpng_radial_disp, dpi=self.fig_dpi)

        ########
        # plot #
        ########
        x_co10 = r_co10_all
        y_co10 = mom0_co10_all
        x_ci10 = r_ci10_all
        y_ci10 = mom0_ci10_all
        x2_co10 = r_co10_cone
        y2_co10 = mom0_co10_cone
        x2_ci10 = r_ci10_cone
        y2_ci10 = mom0_ci10_cone

        xlim   = [0,np.max([np.nanmax(x_co10),np.nanmax(x_ci10)])+0.3]
        ylim   = [np.min([np.nanmin(y_co10),np.nanmin(y_ci10)])-0.3,np.max([np.nanmax(y_co10),np.nanmax(y_ci10)])+0.3]
        title  = "None"
        xlabel = "Distance from center (pc)"
        ylabel = "log$_{10}$ Integrated Intensity (K km s$^{-1}$)"
        alpha  = 1.0
        size   = 30

        # plot
        fig = plt.figure(figsize=(10,10))
        gs  = gridspec.GridSpec(nrows=200, ncols=200)
        ax1 = plt.subplot(gs[30:200,10:170])
        ax2 = plt.subplot(gs[0:30,10:170])
        ax3 = plt.subplot(gs[30:200,170:200])
        ad  = [0.215,0.83,0.10,0.90]
        myax_set(ax1, None, xlim, ylim, None, xlabel, ylabel, adjust=ad)
        myax_set(ax2, None, xlim, None, None, None, None)
        myax_set(ax3, None, None, ylim, None, None, None)

        # ax2 ticks
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.tick_params('x', length=0, which='major')
        ax2.tick_params('y', length=0, which='major')
        ax2.set_xticks([])
        ax2.set_yticks([])

        # ax3 ticks
        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.spines['left'].set_visible(False)
        ax3.tick_params('x', length=0, which='major')
        ax3.tick_params('y', length=0, which='major')
        ax3.set_xticks([])
        ax3.set_yticks([])

        # plot co10 all
        self._scatter_hist(x_co10, y_co10, ax1, ax2, ax3, "deepskyblue", xlim, ylim, "s")

        # plot ci10 all
        self._scatter_hist(x_ci10, y_ci10, ax1, ax2, ax3, "tomato", xlim, ylim, "o", offset=0.15)

        # plot co10 cone
        ax1.scatter(x2_co10, y2_co10, facecolor='lightgrey', edgecolor='blue', lw=2, s=70, marker="s", alpha=1.0, zorder=1e9)

        # plot ci10 cone
        ax1.scatter(x2_ci10, y2_ci10, facecolor='lightgrey', edgecolor='red', lw=2, s=70, marker="o", alpha=1.0, zorder=1e9)

        # pctl bar xaxis
        ypos = 0.97 * (ylim[1] - ylim[0]) + ylim[0]
        ax1.plot([np.percentile(x_ci10,16),np.percentile(x_ci10,84)], [ypos,ypos], '-', color="tomato", lw=2)
        ax1.scatter(np.percentile(x_ci10,50), ypos, marker='o', s=100, color="tomato", zorder=1e9)

        ypos = 0.95 * (ylim[1] - ylim[0]) + ylim[0]
        ax1.plot([np.percentile(x_co10,16),np.percentile(x_co10,84)], [ypos,ypos], '-', color="deepskyblue", lw=2)
        ax1.scatter(np.percentile(x_co10,50), ypos, marker='o', s=100, color="deepskyblue", zorder=1e9)

        ypos = 0.93 * (ylim[1] - ylim[0]) + ylim[0]
        ax1.plot([np.percentile(x2_ci10,16),np.percentile(x2_ci10,84)], [ypos,ypos], '-', color="red", lw=2)
        ax1.scatter(np.percentile(x2_ci10,50), ypos, marker='o', s=100, facecolor='lightgrey', edgecolor='red', lw=2, zorder=1e9)

        ypos = 0.91 * (ylim[1] - ylim[0]) + ylim[0]
        ax1.plot([np.percentile(x2_co10,16),np.percentile(x2_co10,84)], [ypos,ypos], '-', color="blue", lw=2)
        ax1.scatter(np.percentile(x2_co10,50), ypos, marker='o', s=100, facecolor='lightgrey', edgecolor='blue', lw=2, zorder=1e9)

        # pctl bar yaxis
        xpos = 0.97 * (xlim[1] - xlim[0]) + xlim[0]
        ax1.plot([xpos,xpos], [np.percentile(y_ci10,16),np.percentile(y_ci10,84)], '-', color="tomato", lw=2)
        ax1.scatter(xpos, np.percentile(y_ci10,50), marker='o', s=100, color="tomato", zorder=1e9)

        xpos = 0.95 * (xlim[1] - xlim[0]) + xlim[0]
        ax1.plot([xpos,xpos], [np.percentile(y_co10,16),np.percentile(y_co10,84)], '-', color="deepskyblue", lw=2)
        ax1.scatter(xpos, np.percentile(y_co10,50), marker='o', s=100, color="deepskyblue", zorder=1e9)

        xpos = 0.93 * (xlim[1] - xlim[0]) + xlim[0]
        ax1.plot([xpos,xpos], [np.percentile(y2_ci10,16),np.percentile(y2_ci10,84)], '-', color="red", lw=2)
        ax1.scatter(xpos, np.percentile(y2_ci10,50), marker='o', s=100, facecolor='lightgrey', edgecolor='red', lw=2, zorder=1e9)

        xpos = 0.91 * (xlim[1] - xlim[0]) + xlim[0]
        ax1.plot([xpos,xpos], [np.percentile(y2_co10,16),np.percentile(y2_co10,84)], '-', color="blue", lw=2)
        ax1.scatter(xpos, np.percentile(y2_co10,50), marker='o', s=100, facecolor='lightgrey', edgecolor='blue', lw=2, zorder=1e9)

        # save
        os.system("rm -rf " + self.outpng_radial_mom0)
        plt.savefig(self.outpng_radial_mom0, dpi=self.fig_dpi)

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
        v      = tb["VCTR_KMS"]
        s2n    = tb["S2N"]
        radius = tb["RAD_PC"] # tb["RAD_NODC_NOEX"]
        sigv   = tb["SIGV_KMS"] # tb["SIGV_NODC_NOEX"]
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
        s2n_cone    = s2n[cut_cone]
        cone_list = [x_cone, y_cone, v_cone, radius_cone, sigv_cone, mvir_cone, tpeak_cone, mci_cone, xdeg_cone, ydeg_cone, s2n_cone]

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
        s2n_nocone    = s2n[cut_nocone]
        nocone_list = [x_nocone, y_nocone, v_nocone, radius_nocone, sigv_nocone, mvir_nocone, tpeak_nocone, mci_nocone, xdeg_nocone, ydeg_nocone, s2n_nocone]

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
        s2n_sbr    = s2n[cut_sbr]
        sbr_list = [x_sbr, y_sbr, v_sbr, radius_sbr, sigv_sbr, mvir_sbr, tpeak_sbr, mci_sbr, xdeg_sbr, ydeg_sbr, s2n_sbr]

        return cone_list, nocone_list, sbr_list

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

#####################
# end of ToolsCIGMC #
#####################
