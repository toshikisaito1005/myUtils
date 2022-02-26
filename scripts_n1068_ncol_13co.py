"""
Python class for the NGC 1068 13CO-based Ncol project

requirements:
CASA Version 5.4.0-70, ananlysisUtils, astropy

data:
ALMA Band 3 data 2011.0.00061.S
                 2012.1.00657.S
                 2013.1.00060.S
                 2018.1.01506.S
                 2018.1.01684.S
                 2019.1.00130.S
ALMA Band 6 data 2013.1.00221.S
                 2019.2.00129.S (TP and 7m not used)
imaging script   all processed by phangs pipeline v2
                 Leroy et al. 2021, ApJS, 255, 19 (https://ui.adsabs.harvard.edu/abs/2021ApJS..255...19L)

usage:
> import os
> from scripts_n1068_ncol_13co import ToolsNcol as tools
>
> # key
> tl = tools(
>     refresh     = False,
>     keyfile_gal = "/home02/saitots/myUtils/keys_n1068_ncol_13co/key_ngc1068.txt",
>     keyfile_fig = "/home02/saitots/myUtils/keys_n1068_ncol_13co/key_figures.txt",
>     )
>
> # main
> tl.run_ngc1068_ncol(
>     # analysis
>     do_prepare     = False,
>     do_fitting     = False,
>     # plot
>     plot_showcase  = True,
>     do_imagemagick = True,
>     immagick_all   = False,
>     # supplement
>     )
>
> os.system("rm -rf *.last")

paper drafts:
Date         Filename                To

history:
2022-01-30   created
2022-02-04   add simualtion and evaluation on the methods for moment map creation
Toshiki Saito@Nichidai/NAOJ
"""

import os, sys, glob
import numpy as np
from scipy.stats import gaussian_kde

from mycasa_rotation import *
from mycasa_sampling import *
from mycasa_lowess import *
from mycasa_tasks import *
from mycasa_plots import *
from mycasa_pca import *

#############
# ToolsNcol #
#############
class ToolsNcol():
    """
    Class for the NGC 1068 Ncol project.
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
            self.modname = "ToolsNcol."
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

        self.cube_13co10  = self.dir_raw + self._read_key("cube_13co10")
        self.ecube_13co10 = self.dir_raw + self._read_key("ecube_13co10")
        self.cube_13co21  = self.dir_raw + self._read_key("cube_13co21")
        self.ecube_13co21 = self.dir_raw + self._read_key("ecube_13co21")

        self.mom0_12co10  = self.dir_raw + self._read_key("mom0_12co10")
        self.emom0_12co10 = self.dir_raw + self._read_key("emom0_12co10")

        self.vla          = self.dir_other + self._read_key("vla")
        
        """
        self.cube_hcn10   = self.dir_raw + self._read_key("cube_hcn10")
        self.ecube_hcn10  = self.dir_raw + self._read_key("ecube_hcn10")
        self.cube_hcop10  = self.dir_raw + self._read_key("cube_hcop10")
        self.ecube_hcop10 = self.dir_raw + self._read_key("ecube_hcop10")
        """

    def _set_output_fits(self):
        """
        """

        self.outcubes_13co10     = self.dir_ready + self._read_key("outcubes_13co10")
        self.outecubes_13co10    = self.dir_ready + self._read_key("outecubes_13co10")
        self.outcubes_13co21     = self.dir_ready + self._read_key("outcubes_13co21")
        self.outecubes_13co21    = self.dir_ready + self._read_key("outecubes_13co21")

        self.outmaps_12co10      = self.dir_ready + self._read_key("outmaps_12co10")
        self.outemaps_12co10     = self.dir_ready + self._read_key("outemaps_12co10")

        self.outmaps_mom0_13co10 = self.dir_ready + self._read_key("outmaps_13co10")
        self.outmaps_mom0_13co21 = self.dir_ready + self._read_key("outmaps_13co21")
        self.outmaps_mom1        = self.dir_ready + self._read_key("outmaps_mom1")
        self.outmaps_mom2        = self.dir_ready + self._read_key("outmaps_mom2")
        self.outmaps_ratio       = self.dir_ready + self._read_key("outmaps_ratio")
        self.outmaps_13co_trot   = self.dir_ready + self._read_key("outmaps_13co_trot")
        self.outmaps_13co_ncol   = self.dir_ready + self._read_key("outmaps_13co_ncol")
        self.outmaps_residual    = self.dir_ready + self._read_key("outmaps_residual")

        self.outemaps_mom0_13co10 = self.dir_ready + self._read_key("outemaps_13co10")
        self.outemaps_mom0_13co21 = self.dir_ready + self._read_key("outemaps_13co21")
        self.outemaps_mom1        = self.dir_ready + self._read_key("outemaps_mom1")
        self.outemaps_mom2        = self.dir_ready + self._read_key("outemaps_mom2")
        self.outemaps_ratio       = self.dir_ready + self._read_key("outemaps_ratio")
        self.outemaps_13co_trot   = self.dir_ready + self._read_key("outemaps_13co_trot")
        self.outemaps_13co_ncol   = self.dir_ready + self._read_key("outemaps_13co_ncol")

        self.outmaps_aco          = self.dir_ready + self._read_key("outmaps_aco")
        self.outemaps_aco         = self.dir_ready + self._read_key("outemaps_aco")
        self.outmaps_vla          = self.dir_ready + self._read_key("outmaps_vla")
        self.outmaps_pturb        = self.dir_ready + self._read_key("outmaps_pturb")
        self.outmaps_avir         = self.dir_ready + self._read_key("outmaps_avir")

        self.outmodelcube_13co10  = self.dir_ready + self._read_key("outmodelcube_13co10")
        self.outmodelcube_13co21  = self.dir_ready + self._read_key("outmodelcube_13co21")
        self.outmodelmom0_13co10  = self.dir_ready + self._read_key("outmodelmom0_13co10")
        self.outmodelmom0_13co21  = self.dir_ready + self._read_key("outmodelmom0_13co21")
        self.outsimumom0_13co10   = self.dir_ready + self._read_key("outsimumom0_13co10")
        self.outsimumom0_13co21   = self.dir_ready + self._read_key("outsimumom0_13co21")

        """
        self.outcubes_hcn10      = self.dir_ready + self._read_key("outcubes_hcn10")
        self.outecubes_hcn10     = self.dir_ready + self._read_key("outecubes_hcn10")
        self.outcubes_hcop10     = self.dir_ready + self._read_key("outcubes_hcop10")
        self.outecubes_hcop10    = self.dir_ready + self._read_key("outecubes_hcop10")

        self.outmaps_mom0_hcn10  = self.dir_ready + self._read_key("outmaps_hcn10")
        self.outmaps_mom0_hcop10 = self.dir_ready + self._read_key("outmaps_hcop10")
        self.outmaps_hcn10_mom1  = self.dir_ready + self._read_key("outmaps_hcn10_mom1")
        self.outmaps_hcn10_mom2  = self.dir_ready + self._read_key("outmaps_hcn10_mom2")
        self.outmaps_hcn10_ratio = self.dir_ready + self._read_key("outmaps_hcn10_ratio")

        self.outemaps_mom0_hcn10  = self.dir_ready + self._read_key("outemaps_hcn10")
        self.outemaps_mom0_hcop10 = self.dir_ready + self._read_key("outemaps_hcop10")
        self.outemaps_hcn10_mom1  = self.dir_ready + self._read_key("outemaps_hcn10_mom1")
        self.outemaps_hcn10_mom2  = self.dir_ready + self._read_key("outemaps_hcn10_mom2")
        self.outemaps_hcn10_ratio = self.dir_ready + self._read_key("outemaps_hcn10_ratio")
        """

    def _set_input_param(self):
        """
        """

        self.imsize      = float(self._read_key("imsize_as"))
        self.beams       = ["60pc","70pc","80pc","90pc","100pc","110pc","120pc","130pc","140pc","150pc"]

        # ngc1068 properties
        self.ra_agn      = float(self._read_key("ra_agn", "gal").split("deg")[0])
        self.dec_agn     = float(self._read_key("dec_agn", "gal").split("deg")[0])
        self.ra_agn_str  = self._read_key("ra_agn", "gal")
        self.dec_agn_str = self._read_key("dec_agn", "gal")
        self.pa          = float(self._read_key("pa", "gal"))
        self.incl        = float(self._read_key("incl", "gal"))
        self.scale_pc    = float(self._read_key("scale", "gal"))
        self.scale_kpc   = self.scale_pc / 1000.

        self.snr         = float(self._read_key("snr"))
        self.snr_clip    = int(self._read_key("snr_clip"))
        self.snr_model   = float(self._read_key("snr_model"))
        self.snr_fit     = float(self._read_key("snr_fit"))
        self.r_cnd       = float(self._read_key("r_cnd_as")) * self.scale_pc / 1000. # kpc
        self.r_cnd_as    = float(self._read_key("r_cnd_as"))
        self.r_sbr       = float(self._read_key("r_sbr_as")) * self.scale_pc / 1000. # kpc
        self.r_sbr_as    = float(self._read_key("r_sbr_as"))

        self.abundance_12co_h2 = 1e-4   # Cormier et al. 2018
        self.abundance_12co_13co = 100 # starburst asspmtion? or 60.0 # Cormier et al. 2018
        self.abundance_13co_h2 = self.abundance_12co_h2 / self.abundance_12co_13co

    def _set_output_txt_png(self):
        """
        """

        # output txt and png
        self.outpng_mom0_13co10 = self.dir_products + self._read_key("outpng_mom0_13co10")
        self.outpng_mom0_13co21 = self.dir_products + self._read_key("outpng_mom0_13co21")
        self.outpng_mom1        = self.dir_products + self._read_key("outpng_mom1")
        self.outpng_mom2        = self.dir_products + self._read_key("outpng_mom2")
        self.outpng_ratio       = self.dir_products + self._read_key("outpng_ratio")
        self.outpng_13co_trot   = self.dir_products + self._read_key("outpng_13co_trot")
        self.outpng_13co_ncol   = self.dir_products + self._read_key("outpng_13co_ncol")

        self.outpng_emom0_13co10 = self.dir_products + self._read_key("outpng_emom0_13co10")
        self.outpng_emom0_13co21 = self.dir_products + self._read_key("outpng_emom0_13co21")
        self.outpng_emom1        = self.dir_products + self._read_key("outpng_emom1")
        self.outpng_emom2        = self.dir_products + self._read_key("outpng_emom2")
        self.outpng_eratio       = self.dir_products + self._read_key("outpng_eratio")
        self.outpng_e13co_trot   = self.dir_products + self._read_key("outpng_e13co_trot")
        self.outpng_e13co_ncol   = self.dir_products + self._read_key("outpng_e13co_ncol")
        self.outpng_residual     = self.dir_products + self._read_key("outpng_residual")

        self.outpng_modelmom0_13co10 = self.dir_products + self._read_key("outpng_modelmom0_13co10")
        self.outpng_modelmom0_13co21 = self.dir_products + self._read_key("outpng_modelmom0_13co21")
        self.outpng_simumom0_13co10  = self.dir_products + self._read_key("outpng_simumom0_13co10")
        self.outpng_simumom0_13co21  = self.dir_products + self._read_key("outpng_simumom0_13co21")

        self.outpng_13co10_vs_13co21_r = self.dir_products + self._read_key("outpng_13co10_vs_13co21_r")
        self.outpng_13co10_vs_13co21_t = self.dir_products + self._read_key("outpng_13co10_vs_13co21_t")
        self.outpng_13co10_vs_13co21_n = self.dir_products + self._read_key("outpng_13co10_vs_13co21_n")
        self.outpng_trot_vs_int        = self.dir_products + self._read_key("outpng_trot_vs_int")
        self.outpng_ncol_vs_int        = self.dir_products + self._read_key("outpng_ncol_vs_int")
        self.outpng_radial             = self.dir_products + self._read_key("outpng_radial")
        self.outpng_violin             = self.dir_products + self._read_key("outpng_violin")
        self.outpng_12co_vs_nh2        = self.dir_products + self._read_key("outpng_12co_vs_nh2")
        self.outpng_aco_map            = self.dir_products + self._read_key("outpng_aco_map")
        self.outpng_radial_aco         = self.dir_products + self._read_key("outpng_radial_aco")
        self.outpng_12co_vs_aco        = self.dir_products + self._read_key("outpng_12co_vs_aco")
        self.outpng_radio_trot         = self.dir_products + self._read_key("outpng_radio_trot")
        self.outpng_ncol_vs_m2         = self.dir_products + self._read_key("outpng_ncol_vs_m2")
        self.outpng_pturb              = self.dir_products + self._read_key("outpng_pturb")
        self.outpng_avir               = self.dir_products + self._read_key("outpng_avir")
        self.outpng_violin_pturb       = self.dir_products + self._read_key("outpng_violin_pturb")
        self.outpng_violin_avir        = self.dir_products + self._read_key("outpng_violin_avir")
        self.outpng_qqplot             = self.dir_products + self._read_key("outpng_qqplot")

        # finals
        self.final_60pc_obs      = self.dir_final + self._read_key("final_60pc_obs")
        self.final_60pc_rot      = self.dir_final + self._read_key("final_60pc_rot")
        self.final_scatter_int   = self.dir_final + self._read_key("final_scatter_int")
        self.final_scatter_rot   = self.dir_final + self._read_key("final_scatter_rot")
        self.final_radial        = self.dir_final + self._read_key("final_radial")
        self.final_aco           = self.dir_final + self._read_key("final_aco")
        self.final_jet           = self.dir_final + self._read_key("final_jet")
        self.final_gmc           = self.dir_final + self._read_key("final_gmc")
        # appendix
        self.final_qqplot        = self.dir_final + self._read_key("final_qqplot")
        self.final_60pc_err      = self.dir_final + self._read_key("final_60pc_err")
        self.final_sim_input     = self.dir_final + self._read_key("final_sim_input")
        self.final_sim_mom0      = self.dir_final + self._read_key("final_sim_mom0")
        self.final_sim_emom0     = self.dir_final + self._read_key("final_sim_emom0")
        #
        self.final_13co10_mom0   = self.dir_final + self._read_key("final_13co10_mom0")
        self.final_13co21_mom0   = self.dir_final + self._read_key("final_13co21_mom0")
        self.final_mom1          = self.dir_final + self._read_key("final_mom1")
        self.final_mom2          = self.dir_final + self._read_key("final_mom2")
        self.final_ratio         = self.dir_final + self._read_key("final_ratio")
        self.final_trot          = self.dir_final + self._read_key("final_trot")
        self.final_ncol          = self.dir_final + self._read_key("final_ncol")
        self.final_e13co10_mom0  = self.dir_final + self._read_key("final_e13co10_mom0")
        self.final_e13co21_mom0  = self.dir_final + self._read_key("final_e13co21_mom0")
        self.final_emom1         = self.dir_final + self._read_key("final_emom1")
        self.final_emom2         = self.dir_final + self._read_key("final_emom2")
        self.final_eratio        = self.dir_final + self._read_key("final_eratio")
        self.final_etrot         = self.dir_final + self._read_key("final_etrot")
        self.final_encol         = self.dir_final + self._read_key("final_encol")

        # box
        self.box_map            = self._read_key("box_map")
        self.box_map_nox        = self._read_key("box_map_nox")
        self.box_map_noy        = self._read_key("box_map_noy")
        self.box_map_noxy       = self._read_key("box_map_noxy")
        self.box_map_noc        = self._read_key("box_map_noc")
        self.box_map_noxc       = self._read_key("box_map_noxc")
        self.box_map_noyc       = self._read_key("box_map_noyc")
        self.box_map_noxyc      = self._read_key("box_map_noxyc")
        self.box_map2           = self._read_key("box_map2")

    ####################
    # run_ngc1068_ncol #
    ####################

    def run_ngc1068_ncol(
        self,
        do_all           = False,
        # analysis
        do_prepare       = False,
        do_fitting       = False, # after do_prepare
        # mom0 creation simulation
        do_create_models = False, # after do_prepare
        do_simulate_mom  = False, # after do_create_models
        # plot figures in paper
        plot_showcase    = False, # after do_fitting
        plot_showsim     = False, # after do_simulate_mom
        plot_scatter     = False, # after do_fitting
        plot_violin      = False, # after do_fitting
        plot_aco         = False, # after do_fitting
        plot_jet         = False, # after do_fitting
        plot_gmc         = False, # after do_fitting
        do_imagemagick   = False,
        immagick_all     = False,
        # supplement
        ):
        """
        This method runs all the methods which will create figures in the paper.
        """

        if do_all==True:
            do_prepare       = True
            do_fitting       = True
            do_create_models = True
            do_simulate_mom  = True
            plot_showcase    = True
            plot_showsim     = True
            plot_scatter     = True
            plot_violin      = True
            plot_aco         = True
            plot_jet         = True
            plot_gmc         = True
            do_imagemagick   = True
            immagick_all     = True

        # analysis
        if do_prepare==True:
            self.align_maps()

        if do_fitting==True:
            self.multi_fitting()

        if do_create_models==True:
            self.create_model_cubes()
            self.add_noise_to_models()

        if do_simulate_mom==True:
            self.simulate_mom_13co10()
            self.simulate_mom_13co21()
            self.eval_sim()

        # plot figures in paper
        if plot_showcase==True:
            self.showcase()

        if plot_showsim==True:
            self.showsim()

        if plot_scatter==True:
            self.plot_scatter()

        if plot_violin==True:
            self.plot_violin()

        if plot_aco==True:
            self.plot_aco()

        if plot_jet==True:
            self.plot_jet()

        if plot_gmc==True:
            self.plot_gmc()

        if do_imagemagick==True:
            self.immagick_figures(do_all=immagick_all,delin=False)

    ####################
    # immagick_figures #
    ####################

    def immagick_figures(
        self,
        delin                 = False,
        do_all                = False,
        # figure
        do_final_60pc_obs     = False,
        do_final_60pc_rot     = False,
        do_final_scatter_int  = False,
        do_final_scatter_rot  = False,
        do_final_radial       = False,
        do_final_aco          = False,
        do_final_jet          = False,
        do_final_gmc          = False,
        # appendix
        do_final_qqplot       = True,
        do_final_60pc_err     = False,
        do_final_sim_input    = False,
        do_final_sim_mom0     = False,
        do_final_sim_emom0    = False,
        # supplement
        do_final_13co10_mom0  = False,
        do_final_13co21_mom0  = False,
        do_final_ratio        = False,
        do_final_mom1         = False,
        do_final_mom2         = False,
        do_final_trot         = False,
        do_final_ncol         = False,
        do_final_e13co10_mom0 = False,
        do_final_e13co21_mom0 = False,
        do_final_eratio       = False,
        do_final_emom1        = False,
        do_final_emom2        = False,
        do_final_etrot        = False,
        do_final_encol        = False,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outpng_mom0_13co10.replace("???","60pc"),taskname)

        if do_all==True:
            # figure
            do_final_60pc_obs     = True
            do_final_60pc_rot     = True
            do_final_scatter_int  = True
            do_final_scatter_rot  = True
            do_final_radial       = True
            do_final_aco          = True
            do_final_jet          = True
            do_final_gmc          = True
            # appendix
            do_final_qqplot       = True
            do_final_60pc_err     = True
            do_final_sim_input    = True
            do_final_sim_mom0     = True
            do_final_sim_emom0    = True
            # supplement
            do_final_13co10_mom0  = True
            do_final_13co21_mom0  = True
            do_final_ratio        = True
            do_final_mom1         = True
            do_final_mom2         = True
            do_final_trot         = True
            do_final_ncol         = True
            do_final_e13co10_mom0 = True
            do_final_e13co21_mom0 = True
            do_final_eratio       = True
            do_final_emom1        = True
            do_final_emom2        = True
            do_final_etrot        = True
            do_final_encol        = True

        ##########
        # figure #
        ##########
        if do_final_60pc_obs==True:
            print("############################")
            print("# create do_final_60pc_obs #")
            print("############################")

            #
            combine_three_png(
                self.outpng_mom0_13co10.replace("???","60pc"),
                self.outpng_mom0_13co21.replace("???","60pc"),
                self.outpng_ratio.replace("???","60pc"),
                self.final_60pc_obs+"_tmp1.png",
                self.box_map_nox,
                self.box_map_noxy,
                self.box_map_noxy,
                delin=delin,
                )
            combine_two_png(
                self.outpng_mom1.replace("???","60pc"),
                self.outpng_mom2.replace("???","60pc"),
                self.final_60pc_obs+"_tmp2.png",
                self.box_map,
                self.box_map_noy,
                delin=delin,
                )
            combine_two_png(
                self.final_60pc_obs+"_tmp1.png",
                self.final_60pc_obs+"_tmp2.png",
                self.final_60pc_obs,
                "1000000x10000000+0+0",
                "1000000x10000000+0+0",
                axis="column",
                delin=True,
                )

        if do_final_60pc_rot==True:
            print("############################")
            print("# create do_final_60pc_rot #")
            print("############################")

            combine_two_png(
                self.outpng_13co_trot.replace("???","60pc"),
                self.outpng_13co_ncol.replace("???","60pc"),
                self.final_60pc_rot,
                self.box_map,
                self.box_map_noy,
                delin=delin,
                )

        if do_final_scatter_int==True:
            print("###############################")
            print("# create do_final_scatter_int #")
            print("###############################")

            immagick_crop(
                self.outpng_13co10_vs_13co21_r,
                self.final_scatter_int,
                self.box_map_noc,
                )

        if do_final_scatter_rot==True:
            print("###############################")
            print("# create do_final_scatter_rot #")
            print("###############################")

            combine_two_png(
                self.outpng_trot_vs_int,
                self.outpng_ncol_vs_int,
                self.final_scatter_rot,
                self.box_map_noc,
                self.box_map_noc,
                delin=delin,
                )

        if do_final_radial==True:
            print("##########################")
            print("# create do_final_radial #")
            print("##########################")

            combine_two_png(
                self.outpng_radial,
                self.outpng_violin,
                self.final_radial,
                self.box_map,
                self.box_map2,
                )

        if do_final_aco==True:
            print("#######################")
            print("# create do_final_aco #")
            print("#######################")

            combine_two_png(
                self.outpng_12co_vs_nh2,
                self.outpng_aco_map,
                self.final_aco+"_tmp1",
                self.box_map_noc,
                self.box_map,
                )
            combine_two_png(
                self.outpng_radial_aco,
                self.outpng_12co_vs_aco,
                self.final_aco+"_tmp2",
                self.box_map_noc,
                self.box_map_noc,
                )
            combine_two_png(
                self.final_aco+"_tmp1",
                self.final_aco+"_tmp2",
                self.final_aco,
                "1000000x1000000+0+0",
                "1000000x1000000+0+0",
                axis="column",
                delin=True,
                )

        if do_final_jet==True:
            print("#######################")
            print("# create do_final_jet #")
            print("#######################")

            immagick_crop(
                self.outpng_radio_trot,
                self.final_jet,
                self.box_map,
                )

        if do_final_gmc==True:
            print("#######################")
            print("# create do_final_gmc #")
            print("#######################")

            combine_two_png(
                self.outpng_ncol_vs_m2,
                self.outpng_pturb,
                self.final_gmc+"_tmp1",
                self.box_map_noc,
                self.box_map,
                )
            combine_two_png(
                self.outpng_violin_avir,
                self.outpng_violin_pturb,
                self.final_gmc+"_tmp2",
                self.box_map_noc,
                self.box_map_noc,
                )
            combine_two_png(
                self.final_gmc+"_tmp1",
                self.final_gmc+"_tmp2",
                self.final_gmc,
                "1000000x1000000+0+0",
                "1000000x1000000+0+0",
                axis="column",
                delin=True,
                )

        ############
        # appendix #
        ############
        if do_final_qqplot==True:
            print("##########################")
            print("# create do_final_qqplot #")
            print("##########################")

            combine_two_png(
                self.outpng_qqplot.replace("???","60pc"),
                self.outpng_residual.replace("???","60pc"),
                self.final_qqplot,
                self.box_map_noc,
                self.box_map,
                delin=delin,
                )

        if do_final_60pc_err==True:
            print("############################")
            print("# create do_final_60pc_err #")
            print("############################")

            #
            combine_three_png(
                self.outpng_emom0_13co10.replace("???","60pc"),
                self.outpng_emom0_13co21.replace("???","60pc"),
                self.outpng_eratio.replace("???","60pc"),
                self.final_60pc_err+"_tmp1.png",
                self.box_map_nox,
                self.box_map_noxy,
                self.box_map_noxy,
                delin=delin,
                )
            combine_three_png(
                self.outpng_emom1.replace("???","60pc"),
                self.outpng_emom2.replace("???","60pc"),
                self.outpng_e13co_trot.replace("???","60pc"),
                self.final_60pc_err+"_tmp2.png",
                self.box_map_nox,
                self.box_map_noxy,
                self.box_map_noxy,
                delin=delin,
                )
            combine_two_png(
                self.final_60pc_err+"_tmp1.png",
                self.final_60pc_err+"_tmp2.png",
                self.final_60pc_err+"_tmp12.png",
                "1000000x10000000+0+0",
                "1000000x10000000+0+0",
                axis="column",
                delin=True,
                )
            combine_two_png(
                self.final_60pc_err+"_tmp12.png",
                self.outpng_e13co_ncol.replace("???","60pc"),
                self.final_60pc_err,
                "1000000x10000000+0+0",
                self.box_map,
                axis="column",
                delin=delin,
                )
            os.system("rm -rf " + self.final_60pc_err+"_tmp12.png")

        if do_final_sim_input==True:
            print("############################")
            print("# create do_final_sim_mom0 #")
            print("############################")

            immagick_crop(
                self.outpng_modelmom0_13co10,
                self.final_sim_input,
                self.box_map,
                )

        this_snr = "10"
        if do_final_sim_mom0==True:
            print("############################")
            print("# create do_final_sim_mom0 #")
            print("############################")

            combine_three_png(
                self.outpng_simumom0_13co10.replace(".png","_noclip_snr"+this_snr+".png"),
                self.outpng_simumom0_13co10.replace(".png","_clip0_snr"+this_snr+".png"),
                self.outpng_simumom0_13co10.replace(".png","_clip3_snr"+this_snr+".png"),
                self.final_sim_mom0+"_tmp1.png",
                self.box_map_noxc,
                self.box_map_noxyc,
                self.box_map_noxyc,
                delin=delin,
                )
            combine_three_png(
                self.outpng_simumom0_13co10.replace(".png","_noclip_masked_snr"+this_snr+".png"),
                self.outpng_simumom0_13co10.replace(".png","_clip0_masked_snr"+this_snr+".png"),
                self.outpng_simumom0_13co10.replace(".png","_clip3_masked_snr"+this_snr+".png"),
                self.final_sim_mom0+"_tmp2.png",
                self.box_map_noc,
                self.box_map_noyc,
                self.box_map_noy,
                delin=delin,
                )
            combine_two_png(
                self.final_sim_mom0+"_tmp1.png",
                self.final_sim_mom0+"_tmp2.png",
                self.final_sim_mom0,
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                axis="column",
                )

        if do_final_sim_emom0==True:
            print("#############################")
            print("# create do_final_sim_emom0 #")
            print("#############################")

            combine_three_png(
                self.outpng_simumom0_13co10.replace(".png","_noclip_snr"+this_snr+".png").replace("mom0","emom0"),
                self.outpng_simumom0_13co10.replace(".png","_clip0_snr"+this_snr+".png").replace("mom0","emom0"),
                self.outpng_simumom0_13co10.replace(".png","_clip3_snr"+this_snr+".png").replace("mom0","emom0"),
                self.final_sim_emom0+"_tmp1.png",
                self.box_map_noxc,
                self.box_map_noxyc,
                self.box_map_noxyc,
                delin=delin,
                )
            combine_three_png(
                self.outpng_simumom0_13co10.replace(".png","_noclip_masked_snr"+this_snr+".png").replace("mom0","emom0"),
                self.outpng_simumom0_13co10.replace(".png","_clip0_masked_snr"+this_snr+".png").replace("mom0","emom0"),
                self.outpng_simumom0_13co10.replace(".png","_clip3_masked_snr"+this_snr+".png").replace("mom0","emom0"),
                self.final_sim_emom0+"_tmp2.png",
                self.box_map_noc,
                self.box_map_noyc,
                self.box_map_noy,
                delin=delin,
                )
            combine_two_png(
                self.final_sim_emom0+"_tmp1.png",
                self.final_sim_emom0+"_tmp2.png",
                self.final_sim_emom0,
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                axis="column",
                )

        ##############
        # supplement #
        ##############
        if do_final_13co10_mom0==True:
            print("############################")
            print("# create final_13co10_mom0 #")
            print("############################")

            this_prename = self.outpng_mom0_13co10
            this_final   = self.final_13co10_mom0

            combine_two_png(
                this_prename.replace("???","60pc"),
                this_prename.replace("???","70pc"),
                this_final+"_tmp1.png",
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_three_png(
                this_prename.replace("???","80pc"),
                this_prename.replace("???","90pc"),
                this_prename.replace("???","100pc"),
                this_final+"_tmp2.png",
                self.box_map,
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_two_png(
                this_prename.replace("???","110pc"),
                this_prename.replace("???","120pc"),
                this_final+"_tmp3.png",
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_three_png(
                this_prename.replace("???","130pc"),
                this_prename.replace("???","140pc"),
                this_prename.replace("???","150pc"),
                this_final+"_tmp4.png",
                self.box_map,
                self.box_map,
                self.box_map,
                delin=delin,
                )

            combine_two_png(
                this_final+"_tmp1.png",
                this_final+"_tmp2.png",
                this_final+"_tmp12.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )
            combine_two_png(
                this_final+"_tmp3.png",
                this_final+"_tmp4.png",
                this_final+"_tmp34.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )
            combine_two_png(
                this_final+"_tmp12.png",
                this_final+"_tmp34.png",
                this_final,
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                axis="column",
                )

        if do_final_13co21_mom0==True:
            print("############################")
            print("# create final_13co21_mom0 #")
            print("############################")

            this_prename = self.outpng_mom0_13co21
            this_final   = self.final_13co21_mom0

            combine_two_png(
                this_prename.replace("???","60pc"),
                this_prename.replace("???","70pc"),
                this_final+"_tmp1.png",
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_three_png(
                this_prename.replace("???","80pc"),
                this_prename.replace("???","90pc"),
                this_prename.replace("???","100pc"),
                this_final+"_tmp2.png",
                self.box_map,
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_two_png(
                this_prename.replace("???","110pc"),
                this_prename.replace("???","120pc"),
                this_final+"_tmp3.png",
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_three_png(
                this_prename.replace("???","130pc"),
                this_prename.replace("???","140pc"),
                this_prename.replace("???","150pc"),
                this_final+"_tmp4.png",
                self.box_map,
                self.box_map,
                self.box_map,
                delin=delin,
                )

            combine_two_png(
                this_final+"_tmp1.png",
                this_final+"_tmp2.png",
                this_final+"_tmp12.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )
            combine_two_png(
                this_final+"_tmp3.png",
                this_final+"_tmp4.png",
                this_final+"_tmp34.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )
            combine_two_png(
                this_final+"_tmp12.png",
                this_final+"_tmp34.png",
                this_final,
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                axis="column",
                )

        if do_final_ratio==True:
            print("#########################")
            print("# create do_final_ratio #")
            print("#########################")

            this_prename = self.outpng_ratio
            this_final   = self.final_ratio

            combine_two_png(
                this_prename.replace("???","60pc"),
                this_prename.replace("???","70pc"),
                this_final+"_tmp1.png",
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_three_png(
                this_prename.replace("???","80pc"),
                this_prename.replace("???","90pc"),
                this_prename.replace("???","100pc"),
                this_final+"_tmp2.png",
                self.box_map,
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_two_png(
                this_prename.replace("???","110pc"),
                this_prename.replace("???","120pc"),
                this_final+"_tmp3.png",
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_three_png(
                this_prename.replace("???","130pc"),
                this_prename.replace("???","140pc"),
                this_prename.replace("???","150pc"),
                this_final+"_tmp4.png",
                self.box_map,
                self.box_map,
                self.box_map,
                delin=delin,
                )

            combine_two_png(
                this_final+"_tmp1.png",
                this_final+"_tmp2.png",
                this_final+"_tmp12.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )
            combine_two_png(
                this_final+"_tmp3.png",
                this_final+"_tmp4.png",
                this_final+"_tmp34.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )
            combine_two_png(
                this_final+"_tmp12.png",
                this_final+"_tmp34.png",
                this_final,
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                axis="column",
                )

        if do_final_mom1==True:
            print("########################")
            print("# create do_final_mom1 #")
            print("########################")

            this_prename = self.outpng_mom1
            this_final   = self.final_mom1

            combine_two_png(
                this_prename.replace("???","60pc"),
                this_prename.replace("???","70pc"),
                this_final+"_tmp1.png",
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_three_png(
                this_prename.replace("???","80pc"),
                this_prename.replace("???","90pc"),
                this_prename.replace("???","100pc"),
                this_final+"_tmp2.png",
                self.box_map,
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_two_png(
                this_prename.replace("???","110pc"),
                this_prename.replace("???","120pc"),
                this_final+"_tmp3.png",
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_three_png(
                this_prename.replace("???","130pc"),
                this_prename.replace("???","140pc"),
                this_prename.replace("???","150pc"),
                this_final+"_tmp4.png",
                self.box_map,
                self.box_map,
                self.box_map,
                delin=delin,
                )

            combine_two_png(
                this_final+"_tmp1.png",
                this_final+"_tmp2.png",
                this_final+"_tmp12.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )
            combine_two_png(
                this_final+"_tmp3.png",
                this_final+"_tmp4.png",
                this_final+"_tmp34.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )
            combine_two_png(
                this_final+"_tmp12.png",
                this_final+"_tmp34.png",
                this_final,
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                axis="column",
                )

        if do_final_mom2==True:
            print("########################")
            print("# create do_final_mom2 #")
            print("########################")

            this_prename = self.outpng_mom2
            this_final   = self.final_mom2

            combine_two_png(
                this_prename.replace("???","60pc"),
                this_prename.replace("???","70pc"),
                this_final+"_tmp1.png",
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_three_png(
                this_prename.replace("???","80pc"),
                this_prename.replace("???","90pc"),
                this_prename.replace("???","100pc"),
                this_final+"_tmp2.png",
                self.box_map,
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_two_png(
                this_prename.replace("???","110pc"),
                this_prename.replace("???","120pc"),
                this_final+"_tmp3.png",
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_three_png(
                this_prename.replace("???","130pc"),
                this_prename.replace("???","140pc"),
                this_prename.replace("???","150pc"),
                this_final+"_tmp4.png",
                self.box_map,
                self.box_map,
                self.box_map,
                delin=delin,
                )

            combine_two_png(
                this_final+"_tmp1.png",
                this_final+"_tmp2.png",
                this_final+"_tmp12.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )
            combine_two_png(
                this_final+"_tmp3.png",
                this_final+"_tmp4.png",
                this_final+"_tmp34.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )
            combine_two_png(
                this_final+"_tmp12.png",
                this_final+"_tmp34.png",
                this_final,
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                axis="column",
                )

        if do_final_trot==True:
            print("########################")
            print("# create do_final_trot #")
            print("########################")

            this_prename = self.outpng_13co_trot
            this_final   = self.final_trot

            combine_two_png(
                this_prename.replace("???","60pc"),
                this_prename.replace("???","70pc"),
                this_final+"_tmp1.png",
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_three_png(
                this_prename.replace("???","80pc"),
                this_prename.replace("???","90pc"),
                this_prename.replace("???","100pc"),
                this_final+"_tmp2.png",
                self.box_map,
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_two_png(
                this_prename.replace("???","110pc"),
                this_prename.replace("???","120pc"),
                this_final+"_tmp3.png",
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_three_png(
                this_prename.replace("???","130pc"),
                this_prename.replace("???","140pc"),
                this_prename.replace("???","150pc"),
                this_final+"_tmp4.png",
                self.box_map,
                self.box_map,
                self.box_map,
                delin=delin,
                )

            combine_two_png(
                this_final+"_tmp1.png",
                this_final+"_tmp2.png",
                this_final+"_tmp12.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )
            combine_two_png(
                this_final+"_tmp3.png",
                this_final+"_tmp4.png",
                this_final+"_tmp34.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )
            combine_two_png(
                this_final+"_tmp12.png",
                this_final+"_tmp34.png",
                this_final,
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                axis="column",
                )

        if do_final_ncol==True:
            print("########################")
            print("# create do_final_ncol #")
            print("########################")

            this_prename = self.outpng_13co_ncol
            this_final   = self.final_ncol

            combine_two_png(
                this_prename.replace("???","60pc"),
                this_prename.replace("???","70pc"),
                this_final+"_tmp1.png",
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_three_png(
                this_prename.replace("???","80pc"),
                this_prename.replace("???","90pc"),
                this_prename.replace("???","100pc"),
                this_final+"_tmp2.png",
                self.box_map,
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_two_png(
                this_prename.replace("???","110pc"),
                this_prename.replace("???","120pc"),
                this_final+"_tmp3.png",
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_three_png(
                this_prename.replace("???","130pc"),
                this_prename.replace("???","140pc"),
                this_prename.replace("???","150pc"),
                this_final+"_tmp4.png",
                self.box_map,
                self.box_map,
                self.box_map,
                delin=delin,
                )

            combine_two_png(
                this_final+"_tmp1.png",
                this_final+"_tmp2.png",
                this_final+"_tmp12.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )
            combine_two_png(
                this_final+"_tmp3.png",
                this_final+"_tmp4.png",
                this_final+"_tmp34.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )
            combine_two_png(
                this_final+"_tmp12.png",
                this_final+"_tmp34.png",
                this_final,
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                axis="column",
                )

        if do_final_e13co10_mom0==True:
            print("#############################")
            print("# create final_e13co10_mom0 #")
            print("#############################")

            this_prename = self.outpng_emom0_13co10
            this_final   = self.final_e13co10_mom0

            combine_two_png(
                this_prename.replace("???","60pc"),
                this_prename.replace("???","70pc"),
                this_final+"_tmp1.png",
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_three_png(
                this_prename.replace("???","80pc"),
                this_prename.replace("???","90pc"),
                this_prename.replace("???","100pc"),
                this_final+"_tmp2.png",
                self.box_map,
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_two_png(
                this_prename.replace("???","110pc"),
                this_prename.replace("???","120pc"),
                this_final+"_tmp3.png",
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_three_png(
                this_prename.replace("???","130pc"),
                this_prename.replace("???","140pc"),
                this_prename.replace("???","150pc"),
                this_final+"_tmp4.png",
                self.box_map,
                self.box_map,
                self.box_map,
                delin=delin,
                )

            combine_two_png(
                this_final+"_tmp1.png",
                this_final+"_tmp2.png",
                this_final+"_tmp12.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )
            combine_two_png(
                this_final+"_tmp3.png",
                this_final+"_tmp4.png",
                this_final+"_tmp34.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )
            combine_two_png(
                this_final+"_tmp12.png",
                this_final+"_tmp34.png",
                this_final,
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                axis="column",
                )

        if do_final_e13co21_mom0==True:
            print("#############################")
            print("# create final_e13co21_mom0 #")
            print("#############################")

            this_prename = self.outpng_emom0_13co21
            this_final   = self.final_e13co21_mom0

            combine_two_png(
                this_prename.replace("???","60pc"),
                this_prename.replace("???","70pc"),
                this_final+"_tmp1.png",
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_three_png(
                this_prename.replace("???","80pc"),
                this_prename.replace("???","90pc"),
                this_prename.replace("???","100pc"),
                this_final+"_tmp2.png",
                self.box_map,
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_two_png(
                this_prename.replace("???","110pc"),
                this_prename.replace("???","120pc"),
                this_final+"_tmp3.png",
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_three_png(
                this_prename.replace("???","130pc"),
                this_prename.replace("???","140pc"),
                this_prename.replace("???","150pc"),
                this_final+"_tmp4.png",
                self.box_map,
                self.box_map,
                self.box_map,
                delin=delin,
                )

            combine_two_png(
                this_final+"_tmp1.png",
                this_final+"_tmp2.png",
                this_final+"_tmp12.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )
            combine_two_png(
                this_final+"_tmp3.png",
                this_final+"_tmp4.png",
                this_final+"_tmp34.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )
            combine_two_png(
                this_final+"_tmp12.png",
                this_final+"_tmp34.png",
                this_final,
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                axis="column",
                )

        if do_final_eratio==True:
            print("#######################$##")
            print("# create do_final_eratio #")
            print("#######################$##")

            this_prename = self.outpng_eratio
            this_final   = self.final_eratio

            combine_two_png(
                this_prename.replace("???","60pc"),
                this_prename.replace("???","70pc"),
                this_final+"_tmp1.png",
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_three_png(
                this_prename.replace("???","80pc"),
                this_prename.replace("???","90pc"),
                this_prename.replace("???","100pc"),
                this_final+"_tmp2.png",
                self.box_map,
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_two_png(
                this_prename.replace("???","110pc"),
                this_prename.replace("???","120pc"),
                this_final+"_tmp3.png",
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_three_png(
                this_prename.replace("???","130pc"),
                this_prename.replace("???","140pc"),
                this_prename.replace("???","150pc"),
                this_final+"_tmp4.png",
                self.box_map,
                self.box_map,
                self.box_map,
                delin=delin,
                )

            combine_two_png(
                this_final+"_tmp1.png",
                this_final+"_tmp2.png",
                this_final+"_tmp12.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )
            combine_two_png(
                this_final+"_tmp3.png",
                this_final+"_tmp4.png",
                this_final+"_tmp34.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )
            combine_two_png(
                this_final+"_tmp12.png",
                this_final+"_tmp34.png",
                this_final,
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                axis="column",
                )

        if do_final_emom1==True:
            print("#########################")
            print("# create do_final_emom1 #")
            print("#########################")

            this_prename = self.outpng_emom1
            this_final   = self.final_emom1

            combine_two_png(
                this_prename.replace("???","60pc"),
                this_prename.replace("???","70pc"),
                this_final+"_tmp1.png",
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_three_png(
                this_prename.replace("???","80pc"),
                this_prename.replace("???","90pc"),
                this_prename.replace("???","100pc"),
                this_final+"_tmp2.png",
                self.box_map,
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_two_png(
                this_prename.replace("???","110pc"),
                this_prename.replace("???","120pc"),
                this_final+"_tmp3.png",
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_three_png(
                this_prename.replace("???","130pc"),
                this_prename.replace("???","140pc"),
                this_prename.replace("???","150pc"),
                this_final+"_tmp4.png",
                self.box_map,
                self.box_map,
                self.box_map,
                delin=delin,
                )

            combine_two_png(
                this_final+"_tmp1.png",
                this_final+"_tmp2.png",
                this_final+"_tmp12.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )
            combine_two_png(
                this_final+"_tmp3.png",
                this_final+"_tmp4.png",
                this_final+"_tmp34.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )
            combine_two_png(
                this_final+"_tmp12.png",
                this_final+"_tmp34.png",
                this_final,
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                axis="column",
                )

        if do_final_emom2==True:
            print("#########################")
            print("# create do_final_emom2 #")
            print("#########################")

            this_prename = self.outpng_emom2
            this_final   = self.final_emom2

            combine_two_png(
                this_prename.replace("???","60pc"),
                this_prename.replace("???","70pc"),
                this_final+"_tmp1.png",
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_three_png(
                this_prename.replace("???","80pc"),
                this_prename.replace("???","90pc"),
                this_prename.replace("???","100pc"),
                this_final+"_tmp2.png",
                self.box_map,
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_two_png(
                this_prename.replace("???","110pc"),
                this_prename.replace("???","120pc"),
                this_final+"_tmp3.png",
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_three_png(
                this_prename.replace("???","130pc"),
                this_prename.replace("???","140pc"),
                this_prename.replace("???","150pc"),
                this_final+"_tmp4.png",
                self.box_map,
                self.box_map,
                self.box_map,
                delin=delin,
                )

            combine_two_png(
                this_final+"_tmp1.png",
                this_final+"_tmp2.png",
                this_final+"_tmp12.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )
            combine_two_png(
                this_final+"_tmp3.png",
                this_final+"_tmp4.png",
                this_final+"_tmp34.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )
            combine_two_png(
                this_final+"_tmp12.png",
                this_final+"_tmp34.png",
                this_final,
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                axis="column",
                )

        if do_final_etrot==True:
            print("#########################")
            print("# create do_final_etrot #")
            print("#########################")

            this_prename = self.outpng_e13co_trot
            this_final   = self.final_etrot

            combine_two_png(
                this_prename.replace("???","60pc"),
                this_prename.replace("???","70pc"),
                this_final+"_tmp1.png",
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_three_png(
                this_prename.replace("???","80pc"),
                this_prename.replace("???","90pc"),
                this_prename.replace("???","100pc"),
                this_final+"_tmp2.png",
                self.box_map,
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_two_png(
                this_prename.replace("???","110pc"),
                this_prename.replace("???","120pc"),
                this_final+"_tmp3.png",
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_three_png(
                this_prename.replace("???","130pc"),
                this_prename.replace("???","140pc"),
                this_prename.replace("???","150pc"),
                this_final+"_tmp4.png",
                self.box_map,
                self.box_map,
                self.box_map,
                delin=delin,
                )

            combine_two_png(
                this_final+"_tmp1.png",
                this_final+"_tmp2.png",
                this_final+"_tmp12.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )
            combine_two_png(
                this_final+"_tmp3.png",
                this_final+"_tmp4.png",
                this_final+"_tmp34.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )
            combine_two_png(
                this_final+"_tmp12.png",
                this_final+"_tmp34.png",
                this_final,
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                axis="column",
                )

        if do_final_encol==True:
            print("#########################")
            print("# create do_final_encol #")
            print("#########################")

            this_prename = self.outpng_e13co_ncol
            this_final   = self.final_encol

            combine_two_png(
                this_prename.replace("???","60pc"),
                this_prename.replace("???","70pc"),
                this_final+"_tmp1.png",
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_three_png(
                this_prename.replace("???","80pc"),
                this_prename.replace("???","90pc"),
                this_prename.replace("???","100pc"),
                this_final+"_tmp2.png",
                self.box_map,
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_two_png(
                this_prename.replace("???","110pc"),
                this_prename.replace("???","120pc"),
                this_final+"_tmp3.png",
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_three_png(
                this_prename.replace("???","130pc"),
                this_prename.replace("???","140pc"),
                this_prename.replace("???","150pc"),
                this_final+"_tmp4.png",
                self.box_map,
                self.box_map,
                self.box_map,
                delin=delin,
                )

            combine_two_png(
                this_final+"_tmp1.png",
                this_final+"_tmp2.png",
                this_final+"_tmp12.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )
            combine_two_png(
                this_final+"_tmp3.png",
                this_final+"_tmp4.png",
                this_final+"_tmp34.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )
            combine_two_png(
                this_final+"_tmp12.png",
                this_final+"_tmp34.png",
                this_final,
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                axis="column",
                )

    ############
    # plot_gmc #
    ############

    def plot_gmc(
        self,
        ):
        """
        """
        this_beam = "60pc"
        beamr     = 30

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outmaps_13co_ncol.replace("???",this_beam),taskname)

        xlim      = [0.8,3.5]
        ylim      = [0.3,2.5]
        factor    = 1.0 / self.abundance_13co_h2
        title     = "log$_{\mathrm{10}}$ $\\Sigma_{\mathrm{H_2}}$ vs. log$_{\mathrm{10}}$ $\\sigma_{\mathrm{v}}$ at " + this_beam.replace("pc"," pc")
        xlabel    = "log$_{\mathrm{10}}$ $\\Sigma_{\mathrm{H_2}}$ ($M_{\odot}$ pc$^{-2}$)"
        ylabel    = "log$_{\mathrm{10}}$ $\\sigma_{\mathrm{v}}$ (km s$^{-1}$)"
        ximage    = self.outmaps_13co_ncol.replace("???",this_beam)
        xerrimage = self.outemaps_13co_ncol.replace("???",this_beam)
        yimage    = self.outmaps_mom2.replace("???",this_beam)
        yerrimage = self.outemaps_mom2.replace("???",this_beam)

        # cmap = distance
        cblabel   = "Distance (kpc)"
        cimage    = None
        cerrimage = None
        outpng    = self.outpng_ncol_vs_m2

        log_Sh2, log_Sh2_err, mom2, mom2_err, dist = self._plot_scatter5(
            ximage,
            xerrimage,
            yimage,
            yerrimage,
            cimage,
            cerrimage,
            outpng,
            xlim,
            ylim,
            title,
            xlabel,
            ylabel,
            cblabel,
            factor,
            outfits_P=self.outmaps_pturb.replace("???",this_beam),
            outfits_vir=self.outmaps_avir.replace("???",this_beam),
            templatefits=self.outcubes_13co10.replace("???",this_beam),
            )

        self._showcase_one(
            self.outmaps_pturb.replace("???",this_beam),
            self.outmaps_mom0_13co21.replace("???",this_beam),
            self.outpng_pturb,
            "log$_{\mathrm{10}}$ $P_{\mathrm{turb}}$ at " + this_beam.replace("pc"," pc"),
            "K cm$^{-3}$",
            clim=None,
            )

        self._showcase_one(
            self.outmaps_avir.replace("???",this_beam),
            self.outmaps_mom0_13co21.replace("???",this_beam),
            self.outpng_avir,
            "log$_{\mathrm{10}}$ $\\alpha_{\mathrm{vir}}$ at " + this_beam.replace("pc"," pc"),
            "",
            clim=[0,2],
            )

        ################
        # pturb violin #
        ################
        # prepare
        R_as = dist * 1000 / self.scale_pc
        T    = np.log10(61.3 * 10**log_Sh2 * mom2**2 / (beamr/40.))

        tlim    = [4,9]
        t_grid  = np.linspace(tlim[0], tlim[1], num=1000)
        ylabel  = "log$_{\mathrm{10}}$ $P_{\mathrm{turb}}$ (K cm$^{-3}$)"
        title   = "$P_{\mathrm{turb}}$ Distribution"

        # plot
        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        ad  = [0.215,0.83,0.10,0.90]
        myax_set(ax1, "y", [-0.5,8.5], tlim, title, None, ylabel, adjust=ad)

        ax1.set_xticks([1,3,5,7])
        ax1.set_xticklabels(["All","CND","INT","SBR"], rotation=0, ha="center")

        # plot all data
        n = 1
        self._ax_violin(ax1,T,n,t_grid,"grey")

        # plot cnd data
        n = 3
        cut = np.where(R_as<self.r_cnd_as)
        self._ax_violin(ax1,T[cut],n,t_grid,"tomato")

        # plot intermediate data
        n = 5
        cut = np.where((R_as>=self.r_cnd_as)&(R_as<self.r_sbr_as))
        self._ax_violin(ax1,T[cut],n,t_grid,"green")

        # plot sbr data
        n = 7
        cut = np.where(R_as>=self.r_sbr_as)
        self._ax_violin(ax1,T[cut],n,t_grid,"deepskyblue")

        # save
        os.system("rm -rf " + self.outpng_violin_pturb)
        plt.savefig(self.outpng_violin_pturb, dpi=self.fig_dpi)

        ###############
        # avir violin #
        ###############
        # prepare
        R_as = dist * 1000 / self.scale_pc
        T    = np.log10(5.77 * mom2**2 / 10**log_Sh2 / (beamr/40.))

        tlim    = [-0.2,3]
        t_grid  = np.linspace(tlim[0], tlim[1], num=1000)
        ylabel  = "log$_{\mathrm{10}}$ $\\alpha_{\mathrm{vir}}$"
        title   = "$\\alpha_{\mathrm{vir}}$ Distribution"

        # plot
        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        ad  = [0.215,0.83,0.10,0.90]
        myax_set(ax1, "y", [-0.5,8.5], tlim, title, None, ylabel, adjust=ad)

        ax1.set_xticks([1,3,5,7])
        ax1.set_xticklabels(["All","CND","INT","SBR"], rotation=0, ha="center")

        # plot all data
        n = 1
        self._ax_violin(ax1,T,n,t_grid,"grey")

        # plot cnd data
        n = 3
        cut = np.where(R_as<self.r_cnd_as)
        self._ax_violin(ax1,T[cut],n,t_grid,"tomato")

        # plot intermediate data
        n = 5
        cut = np.where((R_as>=self.r_cnd_as)&(R_as<self.r_sbr_as))
        self._ax_violin(ax1,T[cut],n,t_grid,"green")

        # plot sbr data
        n = 7
        cut = np.where(R_as>=self.r_sbr_as)
        self._ax_violin(ax1,T[cut],n,t_grid,"deepskyblue")

        # save
        os.system("rm -rf " + self.outpng_violin_avir)
        plt.savefig(self.outpng_violin_avir, dpi=self.fig_dpi)

    ############
    # plot_jet #
    ############

    def plot_jet(
        self,
        ):
        """
        """
        this_beam = "150pc"
        rms_vla   = 0.001473485

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outmaps_13co_trot.replace("???",this_beam),taskname)

        #
        template = "template.image"
        run_importfits(self.outmaps_13co_trot.replace("???",this_beam),template)
        run_roundsmooth(self.vla,self.vla+"_tmp1",150/72.)
        run_imregrid(self.vla+"_tmp1",template,self.vla+"_tmp2",delin=True)
        run_exportfits(self.vla+"_tmp2",self.outmaps_vla.replace("???",this_beam),dropdeg=True,dropstokes=True,delin=True)
        os.system("rm -rf template.image")

        scalebar = 100. / self.scale_pc
        label_scalebar = "100 pc"

        levels_cont1 = [-2,2,4,8,16,32,64,96]
        width_cont1  = [1.0]
        set_bg_color = "white"

        # plot
        myfig_fits2png(
            imcolor=self.outmaps_13co_trot.replace("???",this_beam),
            outfile=self.outpng_radio_trot,
            imcontour1=self.outmaps_vla.replace("???",this_beam),
            imsize_as=self.imsize,
            ra_cnt=self.ra_agn_str,
            dec_cnt=self.dec_agn_str,
            unit_cont1=rms_vla,
            levels_cont1=[-3,3,6,12,24,48],
            width_cont1=[1.0],
            set_title="$T_{\mathrm{rot}}$ + Radio continuum at " + this_beam.replace("pc"," pc"),
            colorlog=False,
            scalebar=scalebar,
            label_scalebar=label_scalebar,
            set_cbar=True,
            label_cbar="(K)",
            clim=[2.73,6],
            set_bg_color=set_bg_color,
            numann="13co",
            )

    ############
    # plot_aco #
    ############

    def plot_aco(
        self,
        ):
        """
        """
        this_beam = "60pc"

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outmaps_13co_ncol.replace("???",this_beam),taskname)

        ###########
        # prepare #
        ###########
        template = "template.image"
        run_importfits(self.outmaps_13co_ncol.replace("???",this_beam),template)
        run_imregrid(
            self.mom0_12co10.replace("???",this_beam),
            template,
            self.mom0_12co10.replace("???",this_beam)+".regrid",
            axes=-1,
            )
        run_exportfits(self.mom0_12co10.replace("???",this_beam)+".regrid",self.outmaps_12co10.replace("???",this_beam),delin=True)

        #
        pix_before = abs(imhead(imagename=self.mom0_12co10.replace("???",this_beam),mode="list")["cdelt1"]) * 3600 * 180 / np.pi
        pix_after  = abs(imhead(imagename=self.outmaps_12co10.replace("???",this_beam),mode="list")["cdelt1"]) * 3600 * 180 / np.pi
        numpix     = pix_after**2/pix_before**2

        run_immath_one(
            self.emom0_12co10.replace("???",this_beam),
            self.emom0_12co10.replace("???",this_beam)+"_tmp1",
            "IM0*IM0",
            )
        run_imregrid(
            self.emom0_12co10.replace("???",this_beam)+"_tmp1",
            template,
            self.emom0_12co10.replace("???",this_beam)+"_tmp2",
            axes=[0,1],
            delin=True,
            )
        run_immath_one(
            self.emom0_12co10.replace("???",this_beam)+"_tmp2",
            self.emom0_12co10.replace("???",this_beam)+"_tmp3",
            "sqrt(IM0)/sqrt("+str(numpix)+")",
            delin=True,
            )
        run_exportfits(
            self.emom0_12co10.replace("???",this_beam)+"_tmp3",
            self.outemaps_12co10.replace("???",this_beam),
            delin=True,
            )
        os.system("rm -rf template.image")

        ######################
        # plot 12co10 vs NH2 #
        ######################
        xlim      = [1.3,3.4]
        ylim      = [20.7,23.2]
        factor    = 1.0 / self.abundance_13co_h2
        title     = "log$_{\mathrm{10}}$ $I_{\mathrm{^{12}CO(1-0)}}$ vs. log$_{\mathrm{10}}$ $N_{\mathrm{H_2}}$ at " + this_beam.replace("pc"," pc")
        xlabel    = "log$_{\mathrm{10}}$ $I_{\mathrm{^{12}CO(1-0)}}$ (K km s$^{-1}$)"
        ylabel    = "log$_{\mathrm{10}}$ $N_{\mathrm{H_2}}$ (cm$^{-2}$)"
        ximage    = self.outmaps_12co10.replace("???",this_beam)
        xerrimage = self.outemaps_12co10.replace("???",this_beam)
        yimage    = self.outmaps_13co_ncol.replace("???",this_beam)
        yerrimage = self.outemaps_13co_ncol.replace("???",this_beam)

        # plot
        cblabel   = "Distance (kpc)"
        cimage    = None
        cerrimage = None
        outpng    = self.outpng_12co_vs_nh2
        log_co,elog_co,log_nh2,elog_nh2,dist = self._plot_scatter4(
            ximage,
            xerrimage,
            yimage,
            yerrimage,
            cimage,
            cerrimage,
            outpng,
            xlim,
            ylim,
            title,
            xlabel,
            ylabel,
            cblabel,
            factor,
            cmap="rainbow_r",
            outfits=self.outmaps_aco.replace("???",this_beam),
            outefits=self.outemaps_aco.replace("???",this_beam),
            templatefits=self.outcubes_13co10.replace("???",this_beam),
            )

        os.system("rm -rf " + self.mom0_12co10.replace("???",this_beam) + ".regrid")
        os.system("rm -rf " + self.emom0_12co10.replace("???",this_beam) + ".regrid")

        ################
        # plot aco map #
        ################
        self._showcase_one(
            self.outmaps_aco.replace("???",this_beam),
            self.outmaps_12co10.replace("???",this_beam),
            self.outpng_aco_map,
            "$\\alpha_{\mathrm{CO}}$ map at " + this_beam.replace("pc"," pc"),
            "($M_{\\odot}$ (K km s$^{-1}$ pc$^{2}$)$^{-1}$)",
            clim=[0.1,1.3],
            )

        ###################
        # plot radial aco #
        ###################
        x1    = dist
        y1    = (log_nh2-log_co) - np.log10(2e20) + np.log10(4.3)
        yerr1 = np.sqrt(elog_co**2+elog_nh2**2)
        # cut
        x    = x1[y1<0.5]
        y    = y1[y1<0.5]
        yerr = yerr1[y1<0.5]
        # binning
        n,_   = np.histogram(x, bins=10, range=[np.min(x),np.max(x)])
        sy,_  = np.histogram(x, bins=10, range=[np.min(x),np.max(x)], weights=y)
        sy2,_ = np.histogram(x, bins=10, range=[np.min(x),np.max(x)], weights=y*y)
        mean  = sy / n
        std   = np.sqrt(sy2/n - mean*mean)
        binx1 = (_[1:]+_[:-1])/2
        biny1 = mean
        binyerr1 = std

        # plot
        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        ad  = [0.215,0.83,0.10,0.90]
        myax_set(
            ax1,
            "both",
            [0.0,1.3],
            [-1.3,0.8],
            "radial $\\alpha_{\mathrm{CO}}$ at " + this_beam.replace("pc"," pc"),
            "Distance (kpc)",
            "log$_{\mathrm{10}}$ $\\alpha_{\mathrm{CO}}$ (K km s$^{-1}$ pc$^{2}$)$^{-1}$)",
            adjust=ad,
            )

        ax1.scatter(x, y, c="tomato", lw=0, s=40, zorder=1e9)
        ax1.errorbar(x, y, yerr=yerr, lw=1, capsize=0, color="grey", linestyle="None")

        ax1.plot(binx1, biny1, color="red", lw=2.0, zorder=1e11)
        for i in range(len(binx1)):
            this_binx1    = binx1[i]
            this_biny1    = biny1[i]
            this_binyerr1 = binyerr1[i]
            ax1.plot([this_binx1,this_binx1],[this_biny1-this_binyerr1,this_biny1+this_binyerr1], color="red", lw=2.0, zorder=1e11)

        # save
        os.system("rm -rf " + self.outpng_radial_aco)
        plt.savefig(self.outpng_radial_aco, dpi=self.fig_dpi)

        ##############
        # aco violin #
        ##############
        x1    = log_co
        y1    =(log_nh2-log_co) - np.log10(2e20) + np.log10(4.3)
        xerr1 = elog_co
        yerr1 = np.sqrt(elog_co**2+elog_nh2**2)
        c1    = dist * 1000 / self.scale_pc
        # cut
        x    = x1[y1<0.5]
        y    = y1[y1<0.5]
        xerr = xerr1[y1<0.5]
        yerr = yerr1[y1<0.5]
        c    = c1[y1<0.5]

        # prepare
        R_as = c
        T    = y

        tlim    = [-0.8,0.4]
        t_grid  = np.linspace(tlim[0], tlim[1], num=1000)
        ylabel  = "log$_{\mathrm{10}}$ $\\alpha_{\mathrm{CO}}$ (K km s$^{-1}$ pc$^{2}$)$^{-1}$)"
        title   = "$\\alpha_{\mathrm{CO}}$ Distribution"

        # plot
        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        ad  = [0.215,0.83,0.10,0.90]
        myax_set(ax1, "y", [-0.5,8.5], tlim, title, None, ylabel, adjust=ad)

        ax1.set_xticks([1,3,5,7])
        ax1.set_xticklabels(["All","CND","INT","SBR"], rotation=0, ha="center")

        # plot all data
        n = 1
        self._ax_violin(ax1,T,n,t_grid,"grey")

        # plot cnd data
        n = 3
        cut = np.where(R_as<self.r_cnd_as)
        self._ax_violin(ax1,T[cut],n,t_grid,"tomato",vmin=-0.75,vmax=0.05)

        # plot intermediate data
        n = 5
        cut = np.where((R_as>=self.r_cnd_as)&(R_as<self.r_sbr_as))
        self._ax_violin(ax1,T[cut],n,t_grid,"green")

        # plot sbr data
        n = 7
        cut = np.where(R_as>=self.r_sbr_as)
        self._ax_violin(ax1,T[cut],n,t_grid,"deepskyblue")

        # save
        os.system("rm -rf " + self.outpng_12co_vs_aco)
        plt.savefig(self.outpng_12co_vs_aco, dpi=self.fig_dpi)

    ###############
    # plot_violin #
    ###############

    def plot_violin(
        self,
        plot_I_vs_I=True,
        plot_phys_vs_I=True,
        plot_radial=True,
        ):
        """
        """
        this_beam = "60pc"

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outmaps_mom0_13co10.replace("???",this_beam),taskname)

        # fits
        yimage     = self.outmaps_13co_trot.replace("???",this_beam)
        yerrimage  = self.outemaps_13co_trot.replace("???",this_beam)
        y2image    = self.outmaps_13co_ncol.replace("???",this_beam)
        y2errimage = self.outemaps_13co_ncol.replace("???",this_beam)

        # coords
        _,box       = imval_all(yimage)
        data_coords = imval(yimage,box=box)["coords"]
        ra_deg      = data_coords[:,:,0] * 180/np.pi
        ra_deg      = ra_deg.flatten()
        dec_deg     = data_coords[:,:,1] * 180/np.pi
        dec_deg     = dec_deg.flatten()
        dist_pc,_   = get_reldist_pc(ra_deg, dec_deg, self.ra_agn, self.dec_agn, self.scale_pc, 0, 0)
        data_x      = dist_pc / self.scale_pc

        # y1
        data_y1,_ = imval_all(yimage)
        data_y1   = data_y1["data"] * data_y1["mask"]
        data_y1   = data_y1.flatten()
        data_y1[np.isnan(data_y1)] = 0

        err_y1,_ = imval_all(yerrimage)
        err_y1   = err_y1["data"] * err_y1["mask"]
        err_y1   = err_y1.flatten()
        err_y1[np.isnan(err_y1)] = 0

        # y2
        data_y2,_ = imval_all(y2image)
        data_y2   = data_y2["data"] * data_y2["mask"]
        data_y2   = data_y2.flatten()
        data_y2[np.isnan(data_y2)] = 0

        err_y2,_ = imval_all(y2errimage)
        err_y2   = err_y2["data"] * err_y2["mask"]
        err_y2   = err_y2.flatten()
        err_y2[np.isnan(err_y2)] = 0

        # prepare
        cut  = np.where((data_y1>abs(err_y1)*self.snr)&(data_y2>abs(err_y2)*self.snr))
        R_as = data_x[cut]
        T    = np.log10(data_y1[cut])
        N    = data_y2[cut]

        tlim    = [0.38,1.20]
        t_grid  = np.linspace(tlim[0], tlim[1], num=1000)
        ylabel  = "log$_{\mathrm{10}}$ $T_{\mathrm{rot}}$ (K)"
        title   = "$T_{\mathrm{rot}}$ Distribution"

        nlim    = [14.98,16.93]
        n_grid  = np.linspace(nlim[0], nlim[1], num=1000)
        ylabel2 = "log$_{\mathrm{10}}$ $N_{\mathrm{^{13}CO}}$ (cm$^{-2}$)"
        title2  = "$N_{\mathrm{^{13}CO}}$ Distribution"

        ########
        # plot #
        ########
        fig = plt.figure(figsize=(17,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:5])
        ax2 = plt.subplot(gs[0:10,5:10])
        ax3 = ax2.twinx()
        ad  = [0.10,0.90,0.10,0.90]
        myax_set(ax1, "y", [-0.5,8.5], tlim, title, None, ylabel, adjust=ad)
        myax_set(ax2, "y", [-0.5,8.5], nlim, title2, None, None, adjust=ad)
        ax3.set_ylabel(ylabel2)
        ax3.set_ylim(nlim)
        ax2.tick_params(labelleft=False)

        ax1.set_yticks([0.4,0.6,0.8,1.0])
        ax1.set_xticks([1,3,5,7])
        ax1.set_xticklabels(["All","CND","INT","SBR"], rotation=0, ha="center")

        ax2.set_xticks([1,3,5,7])
        ax2.set_xticklabels(["All","CND","INT","SBR"], rotation=0, ha="center")

        # plot all data
        n = 1
        self._ax_violin(ax1,T,n,t_grid,"grey",vmax=1.05)
        self._ax_violin(ax3,N,n,n_grid,"grey",vmin=15.05,vmax=16.9)

        # plot cnd data
        n = 3
        cut = np.where(R_as<self.r_cnd_as)
        self._ax_violin(ax1,T[cut],n,t_grid,"tomato",vmin=0.40,vmax=1.2)
        self._ax_violin(ax3,N[cut],n,n_grid,"tomato",vmin=15.1,vmax=16.8)

        # plot intermediate data
        n = 5
        cut = np.where((R_as>=self.r_cnd_as)&(R_as<self.r_sbr_as))
        self._ax_violin(ax1,T[cut],n,t_grid,"green",vmin=0.40,vmax=0.93)
        self._ax_violin(ax3,N[cut],n,n_grid,"green",vmin=15.0,vmax=16.5)

        # plot sbr data
        n = 7
        cut = np.where(R_as>=self.r_sbr_as)
        self._ax_violin(ax1,T[cut],n,t_grid,"deepskyblue")
        self._ax_violin(ax3,N[cut],n,n_grid,"deepskyblue",vmax=16.9)

        # save
        os.system("rm -rf " + self.outpng_violin)
        plt.savefig(self.outpng_violin, dpi=self.fig_dpi)

    ################
    # plot_scatter #
    ################

    def _ax_violin(
        self,
        ax,
        data,
        n,
        ygrid,
        color,
        vmin=None,
        vmax=None,
        ):
        """
        """

        # prepare
        if vmin==None:
            vmin = np.min(data)

        if vmax==None:
            vmax = np.max(data)

        # percentiles
        p2   = np.nanpercentile(data[data!=0],2)
        p16  = np.nanpercentile(data[data!=0],16)
        p50  = np.nanpercentile(data[data!=0],50)
        p84  = np.nanpercentile(data[data!=0],84)
        p98  = np.nanpercentile(data[data!=0],98)

        # kde
        l = gaussian_kde(data)
        data = np.array(l(ygrid) / np.max(l(ygrid))) / 1.1

        left  = n-data
        right = n+data
        cut = np.where((ygrid<vmax)&(ygrid>vmin))

        ax.plot(right[cut], ygrid[cut], lw=2, color="grey")
        ax.plot(left[cut], ygrid[cut], lw=2, color="grey")
        ax.fill_betweenx(ygrid, left, right, facecolor=color, alpha=0.5, lw=0)

        # percentiles
        ax.plot([n,n],[p2,p98],lw=2,color="grey")
        ax.plot([n,n],[p16,p84],lw=9,color="grey")
        ax.plot(n,p50,".",color="white",markersize=10, markeredgewidth=0)

    ################
    # plot_scatter #
    ################

    def plot_scatter(
        self,
        plot_I_vs_I=True,
        plot_phys_vs_I=True,
        plot_radial=True,
        ):
        """
        References:
        https://stackoverflow.com/questions/10208814/colormap-for-errorbars-in-x-y-scatter-plot-using-matplotlib
        https://sabopy.com/py/matplotlib-79/
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outmaps_mom0_13co10.replace("???","60pc"),taskname)

        ###############
        # plot_I_vs_I #
        ###############
        this_beam = "60pc"
        if plot_I_vs_I==True:
            lim       = [-0.4,2.3]
            title     = "log$_{\mathrm{10}}$ $I_{\mathrm{^{13}CO(1-0)}}$ vs. log$_{\mathrm{10}}$ $I_{\mathrm{^{13}CO(2-1)}}$ at " + this_beam.replace("pc"," pc")
            xlabel    = "log$_{\mathrm{10}}$ $I_{\mathrm{^{13}CO(1-0)}}$ (K km s$^{-1}$)"
            ylabel    = "log$_{\mathrm{10}}$ $I_{\mathrm{^{13}CO(2-1)}}$ (K km s$^{-1}$)"
            ximage    = self.outmaps_mom0_13co10.replace("???",this_beam)
            xerrimage = self.outemaps_mom0_13co10.replace("???",this_beam)
            yimage    = self.outmaps_mom0_13co21.replace("???",this_beam)
            yerrimage = self.outemaps_mom0_13co21.replace("???",this_beam)

            # cmap = distance
            cblabel   = "Distance (kpc)"
            cimage    = None
            cerrimage = None
            outpng    = self.outpng_13co10_vs_13co21_r
            self._plot_scatter1(ximage,xerrimage,yimage,yerrimage,cimage,cerrimage,outpng,lim,title,xlabel,ylabel,cblabel)

            # cmap = Trot
            cblabel   = "$T_{\mathrm{rot}}$ (K)"
            cimage    = self.outmaps_13co_trot.replace("???",this_beam)
            cerrimage = self.outemaps_13co_trot.replace("???",this_beam)
            outpng    = self.outpng_13co10_vs_13co21_t
            self._plot_scatter1(ximage,xerrimage,yimage,yerrimage,cimage,cerrimage,outpng,lim,title,xlabel,ylabel,cblabel,cmap="rainbow")

            # cmap = log Ncol
            cblabel   = "log$_{\mathrm{10}}$ $N_{\mathrm{^{13}CO}}$ (cm$^{-2}$)"
            cimage    = self.outmaps_13co_ncol.replace("???",this_beam)
            cerrimage = self.outemaps_13co_ncol.replace("???",this_beam)
            outpng    = self.outpng_13co10_vs_13co21_n
            self._plot_scatter1(ximage,xerrimage,yimage,yerrimage,cimage,cerrimage,outpng,lim,title,xlabel,ylabel,cblabel,cmap="rainbow")

        ##################
        # plot_phys_vs_I #
        ##################
        this_beam  = "60pc"
        if plot_phys_vs_I==True:
            xlim       = [-0.4,2.3]
            ylim       = [2,13]
            title      = "log$_{\mathrm{10}}$ $I_{\mathrm{^{13}CO}}$ vs. $T_{\mathrm{rot}}$ at " + this_beam.replace("pc"," pc")
            xlabel     = "log$_{\mathrm{10}}$ $I_{\mathrm{^{13}CO}}$ (K km s$^{-1}$)"
            ylabel     = "$T_{\mathrm{rot}}$ (K)"
            x1image    = self.outmaps_mom0_13co10.replace("???",this_beam)
            x1errimage = self.outemaps_mom0_13co10.replace("???",this_beam)
            x2image    = self.outmaps_mom0_13co21.replace("???",this_beam)
            x2errimage = self.outemaps_mom0_13co21.replace("???",this_beam)
            yimage     = self.outmaps_13co_trot.replace("???",this_beam)
            yerrimage  = self.outemaps_13co_trot.replace("???",this_beam)
            outpng     = self.outpng_trot_vs_int
            self._plot_scatter2(x1image,x1errimage,x2image,x2errimage,yimage,yerrimage,outpng,xlim,ylim,title,xlabel,ylabel)

            xlim       = [-0.4,2.3]
            ylim       = [14.7,17.2]
            title      = "log$_{\mathrm{10}}$ $I_{\mathrm{^{13}CO}}$ vs. log$_{\mathrm{10}}$ $N_{\mathrm{^{13}CO}}$ at " + this_beam.replace("pc"," pc")
            xlabel     = "log$_{\mathrm{10}}$ $I_{\mathrm{^{13}CO}}$ (K km s$^{-1}$)"
            ylabel     = "log$_{\mathrm{10}}$ $N_{\mathrm{^{13}CO}}$ (cm$^{-2}$)"
            x1image    = self.outmaps_mom0_13co10.replace("???",this_beam)
            x1errimage = self.outemaps_mom0_13co10.replace("???",this_beam)
            x2image    = self.outmaps_mom0_13co21.replace("???",this_beam)
            x2errimage = self.outemaps_mom0_13co21.replace("???",this_beam)
            yimage     = self.outmaps_13co_ncol.replace("???",this_beam)
            yerrimage  = self.outemaps_13co_ncol.replace("???",this_beam)
            outpng     = self.outpng_ncol_vs_int
            self._plot_scatter2(x1image,x1errimage,x2image,x2errimage,yimage,yerrimage,outpng,xlim,ylim,title,xlabel,ylabel)

        ###############
        # plot_radial #
        ###############
        this_beam  = "60pc"
        if plot_radial==True:
            xlim       = [0.0,1.3]
            ylim       = [2,21]
            ylim2      = [13.5,17.2]
            title      = "radial $T_{\mathrm{rot}}$ and log$_{\mathrm{10}}$ $N_{\mathrm{^{13}CO}}$ at " + this_beam.replace("pc"," pc")
            xlabel     = "Distance (kpc)"
            ylabel     = "$T_{\mathrm{rot}}$ (K)"
            ylabel2    = "log$_{\mathrm{10}}$ $N_{\mathrm{^{13}CO}}$ (cm$^{-2}$)"
            yimage     = self.outmaps_13co_trot.replace("???",this_beam)
            yerrimage  = self.outemaps_13co_trot.replace("???",this_beam)
            y2image    = self.outmaps_13co_ncol.replace("???",this_beam)
            y2errimage = self.outemaps_13co_ncol.replace("???",this_beam)
            outpng     = self.outpng_radial
            self._plot_scatter3(yimage,yerrimage,y2image,y2errimage,outpng,xlim,ylim,ylim2,title,xlabel,ylabel,ylabel2)

    ############
    # eval_sim #
    ############

    def eval_sim(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outmodelcube_13co10.replace(".fits","_snr10.fits"),taskname)

        lim      = [0.75,2.25] # 13co10 range
        lim2     = [-1.0,0.2]  # ratio range
        this_snr = "snr10"
        outpng_mom0_nomask  = "test1_"+this_snr+".png"
        outpng_mom0_mask    = "test2_"+this_snr+".png"
        outpng_ratio_nomask = "test3_"+this_snr+".png"
        outpng_ratio_mask   = "test4_"+this_snr+".png"
        self._eval_a_sim(
            10,
            lim,
            lim2,
            this_snr,
            outpng_mom0_nomask,
            outpng_mom0_mask,
            outpng_ratio_nomask,
            outpng_ratio_mask,
            )

        lim      = [0.75,np.log10(10**2.25*2.5)] # 13co10 range
        lim2     = [np.log10(10**-1.0/2.5),0.2]  # ratio range
        this_snr = "snr25"
        outpng_mom0_nomask  = "test1_"+this_snr+".png"
        outpng_mom0_mask    = "test2_"+this_snr+".png"
        outpng_ratio_nomask = "test3_"+this_snr+".png"
        outpng_ratio_mask   = "test4_"+this_snr+".png"
        self._eval_a_sim(
            10,
            lim,
            lim2,
            this_snr,
            outpng_mom0_nomask,
            outpng_mom0_mask,
            outpng_ratio_nomask,
            outpng_ratio_mask,
            )

        lim      = [0.75,np.log10(10**2.25*5.0)] # 13co10 range
        lim2     = [np.log10(10**-1.0/2.5),0.2]  # ratio range
        this_snr = "snr50"
        outpng_mom0_nomask  = "test1_"+this_snr+".png"
        outpng_mom0_mask    = "test2_"+this_snr+".png"
        outpng_ratio_nomask = "test3_"+this_snr+".png"
        outpng_ratio_mask   = "test4_"+this_snr+".png"
        self._eval_a_sim(
            10,
            lim,
            lim2,
            this_snr,
            outpng_mom0_nomask,
            outpng_mom0_mask,
            outpng_ratio_nomask,
            outpng_ratio_mask,
            )

    #######################
    # simulate_mom_13co10 #
    #######################

    def simulate_mom_13co10(
        self,
        do_noclip=False,
        do_zeroclip=False,
        do_clip=False,
        do_noclip_mask=False,
        do_zeroclip_mask=False,
        do_clip_mask=False,
        do_fitting=True,
        rms=0.227283716202,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outmodelcube_13co10.replace(".fits","_snr10.fits"),taskname)

        modelcube = self.outmodelcube_13co10
        modelmom0 = self.outmodelmom0_13co10
        simmom0   = self.outsimumom0_13co10

        #
        nchan         = imhead(modelcube,mode="list")["shape"][2]
        chanwidth_Hz  = abs(imhead(modelcube,mode="list")["cdelt3"])
        restfreq_Hz   = imhead(modelcube,mode="list")["restfreq"][0]
        chanwidth_kms = chanwidth_Hz / restfreq_Hz * 299792.458 # km/s

        ####################
        # model input mom0 #
        ####################
        infile  = modelcube
        outfile = modelmom0
        os.system("rm -rf " + outfile + ".image")
        immoments(imagename=infile,outfile=outfile+".image")
        run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)

        ###############
        # create mask #
        ###############
        infile = modelcube
        mask   = modelcube.replace(".fits","_mask.fits")
        maxval = imstat(infile)["max"][0]
        clip   = str(maxval / 50.0)
        os.system("rm -rf " + mask + ".image")
        expr = "iif(IM0>=" + clip + ",1,0)"
        run_immath_one(infile,mask+".image",expr)
        run_exportfits(mask+".image",mask,delin=True,dropdeg=True,dropstokes=True)

        #################
        # nchan in mask #
        #################
        map_nchan = "map_nchan_in_mask.fits"
        os.system("rm -rf " + map_nchan + ".image?")
        immoments(mask,outfile=map_nchan+".image1")
        run_immath_one(map_nchan+".image1",map_nchan+".image2","IM0/"+str(chanwidth_kms))
        run_exportfits(map_nchan+".image2",map_nchan,delin=True,dropdeg=True,dropstokes=True)
        os.system("rm -rf " + map_nchan + ".image?")

        #############
        # do_noclip # emom0 done
        #############
        if do_noclip==True:
            # snr = 10
            infile  = modelcube.replace(".fits","_snr10.fits")
            outfile = simmom0.replace(".fits","_noclip_snr10.fits")
            outerr  = simmom0.replace(".fits","_noclip_snr10.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image")
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)
            # error
            run_immath_one(outfile,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt("+str(nchan)+")")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

            # snr = 25
            infile  = modelcube.replace(".fits","_snr25.fits")
            outfile = simmom0.replace(".fits","_noclip_snr25.fits")
            outerr  = simmom0.replace(".fits","_noclip_snr25.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image")
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)
            # error
            run_immath_one(outfile,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt("+str(nchan)+")")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

            # snr = 50
            infile  = modelcube.replace(".fits","_snr50.fits")
            outfile = simmom0.replace(".fits","_noclip_snr50.fits")
            outerr  = simmom0.replace(".fits","_noclip_snr50.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image")
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)
            # error
            run_immath_one(outfile,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt("+str(nchan)+")")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

        ###############
        # do_zeroclip # emom0 done
        ###############
        includepix = [0.0,1000000.]
        if do_zeroclip==True:
            # snr = 10
            infile  = modelcube.replace(".fits","_snr10.fits")
            outfile = simmom0.replace(".fits","_clip0_snr10.fits")
            outerr  = simmom0.replace(".fits","_clip0_snr10.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image",includepix=includepix)
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_one(infile,this_mask+".image1","iif(IM0>0,1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

            # snr = 25
            infile  = modelcube.replace(".fits","_snr25.fits")
            outfile = simmom0.replace(".fits","_clip0_snr25.fits")
            outerr  = simmom0.replace(".fits","_clip0_snr25.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image",includepix=includepix)
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_one(infile,this_mask+".image1","iif(IM0>0,1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

            # snr = 50
            infile  = modelcube.replace(".fits","_snr50.fits")
            outfile = simmom0.replace(".fits","_clip0_snr50.fits")
            outerr  = simmom0.replace(".fits","_clip0_snr50.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image",includepix=includepix)
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_one(infile,this_mask+".image1","iif(IM0>0,1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

        ###########
        # do_clip # emom0 done
        ###########
        includepix = [rms*self.snr_clip,1000000.]
        if do_clip==True:
            # snr = 10
            infile  = modelcube.replace(".fits","_snr10.fits")
            outfile = simmom0.replace(".fits","_clip"+str(self.snr_clip)+"_snr10.fits")
            outerr  = simmom0.replace(".fits","_clip"+str(self.snr_clip)+"_snr10.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image",includepix=includepix)
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_one(infile,this_mask+".image1","iif(IM0>"+str(rms*self.snr_clip)+",1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

            # snr = 25
            infile  = modelcube.replace(".fits","_snr25.fits")
            outfile = simmom0.replace(".fits","_clip"+str(self.snr_clip)+"_snr25.fits")
            outerr  = simmom0.replace(".fits","_clip"+str(self.snr_clip)+"_snr25.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image",includepix=includepix)
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_one(infile,this_mask+".image1","iif(IM0>"+str(rms*self.snr_clip)+",1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

            # snr = 50
            infile  = modelcube.replace(".fits","_snr50.fits")
            outfile = simmom0.replace(".fits","_clip"+str(self.snr_clip)+"_snr50.fits")
            outerr  = simmom0.replace(".fits","_clip"+str(self.snr_clip)+"_snr50.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image",includepix=includepix)
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_one(infile,this_mask+".image1","iif(IM0>"+str(rms*self.snr_clip)+",1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

        ##################
        # do_noclip_mask # emom0 done
        ##################
        if do_noclip_mask==True:
            # snr = 10
            infile  = modelcube.replace(".fits","_snr10.fits")
            outfile = simmom0.replace(".fits","_noclip_masked_snr10.fits")
            outerr  = simmom0.replace(".fits","_noclip_masked_snr10.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image?")
            run_immath_two(infile,mask,outfile+".image1","IM0*IM1")
            immoments(imagename=outfile+".image1",outfile=outfile+".image2")
            run_exportfits(outfile+".image2",outfile,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + outfile + ".image?")
            # error
            run_immath_two(outfile,map_nchan,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

            # snr = 25
            infile  = modelcube.replace(".fits","_snr25.fits")
            outfile = simmom0.replace(".fits","_noclip_masked_snr25.fits")
            outerr  = simmom0.replace(".fits","_noclip_masked_snr25.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image?")
            run_immath_two(infile,mask,outfile+".image1","IM0*IM1")
            immoments(imagename=outfile+".image1",outfile=outfile+".image2")
            run_exportfits(outfile+".image2",outfile,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + outfile + ".image?")
            # error
            run_immath_two(outfile,map_nchan,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

            # snr = 50
            infile  = modelcube.replace(".fits","_snr50.fits")
            outfile = simmom0.replace(".fits","_noclip_masked_snr50.fits")
            outerr  = simmom0.replace(".fits","_noclip_masked_snr50.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image?")
            run_immath_two(infile,mask,outfile+".image1","IM0*IM1")
            immoments(imagename=outfile+".image1",outfile=outfile+".image2")
            run_exportfits(outfile+".image2",outfile,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + outfile + ".image?")
            # error
            run_immath_two(outfile,map_nchan,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

        ####################
        # do_zeroclip_mask # emom0 done
        ####################
        includepix = [0.0,1000000.]
        if do_zeroclip_mask==True:
            # snr = 10
            infile  = modelcube.replace(".fits","_snr10.fits")
            outfile = simmom0.replace(".fits","_clip0_masked_snr10.fits")
            outerr  = simmom0.replace(".fits","_clip0_masked_snr10.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image?")
            run_immath_two(infile,mask,outfile+".image1","IM0*IM1")
            immoments(imagename=outfile+".image1",outfile=outfile+".image2",includepix=includepix)
            run_exportfits(outfile+".image2",outfile,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + outfile + ".image?")
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(infile,mask,this_mask+".image1","iif(IM0*IM1>0,1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

            # snr = 25
            infile  = modelcube.replace(".fits","_snr25.fits")
            outfile = simmom0.replace(".fits","_clip0_masked_snr25.fits")
            outerr  = simmom0.replace(".fits","_clip0_masked_snr25.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image?")
            run_immath_two(infile,mask,outfile+".image1","IM0*IM1")
            immoments(imagename=outfile+".image1",outfile=outfile+".image2",includepix=includepix)
            run_exportfits(outfile+".image2",outfile,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + outfile + ".image?")
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(infile,mask,this_mask+".image1","iif(IM0*IM1>0,1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

            # snr = 50
            infile  = modelcube.replace(".fits","_snr50.fits")
            outfile = simmom0.replace(".fits","_clip0_masked_snr50.fits")
            outerr  = simmom0.replace(".fits","_clip0_masked_snr50.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image?")
            run_immath_two(infile,mask,outfile+".image1","IM0*IM1")
            immoments(imagename=outfile+".image1",outfile=outfile+".image2",includepix=includepix)
            run_exportfits(outfile+".image2",outfile,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + outfile + ".image?")
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(infile,mask,this_mask+".image1","iif(IM0*IM1>0,1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

        ################
        # do_clip_mask # emom0 done
        ################
        includepix = [rms*self.snr_clip,1000000.]
        if do_clip_mask==True:
            # snr = 10
            infile  = modelcube.replace(".fits","_snr10.fits")
            outfile = simmom0.replace(".fits","_clip"+str(self.snr_clip)+"_masked_snr10.fits")
            outerr  = simmom0.replace(".fits","_clip"+str(self.snr_clip)+"_masked_snr10.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image?")
            run_immath_two(infile,mask,outfile+".image1","IM0*IM1")
            immoments(imagename=outfile+".image1",outfile=outfile+".image2",includepix=includepix)
            run_exportfits(outfile+".image2",outfile,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + outfile + ".image?")
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(infile,mask,this_mask+".image1","iif(IM0*IM1>"+str(rms*self.snr_clip)+",1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

            # snr = 25
            infile  = modelcube.replace(".fits","_snr25.fits")
            outfile = simmom0.replace(".fits","_clip"+str(self.snr_clip)+"_masked_snr25.fits")
            outerr  = simmom0.replace(".fits","_clip"+str(self.snr_clip)+"_masked_snr25.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image?")
            run_immath_two(infile,mask,outfile+".image1","IM0*IM1")
            immoments(imagename=outfile+".image1",outfile=outfile+".image2",includepix=includepix)
            run_exportfits(outfile+".image2",outfile,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + outfile + ".image?")
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(infile,mask,this_mask+".image1","iif(IM0*IM1>"+str(rms*self.snr_clip)+",1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

            # snr = 50
            infile  = modelcube.replace(".fits","_snr50.fits")
            outfile = simmom0.replace(".fits","_clip"+str(self.snr_clip)+"_masked_snr50.fits")
            outerr  = simmom0.replace(".fits","_clip"+str(self.snr_clip)+"_masked_snr50.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image?")
            run_immath_two(infile,mask,outfile+".image1","IM0*IM1")
            immoments(imagename=outfile+".image1",outfile=outfile+".image2",includepix=includepix)
            run_exportfits(outfile+".image2",outfile,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + outfile + ".image?")
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(infile,mask,this_mask+".image1","iif(IM0*IM1>"+str(rms*self.snr_clip)+",1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

        ##############
        # do_fitting #
        ##############
        if do_fitting==True:
            print("implement fitting simulation!")

            # get model cubes
            os.system("cp " + self.outmodelcube_13co10.replace(".fits","_snr10.fits") + " model_low.fits")
            os.system("cp " + self.outmodelcube_13co21.replace(".fits","_snr10.fits") + " model_high.fits")
            cubelow  = "model_low.fits"
            cubehigh = "model_high.fits"

            # create two model ecubes with a single value of rms
            run_importfits(cubelow,"noisemodel_low.cube1")
            run_immath_one("noisemodel_low.cube1","noisemodel_low.cube2","IM0*0+"+str(rms))
            run_exportfits("noisemodel_low.cube2","noisemodel_low.fits")
            os.system("rm -rf noisemodel_low.cube1 noisemodel_low.cube2")

            run_importfits(cubehigh,"noisemodel_high.cube1")
            run_immath_one("noisemodel_high.cube1","noisemodel_high.cube2","IM0*0+"+str(rms))
            run_exportfits("noisemodel_high.cube2","noisemodel_high.fits")
            os.system("rm -rf noisemodel_high.cube1 noisemodel_high.cube2")

            rotation_13co21_13co10(
                cubelow,
                cubehigh,
                "noisemodel_low.fits",
                "noisemodel_high.fits",
                ra_cnt=self.ra_agn,
                dec_cnt=self.dec_agn,
                snr=self.snr_fit,
                snr_limit=self.snr_fit,
                restfreq_low=110.20135430,
                restfreq_high=220.39868420c,
                )

    #######################
    # simulate_mom_13co21 #
    #######################

    def simulate_mom_13co21(
        self,
        do_noclip=False,
        do_zeroclip=False,
        do_clip=False,
        do_noclip_mask=False,
        do_zeroclip_mask=False,
        do_clip_mask=False,
        rms=0.227283716202,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outmodelcube_13co21.replace(".fits","_snr10.fits"),taskname)

        modelcube = self.outmodelcube_13co21
        modelmom0 = self.outmodelmom0_13co21
        simmom0   = self.outsimumom0_13co21

        #
        nchan         = imhead(modelcube,mode="list")["shape"][2]
        chanwidth_Hz  = abs(imhead(modelcube,mode="list")["cdelt3"])
        restfreq_Hz   = imhead(modelcube,mode="list")["restfreq"][0]
        chanwidth_kms = chanwidth_Hz / restfreq_Hz * 299792.458 # km/s

        ####################
        # model input mom0 #
        ####################
        infile  = modelcube
        outfile = modelmom0
        os.system("rm -rf " + outfile + ".image")
        immoments(imagename=infile,outfile=outfile+".image")
        run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)

        ###############
        # create mask #
        ###############
        infile = modelcube
        mask   = modelcube.replace(".fits","_mask.fits")
        maxval = imstat(infile)["max"][0]
        clip   = str(maxval / 50.0)
        os.system("rm -rf " + mask + ".image")
        expr = "iif(IM0>=" + clip + ",1,0)"
        run_immath_one(infile,mask+".image",expr)
        run_exportfits(mask+".image",mask,delin=True,dropdeg=True,dropstokes=True)

        #################
        # nchan in mask #
        #################
        map_nchan = "map_nchan_in_mask.fits"
        os.system("rm -rf " + map_nchan + ".image?")
        immoments(mask,outfile=map_nchan+".image1")
        run_immath_one(map_nchan+".image1",map_nchan+".image2","IM0/"+str(chanwidth_kms))
        run_exportfits(map_nchan+".image2",map_nchan,delin=True,dropdeg=True,dropstokes=True)
        os.system("rm -rf " + map_nchan + ".image?")

        #############
        # do_noclip # emom0 done
        #############
        if do_noclip==True:
            # snr = 10
            infile  = modelcube.replace(".fits","_snr10.fits")
            outfile = simmom0.replace(".fits","_noclip_snr10.fits")
            outerr  = simmom0.replace(".fits","_noclip_snr10.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image")
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)
            # error
            run_immath_one(outfile,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt("+str(nchan)+")")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

            # snr = 25
            infile  = modelcube.replace(".fits","_snr25.fits")
            outfile = simmom0.replace(".fits","_noclip_snr25.fits")
            outerr  = simmom0.replace(".fits","_noclip_snr25.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image")
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)
            # error
            run_immath_one(outfile,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt("+str(nchan)+")")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

            # snr = 50
            infile  = modelcube.replace(".fits","_snr50.fits")
            outfile = simmom0.replace(".fits","_noclip_snr50.fits")
            outerr  = simmom0.replace(".fits","_noclip_snr50.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image")
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)
            # error
            run_immath_one(outfile,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt("+str(nchan)+")")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

        ###############
        # do_zeroclip # emom0 done
        ###############
        includepix = [0.0,1000000.]
        if do_zeroclip==True:
            # snr = 10
            infile  = modelcube.replace(".fits","_snr10.fits")
            outfile = simmom0.replace(".fits","_clip0_snr10.fits")
            outerr  = simmom0.replace(".fits","_clip0_snr10.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image",includepix=includepix)
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_one(infile,this_mask+".image1","iif(IM0>0,1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

            # snr = 25
            infile  = modelcube.replace(".fits","_snr25.fits")
            outfile = simmom0.replace(".fits","_clip0_snr25.fits")
            outerr  = simmom0.replace(".fits","_clip0_snr25.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image",includepix=includepix)
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_one(infile,this_mask+".image1","iif(IM0>0,1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

            # snr = 50
            infile  = modelcube.replace(".fits","_snr50.fits")
            outfile = simmom0.replace(".fits","_clip0_snr50.fits")
            outerr  = simmom0.replace(".fits","_clip0_snr50.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image",includepix=includepix)
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_one(infile,this_mask+".image1","iif(IM0>0,1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

        ###########
        # do_clip # emom0 done
        ###########
        includepix = [rms*self.snr_clip,1000000.]
        if do_clip==True:
            # snr = 10
            infile  = modelcube.replace(".fits","_snr10.fits")
            outfile = simmom0.replace(".fits","_clip"+str(self.snr_clip)+"_snr10.fits")
            outerr  = simmom0.replace(".fits","_clip"+str(self.snr_clip)+"_snr10.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image",includepix=includepix)
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_one(infile,this_mask+".image1","iif(IM0>"+str(rms*self.snr_clip)+",1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

            # snr = 25
            infile  = modelcube.replace(".fits","_snr25.fits")
            outfile = simmom0.replace(".fits","_clip"+str(self.snr_clip)+"_snr25.fits")
            outerr  = simmom0.replace(".fits","_clip"+str(self.snr_clip)+"_snr25.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image",includepix=includepix)
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_one(infile,this_mask+".image1","iif(IM0>"+str(rms*self.snr_clip)+",1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

            # snr = 50
            infile  = modelcube.replace(".fits","_snr50.fits")
            outfile = simmom0.replace(".fits","_clip"+str(self.snr_clip)+"_snr50.fits")
            outerr  = simmom0.replace(".fits","_clip"+str(self.snr_clip)+"_snr50.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image",includepix=includepix)
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_one(infile,this_mask+".image1","iif(IM0>"+str(rms*self.snr_clip)+",1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

        ##################
        # do_noclip_mask # emom0 done
        ##################
        if do_noclip_mask==True:
            # snr = 10
            infile  = modelcube.replace(".fits","_snr10.fits")
            outfile = simmom0.replace(".fits","_noclip_masked_snr10.fits")
            outerr  = simmom0.replace(".fits","_noclip_masked_snr10.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image?")
            run_immath_two(infile,mask,outfile+".image1","IM0*IM1")
            immoments(imagename=outfile+".image1",outfile=outfile+".image2")
            run_exportfits(outfile+".image2",outfile,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + outfile + ".image?")
            # error
            run_immath_two(outfile,map_nchan,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

            # snr = 25
            infile  = modelcube.replace(".fits","_snr25.fits")
            outfile = simmom0.replace(".fits","_noclip_masked_snr25.fits")
            outerr  = simmom0.replace(".fits","_noclip_masked_snr25.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image?")
            run_immath_two(infile,mask,outfile+".image1","IM0*IM1")
            immoments(imagename=outfile+".image1",outfile=outfile+".image2")
            run_exportfits(outfile+".image2",outfile,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + outfile + ".image?")
            # error
            run_immath_two(outfile,map_nchan,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

            # snr = 50
            infile  = modelcube.replace(".fits","_snr50.fits")
            outfile = simmom0.replace(".fits","_noclip_masked_snr50.fits")
            outerr  = simmom0.replace(".fits","_noclip_masked_snr50.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image?")
            run_immath_two(infile,mask,outfile+".image1","IM0*IM1")
            immoments(imagename=outfile+".image1",outfile=outfile+".image2")
            run_exportfits(outfile+".image2",outfile,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + outfile + ".image?")
            # error
            run_immath_two(outfile,map_nchan,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

        ####################
        # do_zeroclip_mask # emom0 done
        ####################
        includepix = [0.0,1000000.]
        if do_zeroclip_mask==True:
            # snr = 10
            infile  = modelcube.replace(".fits","_snr10.fits")
            outfile = simmom0.replace(".fits","_clip0_masked_snr10.fits")
            outerr  = simmom0.replace(".fits","_clip0_masked_snr10.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image?")
            run_immath_two(infile,mask,outfile+".image1","IM0*IM1")
            immoments(imagename=outfile+".image1",outfile=outfile+".image2",includepix=includepix)
            run_exportfits(outfile+".image2",outfile,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + outfile + ".image?")
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(infile,mask,this_mask+".image1","iif(IM0*IM1>0,1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

            # snr = 25
            infile  = modelcube.replace(".fits","_snr25.fits")
            outfile = simmom0.replace(".fits","_clip0_masked_snr25.fits")
            outerr  = simmom0.replace(".fits","_clip0_masked_snr25.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image?")
            run_immath_two(infile,mask,outfile+".image1","IM0*IM1")
            immoments(imagename=outfile+".image1",outfile=outfile+".image2",includepix=includepix)
            run_exportfits(outfile+".image2",outfile,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + outfile + ".image?")
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(infile,mask,this_mask+".image1","iif(IM0*IM1>0,1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

            # snr = 50
            infile  = modelcube.replace(".fits","_snr50.fits")
            outfile = simmom0.replace(".fits","_clip0_masked_snr50.fits")
            outerr  = simmom0.replace(".fits","_clip0_masked_snr50.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image?")
            run_immath_two(infile,mask,outfile+".image1","IM0*IM1")
            immoments(imagename=outfile+".image1",outfile=outfile+".image2",includepix=includepix)
            run_exportfits(outfile+".image2",outfile,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + outfile + ".image?")
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(infile,mask,this_mask+".image1","iif(IM0*IM1>0,1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

        ################
        # do_clip_mask # emom0 done
        ################
        includepix = [rms*self.snr_clip,1000000.]
        if do_clip_mask==True:
            # snr = 10
            infile  = modelcube.replace(".fits","_snr10.fits")
            outfile = simmom0.replace(".fits","_clip"+str(self.snr_clip)+"_masked_snr10.fits")
            outerr  = simmom0.replace(".fits","_clip"+str(self.snr_clip)+"_masked_snr10.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image?")
            run_immath_two(infile,mask,outfile+".image1","IM0*IM1")
            immoments(imagename=outfile+".image1",outfile=outfile+".image2",includepix=includepix)
            run_exportfits(outfile+".image2",outfile,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + outfile + ".image?")
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(infile,mask,this_mask+".image1","iif(IM0*IM1>"+str(rms*self.snr_clip)+",1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

            # snr = 25
            infile  = modelcube.replace(".fits","_snr25.fits")
            outfile = simmom0.replace(".fits","_clip"+str(self.snr_clip)+"_masked_snr25.fits")
            outerr  = simmom0.replace(".fits","_clip"+str(self.snr_clip)+"_masked_snr25.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image?")
            run_immath_two(infile,mask,outfile+".image1","IM0*IM1")
            immoments(imagename=outfile+".image1",outfile=outfile+".image2",includepix=includepix)
            run_exportfits(outfile+".image2",outfile,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + outfile + ".image?")
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(infile,mask,this_mask+".image1","iif(IM0*IM1>"+str(rms*self.snr_clip)+",1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

            # snr = 50
            infile  = modelcube.replace(".fits","_snr50.fits")
            outfile = simmom0.replace(".fits","_clip"+str(self.snr_clip)+"_masked_snr50.fits")
            outerr  = simmom0.replace(".fits","_clip"+str(self.snr_clip)+"_masked_snr50.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image?")
            run_immath_two(infile,mask,outfile+".image1","IM0*IM1")
            immoments(imagename=outfile+".image1",outfile=outfile+".image2",includepix=includepix)
            run_exportfits(outfile+".image2",outfile,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + outfile + ".image?")
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(infile,mask,this_mask+".image1","iif(IM0*IM1>"+str(rms*self.snr_clip)+",1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

    #######################
    # add_noise_to_models #
    #######################

    def add_noise_to_models(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outmodelcube_13co10,taskname)

        # create correlated noise
        noise1 = self._create_correlated_noise_cube(self.outmodelcube_13co10)
        noise2 = self._create_correlated_noise_cube(self.outmodelcube_13co10)
        noise3 = self._create_correlated_noise_cube(self.outmodelcube_13co10)
        noise4 = self._create_correlated_noise_cube(self.outmodelcube_13co10)
        noise5 = self._create_correlated_noise_cube(self.outmodelcube_13co10)
        noise6 = self._create_correlated_noise_cube(self.outmodelcube_13co10)

        # snr = 10
        model_snr = self.outmodelcube_13co10.replace(".fits","_snr10.fits")
        im      = pyfits.open(self.outmodelcube_13co10)
        im0     = im[0]
        newdata = im0.data + noise1
        os.system("rm -rf " + model_snr)
        pyfits.writeto(model_snr,data=newdata,header=im0.header)

        model_snr = self.outmodelcube_13co21.replace(".fits","_snr10.fits")
        im      = pyfits.open(self.outmodelcube_13co21)
        im0     = im[0]
        newdata = im0.data + noise2
        os.system("rm -rf " + model_snr)
        pyfits.writeto(model_snr,data=newdata,header=im0.header)

        # snr = 25
        model_snr = self.outmodelcube_13co10.replace(".fits","_snr25.fits")
        im      = pyfits.open(self.outmodelcube_13co10)
        im0     = im[0]
        newdata = im0.data * 2.5 + noise3
        os.system("rm -rf " + model_snr)
        pyfits.writeto(model_snr,data=newdata,header=im0.header)

        model_snr = self.outmodelcube_13co21.replace(".fits","_snr25.fits")
        im      = pyfits.open(self.outmodelcube_13co21)
        im0     = im[0]
        newdata = im0.data * 2.5 + noise4
        os.system("rm -rf " + model_snr)
        pyfits.writeto(model_snr,data=newdata,header=im0.header)

        # snr = 50
        model_snr = self.outmodelcube_13co10.replace(".fits","_snr50.fits")
        im      = pyfits.open(self.outmodelcube_13co10)
        im0     = im[0]
        newdata = im0.data * 5.0 + noise5
        os.system("rm -rf " + model_snr)
        pyfits.writeto(model_snr,data=newdata,header=im0.header)

        model_snr = self.outmodelcube_13co21.replace(".fits","_snr50.fits")
        im      = pyfits.open(self.outmodelcube_13co21)
        im0     = im[0]
        newdata = im0.data * 5.0 + noise6
        os.system("rm -rf " + model_snr)
        pyfits.writeto(model_snr,data=newdata,header=im0.header)

    ######################
    # create_model_cubes #
    ######################

    def create_model_cubes(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outcubes_13co10.replace("???","60pc"),taskname)

        # create 13co10 model cube
        nchan = imhead(self.outcubes_13co10.replace("???","60pc"),mode="list")["shape"][2]
        run_immath_two(
            self.outcubes_13co10.replace("???","60pc"),
            self.outecubes_13co10.replace("???","60pc"),
            self.outmodelcube_13co10 + "_tmp1",
            "iif(IM0/IM1>"+str(self.snr_model)+",IM0,0)",
            chans="1~" + str(nchan-2),
            )
        run_roundsmooth(
            self.outmodelcube_13co10 + "_tmp1",
            self.outmodelcube_13co10 + "_tmp2",
            1.666,
            delin=True,
            )
        run_exportfits(
            self.outmodelcube_13co10 + "_tmp2",
            self.outmodelcube_13co10,
            delin=True,
            velocity=False,
            )

        # create 13co21 model cube based on the 13co10 model
        maxval = str(imstat(self.outmodelcube_13co10)["max"][0])
        run_immath_one(
            self.outmodelcube_13co10,
            self.outmodelcube_13co21 + "_tmp1",
            "IM0*IM0/"+maxval,
            )

        run_exportfits(
            self.outmodelcube_13co21 + "_tmp1",
            self.outmodelcube_13co21,
            delin=True,
            velocity=False,
            )

    ###########
    # showsim #
    ###########

    def showsim(
        self,
        do_noclip=True,
        do_zeroclip=True,
        do_clip=True,
        do_noclip_mask=True,
        do_zeroclip_mask=True,
        do_clip_mask=True,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outsimumom0_13co10.replace(".fits","_just_snr10.fits"),taskname)

        # prepare
        imcontour1 = self.outmodelmom0_13co10

        ####################
        # model input mom0 #
        ####################
        self._showcase_one(
            self.outmodelmom0_13co10,
            imcontour1,
            self.outpng_modelmom0_13co10,
            "Input noise-free model mom0",
            "(K km s$^{-1}$)",
            [0,100],
            )

        maxnoise = imstat(self.outsimumom0_13co10.replace(".fits","_noclip_snr10.fits").replace("mom0","emom0"))["max"]
        #############
        # do_noclip #
        #############
        if do_noclip==True:
            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_noclip_snr10.fits"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_noclip_snr10.png"),
                "noclip: mom0$_{\mathrm{SNR=10}}$",
                "(K km s$^{-1}$)",
                [0,100],
                )
            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_noclip_snr10.fits").replace("mom0","emom0"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_noclip_snr10.png").replace("mom0","emom0"),
                "noclip: $\sigma_{\mathrm{err}}$(mom0$_{\mathrm{SNR=10}}$)",
                "(K km s$^{-1}$)",
                [0,maxnoise],
                )

            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_noclip_snr25.fits"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_noclip_snr25.png"),
                "noclip: mom0$_{\mathrm{SNR=25}}$",
                "(K km s$^{-1}$)",
                [0,250],
                )
            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_noclip_snr25.fits").replace("mom0","emom0"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_noclip_snr25.png").replace("mom0","emom0"),
                "noclip: $\sigma_{\mathrm{err}}$(mom0$_{\mathrm{SNR=25}}$)",
                "(K km s$^{-1}$)",
                [0,maxnoise],
                )

            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_noclip_snr50.fits"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_noclip_snr50.png"),
                "noclip: mom0$_{\mathrm{SNR=50}}$",
                "(K km s$^{-1}$)",
                [0,500],
                )
            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_noclip_snr50.fits").replace("mom0","emom0"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_noclip_snr50.png").replace("mom0","emom0"),
                "noclip: $\sigma_{\mathrm{err}}$(mom0$_{\mathrm{SNR=50}}$)",
                "(K km s$^{-1}$)",
                [0,maxnoise],
                )

        ###############
        # do_zeroclip #
        ###############
        if do_zeroclip==True:
            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_clip0_snr10.fits"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_clip0_snr10.png"),
                "clip0$\sigma$: mom0$_{\mathrm{SNR=10}}$",
                "(K km s$^{-1}$)",
                [0,100],
                )
            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_clip0_snr10.fits").replace("mom0","emom0"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_clip0_snr10.png").replace("mom0","emom0"),
                "clip0$\sigma$: $\sigma_{\mathrm{err}}$(mom0$_{\mathrm{SNR=10}}$)",
                "(K km s$^{-1}$)",
                [0,maxnoise],
                )

            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_clip0_snr25.fits"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_clip0_snr25.png"),
                "clip0$\sigma$: mom0$_{\mathrm{SNR=25}}$",
                "(K km s$^{-1}$)",
                [0,250],
                )
            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_clip0_snr25.fits").replace("mom0","emom0"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_clip0_snr25.png").replace("mom0","emom0"),
                "clip0$\sigma$: $\sigma_{\mathrm{err}}$(mom0$_{\mathrm{SNR=25}}$)",
                "(K km s$^{-1}$)",
                [0,maxnoise],
                )

            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_clip0_snr50.fits"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_clip0_snr50.png"),
                "clip0$\sigma$: mom0$_{\mathrm{SNR=50}}$",
                "(K km s$^{-1}$)",
                [0,500],
                )
            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_clip0_snr50.fits").replace("mom0","emom0"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_clip0_snr50.png").replace("mom0","emom0"),
                "clip0$\sigma$: $\sigma_{\mathrm{err}}$(mom0$_{\mathrm{SNR=50}}$)",
                "(K km s$^{-1}$)",
                [0,maxnoise],
                )

        ###########
        # do_clip #
        ###########
        if do_clip==True:
            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_clip3_snr10.fits"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_clip3_snr10.png"),
                "clip3$\sigma$: mom0$_{\mathrm{SNR=10}}$",
                "(K km s$^{-1}$)",
                [0,100],
                )
            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_clip3_snr10.fits").replace("mom0","emom0"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_clip3_snr10.png").replace("mom0","emom0"),
                "clip3$\sigma$: $\sigma_{\mathrm{err}}$(mom0$_{\mathrm{SNR=10}}$)",
                "(K km s$^{-1}$)",
                [0,maxnoise],
                )

            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_clip3_snr25.fits"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_clip3_snr25.png"),
                "clip3$\sigma$: mom0$_{\mathrm{SNR=25}}$",
                "(K km s$^{-1}$)",
                [0,250],
                )
            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_clip3_snr25.fits").replace("mom0","emom0"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_clip3_snr25.png").replace("mom0","emom0"),
                "clip3$\sigma$: $\sigma_{\mathrm{err}}$(mom0$_{\mathrm{SNR=25}}$)",
                "(K km s$^{-1}$)",
                [0,maxnoise],
                )

            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_clip3_snr50.fits"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_clip3_snr50.png"),
                "clip3$\sigma$: mom0$_{\mathrm{SNR=50}}$",
                "(K km s$^{-1}$)",
                [0,500],
                )
            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_clip3_snr50.fits").replace("mom0","emom0"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_clip3_snr50.png").replace("mom0","emom0"),
                "clip3$\sigma$: $\sigma_{\mathrm{err}}$(mom0$_{\mathrm{SNR=50}}$)",
                "(K km s$^{-1}$)",
                [0,maxnoise],
                )

        ##################
        # do_noclip_mask #
        ##################
        if do_noclip_mask==True:
            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_noclip_masked_snr10.fits"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_noclip_masked_snr10.png"),
                "noclip+masking: mom0$_{\mathrm{SNR=10}}$",
                "(K km s$^{-1}$)",
                [0,100],
                )
            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_noclip_masked_snr10.fits").replace("mom0","emom0"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_noclip_masked_snr10.png").replace("mom0","emom0"),
                "noclip+masking: $\sigma_{\mathrm{err}}$(mom0$_{\mathrm{SNR=10}}$)",
                "(K km s$^{-1}$)",
                [0,maxnoise],
                )

            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_noclip_masked_snr25.fits"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_noclip_masked_snr25.png"),
                "noclip+masking: mom0$_{\mathrm{SNR=25}}$",
                "(K km s$^{-1}$)",
                [0,250],
                )
            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_noclip_masked_snr25.fits").replace("mom0","emom0"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_noclip_masked_snr25.png").replace("mom0","emom0"),
                "noclip+masking: $\sigma_{\mathrm{err}}$(mom0$_{\mathrm{SNR=25}}$)",
                "(K km s$^{-1}$)",
                [0,maxnoise],
                )

            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_noclip_masked_snr50.fits"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_noclip_masked_snr50.png"),
                "noclip+masking: mom0$_{\mathrm{SNR=50}}$",
                "(K km s$^{-1}$)",
                [0,500],
                )
            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_noclip_masked_snr50.fits").replace("mom0","emom0"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_noclip_masked_snr50.png").replace("mom0","emom0"),
                "noclip+masking: $\sigma_{\mathrm{err}}$(mom0$_{\mathrm{SNR=50}}$)",
                "(K km s$^{-1}$)",
                [0,maxnoise],
                )

        ####################
        # do_zeroclip_mask #
        ####################
        if do_zeroclip_mask==True:
            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_clip0_masked_snr10.fits"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_clip0_masked_snr10.png"),
                "clip0$\sigma$+masking: mom0$_{\mathrm{SNR=10}}$",
                "(K km s$^{-1}$)",
                [0,100],
                )
            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_clip0_masked_snr10.fits").replace("mom0","emom0"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_clip0_masked_snr10.png").replace("mom0","emom0"),
                "clip0$\sigma$+masking: $\sigma_{\mathrm{err}}$(mom0$_{\mathrm{SNR=10}}$)",
                "(K km s$^{-1}$)",
                [0,maxnoise],
                )

            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_clip0_masked_snr25.fits"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_clip0_masked_snr25.png"),
                "clip0$\sigma$+masking: mom0$_{\mathrm{SNR=25}}$",
                "(K km s$^{-1}$)",
                [0,250],
                )
            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_clip0_masked_snr25.fits").replace("mom0","emom0"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_clip0_masked_snr25.png").replace("mom0","emom0"),
                "clip0$\sigma$+masking: $\sigma_{\mathrm{err}}$(mom0$_{\mathrm{SNR=25}}$)",
                "(K km s$^{-1}$)",
                [0,maxnoise],
                )

            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_clip0_masked_snr50.fits"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_clip0_masked_snr50.png"),
                "clip0$\sigma$+masking: mom0$_{\mathrm{SNR=50}}$",
                "(K km s$^{-1}$)",
                [0,500],
                )
            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_clip0_masked_snr50.fits").replace("mom0","emom0"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_clip0_masked_snr50.png").replace("mom0","emom0"),
                "clip0$\sigma$+masking: $\sigma_{\mathrm{err}}$(mom0$_{\mathrm{SNR=50}}$)",
                "(K km s$^{-1}$)",
                [0,maxnoise],
                )

        ################
        # do_clip_mask #
        ################
        if do_clip_mask==True:
            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_clip3_masked_snr10.fits"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_clip3_masked_snr10.png"),
                "clip3$\sigma$+masking: mom0$_{\mathrm{SNR=10}}$",
                "(K km s$^{-1}$)",
                [0,100],
                )
            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_clip3_masked_snr10.fits").replace("mom0","emom0"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_clip3_masked_snr10.png").replace("mom0","emom0"),
                "clip3$\sigma$+masking: $\sigma_{\mathrm{err}}$(mom0$_{\mathrm{SNR=10}}$)",
                "(K km s$^{-1}$)",
                [0,maxnoise],
                )

            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_clip3_masked_snr25.fits"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_clip3_masked_snr25.png"),
                "clip3$\sigma$+masking: mom0$_{\mathrm{SNR=25}}$",
                "(K km s$^{-1}$)",
                [0,250],
                )
            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_clip3_masked_snr25.fits").replace("mom0","emom0"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_clip3_masked_snr25.png").replace("mom0","emom0"),
                "clip3$\sigma$+masking: $\sigma_{\mathrm{err}}$(mom0$_{\mathrm{SNR=25}}$)",
                "(K km s$^{-1}$)",
                [0,maxnoise],
                )

            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_clip3_masked_snr50.fits"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_clip3_masked_snr50.png"),
                "clip3$\sigma$+masking: mom0$_{\mathrm{SNR=50}}$",
                "(K km s$^{-1}$)",
                [0,500],
                )
            self._showcase_one(
                self.outsimumom0_13co10.replace(".fits","_clip3_masked_snr50.fits").replace("mom0","emom0"),
                imcontour1,
                self.outpng_simumom0_13co10.replace(".png","_clip3_masked_snr50.png").replace("mom0","emom0"),
                "clip3$\sigma$+masking: $\sigma_{\mathrm{err}}$(mom0$_{\mathrm{SNR=50}}$)",
                "(K km s$^{-1}$)",
                [0,maxnoise],
                )

    ############
    # showcase #
    ############

    def showcase(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outmaps_13co_trot.replace("???","60pc"),taskname)

        scalebar = 100. / self.scale_pc
        label_scalebar = "100 pc"

        for this_beam in self.beams:
            #if this_beam!="60pc":
            #    continue

            print("# myfig_fits2png at " + this_beam)
            imcontour1 = self.outmaps_mom0_13co21.replace("???",this_beam)

            # 13co10 mom0
            maxval = imstat(self.outmaps_mom0_13co10.replace("???",this_beam))["max"]
            self._showcase_one(
                self.outmaps_mom0_13co10.replace("???",this_beam),
                imcontour1,
                self.outpng_mom0_13co10.replace("???",this_beam),
                "$I_{\mathrm{^{13}CO(1-0)}}$ at " + this_beam.replace("pc"," pc"),
                "(K km s$^{-1}$)",
                [0,maxval],
                )

            # 13co10 mom0 err
            maxval = imstat(self.outemaps_mom0_13co10.replace("???",this_beam))["max"]
            self._showcase_one(
                self.outemaps_mom0_13co10.replace("???",this_beam),
                imcontour1,
                self.outpng_emom0_13co10.replace("???",this_beam),
                "$\sigma_{\mathrm{err}}$($I_{\mathrm{^{13}CO(1-0)}}$) at " + this_beam.replace("pc"," pc"),
                "(K km s$^{-1}$)",
                [0,maxval],
                )

            # 13co21 mom0
            maxval = imstat(self.outmaps_mom0_13co21.replace("???",this_beam))["max"]
            self._showcase_one(
                self.outmaps_mom0_13co21.replace("???",this_beam),
                imcontour1,
                self.outpng_mom0_13co21.replace("???",this_beam),
                "$I_{\mathrm{^{13}CO(2-1)}}$ at " + this_beam.replace("pc"," pc"),
                "(K km s$^{-1}$)",
                [0,maxval],
                )

            # 13co21 mom0 err
            maxval = imstat(self.outemaps_mom0_13co21.replace("???",this_beam))["max"]
            self._showcase_one(
                self.outemaps_mom0_13co21.replace("???",this_beam),
                imcontour1,
                self.outpng_emom0_13co21.replace("???",this_beam),
                "$\sigma_{\mathrm{err}}$($I_{\mathrm{^{13}CO(2-1)}}$) at " + this_beam.replace("pc"," pc"),
                "(K km s$^{-1}$)",
                [0,maxval],
                )

            # ratio
            maxval = imstat(self.outmaps_ratio.replace("???",this_beam))["max"]
            self._showcase_one(
                self.outmaps_ratio.replace("???",this_beam),
                imcontour1,
                self.outpng_ratio.replace("???",this_beam),
                "Ratio at " + this_beam.replace("pc"," pc"),
                "Ratio",
                [0,maxval],
                )

            # ratio error
            maxval = imstat(self.outemaps_ratio.replace("???",this_beam))["max"]
            self._showcase_one(
                self.outemaps_ratio.replace("???",this_beam),
                imcontour1,
                self.outpng_eratio.replace("???",this_beam),
                "$\sigma_{\mathrm{err}}$(Ratio) at " + this_beam.replace("pc"," pc"),
                "(K km s$^{-1}$)",
                [0,maxval],
                )

            # mom1
            self._showcase_one(
                self.outmaps_mom1.replace("???",this_beam),
                imcontour1,
                self.outpng_mom1.replace("???",this_beam),
                "$v_{\mathrm{los}}$ at " + this_beam.replace("pc"," pc"),
                "(km s$^{-1}$)",
                )

            # mom1 error
            self._showcase_one(
                self.outemaps_mom1.replace("???",this_beam),
                imcontour1,
                self.outpng_emom1.replace("???",this_beam),
                "$\sigma_{\mathrm{err}}$($v_{\mathrm{los}}$) at " + this_beam.replace("pc"," pc"),
                "(km s$^{-1}$)",
                )

            # mom2
            maxval = imstat(self.outmaps_mom2.replace("???",this_beam))["max"]
            self._showcase_one(
                self.outmaps_mom2.replace("???",this_beam),
                imcontour1,
                self.outpng_mom2.replace("???",this_beam),
                "$\sigma_{v}$ at " + this_beam.replace("pc"," pc"),
                "(km s$^{-1}$)",
                [0,maxval],
                )

            # mom2 error
            maxval = imstat(self.outemaps_mom2.replace("???",this_beam))["max"]
            self._showcase_one(
                self.outemaps_mom2.replace("???",this_beam),
                imcontour1,
                self.outpng_emom2.replace("???",this_beam),
                "$\sigma_{\mathrm{err}}$($\sigma_{v}$) at " + this_beam.replace("pc"," pc"),
                "(km s$^{-1}$)",
                [0,maxval],
                )

            # Trot
            self._showcase_one(
                self.outmaps_13co_trot.replace("???",this_beam),
                imcontour1,
                self.outpng_13co_trot.replace("???",this_beam),
                "$T_{\mathrm{rot}}$ at " + this_beam.replace("pc"," pc"),
                "(K)",
                clim=[2.73,8],
                )

            # Trot error
            maxval = imstat(self.outemaps_13co_trot.replace("???",this_beam))["max"]
            self._showcase_one(
                self.outemaps_13co_trot.replace("???",this_beam),
                imcontour1,
                self.outpng_e13co_trot.replace("???",this_beam),
                "$\sigma_{\mathrm{err}}$($T_{\mathrm{rot}}$) at " + this_beam.replace("pc"," pc"),
                "(K)",
                [0,maxval],
                )

            # log N13co
            self._showcase_one(
                self.outmaps_13co_ncol.replace("???",this_beam),
                imcontour1,
                self.outpng_13co_ncol.replace("???",this_beam),
                "log$_{\mathrm{10}}$ $N_{\mathrm{^{13}CO}}$ at " + this_beam.replace("pc"," pc"),
                "(cm$^{-2}$ in log$_{\mathrm{10}}$)",
                )

            # log N13co error
            self._showcase_one(
                self.outemaps_13co_ncol.replace("???",this_beam),
                imcontour1,
                self.outpng_e13co_ncol.replace("???",this_beam),
                "$\sigma_{\mathrm{err}}$(log$_{\mathrm{10}}$ $N_{\mathrm{^{13}CO}}$) at " + this_beam.replace("pc"," pc"),
                "(cm$^{-2}$ in log$_{\mathrm{10}}$)",
                )

    #################
    # multi_fitting #
    #################

    def multi_fitting(self):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outcubes_13co10.replace("???","60pc"),taskname)

        for this_beam in self.beams:
            if this_beam!="60pc":
                continue

            print("# multi_fitting for cubes at " + this_beam)

            # input
            cube_13co10  = self.outcubes_13co10.replace("???",this_beam)
            cube_13co21  = self.outcubes_13co21.replace("???",this_beam)
            ecube_13co10 = self.outecubes_13co10.replace("???",this_beam)
            ecube_13co21 = self.outecubes_13co21.replace("???",this_beam)

            # output
            self.list_qqdata = rotation_13co21_13co10(
                cube_13co10,
                cube_13co21,
                ecube_13co10,
                ecube_13co21,
                ra_cnt=self.ra_agn,
                dec_cnt=self.dec_agn,
                snr=self.snr_fit,
                snr_limit=self.snr_fit,
                )

            #
            os.system("mv mom0_low.fits " + self.outmaps_mom0_13co10.replace("???",this_beam))
            os.system("mv mom0_high.fits " + self.outmaps_mom0_13co21.replace("???",this_beam))
            os.system("mv mom1.fits " + self.outmaps_mom1.replace("???",this_beam))
            os.system("mv mom2.fits " + self.outmaps_mom2.replace("???",this_beam))
            os.system("mv ratio.fits " + self.outmaps_ratio.replace("???",this_beam))
            os.system("mv Trot.fits " + self.outmaps_13co_trot.replace("???",this_beam))
            os.system("mv logN.fits " + self.outmaps_13co_ncol.replace("???",this_beam))
            os.system("mv residual_snr.fits " + self.outmaps_residual.replace("???",this_beam))

            #
            os.system("mv emom0_low.fits " + self.outemaps_mom0_13co10.replace("???",this_beam))
            os.system("mv emom0_high.fits " + self.outemaps_mom0_13co21.replace("???",this_beam))
            os.system("mv emom1.fits " + self.outemaps_mom1.replace("???",this_beam))
            os.system("mv emom2.fits " + self.outemaps_mom2.replace("???",this_beam))
            os.system("mv eratio.fits " + self.outemaps_ratio.replace("???",this_beam))
            os.system("mv eTrot.fits " + self.outemaps_13co_trot.replace("???",this_beam))
            os.system("mv elogN.fits " + self.outemaps_13co_ncol.replace("???",this_beam))

            #
            os.system("mv mom0_low_all.fits " + self.outmaps_mom0_13co10.replace("???","all_"+this_beam))
            os.system("mv mom0_high_all.fits " + self.outmaps_mom0_13co21.replace("???","all_"+this_beam))
            os.system("mv mom1_all.fits " + self.outmaps_mom1.replace("???","all_"+this_beam))
            os.system("mv mom2_all.fits " + self.outmaps_mom2.replace("???","all_"+this_beam))
            os.system("mv ratio_all.fits " + self.outmaps_ratio.replace("???","all_"+this_beam))
            os.system("mv Trot_all.fits " + self.outmaps_13co_trot.replace("???","all_"+this_beam))
            os.system("mv logN_all.fits " + self.outmaps_13co_ncol.replace("???","all_"+this_beam))

            #
            os.system("mv emom0_low_all.fits " + self.outemaps_mom0_13co10.replace("???","all_"+this_beam))
            os.system("mv emom0_high_all.fits " + self.outemaps_mom0_13co21.replace("???","all_"+this_beam))
            os.system("mv emom1_all.fits " + self.outemaps_mom1.replace("???","all_"+this_beam))
            os.system("mv emom2_all.fits " + self.outemaps_mom2.replace("???","all_"+this_beam))
            os.system("mv eratio_all.fits " + self.outemaps_ratio.replace("???","all_"+this_beam))
            os.system("mv eTrot_all.fits " + self.outemaps_13co_trot.replace("???","all_"+this_beam))
            os.system("mv elogN_all.fits " + self.outemaps_13co_ncol.replace("???","all_"+this_beam))

            # plot qq-plot
            xlim     = ylim = [-0.15,1.15]
            title    = "Quantile-Quantile plot"
            xlabel   = "Gaussian quantiles"
            ylabel   = "Observation quantiles"
            cblabel  = "Residual-to-Gaussian intensity ratio"
            vmax     = 1
            list_res = np.array([s[:,2][0] for s in self.list_qqdata])

            fig = plt.figure(figsize=(13,10))
            gs  = gridspec.GridSpec(nrows=10, ncols=10)
            ax1 = plt.subplot(gs[0:10,0:10])
            ad  = [0.215,0.83,0.10,0.90]
            myax_set(ax1, "both", xlim, ylim, title, xlabel, ylabel, adjust=ad)

            for this_qqdata in self.list_qqdata:
                this_x = this_qqdata[:,0]
                this_y = this_qqdata[:,1]
                this_res_snr = this_qqdata[:,2][0]
                this_c = cm.rainbow(this_res_snr)

                #ax1.plot(this_x, this_y, color=this_c, lw=2, marker=None, alpha=0.3)
                if this_res_snr>=0.2:
                    ax1.plot(this_x, this_y, color="tomato", lw=1, marker=None, alpha=1.0)
                else:
                    ax1.plot(this_x, this_y, color="grey", lw=1, marker=None, alpha=0.3, zorder=1e9)

            #cs = ax1.scatter(list_res*0+2, list_res, c=list_res, cmap="rainbow", s=1, vmin=0, vmax=vmax)

            # colorbar
            #cax = fig.add_axes([0.25, 0.81, 0.33, 0.04])
            #cbar = plt.colorbar(cs, cax=cax, orientation="horizontal")
            #cbar.set_label(cblabel)
            #cbar.set_ticks([0,0.2,0.4,0.6,0.8,1.0])

            # ann
            ax1.plot(xlim,ylim,"--",color="black",lw=1,zorder=1e10)

            # save
            os.system("rm -rf " + self.outpng_qqplot.replace("???",this_beam))
            plt.savefig(self.outpng_qqplot.replace("???",this_beam), dpi=self.fig_dpi)

            self._showcase_one(
                self.outmaps_residual.replace("???",this_beam),
                self.outmaps_mom0_13co21.replace("???",this_beam),
                self.outpng_residual.replace("???",this_beam),
                "Residual-to-Gaussian intensity ratio",
                "Ratio",
                clim=[0,0.2],
                )

        """ hcn10-hcop10 ratio case
        for this_beam in ["60pc"]:
            print("# multi_fitting for cubes at " + this_beam)

            # input
            cube_hcn10   = self.outcubes_hcn10.replace("???",this_beam)
            cube_hcop10  = self.outcubes_hcop10.replace("???",this_beam)
            ecube_hcn10  = self.outecubes_hcn10.replace("???",this_beam)
            ecube_hcop10 = self.outecubes_hcop10.replace("???",this_beam)

            # output
            fit_two_lines(
                cube_hcop10,
                cube_hcn10,
                ecube_hcop10,
                ecube_hcn10,
                ra_cnt=self.ra_agn,
                dec_cnt=self.dec_agn,
                snr=4.0,
                snr_limit=4.0,
                ratio_max=4.0,
                )

            #
            os.system("mv mom0_low.fits " + self.outmaps_mom0_hcop10.replace("???",this_beam))
            os.system("mv mom0_high.fits " + self.outmaps_mom0_hcn10.replace("???",this_beam))
            os.system("mv mom1.fits " + self.outmaps_hcn10_mom1.replace("???",this_beam))
            os.system("mv mom2.fits " + self.outmaps_hcn10_mom2.replace("???",this_beam))
            os.system("mv ratio.fits " + self.outmaps_hcn10_ratio.replace("???",this_beam))

            #
            os.system("mv emom0_low.fits " + self.outemaps_mom0_hcop10.replace("???",this_beam))
            os.system("mv emom0_high.fits " + self.outemaps_mom0_hcn10.replace("???",this_beam))
            os.system("mv emom1.fits " + self.outemaps_hcn10_mom1.replace("???",this_beam))
            os.system("mv emom2.fits " + self.outemaps_hcn10_mom2.replace("???",this_beam))
            os.system("mv eratio.fits " + self.outemaps_hcn10_ratio.replace("???",this_beam))

            #
            os.system("mv mom0_low_all.fits " + self.outmaps_mom0_hcop10.replace("???","all_"+this_beam))
            os.system("mv mom0_high_all.fits " + self.outmaps_mom0_hcn10.replace("???","all_"+this_beam))
            os.system("mv mom1_all.fits " + self.outmaps_hcn10_mom1.replace("???","all_"+this_beam))
            os.system("mv mom2_all.fits " + self.outmaps_hcn10_mom2.replace("???","all_"+this_beam))
            os.system("mv ratio_all.fits " + self.outmaps_hcn10_ratio.replace("???","all_"+this_beam))

            #
            os.system("mv emom0_low_all.fits " + self.outemaps_mom0_hcop10.replace("???","all_"+this_beam))
            os.system("mv emom0_high_all.fits " + self.outemaps_mom0_hcn10.replace("???","all_"+this_beam))
            os.system("mv emom1_all.fits " + self.outemaps_hcn10_mom1.replace("???","all_"+this_beam))
            os.system("mv emom2_all.fits " + self.outemaps_hcn10_mom2.replace("???","all_"+this_beam))
            os.system("mv eratio_all.fits " + self.outemaps_hcn10_ratio.replace("???","all_"+this_beam))
        """

    ##############
    # align_maps #
    ##############

    def align_maps(self):
        """
        """

        for this_beam in self.beams:
            print("# align_maps for cubes at " + this_beam)
            self._align_maps_at_a_res(
                self.cube_13co10.replace("???",this_beam),
                self.cube_13co21.replace("???",this_beam),
                self.outcubes_13co10.replace("???",this_beam),
                self.outcubes_13co21.replace("???",this_beam),
                self.ecube_13co10.replace("???",this_beam),
                self.ecube_13co21.replace("???",this_beam),
                self.outecubes_13co10.replace("???",this_beam),
                self.outecubes_13co21.replace("???",this_beam),
                )
 
        """
        # hcn10 hcop10 case
        for this_beam in ["60pc"]:
            print("# align_maps for cubes at " + this_beam)
            self._align_maps_at_a_res(
                self.cube_hcn10.replace("???",this_beam),
                self.cube_hcop10.replace("???",this_beam),
                self.outcubes_hcn10.replace("???",this_beam),
                self.outcubes_hcop10.replace("???",this_beam),
                self.ecube_hcn10.replace("???",this_beam),
                self.ecube_hcop10.replace("???",this_beam),
                self.outecubes_hcn10.replace("???",this_beam),
                self.outecubes_hcop10.replace("???",this_beam),
                )
        """

    ##################
    # _fits_creation #
    ##################

    def _fits_creation(
        self,
        input_array,
        output_map,
        coords_template,
        bunit="K",
        ):
        """
        Reference:
        https://stackoverflow.com/questions/45744394/write-a-new-fits-file-after-modification-in-pixel-values
        """
        print(output_map)
        os.system("rm -rf " + output_map)
        os.system("cp " + coords_template + " " + output_map)

        obj = pyfits.open(output_map)
        obj[0].data = input_array
        obj[0].header.append(("BUNIT", bunit))
        obj.writeto(output_map, clobber=True)

    ##################
    # _plot_scatter5 #
    ##################

    def _plot_scatter5(
        self,
        ximage,
        xerrimage,
        yimage,
        yerrimage,
        cimage,
        cerrimage,
        outpng,
        xlim,
        ylim,
        title,
        xlabel,
        ylabel,
        cblabel,
        factor,
        cmap="rainbow_r",
        outfits_P=None,
        outfits_vir=None,
        templatefits=None,
        beamr=30, # pc
        ):

        unit_conv = 2 * 3.24078**-2 * 10**38 / (6.02*10**23 * 1.9884 * 10**33)

        # 13co10
        data_13co10,box = imval_all(ximage)
        data_13co10     = data_13co10["data"] * data_13co10["mask"]
        data_13co10     = data_13co10.flatten()
        data_13co10[np.isnan(data_13co10)] = 0

        err_13co10,_ = imval_all(xerrimage)
        err_13co10   = err_13co10["data"] * err_13co10["mask"]
        err_13co10   = err_13co10.flatten()
        err_13co10[np.isnan(err_13co10)] = 0

        # 13co21
        data_13co21,_ = imval_all(yimage)
        data_13co21   = data_13co21["data"] * data_13co21["mask"]
        data_13co21   = data_13co21.flatten()
        data_13co21[np.isnan(data_13co21)] = 0

        err_13co21,_ = imval_all(yerrimage)
        err_13co21   = err_13co21["data"] * err_13co21["mask"]
        err_13co21   = err_13co21.flatten()
        err_13co21[np.isnan(err_13co21)] = 0

        if cimage==None:
            # coords
            data_coords = imval(ximage,box=box)["coords"]
            ra_deg      = data_coords[:,:,0] * 180/np.pi
            ra_deg      = ra_deg.flatten()
            dec_deg     = data_coords[:,:,1] * 180/np.pi
            dec_deg     = dec_deg.flatten()
            dist_pc,_   = get_reldist_pc(ra_deg, dec_deg, self.ra_agn, self.dec_agn, self.scale_pc, 0, 0)
            c           = dist_pc / 1000.0
            # prepare
            cut  = np.where((data_13co10>abs(err_13co10)*self.snr)&(data_13co21>abs(err_13co21)*self.snr))
            x    = data_13co10[cut] + np.log10(factor) + np.log10(unit_conv)
            xerr = err_13co10[cut]
            y    = np.log10(data_13co21[cut])
            yerr = err_13co21[cut] / abs(data_13co21[cut])
            c    = np.array(c)[cut]

        else:
            data_c,_    = imval_all(cimage)
            data_c      = data_c["data"] * data_c["mask"]
            data_c      = data_c.flatten()
            data_c[np.isnan(data_c)] = 0
            c           = data_c
            data_cerr,_ = imval_all(cerrimage)
            data_cerr   = data_cerr["data"] * data_cerr["mask"]
            data_cerr   = data_cerr.flatten()
            data_cerr[np.isnan(data_cerr)] = 0
            cerr        = data_cerr
            # prepare
            cut  = np.where((data_13co10>abs(err_13co10)*self.snr)&(data_13co21>abs(err_13co21)*self.snr)&(c>abs(cerr)*self.snr))
            x    = data_13co10[cut] + np.log10(factor) + np.log10(unit_conv)
            xerr = err_13co10[cut]
            y    = np.log10(data_13co21[cut])
            yerr = err_13co21[cut] / abs(data_13co21[cut])
            c    = np.array(c)[cut]

        # plot
        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        ad  = [0.215,0.83,0.10,0.90]
        myax_set(ax1, None, xlim, ylim, title, xlabel, ylabel, adjust=ad)

        cs = ax1.scatter(x, y, c=c, cmap=cmap, lw=0, s=40, zorder=1e9)
        ax1.errorbar(x, y, xerr=xerr, yerr=yerr, lw=1, capsize=0, color="grey", linestyle="None")

        # colorbar
        cax = fig.add_axes([0.25, 0.81, 0.33, 0.04])
        cbar = plt.colorbar(cs, cax=cax, orientation="horizontal")
        cbar.set_label(cblabel)
        if cimage==None:
            cbar.set_ticks([0,0.3,0.6,0.9,1.2])

        # ann
        # virial paramter: eq 13 of Sun et al. 2018
        vir1 = [np.log10(np.sqrt(1.0/5.77*10**xlim[0]*beamr/40.)),np.log10(np.sqrt(1.0/5.77*10**xlim[1]*beamr/40.))]
        vir2 = [np.log10(np.sqrt(2.0/5.77*10**xlim[0]*beamr/40.)),np.log10(np.sqrt(2.0/5.77*10**xlim[1]*beamr/40.))]
        ax1.plot([xlim[0],xlim[1]],[vir1[0],vir1[1]],"--",lw=1,color="black")
        ax1.plot([xlim[0],xlim[1]],[vir2[0],vir2[1]],"--",lw=1,color="black")

        # internal pressure: eq 15 of Sun et al. 2018
        p3 = [np.log10(np.sqrt(10**3/61.3/10**xlim[0]*beamr/40.)),np.log10(np.sqrt(10**3/61.3/10**xlim[1]*beamr/40.))]
        p4 = [np.log10(np.sqrt(10**4/61.3/10**xlim[0]*beamr/40.)),np.log10(np.sqrt(10**4/61.3/10**xlim[1]*beamr/40.))]
        p5 = [np.log10(np.sqrt(10**5/61.3/10**xlim[0]*beamr/40.)),np.log10(np.sqrt(10**5/61.3/10**xlim[1]*beamr/40.))]
        p6 = [np.log10(np.sqrt(10**6/61.3/10**xlim[0]*beamr/40.)),np.log10(np.sqrt(10**6/61.3/10**xlim[1]*beamr/40.))]
        p7 = [np.log10(np.sqrt(10**7/61.3/10**xlim[0]*beamr/40.)),np.log10(np.sqrt(10**7/61.3/10**xlim[1]*beamr/40.))]
        p8 = [np.log10(np.sqrt(10**8/61.3/10**xlim[0]*beamr/40.)),np.log10(np.sqrt(10**8/61.3/10**xlim[1]*beamr/40.))]
        p9 = [np.log10(np.sqrt(10**9/61.3/10**xlim[0]*beamr/40.)),np.log10(np.sqrt(10**9/61.3/10**xlim[1]*beamr/40.))]
        ax1.plot([xlim[0],xlim[1]],[p3[0],p3[1]],":",lw=1,color="black")
        ax1.plot([xlim[0],xlim[1]],[p4[0],p4[1]],":",lw=1,color="black")
        ax1.plot([xlim[0],xlim[1]],[p5[0],p5[1]],":",lw=1,color="black")
        ax1.plot([xlim[0],xlim[1]],[p6[0],p6[1]],":",lw=1,color="black")
        ax1.plot([xlim[0],xlim[1]],[p7[0],p7[1]],":",lw=1,color="black")
        ax1.plot([xlim[0],xlim[1]],[p8[0],p8[1]],":",lw=1,color="black")
        ax1.plot([xlim[0],xlim[1]],[p9[0],p9[1]],":",lw=1,color="black")

        # save
        os.system("rm -rf " + outpng)
        plt.savefig(outpng, dpi=self.fig_dpi)

        #
        if outfits_P!=None:
            # 13co10
            data_13co10,box = imval_all(ximage)
            data_13co10     = data_13co10["data"] * data_13co10["mask"]
            data_13co10[np.isnan(data_13co10)] = 0

            err_13co10,_ = imval_all(xerrimage)
            err_13co10   = err_13co10["data"] * err_13co10["mask"]
            err_13co10[np.isnan(err_13co10)] = 0

            # 13co21
            data_13co21,_ = imval_all(yimage)
            data_13co21   = data_13co21["data"] * data_13co21["mask"]
            data_13co21[np.isnan(data_13co21)] = 0

            err_13co21,_ = imval_all(yerrimage)
            err_13co21   = err_13co21["data"] * err_13co21["mask"]
            err_13co21[np.isnan(err_13co21)] = 0

            x    = data_13co10 + np.log10(factor) + np.log10(unit_conv)
            xerr = err_13co10
            y    = data_13co21
            yerr = err_13co21

            outarray = np.log10(61.3 * 10**x * y**2 / (beamr/40.))
            outarray = np.rot90(np.fliplr( np.where((x>abs(xerr)*self.snr)&(y>abs(yerr)*self.snr),outarray,np.nan) ))

            self._fits_creation(
                input_array=outarray,
                output_map=outfits_P,
                coords_template=templatefits,
                bunit="K",
                )

        if outfits_vir!=None:
            # 13co10
            data_13co10,box = imval_all(ximage)
            data_13co10     = data_13co10["data"] * data_13co10["mask"]
            data_13co10[np.isnan(data_13co10)] = 0

            err_13co10,_ = imval_all(xerrimage)
            err_13co10   = err_13co10["data"] * err_13co10["mask"]
            err_13co10[np.isnan(err_13co10)] = 0

            # 13co21
            data_13co21,_ = imval_all(yimage)
            data_13co21   = data_13co21["data"] * data_13co21["mask"]
            data_13co21[np.isnan(data_13co21)] = 0

            err_13co21,_ = imval_all(yerrimage)
            err_13co21   = err_13co21["data"] * err_13co21["mask"]
            err_13co21[np.isnan(err_13co21)] = 0

            x    = data_13co10 + np.log10(factor) + np.log10(unit_conv)
            xerr = err_13co10
            y    = data_13co21
            yerr = err_13co21

            outarray = np.log10(5.77 * y**2 / 10**x / (beamr/40.))
            outarray = np.rot90(np.fliplr( np.where((x>abs(xerr)*self.snr)&(y>abs(yerr)*self.snr),outarray,np.nan) ))

            self._fits_creation(
                input_array=outarray,
                output_map=outfits_vir,
                coords_template=templatefits,
                bunit="K",
                )

        # 13co10
        data_13co10,box = imval_all(ximage)
        data_13co10     = data_13co10["data"] * data_13co10["mask"]
        data_13co10     = data_13co10.flatten()
        data_13co10[np.isnan(data_13co10)] = 0

        err_13co10,_ = imval_all(xerrimage)
        err_13co10   = err_13co10["data"] * err_13co10["mask"]
        err_13co10   = err_13co10.flatten()
        err_13co10[np.isnan(err_13co10)] = 0

        # 13co21
        data_13co21,_ = imval_all(yimage)
        data_13co21   = data_13co21["data"] * data_13co21["mask"]
        data_13co21   = data_13co21.flatten()
        data_13co21[np.isnan(data_13co21)] = 0

        err_13co21,_ = imval_all(yerrimage)
        err_13co21   = err_13co21["data"] * err_13co21["mask"]
        err_13co21   = err_13co21.flatten()
        err_13co21[np.isnan(err_13co21)] = 0

        cut  = np.where((data_13co10>abs(err_13co10)*self.snr)&(data_13co21>abs(err_13co21)*self.snr))
        log_Sh2     = data_13co10[cut] + np.log10(factor) + np.log10(unit_conv)
        log_Sh2_err = err_13co10[cut]
        mom2        = data_13co21[cut]
        mom2_err    = err_13co21[cut]

        return log_Sh2, log_Sh2_err, mom2, mom2_err, c

    ##################
    # _plot_scatter4 #
    ##################

    def _plot_scatter4(
        self,
        ximage,
        xerrimage,
        yimage,
        yerrimage,
        cimage,
        cerrimage,
        outpng,
        xlim,
        ylim,
        title,
        xlabel,
        ylabel,
        cblabel,
        factor,
        cmap="rainbow_r",
        outfits=None,
        outefits=None,
        templatefits=None,
        ):
        """
        use only for aco
        """

        # 13co10
        data_13co10,box = imval_all(ximage)
        data_13co10     = data_13co10["data"] * data_13co10["mask"]
        data_13co10     = data_13co10.flatten()
        data_13co10[np.isnan(data_13co10)] = 0

        err_13co10,_ = imval_all(xerrimage)
        err_13co10   = err_13co10["data"] * err_13co10["mask"]
        err_13co10   = err_13co10.flatten()
        err_13co10[np.isnan(err_13co10)] = 0

        # 13co21
        data_13co21,_ = imval_all(yimage)
        data_13co21   = data_13co21["data"] * data_13co21["mask"]
        data_13co21   = data_13co21.flatten()
        data_13co21[np.isnan(data_13co21)] = 0

        err_13co21,_ = imval_all(yerrimage)
        err_13co21   = err_13co21["data"] * err_13co21["mask"]
        err_13co21   = err_13co21.flatten()
        err_13co21[np.isnan(err_13co21)] = 0

        if cimage==None:
            # coords
            data_coords = imval(ximage,box=box)["coords"]
            ra_deg      = data_coords[:,:,0] * 180/np.pi
            ra_deg      = ra_deg.flatten()
            dec_deg     = data_coords[:,:,1] * 180/np.pi
            dec_deg     = dec_deg.flatten()
            dist_pc,_   = get_reldist_pc(ra_deg, dec_deg, self.ra_agn, self.dec_agn, self.scale_pc, 0, 0)
            c           = dist_pc / 1000.0
            # prepare
            cut  = np.where((data_13co10>abs(err_13co10)*self.snr)&(data_13co21>abs(err_13co21)*self.snr))
            x    = np.log10(data_13co10[cut])
            xerr = err_13co10[cut] / abs(data_13co10[cut])
            y    = data_13co21[cut] + np.log10(factor)
            yerr = err_13co21[cut]
            c    = np.array(c)[cut]
        else:
            data_c,_    = imval_all(cimage)
            data_c      = data_c["data"] * data_c["mask"]
            data_c      = data_c.flatten()
            data_c[np.isnan(data_c)] = 0
            c           = data_c
            data_cerr,_ = imval_all(cerrimage)
            data_cerr   = data_cerr["data"] * data_cerr["mask"]
            data_cerr   = data_cerr.flatten()
            data_cerr[np.isnan(data_cerr)] = 0
            cerr        = data_cerr
            # prepare
            cut  = np.where((data_13co10>abs(err_13co10)*self.snr)&(data_13co21>abs(err_13co21)*self.snr)&(c>abs(cerr)*self.snr))
            x    = np.log10(data_13co10[cut])
            xerr = err_13co10[cut] / abs(data_13co10[cut])
            y    = data_13co21[cut] + np.log10(factor)
            yerr = err_13co21[cut]
            c    = np.array(c)[cut]

        # plot
        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        ad  = [0.215,0.83,0.10,0.90]
        myax_set(ax1, "both", xlim, ylim, title, xlabel, ylabel, adjust=ad)

        cs = ax1.scatter(x, y, c=c, cmap=cmap, lw=0, s=40, zorder=1e9)
        ax1.errorbar(x, y, xerr=xerr, yerr=yerr, lw=1, capsize=0, color="grey", linestyle="None")

        # colorbar
        cax = fig.add_axes([0.25, 0.81, 0.33, 0.04])
        cbar = plt.colorbar(cs, cax=cax, orientation="horizontal")
        cbar.set_label(cblabel)
        if cimage==None:
            cbar.set_ticks([0,0.3,0.6,0.9,1.2])

        # ann
        ax1.plot(xlim, [xlim[0]+np.log10(2e20),xlim[1]+np.log10(2e20)], "--", color="black", lw=1)
        ax1.plot(xlim, [xlim[0]+np.log10(2e20)-1.0,xlim[1]+np.log10(2e20)-1.0], "--", color="black", lw=1)

        # text
        ax1.text(1.5,21.9, "$X_{\mathrm{CO}}$=2.0$\\times$ 10$^{20}$ cm$^{-2}$", fontsize=20, ha="left", va="bottom", rotation=40)
        ax1.text(1.5,20.7, "2.0$\\times$ 10$^{19}$ cm$^{-2}$", fontsize=20, ha="left", va="bottom", rotation=40)

        # save
        os.system("rm -rf " + outpng)
        plt.savefig(outpng, dpi=self.fig_dpi)

        #
        if outfits!=None:
            # 12co10
            data_13co10,box = imval_all(ximage)
            data_13co10     = data_13co10["data"] * data_13co10["mask"]
            data_13co10[np.isnan(data_13co10)] = 0

            err_13co10,_ = imval_all(xerrimage)
            err_13co10   = err_13co10["data"] * err_13co10["mask"]
            err_13co10[np.isnan(err_13co10)] = 0

            # log ncol
            data_13co21,_ = imval_all(yimage)
            data_13co21   = data_13co21["data"] * data_13co21["mask"]
            data_13co21[np.isnan(data_13co21)] = 0


            err_13co21,_ = imval_all(yerrimage)
            err_13co21   = err_13co21["data"] * err_13co21["mask"]
            err_13co21[np.isnan(err_13co21)] = 0

            # prepare
            outarray = data_13co21 + np.log10(factor) - np.log10(data_13co10)
            outarray = 10**outarray / 2e20 * 4.3
            outarray = np.rot90(np.fliplr( np.where((data_13co10>abs(err_13co10)*self.snr)&(data_13co21>abs(err_13co21)*self.snr),outarray,np.nan) ))

            self._fits_creation(
                input_array=outarray,
                output_map=outfits,
                coords_template=templatefits,
                bunit="K",
                )

            outarray = np.log(10) * 10**data_13co21 * abs(err_13co21) * factor / 2e20 * 4.3 / abs(data_13co10)
            outarray = np.rot90(np.fliplr( np.where((data_13co10>abs(err_13co10)*self.snr)&(data_13co21>abs(err_13co21)*self.snr),outarray,np.nan) ))

            self._fits_creation(
                input_array=outarray,
                output_map=outefits,
                coords_template=templatefits,
                bunit="K",
                )

        return x, xerr, y, yerr, c

    ##################
    # _plot_scatter3 #
    ##################

    def _plot_scatter3(
        self,
        yimage,
        yerrimage,
        y2image,
        y2errimage,
        outpng,
        xlim,
        ylim,
        ylim2,
        title,
        xlabel,
        ylabel,
        ylabel2,
        ):
        # coords
        _,box       = imval_all(yimage)
        data_coords = imval(yimage,box=box)["coords"]
        ra_deg      = data_coords[:,:,0] * 180/np.pi
        ra_deg      = ra_deg.flatten()
        dec_deg     = data_coords[:,:,1] * 180/np.pi
        dec_deg     = dec_deg.flatten()
        dist_pc,_   = get_reldist_pc(ra_deg, dec_deg, self.ra_agn, self.dec_agn, self.scale_pc, 0, 0)
        data_x      = dist_pc / 1000.0

        # y1
        data_y1,_ = imval_all(yimage)
        data_y1   = data_y1["data"] * data_y1["mask"]
        data_y1   = data_y1.flatten()
        data_y1[np.isnan(data_y1)] = 0

        err_y1,_ = imval_all(yerrimage)
        err_y1   = err_y1["data"] * err_y1["mask"]
        err_y1   = err_y1.flatten()
        err_y1[np.isnan(err_y1)] = 0

        # y2
        data_y2,_ = imval_all(y2image)
        data_y2   = data_y2["data"] * data_y2["mask"]
        data_y2   = data_y2.flatten()
        data_y2[np.isnan(data_y2)] = 0

        err_y2,_ = imval_all(y2errimage)
        err_y2   = err_y2["data"] * err_y2["mask"]
        err_y2   = err_y2.flatten()
        err_y2[np.isnan(err_y2)] = 0

        # prepare
        cut   = np.where((data_y1>abs(err_y1)*self.snr)&(data_y2>abs(err_y2)*self.snr))
        x     = data_x[cut]
        y1    = data_y1[cut]
        y1err = err_y1[cut]
        y2    = data_y2[cut]
        y2err = err_y2[cut]

        # binned
        n,_   = np.histogram(x, bins=10, range=[np.min(x),np.max(x)])
        sy,_  = np.histogram(x, bins=10, range=[np.min(x),np.max(x)], weights=y1)
        sy2,_ = np.histogram(x, bins=10, range=[np.min(x),np.max(x)], weights=y1*y1)
        mean  = sy / n
        std   = np.sqrt(sy2/n - mean*mean)
        binx1 = (_[1:]+_[:-1])/2
        biny1 = mean
        binyerr1 = std

        n,_   = np.histogram(x, bins=10, range=[np.min(x),np.max(x)])
        sy,_  = np.histogram(x, bins=10, range=[np.min(x),np.max(x)], weights=y2)
        sy2,_ = np.histogram(x, bins=10, range=[np.min(x),np.max(x)], weights=y2*y2)
        mean  = sy / n
        std   = np.sqrt(sy2/n - mean*mean)
        binx2 = (_[1:]+_[:-1])/2
        biny2 = mean
        binyerr2 = std

        # plot
        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        ax2 = ax1.twinx()
        ad  = [0.215,0.83,0.10,0.90]
        myax_set(ax1, "both", xlim, ylim, title, xlabel, ylabel, adjust=ad)
        ax2.set_ylabel(ylabel2)
        ax2.set_ylim(ylim2)

        ax1.errorbar(x, y1, yerr=y1err, lw=1, capsize=0, color="grey", linestyle="None")
        ax2.errorbar(x, y2, yerr=y2err, lw=1, capsize=0, color="grey", linestyle="None")
        ax1.scatter(x, y1, c="deepskyblue", lw=0, s=40, zorder=1e9)
        ax2.scatter(x, y2, c="tomato", lw=0, s=40, zorder=1e9)

        ax1.plot(binx1, biny1, color="blue", lw=2.0, zorder=1e11)
        for i in range(len(binx1)):
            this_binx1    = binx1[i]
            this_biny1    = biny1[i]
            this_binyerr1 = binyerr1[i]
            ax1.plot([this_binx1,this_binx1],[this_biny1-this_binyerr1,this_biny1+this_binyerr1], color="blue", lw=2.0, zorder=1e11)

        ax2.plot(binx2, biny2, color="red", lw=2.0, zorder=1e11)
        for i in range(len(binx2)):
            this_binx2    = binx2[i]
            this_biny2    = biny2[i]
            this_binyerr2 = binyerr2[i]
            ax2.plot([this_binx2,this_binx2],[this_biny2-this_binyerr2,this_biny2+this_binyerr2], color="red", lw=2.0, zorder=1e11)

        # text
        ax1.text(0.05,0.93, "$T_{\mathrm{rot}}$", color="deepskyblue", transform=ax1.transAxes, weight="bold", fontsize=26, ha="left")
        ax1.text(0.05,0.88, "log$_{\mathrm{10}}$ $N_{\mathrm{^{13}CO}}$", color="tomato", transform=ax1.transAxes, weight="bold", fontsize=26, ha="left")

        # save
        os.system("rm -rf " + outpng)
        plt.savefig(outpng, dpi=self.fig_dpi)

    ##################
    # _plot_scatter2 #
    ##################

    def _plot_scatter2(
        self,
        x1image,
        x1errimage,
        x2image,
        x2errimage,
        yimage,
        yerrimage,
        outpng,
        xlim,
        ylim,
        title,
        xlabel,
        ylabel,
        colorlog=False,
        ):
        # x1
        data_x1,_ = imval_all(x1image)
        data_x1   = data_x1["data"] * data_x1["mask"]
        data_x1   = data_x1.flatten()
        data_x1[np.isnan(data_x1)] = 0

        err_x1,_ = imval_all(x1errimage)
        err_x1   = err_x1["data"] * err_x1["mask"]
        err_x1   = err_x1.flatten()
        err_x1[np.isnan(err_x1)] = 0

        # x2
        data_x2,_ = imval_all(x2image)
        data_x2   = data_x2["data"] * data_x2["mask"]
        data_x2   = data_x2.flatten()
        data_x2[np.isnan(data_x2)] = 0

        err_x2,_ = imval_all(x2errimage)
        err_x2   = err_x2["data"] * err_x2["mask"]
        err_x2   = err_x2.flatten()
        err_x2[np.isnan(err_x2)] = 0

        # y
        data_y,_ = imval_all(yimage)
        data_y   = data_y["data"] * data_y["mask"]
        data_y   = data_y.flatten()
        data_y[np.isnan(data_y)] = 0

        err_y,_ = imval_all(yerrimage)
        err_y   = err_y["data"] * err_y["mask"]
        err_y   = err_y.flatten()
        err_y[np.isnan(err_y)] = 0

        # prepare
        cut   = np.where((data_x1>abs(err_x1)*self.snr)&(data_x2>abs(err_x2)*self.snr)&(data_y>abs(err_y)*self.snr))
        x1    = np.log10(data_x1[cut])
        x1err = err_x1[cut] / abs(data_x1[cut])
        x2    = np.log10(data_x2[cut])
        x2err = err_x2[cut] / abs(data_x2[cut])
        y     = data_y[cut]
        yerr  = err_y[cut]

        if colorlog==True:
            y    = np.log10(y)
            yerr = yerr / abs(10**y)

        # binned
        n,_   = np.histogram(x1, bins=10, range=[np.min(x1),np.max(x1)])
        sy,_  = np.histogram(x1, bins=10, range=[np.min(x1),np.max(x1)], weights=y)
        sy2,_ = np.histogram(x1, bins=10, range=[np.min(x1),np.max(x1)], weights=y*y)
        mean  = sy / n
        std   = np.sqrt(sy2/n - mean*mean)
        binx1 = (_[1:]+_[:-1])/2
        biny1 = mean
        binyerr1 = std

        n,_   = np.histogram(x2, bins=10, range=[np.min(x2),np.max(x2)])
        sy,_  = np.histogram(x2, bins=10, range=[np.min(x2),np.max(x2)], weights=y)
        sy2,_ = np.histogram(x2, bins=10, range=[np.min(x2),np.max(x2)], weights=y*y)
        mean  = sy / n
        std   = np.sqrt(sy2/n - mean*mean)
        binx2 = (_[1:]+_[:-1])/2
        biny2 = mean
        binyerr2 = std

        # plot
        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        ad  = [0.215,0.83,0.10,0.90]
        myax_set(ax1, "both", xlim, ylim, title, xlabel, ylabel, adjust=ad)

        ax1.errorbar(x1, y, xerr=x1err, yerr=yerr, lw=1, capsize=0, color="grey", linestyle="None")
        ax1.errorbar(x2, y, xerr=x2err, yerr=yerr, lw=1, capsize=0, color="grey", linestyle="None")
        ax1.scatter(x1, y, c="deepskyblue", lw=0, s=40, zorder=1e9)
        ax1.scatter(x2, y, c="tomato", lw=0, s=40, zorder=1e9)

        ax1.plot(binx1, biny1, color="blue", lw=2.0, zorder=1e11)
        for i in range(len(binx1)):
            this_binx1    = binx1[i]
            this_biny1    = biny1[i]
            this_binyerr1 = binyerr1[i]
            ax1.plot([this_binx1,this_binx1],[this_biny1-this_binyerr1,this_biny1+this_binyerr1], color="blue", lw=2.0, zorder=1e11)

        ax1.plot(binx2, biny2, color="red", lw=2.0, zorder=1e11)
        for i in range(len(binx2)):
            this_binx2    = binx2[i]
            this_biny2    = biny2[i]
            this_binyerr2 = binyerr2[i]
            ax1.plot([this_binx2,this_binx2],[this_biny2-this_binyerr2,this_biny2+this_binyerr2], color="red", lw=2.0, zorder=1e11)

        # text
        ax1.text(0.05,0.93, "J = 1-0", color="deepskyblue", transform=ax1.transAxes, weight="bold", fontsize=26, ha="left")
        ax1.text(0.05,0.88, "J = 2-1", color="tomato", transform=ax1.transAxes, weight="bold", fontsize=26, ha="left")

        # save
        os.system("rm -rf " + outpng)
        plt.savefig(outpng, dpi=self.fig_dpi)

    ##################
    # _plot_scatter1 #
    ##################

    def _plot_scatter1(
        self,
        ximage,
        xerrimage,
        yimage,
        yerrimage,
        cimage,
        cerrimage,
        outpng,
        lim,
        title,
        xlabel,
        ylabel,
        cblabel,
        ylim=None,
        cmap="rainbow_r",
        ):

        # 13co10
        data_13co10,box = imval_all(ximage)
        data_13co10     = data_13co10["data"] * data_13co10["mask"]
        data_13co10     = data_13co10.flatten()
        data_13co10[np.isnan(data_13co10)] = 0

        err_13co10,_ = imval_all(xerrimage)
        err_13co10   = err_13co10["data"] * err_13co10["mask"]
        err_13co10   = err_13co10.flatten()
        err_13co10[np.isnan(err_13co10)] = 0

        # 13co21
        data_13co21,_ = imval_all(yimage)
        data_13co21   = data_13co21["data"] * data_13co21["mask"]
        data_13co21   = data_13co21.flatten()
        data_13co21[np.isnan(data_13co21)] = 0

        err_13co21,_ = imval_all(yerrimage)
        err_13co21   = err_13co21["data"] * err_13co21["mask"]
        err_13co21   = err_13co21.flatten()
        err_13co21[np.isnan(err_13co21)] = 0

        if cimage==None:
            # coords
            data_coords = imval(ximage,box=box)["coords"]
            ra_deg      = data_coords[:,:,0] * 180/np.pi
            ra_deg      = ra_deg.flatten()
            dec_deg     = data_coords[:,:,1] * 180/np.pi
            dec_deg     = dec_deg.flatten()
            dist_pc,_   = get_reldist_pc(ra_deg, dec_deg, self.ra_agn, self.dec_agn, self.scale_pc, 0, 0)
            c           = dist_pc / 1000.0
            # prepare
            cut  = np.where((data_13co10>abs(err_13co10)*self.snr)&(data_13co21>abs(err_13co21)*self.snr))
            x    = np.log10(data_13co10[cut])
            xerr = err_13co10[cut] / abs(data_13co10[cut])
            y    = np.log10(data_13co21[cut])
            yerr = err_13co21[cut] / abs(data_13co21[cut])
            c    = np.array(c)[cut]

        else:
            data_c,_    = imval_all(cimage)
            data_c      = data_c["data"] * data_c["mask"]
            data_c      = data_c.flatten()
            data_c[np.isnan(data_c)] = 0
            c           = data_c
            data_cerr,_ = imval_all(cerrimage)
            data_cerr   = data_cerr["data"] * data_cerr["mask"]
            data_cerr   = data_cerr.flatten()
            data_cerr[np.isnan(data_cerr)] = 0
            cerr        = data_cerr
            # prepare
            cut  = np.where((data_13co10>abs(err_13co10)*self.snr)&(data_13co21>abs(err_13co21)*self.snr)&(c>abs(cerr)*self.snr))
            x    = np.log10(data_13co10[cut])
            xerr = err_13co10[cut] / abs(data_13co10[cut])
            y    = np.log10(data_13co21[cut])
            yerr = err_13co21[cut] / abs(data_13co21[cut])
            c    = np.array(c)[cut]

        # plot
        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        ad  = [0.215,0.83,0.10,0.90]
        myax_set(ax1, "both", lim, lim, title, xlabel, ylabel, adjust=ad)

        cs = ax1.scatter(x, y, c=c, cmap=cmap, lw=0, s=40, zorder=1e9)
        ax1.errorbar(x, y, xerr=xerr, yerr=yerr, lw=1, capsize=0, color="grey", linestyle="None")

        # colorbar
        cax = fig.add_axes([0.25, 0.81, 0.33, 0.04])
        cbar = plt.colorbar(cs, cax=cax, orientation="horizontal")
        cbar.set_label(cblabel)
        if cimage==None:
            cbar.set_ticks([0,0.3,0.6,0.9,1.2])

        """ cmap for errorbar
        clb   = plt.colorbar(sc)
        color = clb.to_rgba(r)
        for this_x, this_y, this_xerr, this_yerr, this_c in zip(x, y, xerr, yerr, color):
            plt.errorbar(this_x, this_y, this_xerr, this_yerr, lw=1, capsize=0, color=this_c)
        """

        # ann
        ax1.plot(lim, lim, "--", color="black", lw=1)

        # save
        os.system("rm -rf " + outpng)
        plt.savefig(outpng, dpi=self.fig_dpi)

    ###############
    # _eval_a_sim #
    ###############

    def _eval_a_sim(
        self,
        nbins,
        lim,
        lim2,
        this_snr,
        outpng_mom0_nomask,
        outpng_mom0_mask,
        outpng_ratio_nomask,
        outpng_ratio_mask,
        ):

        snrtext = this_snr.replace("snr","")
        snrfloat = float(this_snr.replace("snr",""))

        # hereafter
        simumom0a_1  = self.outsimumom0_13co10.replace(".fits","_noclip_"+this_snr+".fits")
        simumom0a_2  = self.outsimumom0_13co10.replace(".fits","_noclip_"+this_snr+".fits").replace("mom0","emom0")
        simumom0a_3  = self.outsimumom0_13co10.replace(".fits","_clip0_"+this_snr+".fits")
        simumom0a_4  = self.outsimumom0_13co10.replace(".fits","_clip0_"+this_snr+".fits").replace("mom0","emom0")
        simumom0a_5  = self.outsimumom0_13co10.replace(".fits","_clip3_"+this_snr+".fits")
        simumom0a_6  = self.outsimumom0_13co10.replace(".fits","_clip3_"+this_snr+".fits").replace("mom0","emom0")
        simumom0a_1m = self.outsimumom0_13co10.replace(".fits","_noclip_masked_"+this_snr+".fits")
        simumom0a_2m = self.outsimumom0_13co10.replace(".fits","_noclip_masked_"+this_snr+".fits").replace("mom0","emom0")
        simumom0a_3m = self.outsimumom0_13co10.replace(".fits","_clip0_masked_"+this_snr+".fits")
        simumom0a_4m = self.outsimumom0_13co10.replace(".fits","_clip0_masked_"+this_snr+".fits").replace("mom0","emom0")
        simumom0a_5m = self.outsimumom0_13co10.replace(".fits","_clip3_masked_"+this_snr+".fits")
        simumom0a_6m = self.outsimumom0_13co10.replace(".fits","_clip3_masked_"+this_snr+".fits").replace("mom0","emom0")
        modelmom0a   = self.outmodelmom0_13co10

        simumom0b_1  = self.outsimumom0_13co21.replace(".fits","_noclip_"+this_snr+".fits")
        simumom0b_2  = self.outsimumom0_13co21.replace(".fits","_noclip_"+this_snr+".fits").replace("mom0","emom0")
        simumom0b_3  = self.outsimumom0_13co21.replace(".fits","_clip0_"+this_snr+".fits")
        simumom0b_4  = self.outsimumom0_13co21.replace(".fits","_clip0_"+this_snr+".fits").replace("mom0","emom0")
        simumom0b_5  = self.outsimumom0_13co21.replace(".fits","_clip3_"+this_snr+".fits")
        simumom0b_6  = self.outsimumom0_13co21.replace(".fits","_clip3_"+this_snr+".fits").replace("mom0","emom0")
        simumom0b_1m = self.outsimumom0_13co21.replace(".fits","_noclip_masked_"+this_snr+".fits")
        simumom0b_2m = self.outsimumom0_13co21.replace(".fits","_noclip_masked_"+this_snr+".fits").replace("mom0","emom0")
        simumom0b_3m = self.outsimumom0_13co21.replace(".fits","_clip0_masked_"+this_snr+".fits")
        simumom0b_4m = self.outsimumom0_13co21.replace(".fits","_clip0_masked_"+this_snr+".fits").replace("mom0","emom0")
        simumom0b_5m = self.outsimumom0_13co21.replace(".fits","_clip3_masked_"+this_snr+".fits")
        simumom0b_6m = self.outsimumom0_13co21.replace(".fits","_clip3_masked_"+this_snr+".fits").replace("mom0","emom0")
        modelmom0b   = self.outmodelmom0_13co21

        #################
        # import 13co10 #
        #################
        x1,y1,binx1,biny1,bine1 = self._get_sim_data(simumom0a_1,simumom0a_2,modelmom0a,lim)   # noclip
        x2,y2,binx2,biny2,bine2 = self._get_sim_data(simumom0a_3,simumom0a_4,modelmom0a,lim)   # clip0
        x3,y3,binx3,biny3,bine3 = self._get_sim_data(simumom0a_5,simumom0a_6,modelmom0a,lim)   # clip3
        x4,y4,binx4,biny4,bine4 = self._get_sim_data(simumom0a_1m,simumom0a_2m,modelmom0a,lim) # noclip+mask
        x5,y5,binx5,biny5,bine5 = self._get_sim_data(simumom0a_3m,simumom0a_4m,modelmom0a,lim) # clip0+mask
        x6,y6,binx6,biny6,bine6 = self._get_sim_data(simumom0a_5m,simumom0a_6m,modelmom0a,lim) # clip3+mask

        ################
        # import ratio #
        ################
        # model
        l,_ = imval_all(modelmom0a)
        a0 = l["data"] * l["mask"]
        a0 = np.log10(np.array(a0.flatten()))

        l,_ = imval_all(modelmom0b)
        a0b = l["data"] * l["mask"]
        a0b = np.array(a0b.flatten())
        b0   = np.log10(a0b / 10**a0)

        n,_   = np.histogram(a0, bins=nbins, range=lim)
        sy,_  = np.histogram(a0, bins=nbins, range=lim, weights=b0)
        sy2,_ = np.histogram(a0, bins=nbins, range=lim, weights=b0*b0)
        mean  = sy / n
        std   = np.sqrt(sy2/n - mean*mean)
        bina0 = (_[1:]+_[:-1])/2
        binb0 = mean
        binc0 = std

        a1,b1,bina1,binb1,binc1 = self._get_sim_ratio(simumom0a_1,simumom0a_2,modelmom0a,simumom0b_1,simumom0b_2,modelmom0b,lim)     # noclip
        a2,b2,bina2,binb2,binc2 = self._get_sim_ratio(simumom0a_3,simumom0a_4,modelmom0a,simumom0b_3,simumom0b_4,modelmom0b,lim)     # clip0
        a3,b3,bina3,binb3,binc3 = self._get_sim_ratio(simumom0a_5,simumom0a_6,modelmom0a,simumom0b_5,simumom0b_6,modelmom0b,lim)     # clip3
        a4,b4,bina4,binb4,binc4 = self._get_sim_ratio(simumom0a_1m,simumom0a_2m,modelmom0a,simumom0b_1m,simumom0b_2m,modelmom0b,lim) # noclip+mask
        a5,b5,bina5,binb5,binc5 = self._get_sim_ratio(simumom0a_3m,simumom0a_4m,modelmom0a,simumom0b_3m,simumom0b_4m,modelmom0b,lim) # clip0+mask
        a6,b6,bina6,binb6,binc6 = self._get_sim_ratio(simumom0a_5m,simumom0a_6m,modelmom0a,simumom0b_5m,simumom0b_6m,modelmom0b,lim) # clip3+mask

        ########
        # plot #
        ########
        # set plt, ax
        fig  = plt.figure(figsize=(13,10))
        plt.rcParams["font.size"] = 16
        gs   = gridspec.GridSpec(nrows=11, ncols=11)
        ax   = plt.subplot(gs[0:10,0:10])

        # set ax parameter
        myax_set(
        ax,
        grid="both",
        xlim=lim,
        ylim=lim,
        xlabel="log$_{\mathrm{10}}$ $I_{\mathrm{^{13}CO(1-0),model}}$",
        ylabel="log$_{\mathrm{10}}$ $I_{\mathrm{^{13}CO(1-0),reconstructed}}$",
        adjust=[0.215,0.83,0.10,0.90],
        )

        ax.scatter(x1+np.log10(snrfloat/10.0), y1, marker=".", color="green", lw=0.5, alpha=0.2)
        ax.scatter(x2+np.log10(snrfloat/10.0), y2, marker=".", color="deepskyblue", lw=0.5, alpha=0.2)
        ax.scatter(x3+np.log10(snrfloat/10.0), y3, marker=".", color="tomato", lw=0.5, alpha=0.2)

        ax.errorbar(binx1+np.log10(snrfloat/10.0), biny1, yerr=bine1, color="green", capsize=0, lw=2.0)
        ax.errorbar(binx2+np.log10(snrfloat/10.0), biny2, yerr=bine1, color="blue", capsize=0, lw=2.0)
        ax.errorbar(binx3+np.log10(snrfloat/10.0), biny3, yerr=bine1, color="red", capsize=0, lw=2.0)

        # ann
        ax.plot(lim,lim,"--",color="black",lw=1)

        # text
        ax.text(0.05,0.90, "mom0$_{\mathrm{SNR="+snrtext+"}}$", transform=ax.transAxes, weight="bold", fontsize=26, ha="left")
        ax.text(0.95, 0.15, "noclip", color="green", transform=ax.transAxes, weight="bold", fontsize=22, ha="right")
        ax.text(0.95, 0.10, "clip0$\sigma$", color="deepskyblue", transform=ax.transAxes, weight="bold", fontsize=22, ha="right")
        ax.text(0.95, 0.05, "clip3$\sigma$", color="tomato", transform=ax.transAxes, weight="bold", fontsize=22, ha="right")

        # save
        os.system("rm -rf " + outpng_mom0_nomask)
        plt.savefig(outpng_mom0_nomask, dpi=300)

        ########
        # plot #
        ########
        # set plt, ax
        fig  = plt.figure(figsize=(13,10))
        plt.rcParams["font.size"] = 16
        gs   = gridspec.GridSpec(nrows=11, ncols=11)
        ax   = plt.subplot(gs[0:10,0:10])

        # set ax parameter
        myax_set(
        ax,
        grid="both",
        xlim=lim,
        ylim=lim,
        xlabel="log$_{\mathrm{10}}$ $I_{\mathrm{^{13}CO(1-0),model}}$",
        ylabel="log$_{\mathrm{10}}$ $I_{\mathrm{^{13}CO(1-0),reconstructed}}$",
        adjust=[0.215,0.83,0.10,0.90],
        )

        ax.scatter(x4+np.log10(snrfloat/10.0), y4, marker=".", color="green", lw=0.5, alpha=0.2)
        ax.scatter(x5+np.log10(snrfloat/10.0), y5, marker=".", color="deepskyblue", lw=0.5, alpha=0.2)
        ax.scatter(x6+np.log10(snrfloat/10.0), y6, marker=".", color="tomato", lw=0.5, alpha=0.2)

        ax.errorbar(binx4+np.log10(snrfloat/10.0), biny4, yerr=bine1, color="green", capsize=0, lw=2.0)
        ax.errorbar(binx5+np.log10(snrfloat/10.0), biny5, yerr=bine1, color="blue", capsize=0, lw=2.0)
        ax.errorbar(binx6+np.log10(snrfloat/10.0), biny6, yerr=bine1, color="red", capsize=0, lw=2.0)

        # ann
        ax.plot(lim,lim,"--",color="black",lw=1)

        # text
        ax.text(0.05,0.90, "mom0$_{\mathrm{SNR="+snrtext+"}}$", transform=ax.transAxes, weight="bold", fontsize=26, ha="left")
        ax.text(0.95, 0.15, "noclip+masking", color="green", transform=ax.transAxes, weight="bold", fontsize=22, ha="right")
        ax.text(0.95, 0.10, "clip0$\sigma$+masking", color="deepskyblue", transform=ax.transAxes, weight="bold", fontsize=22, ha="right")
        ax.text(0.95, 0.05, "clip3$\sigma$+masking", color="tomato", transform=ax.transAxes, weight="bold", fontsize=22, ha="right")

        # save
        os.system("rm -rf " + outpng_mom0_mask)
        plt.savefig(outpng_mom0_mask, dpi=300)

        ########
        # plot #
        ########
        # set plt, ax
        fig  = plt.figure(figsize=(13,10))
        plt.rcParams["font.size"] = 16
        gs   = gridspec.GridSpec(nrows=11, ncols=11)
        ax   = plt.subplot(gs[0:10,0:10])

        # set ax parameter
        myax_set(
        ax,
        grid="both",
        xlim=lim,
        ylim=lim2,
        xlabel="log$_{\mathrm{10}}$ $I_{\mathrm{^{13}CO(1-0),model}}$",
        ylabel="log$_{\mathrm{10}}$ Ratio$_{\mathrm{reconstructed}}$",
        adjust=[0.215,0.83,0.10,0.90],
        )

        ax.scatter(a0+np.log10(snrfloat/10.0), b0, marker=".", color="grey", lw=0.5, alpha=0.2)
        ax.scatter(a1+np.log10(snrfloat/10.0), b1, marker=".", color="green", lw=0.5, alpha=0.2)
        ax.scatter(a2+np.log10(snrfloat/10.0), b2, marker=".", color="deepskyblue", lw=0.5, alpha=0.2)
        ax.scatter(a3+np.log10(snrfloat/10.0), b3, marker=".", color="tomato", lw=0.5, alpha=0.2)

        ax.errorbar(bina0+np.log10(snrfloat/10.0), binb0, yerr=binc0, color="black", capsize=0, lw=2.0)
        ax.errorbar(bina1+np.log10(snrfloat/10.0), binb1, yerr=binc1, color="green", capsize=0, lw=2.0)
        ax.errorbar(bina2+np.log10(snrfloat/10.0), binb2, yerr=binc1, color="blue", capsize=0, lw=2.0)
        ax.errorbar(bina3+np.log10(snrfloat/10.0), binb3, yerr=binc1, color="red", capsize=0, lw=2.0)

        # text
        ax.text(0.05,0.90, "mom0$_{\mathrm{SNR="+snrtext+"}}$", transform=ax.transAxes, weight="bold", fontsize=26, ha="left")
        ax.text(0.95, 0.20, "model", color="grey", transform=ax.transAxes, weight="bold", fontsize=22, ha="right")
        ax.text(0.95, 0.15, "noclip", color="green", transform=ax.transAxes, weight="bold", fontsize=22, ha="right")
        ax.text(0.95, 0.10, "clip0$\sigma$", color="deepskyblue", transform=ax.transAxes, weight="bold", fontsize=22, ha="right")
        ax.text(0.95, 0.05, "clip3$\sigma$", color="tomato", transform=ax.transAxes, weight="bold", fontsize=22, ha="right")

        # save
        os.system("rm -rf " + outpng_ratio_nomask)
        plt.savefig(outpng_ratio_nomask, dpi=300)

        ########
        # plot #
        ########
        # set plt, ax
        fig  = plt.figure(figsize=(13,10))
        plt.rcParams["font.size"] = 16
        gs   = gridspec.GridSpec(nrows=11, ncols=11)
        ax   = plt.subplot(gs[0:10,0:10])

        # set ax parameter
        myax_set(
        ax,
        grid="both",
        xlim=lim,
        ylim=lim2,
        xlabel="log$_{\mathrm{10}}$ $I_{\mathrm{^{13}CO(1-0),model}}$",
        ylabel="log$_{\mathrm{10}}$ Ratio$_{\mathrm{reconstructed}}$",
        adjust=[0.215,0.83,0.10,0.90],
        )

        ax.scatter(a0+np.log10(snrfloat/10.0), b0, marker=".", color="grey", lw=0.5, alpha=0.2)
        ax.scatter(a4+np.log10(snrfloat/10.0), b4, marker=".", color="green", lw=0.5, alpha=0.2)
        ax.scatter(a5+np.log10(snrfloat/10.0), b5, marker=".", color="deepskyblue", lw=0.5, alpha=0.2)
        ax.scatter(a6+np.log10(snrfloat/10.0), b6, marker=".", color="tomato", lw=0.5, alpha=0.2)

        ax.errorbar(bina0+np.log10(snrfloat/10.0), binb0, yerr=binc0, color="black", capsize=0, lw=2.0)
        ax.errorbar(bina4+np.log10(snrfloat/10.0), binb4, yerr=binc1, color="green", capsize=0, lw=2.0)
        ax.errorbar(bina5+np.log10(snrfloat/10.0), binb5, yerr=binc1, color="blue", capsize=0, lw=2.0)
        ax.errorbar(bina6+np.log10(snrfloat/10.0), binb6, yerr=binc1, color="red", capsize=0, lw=2.0)

        # text
        ax.text(0.05,0.90, "mom0$_{\mathrm{SNR="+snrtext+"}}$", transform=ax.transAxes, weight="bold", fontsize=26, ha="left")
        ax.text(0.95, 0.20, "model", color="grey", transform=ax.transAxes, weight="bold", fontsize=22, ha="right")
        ax.text(0.95, 0.15, "noclip+masking", color="green", transform=ax.transAxes, weight="bold", fontsize=22, ha="right")
        ax.text(0.95, 0.10, "clip0$\sigma$+masking", color="deepskyblue", transform=ax.transAxes, weight="bold", fontsize=22, ha="right")
        ax.text(0.95, 0.05, "clip3$\sigma$+masking", color="tomato", transform=ax.transAxes, weight="bold", fontsize=22, ha="right")

        # save
        os.system("rm -rf " + outpng_ratio_mask)
        plt.savefig(outpng_ratio_mask, dpi=300)

    ##################
    # _get_sim_ratio #
    ##################

    def _get_sim_ratio(
        self,
        mom0_13co10,
        emom0_13co10,
        input_mom0_13co10,
        mom0_13co21,
        emom0_13co21,
        input_mom0_13co21,
        lim,
        nbins=10,
        ):
        """
        """

        # input
        l,_ = imval_all(input_mom0_13co10)
        model_mom0_13co10 = l["data"] * l["mask"]
        model_mom0_13co10 = np.array(model_mom0_13co10.flatten())

        l,_ = imval_all(input_mom0_13co21)
        model_mom0_13co21 = l["data"] * l["mask"]
        model_mom0_13co21 = np.array(model_mom0_13co21.flatten())

        #
        l,_  = imval_all(mom0_13co10)
        l = l["data"] * l["mask"]
        sim_mom0_13co10 = np.array(l.flatten())

        l,_  = imval_all(mom0_13co21)
        l = l["data"] * l["mask"]
        sim_mom0_13co21 = np.array(l.flatten())

        l,_ = imval_all(emom0_13co10)
        l = l["data"] * l["mask"]
        sim_emom0_13co10 = np.array(l.flatten())

        l,_ = imval_all(emom0_13co21)
        l = l["data"] * l["mask"]
        sim_emom0_13co21 = np.array(l.flatten())

        cut = np.where((sim_mom0_13co10>=sim_emom0_13co10*self.snr)&(~np.isnan(np.log10(model_mom0_13co10)))&(~np.isnan(np.log10(sim_mom0_13co10)))&(sim_mom0_13co21>=sim_emom0_13co21*self.snr)&(~np.isnan(np.log10(model_mom0_13co21)))&(~np.isnan(np.log10(sim_mom0_13co21))))
        x = np.log10(model_mom0_13co10[cut])
        y = np.log10(sim_mom0_13co21[cut]/sim_mom0_13co10[cut])

        # binning
        n,_ = np.histogram(x, bins=nbins, range=lim)
        sy,_ = np.histogram(x, bins=nbins, range=lim, weights=y)
        sy2,_ = np.histogram(x, bins=nbins, range=lim, weights=y*y)
        mean = sy / n
        std = np.sqrt(sy2/n - mean*mean)

        return x, y, (_[1:]+_[:-1])/2, mean, std

    #################
    # _get_sim_data #
    #################

    def _get_sim_data(
        self,
        mom0,
        emom0,
        input_mom0,
        lim,
        nbins=10,
        ):
        """
        """

        # input
        l,_ = imval_all(input_mom0)
        model_mom0 = l["data"] * l["mask"]
        model_mom0 = np.array(model_mom0.flatten())

        # noclip
        l,_  = imval_all(mom0)
        l = l["data"] * l["mask"]
        sim_mom0 = np.array(l.flatten())

        l,_ = imval_all(emom0)
        l = l["data"] * l["mask"]
        sim_emom0 = np.array(l.flatten())

        cut = np.where((sim_mom0>=sim_emom0*self.snr)&(~np.isnan(np.log10(model_mom0)))&(~np.isnan(np.log10(sim_mom0))))
        x = np.log10(model_mom0[cut])
        y = np.log10(sim_mom0[cut])
        #e = sim_emom0[cut]/abs(sim_mom0[cut])

        # binning
        n,_ = np.histogram(x, bins=nbins, range=lim)
        sy,_ = np.histogram(x, bins=nbins, range=lim, weights=y)
        sy2,_ = np.histogram(x, bins=nbins, range=lim, weights=y*y)
        mean = sy / n
        std = np.sqrt(sy2/n - mean*mean)

        return x, y, (_[1:]+_[:-1])/2, mean, std

    #################################
    # _create_correlated_noise_cube #
    #################################

    def _create_correlated_noise_cube(
        self,
        template, # self.outmodelcube_13co10
        snr=10.0,
        ):
        """
        """

        im    = pyfits.open(template)
        im0   = im[0]
        size  = im0.data.shape
        immax = np.nanmax(im0.data)
        scale = immax / snr
        pix   = abs(imhead(template,mode="list")["cdelt1"])
        beam  = imhead(template,mode="list")["beammajor"]["value"]
        print("# aimed noise rms = " + str(scale))

        noise   = np.random.normal(loc=0, scale=scale, size=size)
        im      = pyfits.open(template)
        im0     = im[0]
        im0.header["BMAJ"] = pix
        im0.header["BMIN"] = pix

        os.system("rm -rf noise.fits")
        pyfits.writeto("noise.fits",data=noise,header=im0.header,clobber=True)
        run_roundsmooth(
            "noise.fits",
            "noise_correlated.image",
            beam, # float, arcsec unit
            inputbeam=0.2,
            delin=True,
            )
        run_exportfits("noise_correlated.image","noise_correlated.fits",delin=True)

        im    = pyfits.open("noise_correlated.fits")
        im0   = im[0]
        noise = im0.data * scale / np.nanstd(im0.data)
        #pyfits.writeto("noise_correlated.fits",data=newdata,header=im0.header,clobber=True)
        os.system("rm -rf noise.fits noise_correlated.fits")

        return noise

    #################
    # _showcase_one #
    #################

    def _showcase_one(
        self,
        imcolor,
        imcontour1,
        outfile,
        set_title,
        label_cbar,
        clim=None,
        ):
        """
        """

        scalebar = 100. / self.scale_pc
        label_scalebar = "100 pc"

        levels_cont1 = [0.05, 0.1, 0.2, 0.4, 0.8, 0.96]
        width_cont1  = [1.0]
        set_bg_color = "white" # cm.rainbow(0)

        # plot
        myfig_fits2png(
            imcolor=imcolor,
            outfile=outfile,
            imcontour1=imcontour1,
            imsize_as=self.imsize,
            ra_cnt=self.ra_agn_str,
            dec_cnt=self.dec_agn_str,
            levels_cont1=levels_cont1,
            width_cont1=width_cont1,
            set_title=set_title,
            colorlog=False,
            scalebar=scalebar,
            label_scalebar=label_scalebar,
            set_cbar=True,
            label_cbar=label_cbar,
            clim=clim,
            set_bg_color=set_bg_color,
            numann="13co",
            )

    ########################
    # _align_maps_at_a_res #
    ########################

    def _align_maps_at_a_res(
        self,
        input_13co10,
        input_13co21,
        output_13co10,
        output_13co21,
        input_e13co10,
        input_e13co21,
        output_e13co10,
        output_e13co21,
        ):
        """
        """

        ra  = str(self.ra_agn)+"deg"
        dec = str(self.dec_agn)+"deg"

        # regrid 13co10 mom0
        imrebin2(input_13co10,output_13co10+".image",imsize=self.imsize,direction_ra=ra,direction_dec=dec)

        # regrid 13co21 mom0
        run_imregrid(input_13co21,output_13co10+".image",output_13co21+".image",axes=[0,1])

        # prepare for emom0
        pix_before = abs(imhead(imagename=input_13co10,mode="list")["cdelt1"]) * 3600 * 180 / np.pi
        pix_after  = abs(imhead(imagename=output_13co10+".image",mode="list")["cdelt1"]) * 3600 * 180 / np.pi
        numpix     = pix_after**2/pix_before**2

        # regrid 13co10 emom0
        run_immath_one(input_e13co10,input_e13co10+"_tmp1","IM0*IM0")
        run_imregrid(input_e13co10+"_tmp1",output_13co10+".image",input_e13co10+"_tmp2",axes=[0,1],delin=True)
        run_immath_one(input_e13co10+"_tmp2",output_e13co10+".image","sqrt(IM0)/sqrt("+str(numpix)+")",delin=True)

        # regrid 13co21 emom0
        run_immath_one(input_e13co21,input_e13co21+"_tmp1","IM0*IM0")
        run_imregrid(input_e13co21+"_tmp1",output_13co21+".image",input_e13co21+"_tmp2",axes=[0,1],delin=True)
        run_immath_one(input_e13co21+"_tmp2",output_e13co21+".image","sqrt(IM0)/sqrt("+str(numpix)+")",delin=True)

        # exportfits
        run_exportfits(output_13co10+".image",output_13co10,delin=True)
        run_exportfits(output_13co21+".image",output_13co21,delin=True)
        run_exportfits(output_e13co10+".image",output_e13co10,delin=True)
        run_exportfits(output_e13co21+".image",output_e13co21,delin=True)

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

####################
# end of ToolsNcol #
####################