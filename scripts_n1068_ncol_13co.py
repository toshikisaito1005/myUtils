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
Toshiki Saito@Nichidai/NAOJ
"""

import os, sys, glob
import numpy as np

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

        self.cube_13co10 = self.dir_raw + self._read_key("cube_13co10")
        self.ecube_13co10 = self.dir_raw + self._read_key("ecube_13co10")
        self.cube_13co21 = self.dir_raw + self._read_key("cube_13co21")
        self.ecube_13co21 = self.dir_raw + self._read_key("ecube_13co21")

    def _set_output_fits(self):
        """
        """

        self.outcubes_13co10     = self.dir_ready + self._read_key("outcubes_13co10")
        self.outecubes_13co10    = self.dir_ready + self._read_key("outecubes_13co10")
        self.outcubes_13co21     = self.dir_ready + self._read_key("outcubes_13co21")
        self.outecubes_13co21    = self.dir_ready + self._read_key("outecubes_13co21")

        self.outmaps_mom0_13co10 = self.dir_ready + self._read_key("outmaps_13co10")
        self.outmaps_mom0_13co21 = self.dir_ready + self._read_key("outmaps_13co21")
        self.outmaps_mom1        = self.dir_ready + self._read_key("outmaps_mom1")
        self.outmaps_mom2        = self.dir_ready + self._read_key("outmaps_mom2")
        self.outmaps_ratio       = self.dir_ready + self._read_key("outmaps_ratio")
        self.outmaps_13co_trot   = self.dir_ready + self._read_key("outmaps_13co_trot")
        self.outmaps_13co_ncol   = self.dir_ready + self._read_key("outmaps_13co_ncol")

        self.outemaps_mom0_13co10 = self.dir_ready + self._read_key("outemaps_13co10")
        self.outemaps_mom0_13co21 = self.dir_ready + self._read_key("outemaps_13co21")
        self.outemaps_mom1        = self.dir_ready + self._read_key("outemaps_mom1")
        self.outemaps_mom2        = self.dir_ready + self._read_key("outemaps_mom2")
        self.outemaps_ratio       = self.dir_ready + self._read_key("outemaps_ratio")
        self.outemaps_13co_trot   = self.dir_ready + self._read_key("outemaps_13co_trot")
        self.outemaps_13co_ncol   = self.dir_ready + self._read_key("outemaps_13co_ncol")

        self.outmodelcube_13co10  = self.dir_ready + self._read_key("outmodelcube_13co10")
        self.outmodelcube_13co21  = self.dir_ready + self._read_key("outmodelcube_13co21")
        self.outmodelmom0_13co10  = self.dir_ready + self._read_key("outmodelmom0_13co10")
        self.outmodelmom0_13co21  = self.dir_ready + self._read_key("outmodelmom0_13co21")
        self.outsimumom0_13co10   = self.dir_ready + self._read_key("outsimumom0_13co10")
        self.outsimumom0_13co21   = self.dir_ready + self._read_key("outsimumom0_13co21")

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
        self.scale_pc    = float(self._read_key("scale", "gal"))
        self.scale_kpc   = self.scale_pc / 1000.

        self.snr_mom     = float(self._read_key("snr_mom"))
        self.r_cnd       = float(self._read_key("r_cnd_as")) * self.scale_pc / 1000. # kpc
        self.r_cnd_as    = float(self._read_key("r_cnd_as"))
        self.r_sbr       = float(self._read_key("r_sbr_as")) * self.scale_pc / 1000. # kpc
        self.r_sbr_as    = float(self._read_key("r_sbr_as"))

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

        self.outpng_modelmom0_13co10 = self.dir_products + self._read_key("outpng_modelmom0_13co10")
        self.outpng_modelmom0_13co21 = self.dir_products + self._read_key("outpng_modelmom0_13co21")
        self.outpng_simumom0_13co10  = self.dir_products + self._read_key("outpng_simumom0_13co10")
        self.outpng_simumom0_13co21  = self.dir_products + self._read_key("outpng_simumom0_13co21")

        # finals
        self.final_60pc_obs      = self.dir_final + self._read_key("final_60pc_obs")
        self.final_60pc_rot      = self.dir_final + self._read_key("final_60pc_rot")
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

    ####################
    # run_ngc1068_ncol #
    ####################

    def run_ngc1068_ncol(
        self,
        # analysis
        do_prepare       = False,
        do_fitting       = False, # after do_prepare
        # mom0 creation simulation
        do_create_models = False, # after do_prepare
        do_simulate_mom  = False, # after do_create_models
        # plot figures in paper
        plot_showcase    = False, # after do_fitting
        plot_showsim     = False, # after do_simulate_mom
        do_imagemagick   = False,
        immagick_all     = False,
        # supplement
        ):
        """
        This method runs all the methods which will create figures in the paper.
        """

        # analysis
        if do_prepare==True:
            self.align_maps()

        if do_fitting==True:
            self.multi_fitting()

        if do_create_models==True:
            self.create_model_cubes()
            self.add_noise_to_models()

        if do_simulate_mom==True:
            self.simualte_mom()

        # plot figures in paper
        if plot_showcase==True:
            self.showcase()

        if plot_showsim==True:
            self.showsim()

        if do_imagemagick==True:
            self.immagick_figures(do_all=immagick_all,delin=False)

    ####################
    # immagick_figures #
    ####################

    def immagick_figures(
        self,
        delin                 = False,
        do_all                = False,
        #
        do_final_60pc_obs     = False,
        do_final_60pc_rot     = False,
        #
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
            #
            do_final_60pc_obs     = True
            do_final_60pc_rot     = True
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
                self.box_map_nox,
                self.box_map_noxy,
                delin=delin,
                )
            combine_three_png(
                self.outpng_emom0_13co10.replace("???","60pc"),
                self.outpng_emom0_13co21.replace("???","60pc"),
                self.outpng_eratio.replace("???","60pc"),
                self.final_60pc_obs+"_tmp3.png",
                self.box_map_nox,
                self.box_map_noxy,
                self.box_map_noxy,
                delin=delin,
                )
            combine_two_png(
                self.outpng_emom1.replace("???","60pc"),
                self.outpng_emom2.replace("???","60pc"),
                self.final_60pc_obs+"_tmp4.png",
                self.box_map,
                self.box_map_noy,
                delin=delin,
                )

            #
            combine_two_png(
                self.final_60pc_obs+"_tmp1.png",
                self.final_60pc_obs+"_tmp2.png",
                self.final_60pc_obs+"_tmp12.png",
                "1000000x10000000+0+0",
                "1000000x10000000+0+0",
                axis="column",
                delin=True,
                )
            combine_two_png(
                self.final_60pc_obs+"_tmp3.png",
                self.final_60pc_obs+"_tmp4.png",
                self.final_60pc_obs+"_tmp34.png",
                "1000000x10000000+0+0",
                "1000000x10000000+0+0",
                axis="column",
                delin=True,
                )
            combine_two_png(
                self.final_60pc_obs+"_tmp12.png",
                self.final_60pc_obs+"_tmp34.png",
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
                self.final_60pc_rot+"_tmp1.png",
                self.box_map_nox,
                self.box_map_noxy,
                delin=delin,
                )
            combine_two_png(
                self.outpng_e13co_trot.replace("???","60pc"),
                self.outpng_e13co_ncol.replace("???","60pc"),
                self.final_60pc_rot+"_tmp2.png",
                self.box_map,
                self.box_map_noy,
                delin=delin,
                )

            #
            combine_two_png(
                self.final_60pc_rot+"_tmp1.png",
                self.final_60pc_rot+"_tmp2.png",
                self.final_60pc_rot,
                "1000000x10000000+0+0",
                "1000000x10000000+0+0",
                axis="column",
                delin=True,
                )

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

    ################
    # simualte_mom #
    ################

    def simualte_mom(
        self,
        do_noclip=True,
        do_zeroclip=True,
        snr=3,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outmodelcube_13co10.replace(".fits","_snr10.fits"),taskname)

        ####################
        # model input mom0 #
        ####################
        infile  = self.outmodelcube_13co10
        outfile = self.outmodelmom0_13co10
        os.system("rm -rf " + outfile + ".image")
        immoments(imagename=infile,outfile=outfile+".image")
        run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)

        #############
        # do_noclip #
        #############
        if do_noclip==True:
            # snr = 5
            infile  = self.outmodelcube_13co10.replace(".fits","_snr10.fits")
            outfile = self.outsimumom0_13co10.replace(".fits","_noclip_snr10.fits")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image")
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)
            # snr = 25
            infile  = self.outmodelcube_13co10.replace(".fits","_snr25.fits")
            outfile = self.outsimumom0_13co10.replace(".fits","_noclip_snr25.fits")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image")
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)
            # snr = 125
            infile  = self.outmodelcube_13co10.replace(".fits","_snr50.fits")
            outfile = self.outsimumom0_13co10.replace(".fits","_noclip_snr50.fits")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image")
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)

        ###############
        # do_zeroclip # 0.227283716202
        ###############
        includepix = [0.0,1000000.]
        if do_noclip==True:
            # snr = 5
            infile  = self.outmodelcube_13co10.replace(".fits","_snr10.fits")
            outfile = self.outsimumom0_13co10.replace(".fits","_zeroclip_snr10.fits")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image",includepix=includepix)
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)
            # snr = 25
            infile  = self.outmodelcube_13co10.replace(".fits","_snr25.fits")
            outfile = self.outsimumom0_13co10.replace(".fits","_zeroclip_snr25.fits")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image",includepix=includepix)
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)
            # snr = 125
            infile  = self.outmodelcube_13co10.replace(".fits","_snr50.fits")
            outfile = self.outsimumom0_13co10.replace(".fits","_zeroclip_snr50.fits")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image",includepix=includepix)
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)

        ###########
        # do_clip #
        ###########
        includepix = [0.227283716202*snr,1000000.]
        if do_noclip==True:
            # snr = 5
            infile  = self.outmodelcube_13co10.replace(".fits","_snr10.fits")
            outfile = self.outsimumom0_13co10.replace(".fits","_clip"+str(snr)+"_snr10.fits")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image",includepix=includepix)
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)
            # snr = 25
            infile  = self.outmodelcube_13co10.replace(".fits","_snr25.fits")
            outfile = self.outsimumom0_13co10.replace(".fits","_clip"+str(snr)+"_snr25.fits")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image",includepix=includepix)
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)
            # snr = 125
            infile  = self.outmodelcube_13co10.replace(".fits","_snr50.fits")
            outfile = self.outsimumom0_13co10.replace(".fits","_clip"+str(snr)+"_snr50.fits")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image",includepix=includepix)
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)

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
        snr_cut=5.0,
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
            "iif(IM0/IM1>"+str(snr_cut)+",IM0,0)",
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
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outsimumom0_13co10.replace(".fits","_just_snr10.fits"),taskname)

        # prepare
        imcontour1 = self.outmodelmom0_13co10

        self._showcase_one(
            self.outmodelmom0_13co10,
            imcontour1,
            self.outpng_modelmom0_13co10,
            "Input model mom0",
            "(K km s$^{-1}$)",
            [0,100],
            )

        self._showcase_one(
            self.outsimumom0_13co10.replace(".fits","_noclip_snr10.fits"),
            imcontour1,
            self.outpng_simumom0_13co10.replace(".png","_noclip_snr10.png"),
            "no-clip: SNR=10 mom0",
            "(K km s$^{-1}$)",
            [0,100],
            )

        self._showcase_one(
            self.outsimumom0_13co10.replace(".fits","_noclip_snr25.fits"),
            imcontour1,
            self.outpng_simumom0_13co10.replace(".png","_noclip_snr25.png"),
            "no-clip: SNR=25 mom0",
            "(K km s$^{-1}$)",
            [0,250],
            )

        self._showcase_one(
            self.outsimumom0_13co10.replace(".fits","_noclip_snr50.fits"),
            imcontour1,
            self.outpng_simumom0_13co10.replace(".png","_noclip_snr50.png"),
            "no-clip: SNR=50 mom0",
            "(K km s$^{-1}$)",
            [0,500],
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
            if this_beam!="60pc":
                continue

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
            #if this_beam!="60pc":
            #    continue

            print("# multi_fitting for cubes at " + this_beam)

            # input
            cube_13co10  = self.outcubes_13co10.replace("???",this_beam)
            cube_13co21  = self.outcubes_13co21.replace("???",this_beam)
            ecube_13co10 = self.outecubes_13co10.replace("???",this_beam)
            ecube_13co21 = self.outecubes_13co21.replace("???",this_beam)

            # output
            rotation_13co21_13co10(
                cube_13co10,
                cube_13co21,
                ecube_13co10,
                ecube_13co21,
                ra_cnt=self.ra_agn,
                dec_cnt=self.dec_agn,
                snr=3.0,
                snr_limit=3.0,
                )

            #
            os.system("mv mom0_low.fits " + self.outmaps_mom0_13co10.replace("???",this_beam))
            os.system("mv mom0_high.fits " + self.outmaps_mom0_13co21.replace("???",this_beam))
            os.system("mv mom1.fits " + self.outmaps_mom1.replace("???",this_beam))
            os.system("mv mom2.fits " + self.outmaps_mom2.replace("???",this_beam))
            os.system("mv ratio.fits " + self.outmaps_ratio.replace("???",this_beam))
            os.system("mv Trot.fits " + self.outmaps_13co_trot.replace("???",this_beam))
            os.system("mv logN.fits " + self.outmaps_13co_ncol.replace("???",this_beam))

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