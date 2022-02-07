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

        self.outpng_13co10_vs_13co21 = self.dir_products + self._read_key("outpng_13co10_vs_13co21")

        # finals
        self.final_60pc_obs      = self.dir_final + self._read_key("final_60pc_obs")
        self.final_60pc_rot      = self.dir_final + self._read_key("final_60pc_rot")
        # appendix
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
        plot_scatter     = False,
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
        # appendix_err
        do_final_60pc_err     = False,
        do_final_sim_input    = True,
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
            #
            do_final_60pc_obs     = True
            do_final_60pc_rot     = True
            # appendix
            do_final_60pc_obs_err = True
            do_final_60pc_rot_err = True
            do_final_sim_mom0     = True
            do_final_sim_emom0    = True
            #
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

        ############
        # appendix #
        ############
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

    ################
    # plot_scatter #
    ################

    def plot_scatter(
        self,
        snr=3.0,
        ):
        """
        Reference:
        https://stackoverflow.com/questions/10208814/colormap-for-errorbars-in-x-y-scatter-plot-using-matplotlib
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outmodelcube_13co10.replace(".fits","_snr10.fits"),taskname)

        this_beam = "60pc"
        lim       = [-0.4,2.3]
        xlabel    = "log$_{\mathrm{10}}$ $I_{\mathrm{^{13}CO(1-0)}}$ at " + this_beam.replace("pc"," pc")
        ylabel    = "log$_{\mathrm{10}}$ $I_{\mathrm{^{13}CO(2-1)}}$ at " + this_beam.replace("pc"," pc")

        # 13co10
        data_13co10,box = imval_all(self.outmaps_mom0_13co10.replace("???",this_beam))
        data_13co10     = data_13co10["data"] * data_13co10["mask"]
        data_13co10     = data_13co10.flatten()
        data_13co10[np.isnan(data_13co10)] = 0

        err_13co10,_ = imval_all(self.outemaps_mom0_13co10.replace("???",this_beam))
        err_13co10   = err_13co10["data"] * err_13co10["mask"]
        err_13co10   = err_13co10.flatten()
        err_13co10[np.isnan(err_13co10)] = 0

        # 13co21
        data_13co21,_ = imval_all(self.outmaps_mom0_13co21.replace("???",this_beam))
        data_13co21   = data_13co21["data"] * data_13co21["mask"]
        data_13co21   = data_13co21.flatten()
        data_13co21[np.isnan(data_13co21)] = 0

        err_13co21,_ = imval_all(self.outemaps_mom0_13co21.replace("???",this_beam))
        err_13co21   = err_13co21["data"] * err_13co21["mask"]
        err_13co21   = err_13co21.flatten()
        err_13co21[np.isnan(err_13co21)] = 0

        # coords
        data_coords = imval(self.outmaps_mom0_13co10.replace("???",this_beam),box=box)["coords"]
        ra_deg      = data_coords[:,:,0] * 180/np.pi
        ra_deg      = ra_deg.flatten()
        dec_deg     = data_coords[:,:,1] * 180/np.pi
        dec_deg     = dec_deg.flatten()
        dist_pc,_   = get_reldist_pc(ra_deg, dec_deg, self.ra_agn, self.dec_agn, self.scale_pc, 0, 0)

        # prepare
        cut  = np.where((data_13co10>abs(err_13co10)*snr)&(data_13co21>abs(err_13co21)*snr))
        x    = np.log10(data_13co10[cut])
        xerr = err_13co10[cut] / abs(data_13co10[cut])
        y    = np.log10(data_13co21[cut])
        yerr = err_13co21[cut] / abs(data_13co21[cut])
        r    = np.array(dist_pc)[cut]

        # plot
        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        ad  = [0.215,0.83,0.10,0.90]
        myax_set(ax1, "both", lim, lim, None, xlabel, ylabel, adjust=ad)

        sc = ax1.scatter(x, y, c=r, cmap="rainbow_r", lw=0, s=20, zorder=1e9)
        plt.errorbar(x, y, xerr, yerr, lw=1, capsize=0, color="grey", linestyle="None")

        """ cmap for errorbar
        clb   = plt.colorbar(sc)
        color = clb.to_rgba(r)
        for this_x, this_y, this_xerr, this_yerr, this_c in zip(x, y, xerr, yerr, color):
            plt.errorbar(this_x, this_y, this_xerr, this_yerr, lw=1, capsize=0, color=this_c)
        """

        # ann
        ax1.plot(lim, lim, "--", color="black", lw=1)

        # save
        os.system("rm -rf " + self.outpng_13co10_vs_13co21)
        plt.savefig(self.outpng_13co10_vs_13co21, dpi=self.fig_dpi)

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
        do_noclip=True,
        do_zeroclip=True,
        do_clip=True,
        do_noclip_mask=True,
        do_zeroclip_mask=True,
        do_clip_mask=True,
        rms=0.227283716202,
        snr=3,
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
        includepix = [rms*snr,1000000.]
        if do_clip==True:
            # snr = 10
            infile  = modelcube.replace(".fits","_snr10.fits")
            outfile = simmom0.replace(".fits","_clip"+str(snr)+"_snr10.fits")
            outerr  = simmom0.replace(".fits","_clip"+str(snr)+"_snr10.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image",includepix=includepix)
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_one(infile,this_mask+".image1","iif(IM0>"+str(rms*snr)+",1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

            # snr = 25
            infile  = modelcube.replace(".fits","_snr25.fits")
            outfile = simmom0.replace(".fits","_clip"+str(snr)+"_snr25.fits")
            outerr  = simmom0.replace(".fits","_clip"+str(snr)+"_snr25.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image",includepix=includepix)
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_one(infile,this_mask+".image1","iif(IM0>"+str(rms*snr)+",1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

            # snr = 50
            infile  = modelcube.replace(".fits","_snr50.fits")
            outfile = simmom0.replace(".fits","_clip"+str(snr)+"_snr50.fits")
            outerr  = simmom0.replace(".fits","_clip"+str(snr)+"_snr50.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image",includepix=includepix)
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_one(infile,this_mask+".image1","iif(IM0>"+str(rms*snr)+",1,0)")
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
        includepix = [rms*snr,1000000.]
        if do_clip_mask==True:
            # snr = 10
            infile  = modelcube.replace(".fits","_snr10.fits")
            outfile = simmom0.replace(".fits","_clip"+str(snr)+"_masked_snr10.fits")
            outerr  = simmom0.replace(".fits","_clip"+str(snr)+"_masked_snr10.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image?")
            run_immath_two(infile,mask,outfile+".image1","IM0*IM1")
            immoments(imagename=outfile+".image1",outfile=outfile+".image2",includepix=includepix)
            run_exportfits(outfile+".image2",outfile,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + outfile + ".image?")
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(infile,mask,this_mask+".image1","iif(IM0*IM1>"+str(rms*snr)+",1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

            # snr = 25
            infile  = modelcube.replace(".fits","_snr25.fits")
            outfile = simmom0.replace(".fits","_clip"+str(snr)+"_masked_snr25.fits")
            outerr  = simmom0.replace(".fits","_clip"+str(snr)+"_masked_snr25.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image?")
            run_immath_two(infile,mask,outfile+".image1","IM0*IM1")
            immoments(imagename=outfile+".image1",outfile=outfile+".image2",includepix=includepix)
            run_exportfits(outfile+".image2",outfile,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + outfile + ".image?")
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(infile,mask,this_mask+".image1","iif(IM0*IM1>"+str(rms*snr)+",1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

            # snr = 50
            infile  = modelcube.replace(".fits","_snr50.fits")
            outfile = simmom0.replace(".fits","_clip"+str(snr)+"_masked_snr50.fits")
            outerr  = simmom0.replace(".fits","_clip"+str(snr)+"_masked_snr50.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image?")
            run_immath_two(infile,mask,outfile+".image1","IM0*IM1")
            immoments(imagename=outfile+".image1",outfile=outfile+".image2",includepix=includepix)
            run_exportfits(outfile+".image2",outfile,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + outfile + ".image?")
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(infile,mask,this_mask+".image1","iif(IM0*IM1>"+str(rms*snr)+",1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

    #######################
    # simulate_mom_13co21 #
    #######################

    def simulate_mom_13co21(
        self,
        do_noclip=True,
        do_zeroclip=True,
        do_clip=True,
        do_noclip_mask=True,
        do_zeroclip_mask=True,
        do_clip_mask=True,
        rms=0.227283716202,
        snr=3,
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
        includepix = [rms*snr,1000000.]
        if do_clip==True:
            # snr = 10
            infile  = modelcube.replace(".fits","_snr10.fits")
            outfile = simmom0.replace(".fits","_clip"+str(snr)+"_snr10.fits")
            outerr  = simmom0.replace(".fits","_clip"+str(snr)+"_snr10.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image",includepix=includepix)
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_one(infile,this_mask+".image1","iif(IM0>"+str(rms*snr)+",1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

            # snr = 25
            infile  = modelcube.replace(".fits","_snr25.fits")
            outfile = simmom0.replace(".fits","_clip"+str(snr)+"_snr25.fits")
            outerr  = simmom0.replace(".fits","_clip"+str(snr)+"_snr25.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image",includepix=includepix)
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_one(infile,this_mask+".image1","iif(IM0>"+str(rms*snr)+",1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

            # snr = 50
            infile  = modelcube.replace(".fits","_snr50.fits")
            outfile = simmom0.replace(".fits","_clip"+str(snr)+"_snr50.fits")
            outerr  = simmom0.replace(".fits","_clip"+str(snr)+"_snr50.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image")
            immoments(imagename=infile,outfile=outfile+".image",includepix=includepix)
            run_exportfits(outfile+".image",outfile,delin=True,dropdeg=True,dropstokes=True)
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_one(infile,this_mask+".image1","iif(IM0>"+str(rms*snr)+",1,0)")
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
        includepix = [rms*snr,1000000.]
        if do_clip_mask==True:
            # snr = 10
            infile  = modelcube.replace(".fits","_snr10.fits")
            outfile = simmom0.replace(".fits","_clip"+str(snr)+"_masked_snr10.fits")
            outerr  = simmom0.replace(".fits","_clip"+str(snr)+"_masked_snr10.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image?")
            run_immath_two(infile,mask,outfile+".image1","IM0*IM1")
            immoments(imagename=outfile+".image1",outfile=outfile+".image2",includepix=includepix)
            run_exportfits(outfile+".image2",outfile,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + outfile + ".image?")
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(infile,mask,this_mask+".image1","iif(IM0*IM1>"+str(rms*snr)+",1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

            # snr = 25
            infile  = modelcube.replace(".fits","_snr25.fits")
            outfile = simmom0.replace(".fits","_clip"+str(snr)+"_masked_snr25.fits")
            outerr  = simmom0.replace(".fits","_clip"+str(snr)+"_masked_snr25.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image?")
            run_immath_two(infile,mask,outfile+".image1","IM0*IM1")
            immoments(imagename=outfile+".image1",outfile=outfile+".image2",includepix=includepix)
            run_exportfits(outfile+".image2",outfile,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + outfile + ".image?")
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(infile,mask,this_mask+".image1","iif(IM0*IM1>"+str(rms*snr)+",1,0)")
            immoments(this_mask+".image1",outfile=this_mask+".image2")
            run_immath_one(this_mask+".image2",this_mask+".image3","IM0/"+str(chanwidth_kms))
            run_exportfits(this_mask+".image3",this_mask,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(outfile,this_mask,outerr+".image","IM0*0+"+str(rms)+"*"+str(chanwidth_kms)+"*sqrt(IM1)")
            run_exportfits(outerr+".image",outerr,delin=True,dropdeg=True,dropstokes=True)

            # snr = 50
            infile  = modelcube.replace(".fits","_snr50.fits")
            outfile = simmom0.replace(".fits","_clip"+str(snr)+"_masked_snr50.fits")
            outerr  = simmom0.replace(".fits","_clip"+str(snr)+"_masked_snr50.fits").replace("mom0","emom0")
            os.system("rm -rf " + outfile + ".image?")
            run_immath_two(infile,mask,outfile+".image1","IM0*IM1")
            immoments(imagename=outfile+".image1",outfile=outfile+".image2",includepix=includepix)
            run_exportfits(outfile+".image2",outfile,delin=True,dropdeg=True,dropstokes=True)
            os.system("rm -rf " + outfile + ".image?")
            # error
            this_mask = "this_mask.fits"
            os.system("rm -rf " + this_mask + ".image?")
            run_immath_two(infile,mask,this_mask+".image1","iif(IM0*IM1>"+str(rms*snr)+",1,0)")
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

        """
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
        snr=3,
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

        cut = np.where((sim_mom0_13co10>=sim_emom0_13co10*snr)&(~np.isnan(np.log10(model_mom0_13co10)))&(~np.isnan(np.log10(sim_mom0_13co10)))&(sim_mom0_13co21>=sim_emom0_13co21*snr)&(~np.isnan(np.log10(model_mom0_13co21)))&(~np.isnan(np.log10(sim_mom0_13co21))))
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
        snr=3,
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

        cut = np.where((sim_mom0>=sim_emom0*snr)&(~np.isnan(np.log10(model_mom0)))&(~np.isnan(np.log10(sim_mom0))))
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