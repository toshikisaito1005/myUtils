"""
Python class for the PHANGS-R21 project

requirements:
CASA Version 5.4.0-70, ananlysisUtils, astropy

data:


usage:
> import os
> from scripts_phangs_r21 import ToolsR21 as tools
>
> # key
> tl = tools(
>     refresh     = False,
>     keyfile_gal = "/home02/saitots/myUtils/keys_projects/galkey_phangs.txt",
>     keyfile_fig = "/home02/saitots/myUtils/keys_projects/key_phangs_r21.txt",
>     )
>
> # main
> tl.run_phangs_r21(
>     do_all           = True,
>     # analysis
>     do_prepare       = True,
>     # plot
>     plot_showcase    = True,
>     # supplement
>     )
>
> os.system("rm -rf *.last")

paper drafts:
Date         Filename                To

history:
2017-11-01   created
2022-07-28   constructed align_cubes
2022-07-29   constructed multismooth
2022-10-11   constructed multimoments
2022-10-14   constrcuted do_align_other
2022-10-17   constrcuted plot_noise
Toshiki Saito@NAOJ
"""

import os, sys, glob
import numpy as np

from mycasa_rotation import *
from mycasa_sampling import *
from mycasa_lowess import *
from mycasa_tasks import *
from mycasa_plots import *
from mycasa_pca import *

############
# ToolsR21 #2
############
class ToolsR21():
    """
    Class for the PHANGS-R21 project.
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
        self.dir_raw         = None
        self.dir_ready       = None
        self.dir_other       = None
        self.dir_products    = None
        self.fig_dpi         = 200
        self.legend_fontsize = 20

        # import parameters
        if keyfile_fig is not None:
            self.modname = "ToolsR21."
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
        self.dir_cprops   = self.dir_proj + self._read_key("dir_cprops")
        self.dir_env      = self.dir_proj + self._read_key("dir_env")
        self.dir_halpha   = self.dir_proj + self._read_key("dir_halpha")
        self.dir_wise     = self.dir_proj + self._read_key("dir_wise")
        self.dir_ready    = self.dir_proj + self._read_key("dir_ready")
        self.dir_products = self.dir_proj + self._read_key("dir_products")
        self.dir_products_txt = self.dir_proj + self._read_key("dir_products_txt")
        self.dir_final    = self.dir_proj + self._read_key("dir_final")

        self._create_dir(self.dir_ready)
        self._create_dir(self.dir_products)
        self._create_dir(self.dir_products_txt)
        self._create_dir(self.dir_final)

    def _set_input_fits(self):
        """
        """

        self.cube_co10_n0628 = self.dir_raw + self._read_key("cube_co10_n0628")
        self.cube_co10_n3627 = self.dir_raw + self._read_key("cube_co10_n3627")
        self.cube_co10_n4254 = self.dir_raw + self._read_key("cube_co10_n4254")
        self.cube_co10_n4321 = self.dir_raw + self._read_key("cube_co10_n4321")
        self.cube_co21_n0628 = self.dir_raw + self._read_key("cube_co21_n0628")
        self.cube_co21_n3627 = self.dir_raw + self._read_key("cube_co21_n3627")
        self.cube_co21_n4254 = self.dir_raw + self._read_key("cube_co21_n4254")
        self.cube_co21_n4321 = self.dir_raw + self._read_key("cube_co21_n4321")

        self.wise1_n0628     = self.dir_wise + self._read_key("wise1_n0628")
        self.wise1_n3627     = self.dir_wise + self._read_key("wise1_n3627")
        self.wise1_n4254     = self.dir_wise + self._read_key("wise1_n4254")
        self.wise1_n4321     = self.dir_wise + self._read_key("wise1_n4321")
        self.wise2_n0628     = self.dir_wise + self._read_key("wise2_n0628")
        self.wise2_n3627     = self.dir_wise + self._read_key("wise2_n3627")
        self.wise2_n4254     = self.dir_wise + self._read_key("wise2_n4254")
        self.wise2_n4321     = self.dir_wise + self._read_key("wise2_n4321")
        self.wise3_n0628     = self.dir_wise + self._read_key("wise3_n0628")
        self.wise3_n3627     = self.dir_wise + self._read_key("wise3_n3627")
        self.wise3_n4254     = self.dir_wise + self._read_key("wise3_n4254")
        self.wise3_n4321     = self.dir_wise + self._read_key("wise3_n4321")

        self.cprops_table_n0628 = self.dir_cprops + self._read_key("cprops_n0628")
        self.cprops_table_n3627 = self.dir_cprops + self._read_key("cprops_n3627")
        self.cprops_table_n4254 = self.dir_cprops + self._read_key("cprops_n4254")
        self.cprops_table_n4321 = self.dir_cprops + self._read_key("cprops_n4321")

        self.env_bulge_n0628 = self.dir_env + self._read_key("env_bulge_n0628")
        self.env_bulge_n3627 = self.dir_env + self._read_key("env_bulge_n3627")
        self.env_bulge_n4254 = self.dir_env + self._read_key("env_bulge_n4254")
        self.env_bulge_n4321 = self.dir_env + self._read_key("env_bulge_n4321")
        self.env_arm_n0628   = self.dir_env + self._read_key("env_arm_n0628")
        self.env_arm_n3627   = self.dir_env + self._read_key("env_arm_n3627")
        self.env_arm_n4254   = self.dir_env + self._read_key("env_arm_n4254")
        self.env_arm_n4321   = self.dir_env + self._read_key("env_arm_n4321")
        self.env_bar_n0628   = self.dir_env + self._read_key("env_bar_n0628")
        self.env_bar_n3627   = self.dir_env + self._read_key("env_bar_n3627")
        self.env_bar_n4254   = self.dir_env + self._read_key("env_bar_n4254")
        self.env_bar_n4321   = self.dir_env + self._read_key("env_bar_n4321")

        self.halpha_mask_n0628 = self.dir_halpha + self._read_key("halpha_mask_n0628")
        self.halpha_mask_n3627 = self.dir_halpha + self._read_key("halpha_mask_n3627")
        self.halpha_mask_n4254 = self.dir_halpha + self._read_key("halpha_mask_n4254")
        self.halpha_mask_n4321 = self.dir_halpha + self._read_key("halpha_mask_n4321")

    def _set_output_fits(self):
        """
        """

        self.outcube_co10_n0628   = self.dir_ready + self._read_key("outcube_co10_n0628")
        self.outcube_co10_n3627   = self.dir_ready + self._read_key("outcube_co10_n3627")
        self.outcube_co10_n4254   = self.dir_ready + self._read_key("outcube_co10_n4254")
        self.outcube_co10_n4321   = self.dir_ready + self._read_key("outcube_co10_n4321")

        self.outcube_co21_n0628   = self.dir_ready + self._read_key("outcube_co21_n0628")
        self.outcube_co21_n3627   = self.dir_ready + self._read_key("outcube_co21_n3627")
        self.outcube_co21_n4254   = self.dir_ready + self._read_key("outcube_co21_n4254")
        self.outcube_co21_n4321   = self.dir_ready + self._read_key("outcube_co21_n4321")

        self.outfits_wise1_n0628  = self.dir_ready + self._read_key("outfits_wise1_n0628")
        self.outfits_wise1_n3627  = self.dir_ready + self._read_key("outfits_wise1_n3627")
        self.outfits_wise1_n4254  = self.dir_ready + self._read_key("outfits_wise1_n4254")
        self.outfits_wise1_n4321  = self.dir_ready + self._read_key("outfits_wise1_n4321")

        self.outfits_wise2_n0628  = self.dir_ready + self._read_key("outfits_wise2_n0628")
        self.outfits_wise2_n3627  = self.dir_ready + self._read_key("outfits_wise2_n3627")
        self.outfits_wise2_n4254  = self.dir_ready + self._read_key("outfits_wise2_n4254")
        self.outfits_wise2_n4321  = self.dir_ready + self._read_key("outfits_wise2_n4321")

        self.outfits_wise3_n0628  = self.dir_ready + self._read_key("outfits_wise3_n0628")
        self.outfits_wise3_n3627  = self.dir_ready + self._read_key("outfits_wise3_n3627")
        self.outfits_wise3_n4254  = self.dir_ready + self._read_key("outfits_wise3_n4254")
        self.outfits_wise3_n4321  = self.dir_ready + self._read_key("outfits_wise3_n4321")

        self.outmom_co10_n0628    = self.outcube_co10_n0628.replace(".image",".momX")
        self.outmom_co10_n3627    = self.outcube_co10_n3627.replace(".image",".momX")
        self.outmom_co10_n4254    = self.outcube_co10_n4254.replace(".image",".momX")
        self.outmom_co10_n4321    = self.outcube_co10_n4321.replace(".image",".momX")

        self.outmom_co21_n0628    = self.outcube_co21_n0628.replace(".image",".momX")
        self.outmom_co21_n3627    = self.outcube_co21_n3627.replace(".image",".momX")
        self.outmom_co21_n4254    = self.outcube_co21_n4254.replace(".image",".momX")
        self.outmom_co21_n4321    = self.outcube_co21_n4321.replace(".image",".momX")

        self.outfits_cprops_n0628 = self.dir_ready + self._read_key("outfits_cprops_n0628")
        self.outfits_cprops_n3627 = self.dir_ready + self._read_key("outfits_cprops_n3627")
        self.outfits_cprops_n4254 = self.dir_ready + self._read_key("outfits_cprops_n4254")
        self.outfits_cprops_n4321 = self.dir_ready + self._read_key("outfits_cprops_n4321")

        self.outfits_env_n0628    = self.dir_ready + self._read_key("outfits_env_n0628")
        self.outfits_env_n3627    = self.dir_ready + self._read_key("outfits_env_n3627")
        self.outfits_env_n4254    = self.dir_ready + self._read_key("outfits_env_n4254")
        self.outfits_env_n4321    = self.dir_ready + self._read_key("outfits_env_n4321")

        self.outfits_halpha_n0628 = self.dir_ready + self._read_key("outfits_halpha_n0628")
        self.outfits_halpha_n3627 = self.dir_ready + self._read_key("outfits_halpha_n3627")
        self.outfits_halpha_n4254 = self.dir_ready + self._read_key("outfits_halpha_n4254")
        self.outfits_halpha_n4321 = self.dir_ready + self._read_key("outfits_halpha_n4321")

        self.outfits_r21_n0628    = self.dir_ready + self._read_key("outfits_r21_n0628")
        self.outfits_r21_n3627    = self.dir_ready + self._read_key("outfits_r21_n3627")
        self.outfits_r21_n4254    = self.dir_ready + self._read_key("outfits_r21_n4254")
        self.outfits_r21_n4321    = self.dir_ready + self._read_key("outfits_r21_n4321")
        self.outfits_t21_n0628    = self.dir_ready + self._read_key("outfits_t21_n0628")
        self.outfits_t21_n3627    = self.dir_ready + self._read_key("outfits_t21_n3627")
        self.outfits_t21_n4254    = self.dir_ready + self._read_key("outfits_t21_n4254")
        self.outfits_t21_n4321    = self.dir_ready + self._read_key("outfits_t21_n4321")

        self.outfits_er21_n0628   = self.dir_ready + self._read_key("outfits_er21_n0628")
        self.outfits_er21_n3627   = self.dir_ready + self._read_key("outfits_er21_n3627")
        self.outfits_er21_n4254   = self.dir_ready + self._read_key("outfits_er21_n4254")
        self.outfits_er21_n4321   = self.dir_ready + self._read_key("outfits_er21_n4321")
        self.outfits_et21_n0628   = self.dir_ready + self._read_key("outfits_et21_n0628")
        self.outfits_et21_n3627   = self.dir_ready + self._read_key("outfits_et21_n3627")
        self.outfits_et21_n4254   = self.dir_ready + self._read_key("outfits_et21_n4254")
        self.outfits_et21_n4321   = self.dir_ready + self._read_key("outfits_et21_n4321")

        self.outfits_r21hl_n0628  = self.dir_ready + self._read_key("outfits_r21hl_n0628")
        self.outfits_r21hl_n3627  = self.dir_ready + self._read_key("outfits_r21hl_n3627")
        self.outfits_r21hl_n4254  = self.dir_ready + self._read_key("outfits_r21hl_n4254")
        self.outfits_r21hl_n4321  = self.dir_ready + self._read_key("outfits_r21hl_n4321")

        self.outfits_bulge_n0628  = self.dir_ready + self._read_key("outfits_bulge_n0628")
        self.outfits_bulge_n3627  = self.dir_ready + self._read_key("outfits_bulge_n3627")
        self.outfits_bulge_n4254  = self.dir_ready + self._read_key("outfits_bulge_n4254")
        self.outfits_bulge_n4321  = self.dir_ready + self._read_key("outfits_bulge_n4321")

    def _set_input_param(self):
        """
        """

        self.ra_n0628          = self._read_key("ra_n0628", "gal").split("deg")[0]
        self.ra_n3627          = self._read_key("ra_n3627", "gal").split("deg")[0]
        self.ra_n4254          = self._read_key("ra_n4254", "gal").split("deg")[0]
        self.ra_n4321          = self._read_key("ra_n4321", "gal").split("deg")[0]

        self.dec_n0628         = self._read_key("dec_n0628", "gal").split("deg")[0]
        self.dec_n3627         = self._read_key("dec_n3627", "gal").split("deg")[0]
        self.dec_n4254         = self._read_key("dec_n4254", "gal").split("deg")[0]
        self.dec_n4321         = self._read_key("dec_n4321", "gal").split("deg")[0]

        self.scale_n0628       = float(self._read_key("scale_n0628", "gal"))
        self.scale_n3627       = float(self._read_key("scale_n3627", "gal"))
        self.scale_n4254       = float(self._read_key("scale_n4254", "gal"))
        self.scale_n4321       = float(self._read_key("scale_n4321", "gal"))

        self.basebeam_n0628    = float(self._read_key("basebeam_n0628"))
        self.basebeam_n3627    = float(self._read_key("basebeam_n3627"))
        self.basebeam_n4254    = float(self._read_key("basebeam_n4254"))
        self.basebeam_n4321    = float(self._read_key("basebeam_n4321"))

        self.imsize_n0628      = float(self._read_key("imsize_n0628"))
        self.imsize_n3627      = float(self._read_key("imsize_n3627"))
        self.imsize_n4254      = float(self._read_key("imsize_n4254"))
        self.imsize_n4321      = float(self._read_key("imsize_n4321"))

        self.chans_n0628       = self._read_key("chans_n0628")
        self.chans_n3627       = self._read_key("chans_n3627")
        self.chans_n4254       = self._read_key("chans_n4254")
        self.chans_n4321       = self._read_key("chans_n4321")

        self.beams_n0628       = [float(s) for s in self._read_key("beams_n0628").split(",")]
        self.beams_n3627       = [float(s) for s in self._read_key("beams_n3627").split(",")]
        self.beams_n4254       = [float(s) for s in self._read_key("beams_n4254").split(",")]
        self.beams_n4321       = [float(s) for s in self._read_key("beams_n4321").split(",")]

        self.freq_co10         = 115.27120
        self.freq_co21         = 230.53800

        self.snr_mom           = 4.0
        self.snr_gmc           = 5.0
        self.snr_ratio         = 3.0

        self.nchan_thres_n0628 = 2
        self.nchan_thres_n3627 = 3
        self.nchan_thres_n4254 = 3
        self.nchan_thres_n4321 = 2

        self.beam_wise_n0628   = 11.5
        self.beam_wise_n3627   = 10.0
        self.beam_wise_n4254   = 8.7
        self.beam_wise_n4321   = 7.5

        self.id_bulge_n0628    = self._read_key("id_bulge_n0628")
        self.id_bulge_n3627    = self._read_key("id_bulge_n3627")
        self.id_bulge_n4254    = self._read_key("id_bulge_n4254")
        self.id_bulge_n4321    = self._read_key("id_bulge_n4321")
        self.id_arm_n0628      = self._read_key("id_arm_n0628")
        self.id_arm_n3627      = self._read_key("id_arm_n3627")
        self.id_arm_n4254      = self._read_key("id_arm_n4254")
        self.id_arm_n4321      = self._read_key("id_arm_n4321")
        self.id_bar_n0628      = self._read_key("id_bar_n0628")
        self.id_bar_n3627      = self._read_key("id_bar_n3627")
        self.id_bar_n4254      = self._read_key("id_bar_n4254")
        self.id_bar_n4321      = self._read_key("id_bar_n4321")

    def _set_output_txt_png(self):
        """
        """

        # common set
        self.c_n0628         = "tomato"
        self.c_n3627         = "purple"
        self.c_n4254         = "forestgreen"
        self.c_n4321         = "deepskyblue"
        self.text_back_alpha = 0.9

        # output txt and png
        self.outpng_noise_hist   = self.dir_products + self._read_key("outpng_noise_hist")
        self.noise_hist_xmax_snr = 7.5
        self.noise_hist_bins     = 500
        self.noise_hist_snr4plt  = 2.5

        self.outpng_noise_vs_beam     = self.dir_products + self._read_key("outpng_noise_vs_beam")
        self.noise_vs_beam_co10_n0628 = self.dir_products_txt + self._read_key("noise_vs_beam_co10_n0628")
        self.noise_vs_beam_co10_n3627 = self.dir_products_txt + self._read_key("noise_vs_beam_co10_n3627")
        self.noise_vs_beam_co10_n4254 = self.dir_products_txt + self._read_key("noise_vs_beam_co10_n4254")
        self.noise_vs_beam_co10_n4321 = self.dir_products_txt + self._read_key("noise_vs_beam_co10_n4321")
        self.noise_vs_beam_co21_n0628 = self.dir_products_txt + self._read_key("noise_vs_beam_co21_n0628")
        self.noise_vs_beam_co21_n3627 = self.dir_products_txt + self._read_key("noise_vs_beam_co21_n3627")
        self.noise_vs_beam_co21_n4254 = self.dir_products_txt + self._read_key("noise_vs_beam_co21_n4254")
        self.noise_vs_beam_co21_n4321 = self.dir_products_txt + self._read_key("noise_vs_beam_co21_n4321")
        self.noise_vs_beam_snr4fit    = 0.5

    ##################
    # run_phangs_r21 #
    ##################

    def run_phangs_r21(
        self,
        do_all         = False,
        # analysis
        do_align       = False,
        do_multismooth = False,
        do_moments     = False,
        do_align_other = False,
        # plot figures in paper
        plot_noise     = False,
        # supplement
        ):
        """
        This method runs all the methods which will create figures in the paper.
        """

        self.do_ngc0628 = True
        self.do_ngc3627 = True
        self.do_ngc4254 = True
        self.do_ngc4321 = True

        if do_all==True:
            self.do_ngc0628 = True
            self.do_ngc3627 = True
            self.do_ngc4254 = True
            self.do_ngc4321 = True
            do_align        = True
            do_multismooth  = True
            do_moments      = True
            do_align_other  = True

        # analysis
        if do_align==True:
            self.align_cubes()
        
        if do_multismooth==True:
            self.multismooth()

        if do_moments==True:
            self.multimoments()

        if do_align_other==True:
            self.align_wise(skip=False)
            self.align_cprops(skip=False)
            self.align_env(skip=False)
            self.align_halpha(skip=False)
            self.align_r21(skip=False)
            self.align_bulge(skip=False)

        # plot figures in paper
        if plot_noise==True:
            #self.plot_noise_hist()
            self.plot_noise_vs_beam()

    #####################
    #####################
    ### plotting part ###
    #####################
    #####################

    ######################
    # plot_noise_vs_beam #
    ######################

    def plot_noise_vs_beam(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outcube_co10_n0628,taskname)

        ###########
        # prepare #
        ###########

        list_rms_co10_n0628 = self._measure_log_rms(
            self.outcube_co10_n0628.replace(str(self.basebeam_n0628).replace(".","p").zfill(4),"????").replace(".image","_k.image"),
            self.beams_n0628[:-2],
            self.noise_vs_beam_co10_n0628,
            )
        list_rms_co21_n0628 = self._measure_log_rms(
            self.outcube_co21_n0628.replace(str(self.basebeam_n0628).replace(".","p").zfill(4),"????").replace(".image","_k.image"),
            self.beams_n0628[:-2],
            self.noise_vs_beam_co21_n0628,
            )

        list_rms_co10_n3627 = self._measure_log_rms(
            self.outcube_co10_n3627.replace(str(self.basebeam_n3627).replace(".","p").zfill(4),"????").replace(".image","_k.image"),
            self.beams_n3627[:-2],
            self.noise_vs_beam_co10_n3627,
            )
        list_rms_co21_n3627 = self._measure_log_rms(
            self.outcube_co21_n3627.replace(str(self.basebeam_n3627).replace(".","p").zfill(4),"????").replace(".image","_k.image"),
            self.beams_n3627[:-2],
            self.noise_vs_beam_co21_n3627,
            )

        list_rms_co10_n4254 = self._measure_log_rms(
            self.outcube_co10_n4254.replace(str(self.basebeam_n4254).replace(".","p").zfill(4),"????").replace(".image","_k.image"),
            self.beams_n4254[:-2],
            self.noise_vs_beam_co10_n4254,
            )
        list_rms_co21_n4254 = self._measure_log_rms(
            self.outcube_co21_n4254.replace(str(self.basebeam_n4254).replace(".","p").zfill(4),"????").replace(".image","_k.image"),
            self.beams_n4254[:-2],
            self.noise_vs_beam_co21_n4254,
            )

        list_rms_co10_n4321 = self._measure_log_rms(
            self.outcube_co10_n4321.replace(str(self.basebeam_n4321).replace(".","p").zfill(4),"????").replace(".image","_k.image"),
            self.beams_n4321[:-2],
            self.noise_vs_beam_co10_n4321,
            )
        list_rms_co21_n4321 = self._measure_log_rms(
            self.outcube_co21_n4321.replace(str(self.basebeam_n4321).replace(".","p").zfill(4),"????").replace(".image","_k.image"),
            self.beams_n4321[:-2],
            self.noise_vs_beam_co21_n4321,
            )

        xlim   = [2,28]
        ylim   = [-3.6,-0.8]
        title  = "(b) Sensitivity vs. Beam Size"
        xlabel = "Beam size (arcsec)"
        ylabel = "log rms per voxel (K)"

        ########
        # plot #
        ########

        plt.figure(figsize=(13,10))
        gs = gridspec.GridSpec(nrows=10, ncols=10)
        ax = plt.subplot(gs[0:10,0:10])

        ad = [0.215,0.83,0.10,0.90]
        myax_set(ax, "both", xlim, ylim, title, xlabel, ylabel, adjust=ad)

        print(self.beams_n0628[:-2], list_rms_co10_n0628[:,1])
        # plot co10 rms
        ax.plot(self.beams_n0628[:-2], list_rms_co10_n0628[:,1], "o-", color=self.c_n0628, markeredgewidth=0, markersize = 20, lw=3, label = "NGC 0628 CO(1-0)")
        ax.plot(self.beams_n3627[:-2], list_rms_co10_n3627[:,1], "o-", color=self.c_n3627, markeredgewidth=0, markersize = 20, lw=3, label = "NGC 3627 CO(1-0)")
        ax.plot(self.beams_n4254[:-2], list_rms_co10_n4254[:,1], "o-", color=self.c_n4254, markeredgewidth=0, markersize = 20, lw=3, label = "NGC 4254 CO(1-0)")
        ax.plot(self.beams_n4321[:-2], list_rms_co10_n4321[:,1], "o-", color=self.c_n4321, markeredgewidth=0, markersize = 20, lw=3, label = "NGC 4321 CO(1-0)")
        # plot co21 rms
        ax.plot(self.beams_n0628[:-2], list_rms_co21_n0628[:,1], "s--", color=self.c_n0628, markeredgewidth=0, markersize = 20, lw=3, label = "NGC 0628 CO(2-1)")
        ax.plot(self.beams_n3627[:-2], list_rms_co21_n3627[:,1], "s--", color=self.c_n3627, markeredgewidth=0, markersize = 20, lw=3, label = "NGC 3627 CO(2-1)")
        ax.plot(self.beams_n4254[:-2], list_rms_co21_n4254[:,1], "s--", color=self.c_n4254, markeredgewidth=0, markersize = 20, lw=3, label = "NGC 4254 CO(2-1)")
        ax.plot(self.beams_n4321[:-2], list_rms_co21_n4321[:,1], "s--", color=self.c_n4321, markeredgewidth=0, markersize = 20, lw=3, label = "NGC 4321 CO(2-1)")

        # text
        t=ax.text(0.95, 0.93, "NGC 0628", color=self.c_n0628, horizontalalignment="right", transform=ax.transAxes, size=self.legend_fontsize, fontweight="bold")
        t.set_bbox(dict(facecolor="white", alpha=self.text_back_alpha, lw=0))
        t=ax.text(0.95, 0.88, "NGC 3627", color=self.c_n3627, horizontalalignment="right", transform=ax.transAxes, size=self.legend_fontsize, fontweight="bold")
        t.set_bbox(dict(facecolor="white", alpha=self.text_back_alpha, lw=0))
        t=ax.text(0.95, 0.83, "NGC 4254", color=self.c_n4254, horizontalalignment="right", transform=ax.transAxes, size=self.legend_fontsize, fontweight="bold")
        t.set_bbox(dict(facecolor="white", alpha=self.text_back_alpha, lw=0))
        t=ax.text(0.95, 0.78, "NGC 4321", color=self.c_n4321, horizontalalignment="right", transform=ax.transAxes, size=self.legend_fontsize, fontweight="bold")
        t.set_bbox(dict(facecolor="white", alpha=self.text_back_alpha, lw=0))

        ax.text(0.55, 0.90, "CO(1-0) datacubes", color="black", horizontalalignment="right", transform=ax.transAxes, size=self.legend_fontsize, fontweight="bold")
        ax.text(0.48, 0.25, "CO(2-1) datacubes", color="black", horizontalalignment="right", transform=ax.transAxes, size=self.legend_fontsize, fontweight="bold")

        plt.savefig(self.outpng_noise_vs_beam, dpi=self.fig_dpi)

    ####################
    # _measure_log_rms #
    ####################

    def _measure_log_rms(
        self,
        incubes,
        beams,
        outtxt,
        ):
        """
        plot_noise_vs_beam
        """

        if not glob.glob(outtxt):
            print("# loop_meausre_rms")
            list_log_rms = []
            list_log_p84 = []
            for i in range(len(beams)):
                this_beam    = beams[i]
                this_beamstr = str(this_beam).replace(".","p").zfill(4)
                this_cube    = incubes.replace("????",this_beamstr)

                this_data,_ = imval_all(this_cube)
                this_data   = this_data["data"]
                this_data[np.isnan(this_data)] = 0
                this_data[np.isinf(this_data)] = 0
                this_data   = this_data[this_data!=0]
                this_bins   = (np.ceil(np.log2(len(this_data))) + 1) * 20 # Sturgess equation * 20

                _,_,_,_,this_rms,_,_,this_p84 = self._gaussfit_noise(this_data)
                list_log_rms.append(np.log10(this_rms))
                list_log_p84.append(np.log10(this_p84))

            header="Column 1 = beam size (arcsec)\nColumn 2 = log best-fit rms (K)\nColumn 3 = log 84th percentile of inversed histogram (K)"
            np.savetxt(outtxt, np.c_[beams,list_log_rms,list_log_p84], fmt ="%4.1f %4.4f %4.4f", header=header, delimiter="   ")
        else:
            print("# read " + outtxt)
            data = np.loadtxt(outtxt)
            list_log_rms = data[:,1]
            list_log_p84 = data[:,2]

        return np.c_[list_log_rms, list_log_p84]

    #

    ###################
    # plot_noise_hist #
    ###################

    def plot_noise_hist(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outcube_co10_n0628.replace(".image","_k.image"),taskname)

        ###########
        # prepare #
        ###########

        data,_ = imval_all(self.outcube_co10_n0628.replace(".image","_k.image"))
        data   = data["data"].flatten()
        histx, histy, histrange, peak, rms, x_bestfit, y_bestfit, _ = self._gaussfit_noise(data)

        xlim     = [0, self.noise_hist_xmax_snr*rms]
        ylim     = [0, np.max(histy)*1.02]
        title    = "(a) 4.0\" CO(1-0) Cube (NGC 0628)"
        xlabel   = "Absolute voxel value (K)"
        ylabel   = "Count"
        binwidth = (histrange[1]-histrange[0]) / self.noise_hist_bins
        c_pos    = "tomato"
        c_neg    ="deepskyblue"

        ########
        # plot #
        ########

        plt.figure(figsize=(13,10))
        gs = gridspec.GridSpec(nrows=10, ncols=10)
        ax = plt.subplot(gs[0:10,0:10])

        ad = [0.215,0.83,0.10,0.90]
        myax_set(ax, None, xlim, ylim, title, xlabel, ylabel, adjust=ad)
        ax.set_yticks(np.linspace(0,20000,3)[1:])

        # plot hists
        ax.bar(histx, histy, width=binwidth, align="center", lw=0, color=c_pos)
        ax.bar(-1*histx, histy, width=binwidth, align="center", lw=0, color=c_neg)

        # plot bestfit
        ax.plot(x_bestfit, y_bestfit, "-", c="black", lw=5)

        # # plot 1 sigma and 2.5sigma dashed vertical lines
        ax.plot([rms, rms], ylim, "--", color='black', lw=2)
        ax.plot([rms*self.noise_hist_snr4plt, rms*self.noise_hist_snr4plt], ylim, "--", color='black', lw=2)

        # legend
        ax.text(0.95, 0.93, "positive voxel histogram", color=c_pos, horizontalalignment="right", transform=ax.transAxes, size=self.legend_fontsize, fontweight="bold")
        ax.text(0.95, 0.88, "negative voxel histogram", color=c_neg, horizontalalignment="right", transform=ax.transAxes, size=self.legend_fontsize, fontweight="bold")
        ax.text(0.95, 0.83, "best-fit Gaussian", color="black", horizontalalignment="right", transform=ax.transAxes, size=self.legend_fontsize, fontweight="bold")
        #
        x    = rms / (xlim[1]-xlim[0]) + 0.01
        text = r"1$\sigma$ = "+str(rms).ljust(5, "0") + " K"
        ax.text(x, 0.96, text, color="black", horizontalalignment="left", verticalalignment="top", transform=ax.transAxes, rotation=90)
        #
        x    = rms*self.noise_hist_snr4plt / (xlim[1]-xlim[0]) + 0.01
        text = str(self.noise_hist_snr4plt)+r"$\sigma$ = "+str(np.round(rms*self.noise_hist_snr4plt,3)).ljust(5, "0") + " K"
        ax.text(x, 0.96, text, color="black", horizontalalignment="left", verticalalignment="top", transform=ax.transAxes, rotation=90)

        plt.savefig(self.outpng_noise_hist, dpi=self.fig_dpi)

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
        rms          = abs(np.round(popt[1], 3))
        y_bestfit    = self._func1(x_bestfit, peak, rms)

        return histx, histy, histrange, peak, rms, x_bestfit, y_bestfit, p84_data

    def _func1(self, x, a, c):
        """
        """
        return a*np.exp(-(x)**2/(2*c**2))

    #####################
    #####################
    ### analysis part ###
    #####################
    #####################

    ###############
    # align_bulge #
    ###############

    def align_bulge(
        self,
        skip=False,
        ):
        """
        """

        if skip==False:
            taskname = self.modname + sys._getframe().f_code.co_name
            check_first(self.outcube_co10_n0628,taskname)

            if self.do_ngc0628==True:
                this_beam     = self.basebeam_n0628
                this_convbeam = self.beam_wise_n0628
                this_template = self.outmom_co10_n0628.replace("momX","mom0")
                this_bulge    = self.env_bulge_n0628
                this_output   = self.outfits_bulge_n0628
                self._import_bulge(this_bulge,this_output,this_beam,this_convbeam,this_template)

            if self.do_ngc3627==True:
                this_beam     = self.basebeam_n3627
                this_convbeam = self.beam_wise_n3627
                this_template = self.outmom_co10_n3627.replace("momX","mom0")
                this_bulge    = self.env_bulge_n3627
                this_output   = self.outfits_bulge_n3627
                self._import_bulge(this_bulge,this_output,this_beam,this_convbeam,this_template)

            if self.do_ngc4254==True:
                this_beam     = self.basebeam_n4254
                this_convbeam = self.beam_wise_n4254
                this_template = self.outmom_co10_n4254.replace("momX","mom0")
                this_bulge    = self.env_bulge_n4254
                this_output   = self.outfits_bulge_n4254
                self._import_bulge(this_bulge,this_output,this_beam,this_convbeam,this_template)

            if self.do_ngc4321==True:
                this_beam     = self.basebeam_n4321
                this_convbeam = self.beam_wise_n4321
                this_template = self.outmom_co10_n4321.replace("momX","mom0")
                this_bulge    = self.env_bulge_n4321
                this_output   = self.outfits_bulge_n4321
                self._import_bulge(this_bulge,this_output,this_beam,this_convbeam,this_template)

    #################
    # _import_bulge #
    #################

    def _import_bulge(
        self,
        infile,
        outfile,
        beam,
        beamto,
        template,
        index=1,
        ):
        """
        align_bulge
        """

        expr = "iif(IM0=="+str(index)+",1,0)"
        run_immath_one(infile,outfile+"_tmp1",expr)
        run_roundsmooth(outfile+"_tmp1",outfile+"_tmp2",beamto,inputbeam=beam,delin=True)
        run_imregrid(outfile+"_tmp2",template,outfile+"_tmp3",delin=True)
        run_immath_one(outfile+"_tmp3",outfile,"iif( IM0>0.01,1,0 )",delin=True)

    #

    #############
    # align_r21 #
    #############

    def align_r21(
        self,
        skip=False,
        ):
        """
        """

        if skip==False:
            taskname = self.modname + sys._getframe().f_code.co_name
            check_first(self.outcube_co10_n0628,taskname)

            if self.do_ngc0628==True:
                this_beam   = self.basebeam_n0628
                this_input  = self.outfits_r21_n0628
                this_output = self.outfits_r21hl_n0628
                self._import_r21(this_input,this_output,this_beam)

            if self.do_ngc3627==True:
                this_beam   = self.basebeam_n3627
                this_input  = self.outfits_r21_n3627
                this_output = self.outfits_r21hl_n3627
                self._import_r21(this_input,this_output,this_beam)

            if self.do_ngc4254==True:
                this_beam   = self.basebeam_n4254
                this_input  = self.outfits_r21_n4254
                this_output = self.outfits_r21hl_n4254
                self._import_r21(this_input,this_output,this_beam)

            if self.do_ngc4321==True:
                this_beam   = self.basebeam_n4321
                this_input  = self.outfits_r21_n4321
                this_output = self.outfits_r21hl_n4321
                self._import_r21(this_input,this_output,this_beam)

    ###############
    # _import_r21 #
    ###############

    def _import_r21(
        self,
        infile,
        outfile,
        beam,
        ):
        """
        align_r21
        """

        data,_ = imval_all(infile)
        data = data["data"]
        data[np.isnan(data)] = 0
        data[np.isinf(data)] = 0
        data = data[data!=0]

        p33  = str(np.percentile(data,33.3333333))
        p66  = str(np.percentile(data,66.6666667))

        run_immath_one(infile,outfile+"_tmp1","iif( IM0>0,2,0 )")
        run_immath_two(infile,outfile+"_tmp1",outfile+"_tmp2","iif( IM0<"+p33+",1,IM1 )")
        run_immath_two(infile,outfile+"_tmp2",outfile,"iif( IM0>="+p66+",3,IM1 )")
        os.system("rm -rf " + outfile + "_tmp1")
        os.system("rm -rf " + outfile + "_tmp2")

        beamstr = str(beam) + "arcsec"
        imhead(imagename=outfile,mode="put",hdkey="beamminor",hdvalue=beamstr)
        imhead(imagename=outfile,mode="put",hdkey="beammajor",hdvalue=beamstr)

    #

    ################
    # align_halpha #
    ################

    def align_halpha(
        self,
        skip=False,
        ):
        """
        """

        if skip==False:
            taskname = self.modname + sys._getframe().f_code.co_name
            check_first(self.outcube_co10_n0628,taskname)

            if self.do_ngc0628==True:
                this_halpha   = self.halpha_mask_n0628
                print(self.halpha_mask_n0628)
                this_output   = self.outfits_halpha_n0628
                this_beam     = self.basebeam_n0628
                this_template = self.outmom_co10_n0628.replace("momX","mom0")
                self._import_halpha(this_halpha,this_output,this_template,this_beam)

            if self.do_ngc3627==True:
                this_halpha   = self.halpha_mask_n3627
                this_output   = self.outfits_halpha_n3627
                this_beam     = self.basebeam_n3627
                this_template = self.outmom_co10_n3627.replace("momX","mom0")
                self._import_halpha(this_halpha,this_output,this_template,this_beam)

            if self.do_ngc4254==True:
                this_halpha   = self.halpha_mask_n4254
                this_output   = self.outfits_halpha_n4254
                this_beam     = self.basebeam_n4254
                this_template = self.outmom_co10_n4254.replace("momX","mom0")
                self._import_halpha(this_halpha,this_output,this_template,this_beam)

            if self.do_ngc4321==True:
                this_halpha   = self.halpha_mask_n4321
                this_output   = self.outfits_halpha_n4321
                this_beam     = self.basebeam_n4321
                this_template = self.outmom_co10_n4321.replace("momX","mom0")
                self._import_halpha(this_halpha,this_output,this_template,this_beam)

    ##################
    # _import_halpha #
    ##################

    def _import_halpha(
        self,
        infits,
        output,
        template,
        beam,
        ):
        """
        align_halpha
        """

        run_importfits(infits,output+"_tmp1")
        relabelimage(output+"_tmp1",icrs_to_j2000=True)
        run_imregrid(output+"_tmp1",template,output+"_tmp2",delin=True)
        run_immath_one(output+"_tmp2",output,"iif( IM0>0,1,0 )",delin=True)

        beamstr = str(beam) + "arcsec"
        imhead(imagename = output, mode="put", hdkey="beamminor", hdvalue=beamstr)
        imhead(imagename = output, mode="put", hdkey="beammajor", hdvalue=beamstr)

    #

    #############
    # align_env #
    #############

    def align_env(
        self,
        skip=False,
        ):
        """
        """

        if skip==False:
            taskname = self.modname + sys._getframe().f_code.co_name
            check_first(self.outcube_co10_n0628,taskname)

            if self.do_ngc0628==True:
                this_beam     = self.basebeam_n0628
                this_fits_env = [self.env_arm_n0628,self.env_bar_n0628,self.env_bulge_n0628]
                this_id_env   = [self.id_arm_n0628,self.id_bar_n0628,self.id_bulge_n0628]
                this_template = self.outmom_co10_n0628.replace("momX","mom0")
                this_output   = self.outfits_env_n0628
                self._import_env(this_fits_env,this_id_env,this_template,this_output,this_beam)

            if self.do_ngc3627==True:
                this_beam     = self.basebeam_n3627
                this_fits_env = [self.env_arm_n3627,self.env_bar_n3627,self.env_bulge_n3627]
                this_id_env   = [self.id_arm_n3627,self.id_bar_n3627,self.id_bulge_n3627]
                this_template = self.outmom_co10_n3627.replace("momX","mom0")
                this_output   = self.outfits_env_n3627
                self._import_env(this_fits_env,this_id_env,this_template,this_output,this_beam)

            if self.do_ngc4254==True:
                this_beam     = self.basebeam_n4254
                this_fits_env = [self.env_arm_n4254,self.env_bar_n4254,self.env_bulge_n4254]
                this_id_env   = [self.id_arm_n4254,self.id_bar_n4254,self.id_bulge_n4254]
                this_template = self.outmom_co10_n4254.replace("momX","mom0")
                this_output   = self.outfits_env_n4254
                self._import_env(this_fits_env,this_id_env,this_template,this_output,this_beam)

            if self.do_ngc4321==True:
                this_beam     = self.basebeam_n4321
                this_fits_env = [self.env_arm_n4321,self.env_bar_n4321,self.env_bulge_n4321]
                this_id_env   = [self.id_arm_n4321,self.id_bar_n4321,self.id_bulge_n4321]
                this_template = self.outmom_co10_n4321.replace("momX","mom0")
                this_output   = self.outfits_env_n4321
                self._import_env(this_fits_env,this_id_env,this_template,this_output,this_beam)

    ###############
    # _import_env #
    ###############

    def _import_env(
        self,
        list_fits_env,
        list_id_env,
        template,
        output,
        beam,
        ):
        """
        align_env
        """

        beamstr = str(beam) + "arcsec"

        # staging
        list_image_mask = []
        for i in range(len(list_fits_env)):
            this_fits = list_fits_env[i]
            this_expr = "iif( IM0" + list_id_env[i] + ", 1, 0 )"

            if this_fits.split("/")[-1]!="None":
                if this_fits.endswith(".fits"):
                    run_importfits(this_fits,this_fits+"_env"+str(i)+"_tmp1")
                else:
                    os.system("cp -r " + this_fits + " " + this_fits + "_env" + str(i) + "_tmp1")

                relabelimage(this_fits+"_env"+str(i)+"_tmp1",icrs_to_j2000=True)
                run_immath_one(this_fits+"_env"+str(i)+"_tmp1",this_fits+"_env"+str(i)+"_tmp2",
                    this_expr,delin=True)
                run_imregrid(this_fits+"_env"+str(i)+"_tmp2",template,this_fits+"_env"+str(i)+"_tmp3",
                    delin=True)
                run_immath_one(this_fits+"_env"+str(i)+"_tmp3",this_fits+"_env"+str(i),
                    "iif( IM0>0,"+str(i+1)+",0 )",delin=True)
                list_image_mask.append(this_fits+"_env"+str(i))

        os.system("rm -rf " + output)
        if len(list_image_mask)==1:
            print("## only one environment is found. No concatnation.")
            os.system("mv " + list_image_mask[0] + " " + output)
        else:
            mask1 = list_image_mask[0]
            masks = list_image_mask[1:]
            for i in range(len(masks)):
                if i>0:
                    mask1 = this_output

                this_mask   = masks[i]
                this_output = output+"_tmp"+str(i+1)
                run_immath_two(mask1,this_mask,this_output,"iif( IM1>0,IM1,IM0 )",delin=False)

        os.system("mv " + this_output + " " + output)
        imhead(imagename = output, mode="put", hdkey="beamminor", hdvalue=beamstr)
        imhead(imagename = output, mode="put", hdkey="beammajor", hdvalue=beamstr)

        os.system("rm -rf " + output + "_tmp1")
        os.system("rm -rf " + output + "_tmp2")
        os.system("rm -rf " + output + "_tmp3")

    #

    ################
    # align_cprops #
    ################

    def align_cprops(
        self,
        skip=False,
        ):
        """
        """

        if skip==False:
            taskname = self.modname + sys._getframe().f_code.co_name
            check_first(self.outcube_co10_n0628,taskname)

            if self.do_ngc0628==True:
                this_beam     = self.basebeam_n0628
                this_beamstr  = str(self.basebeam_n0628).replace(".","p").zfill(4) + "arcsec"
                this_output   = self.outfits_cprops_n0628
                this_table    = self.cprops_table_n0628
                this_scale    = self.scale_n0628
                this_template = self.outmom_co10_n0628.replace("momX","mom0")
                this_ra       = self.ra_n0628
                this_dec      = self.dec_n0628
                this_convbeam = np.sqrt((this_scale*this_beam)**2 - 120**2)
                self._cprops_table2fits(this_template,this_output,this_table,self.snr_gmc,
                    this_scale,this_convbeam,this_ra,this_dec,this_beamstr)

            if self.do_ngc3627==True:
                this_beam     = self.basebeam_n3627
                this_beamstr  = str(self.basebeam_n3627).replace(".","p").zfill(4) + "arcsec"
                this_output   = self.outfits_cprops_n3627
                this_table    = self.cprops_table_n3627
                this_scale    = self.scale_n3627
                this_template = self.outmom_co10_n3627.replace("momX","mom0")
                this_ra       = self.ra_n3627
                this_dec      = self.dec_n3627
                this_convbeam = np.sqrt((this_scale*this_beam)**2 - 120**2)
                self._cprops_table2fits(this_template,this_output,this_table,self.snr_gmc,
                    this_scale,this_convbeam,this_ra,this_dec,this_beamstr)

            if self.do_ngc4254==True:
                this_beam     = self.basebeam_n4254
                this_beamstr  = str(self.basebeam_n4254).replace(".","p").zfill(4) + "arcsec"
                this_output   = self.outfits_cprops_n4254
                this_table    = self.cprops_table_n4254
                this_scale    = self.scale_n4254
                this_template = self.outmom_co10_n4254.replace("momX","mom0")
                this_ra       = self.ra_n4254
                this_dec      = self.dec_n4254
                this_convbeam = np.sqrt((this_scale*this_beam)**2 - 120**2)
                self._cprops_table2fits(this_template,this_output,this_table,self.snr_gmc,
                    this_scale,this_convbeam,this_ra,this_dec,this_beamstr)

            if self.do_ngc4321==True:
                this_beam     = self.basebeam_n4321
                this_beamstr  = str(self.basebeam_n4321).replace(".","p").zfill(4) + "arcsec"
                this_output   = self.outfits_cprops_n4321
                this_table    = self.cprops_table_n4321
                this_scale    = self.scale_n4321
                this_template = self.outmom_co10_n4321.replace("momX","mom0")
                this_ra       = self.ra_n4321
                this_dec      = self.dec_n4321
                this_convbeam = np.sqrt((this_scale*this_beam)**2 - 120**2)
                self._cprops_table2fits(this_template,this_output,this_table,self.snr_gmc,
                    this_scale,this_convbeam,this_ra,this_dec,this_beamstr)

    ######################
    # _cprops_table2fits #
    ######################

    def _cprops_table2fits(
        self,
        template,
        outputfits,
        fits_table_cprops,
        snr_gmc,
        scale,
        convolving_beam,
        image_ra_cnt,
        image_dec_cnt,
        this_beamstr,
        ):
        """
        align_gmc
        """

        # reading_cprops_table
        gmc_ra_dgr, gmc_dec_dgr, gmc_radius_arcsec, gmc_pa, gmc_major_arcsec, gmc_minor_arcsec = \
            self._reading_cprops_table(fits_table_cprops, snr_gmc, scale, convolving_beam)

        # get template grid information
        size_x   = imhead(template,mode="list")["shape"][0]
        size_y   = imhead(template,mode="list")["shape"][1]
        pix_rad  = imhead(template,mode="list")["cdelt2"]
        pix_size = round(pix_rad * 3600 * 180 / np.pi, 3)
        freq     = str(imhead(template,mode="list")["restfreq"][0]/1e9)+"GHz"

        # construct fits
        for i in range(len(gmc_ra_dgr)):
            this_maj = str(gmc_radius_arcsec[i]) + "arcsec"
            this_min = str(gmc_minor_arcsec[i] * gmc_radius_arcsec[i] / gmc_major_arcsec[i]) + "arcsec"
            this_dir = "J2000 " + str(gmc_ra_dgr[i]) + "deg " + str(gmc_dec_dgr[i]) + "deg"
            this_pa  = str(gmc_pa[i])+"deg"
            #
            mycl.addcomponent(
                dir=this_dir,
                flux=1.0,
                fluxunit="Jy",
                freq=freq,
                shape="disk",
                majoraxis=this_maj,
                minoraxis=this_min,
                positionangle=this_pa,
                )

        c = SkyCoord(float(image_ra_cnt), float(image_dec_cnt), unit="deg")
        ra_dgr = str(c.ra.degree/(180/np.pi))
        dec_dgr = str(c.dec.degree/(180/np.pi))

        myia.fromshape(outputfits+"_tmp1", [size_x,size_y,1,1], overwrite=True)
        mycs = myia.coordsys()
        mycs.setunits(["rad", "rad", "", "Hz"])
        cell_rad = myqa.convert(myqa.quantity(str(pix_size)+"arcsec"), "rad")["value"]
        mycs.setincrement([-cell_rad, cell_rad], "direction")
        mycs.setreferencevalue([myqa.convert(ra_dgr,"rad")["value"],
                                myqa.convert(dec_dgr,"rad")["value"]],
                                type="direction")
        mycs.setreferencevalue(freq, "spectral")
        mycs.setincrement("1GHz" ,"spectral")
        myia.setcoordsys(mycs.torecord())
        myia.setbrightnessunit("Jy/pixel")
        myia.modify(mycl.torecord(), subtract=False)
        myia.close()
        mycl.close()

        # exportfits
        run_immath_one(outputfits+"_tmp1",outputfits,"iif( IM0>0,1,0 )",delin=True)
        imhead(imagename=outputfits,mode="put",hdkey="beamminor",hdvalue=this_beamstr)
        imhead(imagename=outputfits,mode="put",hdkey="beammajor",hdvalue=this_beamstr)
        #run_exportfits(outputfits+"_tmp2", outputfits, delin=True)

    #########################
    # _reading_cprops_table #
    #########################

    def _reading_cprops_table(
        self,
        fits_table_cprops,
        snr_gmc,
        scale,
        convolving_beam,
        ):
        """
        align_gmc - _cprops_table2fits
        """

        # reading crpops table
        hdu_list      = fits.open(fits_table_cprops, memmap=True)
        data          = Table(hdu_list[1].data)
        gmc_ra_dgr    = data["XCTR_DEG"] # center ra position of the cloud in decimal degrees
        gmc_dec_dgr   = data["YCTR_DEG"] # center decl position of the cloud in decimal degrees
        gmc_radius_pc = data["RAD_NODC_NOEX"] # the radius without deconvolution or extrapolation in parsecs
        gmc_sn_ratio  = data["S2N"] # the peak signal-to-noise ratio in the cloud
        gmc_pa        = data["POSANG"] * 180 / np.pi
        gmc_num       = data["CLOUDNUM"]
        gmc_npix      = data["NPIX"]
        dx            = np.sqrt(2*np.pi/(8*np.log(2)) * data["BEAMMAJ_PC"]**2 / data["PPBEAM"])
        gmc_major     = np.sqrt((data['MOMMAJPIX_NOEX'] * dx)**2 - (data['BEAMMAJ_PC']**2/8/np.log(2)))
        gmc_minor     = np.sqrt((data['MOMMINPIX_NOEX'] * dx)**2 - (data['BEAMMIN_PC']**2/8/np.log(2)))

        # constrain data
        gmc_major[np.isnan(gmc_major)] = 0
        gmc_major[np.isinf(gmc_major)] = 0
        gmc_minor[np.isnan(gmc_minor)] = 0
        gmc_minor[np.isinf(gmc_minor)] = 0

        cut               = (gmc_radius_pc > 0.) & (gmc_sn_ratio > snr_gmc) & (gmc_minor > 0.) & (gmc_major > 0.)
        gmc_ra_dgr        = gmc_ra_dgr[cut]
        gmc_dec_dgr       = gmc_dec_dgr[cut]
        gmc_radius_arcsec = np.sqrt((gmc_radius_pc[cut] / scale)**2 + (convolving_beam / scale)**2)
        gmc_pa            = gmc_pa[cut]
        gmc_major_arcsec  = gmc_major[cut] / scale
        gmc_minor_arcsec  = gmc_minor[cut] / scale

        return gmc_ra_dgr, gmc_dec_dgr, gmc_radius_arcsec, gmc_pa, gmc_major_arcsec, gmc_minor_arcsec

    #

    ##############
    # align_wise #
    ##############

    def align_wise(
        self,
        skip=False,
        ):
        """
        """

        if skip==False:
            taskname = self.modname + sys._getframe().f_code.co_name
            check_first(self.outcube_co10_n0628,taskname)

            if self.do_ngc0628==True:
                input_w1  = self.wise1_n0628
                input_w2  = self.wise2_n0628
                input_w3  = self.wise3_n0628
                output_w1 = self.outfits_wise1_n0628
                output_w2 = self.outfits_wise2_n0628
                output_w3 = self.outfits_wise3_n0628
                this_beam = self.beam_wise_n0628
                template  = self.outmom_co10_n0628.replace( str(self.basebeam_n0628).replace(".","p").zfill(4) , str(this_beam).replace(".","p").zfill(4) ).replace("momX","mom0")
                self._import_wise(input_w1,output_w1,this_beam,template)
                self._import_wise(input_w2,output_w2,this_beam,template)
                self._import_wise(input_w3,output_w3,this_beam,template)

            if self.do_ngc3627==True:
                input_w1  = self.wise1_n3627
                input_w2  = self.wise2_n3627
                input_w3  = self.wise3_n3627
                output_w1 = self.outfits_wise1_n3627
                output_w2 = self.outfits_wise2_n3627
                output_w3 = self.outfits_wise3_n3627
                this_beam = self.beam_wise_n3627
                template  = self.outmom_co10_n3627.replace( str(self.basebeam_n3627).replace(".","p").zfill(4) , str(this_beam).replace(".","p").zfill(4) ).replace("momX","mom0")
                self._import_wise(input_w1,output_w1,this_beam,template)
                self._import_wise(input_w2,output_w2,this_beam,template)
                self._import_wise(input_w3,output_w3,this_beam,template)

            if self.do_ngc4254==True:
                input_w1  = self.wise1_n4254
                input_w2  = self.wise2_n4254
                input_w3  = self.wise3_n4254
                output_w1 = self.outfits_wise1_n4254
                output_w2 = self.outfits_wise2_n4254
                output_w3 = self.outfits_wise3_n4254
                this_beam = self.beam_wise_n4254
                template  = self.outmom_co10_n4254.replace( str(self.basebeam_n4254).replace(".","p").zfill(4) , str(this_beam).replace(".","p").zfill(4) ).replace("momX","mom0")
                self._import_wise(input_w1,output_w1,this_beam,template)
                self._import_wise(input_w2,output_w2,this_beam,template)
                self._import_wise(input_w3,output_w3,this_beam,template)

            if self.do_ngc4321==True:
                input_w1  = self.wise1_n4321
                input_w2  = self.wise2_n4321
                input_w3  = self.wise3_n4321
                output_w1 = self.outfits_wise1_n4321
                output_w2 = self.outfits_wise2_n4321
                output_w3 = self.outfits_wise3_n4321
                this_beam = self.beam_wise_n4321
                template  = self.outmom_co10_n4321.replace( str(self.basebeam_n4321).replace(".","p").zfill(4) , str(this_beam).replace(".","p").zfill(4) ).replace("momX","mom0")
                self._import_wise(input_w1,output_w1,this_beam,template)
                self._import_wise(input_w2,output_w2,this_beam,template)
                self._import_wise(input_w3,output_w3,this_beam,template)

    ################
    # _import_wise #
    ################

    def _import_wise(
        self,
        imagename,
        outfile,
        this_beam,
        template,
        wise_beam=7.5,
        ):
        """
        align_wise
        """

        # staging
        run_importfits(imagename,outfile+"_tmp1")
        imhead(outfile+"_tmp1", mode="add", hdkey="beammajor", hdvalue=str(wise_beam)+"arcsec")
        imhead(outfile+"_tmp1", mode="add", hdkey="beamminor", hdvalue=str(wise_beam)+"arcsec")
        beamarea = (2*np.pi / (8*np.log(2))) * (wise_beam)**2 / 4.25e10
        run_immath_one(outfile+"_tmp1",outfile+"_tmp2","IM0*1e6*"+str(beamarea),delin=True)
        imhead(outfile+"_tmp2", mode="put", hdkey="bunit", hdvalue="Jy/beam")

        # smoothing
        if this_beam<wise_beam:
            print("# skip smoothing " + imagename.split("/")[-1])
            os.system("rm -rf " + outfile + "_tmp2")
            os.system("mv " + outfile + "_tmp2 " + outfile + "_tmp3")
        else:
            print("# run smoothing " + imagename.split("/")[-1])
            run_roundsmooth(outfile+"_tmp2",outfile+"_tmp3",this_beam,inputbeam=wise_beam,delin=True)

        # alignment to co mom0 map
        run_imregrid(outfile+"_tmp3",template,outfile,delin=True)

    #

    ################
    # multimoments #
    ################

    def multimoments(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outcube_co10_n0628,taskname)

        if self.do_ngc0628==True:
            self._loop_immoments(
                self.outcube_co10_n0628.replace(str(self.basebeam_n0628).replace(".","p").zfill(4),"????").replace(".image","_k.image"),
                self.outcube_co21_n0628.replace(str(self.basebeam_n0628).replace(".","p").zfill(4),"????").replace(".image","_k.image"),
                self.outmom_co10_n0628.replace(str(self.basebeam_n0628).replace(".","p").zfill(4),"????"),
                self.outmom_co21_n0628.replace(str(self.basebeam_n0628).replace(".","p").zfill(4),"????"),
                self.outfits_r21_n0628.replace(str(self.basebeam_n0628).replace(".","p").zfill(4),"????"),
                self.outfits_t21_n0628.replace(str(self.basebeam_n0628).replace(".","p").zfill(4),"????"),
                self.outfits_er21_n0628.replace(str(self.basebeam_n0628).replace(".","p").zfill(4),"????"),
                self.outfits_et21_n0628.replace(str(self.basebeam_n0628).replace(".","p").zfill(4),"????"),
                self.beams_n0628,
                self.nchan_thres_n0628,
                )

        if self.do_ngc3627==True:
            self._loop_immoments(
                self.outcube_co10_n3627.replace(str(self.basebeam_n3627).replace(".","p").zfill(4),"????").replace(".image","_k.image"),
                self.outcube_co21_n3627.replace(str(self.basebeam_n3627).replace(".","p").zfill(4),"????").replace(".image","_k.image"),
                self.outmom_co10_n3627.replace(str(self.basebeam_n3627).replace(".","p").zfill(4),"????"),
                self.outmom_co21_n3627.replace(str(self.basebeam_n3627).replace(".","p").zfill(4),"????"),
                self.outfits_r21_n3627.replace(str(self.basebeam_n3627).replace(".","p").zfill(4),"????"),
                self.outfits_t21_n3627.replace(str(self.basebeam_n3627).replace(".","p").zfill(4),"????"),
                self.outfits_er21_n3627.replace(str(self.basebeam_n3627).replace(".","p").zfill(4),"????"),
                self.outfits_et21_n3627.replace(str(self.basebeam_n3627).replace(".","p").zfill(4),"????"),
                self.beams_n3627,
                self.nchan_thres_n3627,
                )

        if self.do_ngc4254==True:
            self._loop_immoments(
                self.outcube_co10_n4254.replace(str(self.basebeam_n4254).replace(".","p").zfill(4),"????").replace(".image","_k.image"),
                self.outcube_co21_n4254.replace(str(self.basebeam_n4254).replace(".","p").zfill(4),"????").replace(".image","_k.image"),
                self.outmom_co10_n4254.replace(str(self.basebeam_n4254).replace(".","p").zfill(4),"????"),
                self.outmom_co21_n4254.replace(str(self.basebeam_n4254).replace(".","p").zfill(4),"????"),
                self.outfits_r21_n4254.replace(str(self.basebeam_n4254).replace(".","p").zfill(4),"????"),
                self.outfits_t21_n4254.replace(str(self.basebeam_n4254).replace(".","p").zfill(4),"????"),
                self.outfits_er21_n4254.replace(str(self.basebeam_n4254).replace(".","p").zfill(4),"????"),
                self.outfits_et21_n4254.replace(str(self.basebeam_n4254).replace(".","p").zfill(4),"????"),
                self.beams_n4254,
                self.nchan_thres_n4254,
                )

        if self.do_ngc4321==True:
            self._loop_immoments(
                self.outcube_co10_n4321.replace(str(self.basebeam_n4321).replace(".","p").zfill(4),"????").replace(".image","_k.image"),
                self.outcube_co21_n4321.replace(str(self.basebeam_n4321).replace(".","p").zfill(4),"????").replace(".image","_k.image"),
                self.outmom_co10_n4321.replace(str(self.basebeam_n4321).replace(".","p").zfill(4),"????"),
                self.outmom_co21_n4321.replace(str(self.basebeam_n4321).replace(".","p").zfill(4),"????"),
                self.outfits_r21_n4321.replace(str(self.basebeam_n4321).replace(".","p").zfill(4),"????"),
                self.outfits_t21_n4321.replace(str(self.basebeam_n4321).replace(".","p").zfill(4),"????"),
                self.outfits_er21_n4321.replace(str(self.basebeam_n4321).replace(".","p").zfill(4),"????"),
                self.outfits_et21_n4321.replace(str(self.basebeam_n4321).replace(".","p").zfill(4),"????"),
                self.beams_n4321,
                self.nchan_thres_n4321,
                )

    ###################
    # _loop_immoments #
    ###################

    def _loop_immoments(
        self,
        incube_co10,
        incube_co21,
        outmom_co10,
        outmom_co21,
        outmom_r21,
        outmom_t21,
        outmom_er21,
        outmom_et21,
        this_beams,
        nchan_thres,
        ):
        """
        multimoments
        """

        for i in range(len(this_beams)):
            this_beam        = this_beams[i]
            this_beamstr     = str(this_beam).replace(".","p").zfill(4)
            mask_co10        = "co10.mask"
            mask_co21        = "co21.mask"
            mask_combine     = "comb_" + this_beamstr + ".mask"
            this_input_co10  = incube_co10.replace("????",this_beamstr)
            this_input_co21  = incube_co21.replace("????",this_beamstr)
            this_output_co10 = outmom_co10.replace("????",this_beamstr)
            this_output_co21 = outmom_co21.replace("????",this_beamstr)
            this_output_r21  = outmom_r21.replace("????",this_beamstr)
            this_output_t21  = outmom_t21.replace("????",this_beamstr)
            this_output_er21 = outmom_er21.replace("????",this_beamstr)
            this_output_et21 = outmom_et21.replace("????",this_beamstr)

            print("# create " + this_output_co10.split("/")[-1])
            print("# create " + this_output_co21.split("/")[-1])
            print("# create " + this_output_r21.split("/")[-1])
            print("# create " + this_output_t21.split("/")[-1])
            print("# create " + this_output_er21.split("/")[-1])
            print("# create " + this_output_et21.split("/")[-1])

            # snr-based masking
            self._maskig_cube_snr(this_input_co10,mask_co10+"_snr")
            self._maskig_cube_snr(this_input_co21,mask_co21+"_snr")
            run_immath_two(mask_co10+"_snr",mask_co21+"_snr",mask_combine,"IM0*IM1",delin=True)

            # nchan-based masking
            self._masking_cube_nchan(this_input_co10,mask_co10+"_nchan",nchan_thres=nchan_thres)
            run_immath_two(mask_combine,mask_co10+"_nchan",mask_co10,"IM0*IM1",delin=False)
            os.system("rm -rf " + mask_co10 + "_nchan")

            self._masking_cube_nchan(this_input_co21,mask_co21+"_nchan",nchan_thres=nchan_thres)
            run_immath_two(mask_combine,mask_co21+"_nchan",mask_co21,"IM0*IM1",delin=True)

            # mom creation
            mom0_co10, emom0_co10, mom8_co10, emom8_co10 = \
                self._eazy_immoments(this_input_co10,mask_co10,this_output_co10)
            mom0_co21, emom0_co21, mom8_co21, emom8_co21 = \
                self._eazy_immoments(this_input_co21,mask_co21,this_output_co21)

            # clean up
            os.system("rm -rf " + mask_combine)

            # r21 and t21
            self._eazy_r21(mom0_co10,mom0_co21,emom0_co10,emom0_co21,this_output_r21,this_output_er21)
            self._eazy_t21(mom8_co10,mom8_co21,emom8_co10,emom8_co21,this_output_t21,this_output_et21)

    #############
    # _eazy_t21 #
    #############

    def _eazy_t21(
        self,
        mom8_co10,
        mom8_co21,
        emom8_co10,
        emom8_co21,
        output_t21,
        output_et21,
        ):
        """
        multimoments - _loop_immoments
        """

        # r21
        run_immath_two(mom8_co21,mom8_co10,output_t21+"_tmp1","IM0/IM1")

        # error r21
        expr = str(emom8_co21**2) + " / (IM0*IM0)"
        run_immath_one(mom8_co21,output_et21+"_term1",expr)

        expr = str(emom8_co10**2) + " / (IM0*IM0)"
        run_immath_one(mom8_co10,output_et21+"_term2",expr)

        expr = "sqrt(IM0 + IM1)"
        run_immath_two(output_et21+"_term1",output_et21+"_term2",output_et21+"_term3",expr,delin=True)

        expr = "IM0 * IM1"
        run_immath_two(output_t21+"_tmp1",output_et21+"_term3",output_et21+"_tmp1",expr)
        os.system("rm -rf " + output_et21 + "_term3")

        # clipping at snr
        expr = "iif( IM0>=IM1*"+str(self.snr_ratio)+", 1, 0 )"
        run_immath_two(output_t21+"_tmp1",output_et21+"_tmp1",output_t21+"_snrmask_tmp1",expr)
        boolean_masking(output_t21+"_snrmask_tmp1",output_t21+"_snrmask",delin=True)

        expr = "iif(IM1>0, IM0, IM1)"
        run_immath_two(output_t21+"_tmp1",output_t21+"_snrmask",output_t21,expr,delin=False)
        run_immath_two(output_et21+"_tmp1",output_t21+"_snrmask",output_et21,expr,delin=True)
        os.system("rm -rf " + output_t21 + "_tmp1")

    #############
    # _eazy_r21 #
    #############

    def _eazy_r21(
        self,
        mom0_co10,
        mom0_co21,
        emom0_co10,
        emom0_co21,
        output_r21,
        output_er21,
        ):
        """
        multimoments - _loop_immoments
        """

        # r21
        run_immath_two(mom0_co21,mom0_co10,output_r21+"_tmp1","IM0/IM1")

        # error r21
        expr = "(IM0*IM0) / (IM1*IM1)"
        run_immath_two(emom0_co21,mom0_co21,output_er21+"_term1",expr)

        expr = "(IM0*IM0) / (IM1*IM1)"
        run_immath_two(emom0_co10,mom0_co10,output_er21+"_term2",expr)

        expr = "sqrt(IM0 + IM1)"
        run_immath_two(output_er21+"_term1",output_er21+"_term2",output_er21+"_term3",expr,delin=True)

        expr = "IM0 * IM1"
        run_immath_two(output_r21+"_tmp1",output_er21+"_term3",output_er21+"_tmp1",expr)
        os.system("rm -rf " + output_er21 + "_term3")

        # clipping at snr
        expr = "iif( IM0>=IM1*"+str(self.snr_ratio)+", 1, 0 )"
        run_immath_two(output_r21+"_tmp1",output_er21+"_tmp1",output_r21+"_snrmask_tmp1",expr)
        boolean_masking(output_r21+"_snrmask_tmp1",output_r21+"_snrmask",delin=True)

        expr = "iif(IM1>0, IM0, IM1)"
        run_immath_two(output_r21+"_tmp1",output_r21+"_snrmask",output_r21,expr,delin=False)
        run_immath_two(output_er21+"_tmp1",output_r21+"_snrmask",output_er21,expr,delin=True)
        os.system("rm -rf " + output_r21 + "_tmp1")

    ###################
    # _eazy_immoments #
    ###################

    def _eazy_immoments(
        self,
        incube,
        inmask,
        baseoutmom,
        ):
        """
        multimoments - _loop_immoments
        """

        rms = measure_rms(incube)

        outfile  = baseoutmom.replace("momX","mom0")
        outefile = baseoutmom.replace("momX","emom0")
        run_immoments(incube,inmask,outfile,mom=0,rms=rms,snr=self.snr_mom,outfile_err=outefile,vdim=3)
        outfile  = baseoutmom.replace("momX","mom1")
        run_immoments(incube,inmask,outfile,mom=1,rms=rms,snr=self.snr_mom,vdim=3)
        outfile  = baseoutmom.replace("momX","mom2")
        run_immoments(incube,inmask,outfile,mom=2,rms=rms,snr=self.snr_mom,vdim=3)
        outfile  = baseoutmom.replace("momX","mom8")
        run_immoments(incube,inmask,outfile,mom=8,rms=rms,snr=self.snr_mom,vdim=3)

        return baseoutmom.replace("momX","mom0"), baseoutmom.replace("momX","emom0"), baseoutmom.replace("momX","mom8"), rms

    #######################
    # _masking_cube_nchan #
    #######################

    def _masking_cube_nchan(
        self,
        incube,
        outmask,
        pixelmin=1,
        nchan_thres=2,
        ):
        """
        multimoments - _loop_immoments
        """

        thres  = str( measure_rms(incube) * self.snr_mom )
        data   = imval(incube)["coords"][:,3]
        cwidth = str(np.round(abs(data[1]-data[0])/imhead(incube,mode="list")["restfreq"][0] * 299792.458, 2))

        # create nchan 3d mask
        expr = "iif( IM0>=" + thres + ",1.0/" + cwidth + ",0.0 )"
        run_immath_one(incube,incube+"_tmp1",expr)
        immoments(imagename=incube+"_tmp1",moments=[0],outfile=incube+"_tmp2")

        # remove islands
        maskfile = incube + "_tmp2"
        beamarea = beam_area(maskfile)

        myia.open(maskfile)
        mask           = myia.getchunk()
        labeled, j     = scipy.ndimage.label(mask)
        myhistogram    = scipy.ndimage.measurements.histogram(labeled,0,j+1,j+1)
        object_slices  = scipy.ndimage.find_objects(labeled)
        threshold_area = beamarea*pixelmin
        for i in range(j):
            if myhistogram[i+1]<threshold_area:
                mask[object_slices[i]] = 0
        myia.putchunk(mask)
        myia.done()

        # create nchan 2d mask
        expr = "iif( IM0>="+str(nchan_thres)+", 1, 0 )"
        run_immath_one(incube+"_tmp2",incube+"_tmp3",expr,delin=True)
        boolean_masking(incube+"_tmp3",outmask,delin=True)

        os.system("rm -rf " + incube + "_tmp1")

    ####################
    # _maskig_cube_snr #
    ####################

    def _maskig_cube_snr(
        self,
        incube,
        outmask,
        convtos=[3.0,5.0,7.0],
        snrs=[1.0,1.0,1.0],
        ):
        """
        multimoments - _loop_immoments
        """

        smcubes = [incube+".sm1",incube+".sm2",incube+".sm3"]
        smmasks = [s+".mask" for s in smcubes]
        bmaj    = imhead(incube, mode="list")["beammajor"]["value"]

        # multi smooth
        for i in range(len(smcubes)):
            this_smcube = smcubes[i]
            this_smmask = smmasks[i]
            this_sm     = convtos[i]
            this_snr    = snrs[i]
            this_smbeam = bmaj*this_sm # float, arcsec

            run_roundsmooth(incube,this_smcube,this_smbeam,inputbeam=bmaj)
            this_smrms = measure_rms(this_smcube)
            signal_masking(this_smcube,this_smmask,this_smrms*this_snr,delin=True)

        # combine
        expr = "iif( IM0+IM1+IM2>=2.0, 1, 0 )"
        run_immath_three(smmasks[0],smmasks[1],smmasks[2],outmask,expr,delin=True)

    #

    ###############
    # multismooth #
    ###############

    def multismooth(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outcube_co10_n0628,taskname)

        if self.do_ngc0628==True:
            self._loop_roundsmooth(
                self.outcube_co10_n0628,self.beams_n0628[1:],self.basebeam_n0628,
                self.imsize_n0628,self.ra_n0628,self.dec_n0628,self.freq_co10)
            self._loop_roundsmooth(
                self.outcube_co21_n0628,self.beams_n0628[1:],self.basebeam_n0628,
                self.imsize_n0628,self.ra_n0628,self.dec_n0628,self.freq_co21)

        if self.do_ngc3627==True:
            self._loop_roundsmooth(
                self.outcube_co10_n3627,self.beams_n3627[1:],self.basebeam_n3627,
                self.imsize_n3627,self.ra_n3627,self.dec_n3627,self.freq_co10)
            self._loop_roundsmooth(
                self.outcube_co21_n3627,self.beams_n3627[1:],self.basebeam_n3627,
                self.imsize_n3627,self.ra_n3627,self.dec_n3627,self.freq_co21)

        if self.do_ngc4254==True:
            self._loop_roundsmooth(
                self.outcube_co10_n4254,self.beams_n4254[1:],self.basebeam_n4254,
                self.imsize_n4254,self.ra_n4254,self.dec_n4254,self.freq_co10)
            self._loop_roundsmooth(
                self.outcube_co21_n4254,self.beams_n4254[1:],self.basebeam_n4254,
                self.imsize_n4254,self.ra_n4254,self.dec_n4254,self.freq_co21)

        if self.do_ngc4321==True:
            self._loop_roundsmooth(
                self.outcube_co10_n4321,self.beams_n4321[1:],self.basebeam_n4321,
                self.imsize_n4321,self.ra_n4321,self.dec_n4321,self.freq_co10)
            self._loop_roundsmooth(
                self.outcube_co21_n4321,self.beams_n4321[1:],self.basebeam_n4321,
                self.imsize_n4321,self.ra_n4321,self.dec_n4321,self.freq_co21)

    #####################
    # _loop_roundsmooth #
    #####################

    def _loop_roundsmooth(
        self,
        incube,
        beams,
        basebeam,
        imsize,
        ra,
        dec,
        freq,
        ):
        """
        multismooth
        """

        outcube_template = incube.replace(str(basebeam).replace(".","p").zfill(4),"????")
        this_beams       = beams

        unitconv_Jyb_K(incube,incube.replace(".image","_k.image"),freq)

        for i in range(len(this_beams)):
            this_beam    = this_beams[i]
            this_beamstr = str(this_beam).replace(".","p").zfill(4)
            this_outfile = outcube_template.replace("????",this_beamstr)

            print("# create " + this_outfile.split("/")[-1])

            run_roundsmooth(incube,this_outfile+"_tmp1",this_beam,inputbeam=basebeam)
            make_gridtemplate(this_outfile+"_tmp1",this_outfile,imsize,ra,dec,this_beam)
            unitconv_Jyb_K(this_outfile,this_outfile.replace(".image","_k.image"),freq)
            os.system("rm -rf template.image")
            os.system("rm -rf " + this_outfile + "_tmp1")

    #

    ###############
    # align_cubes #
    ###############

    def align_cubes(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.cube_co10_n0628,taskname)

        if self.do_ngc0628==True:
            self._align_cube_gal(self.cube_co10_n0628,self.cube_co21_n0628,
                self.outcube_co10_n0628,self.outcube_co21_n0628,self.basebeam_n0628,
                self.imsize_n0628,self.ra_n0628,self.dec_n0628,self.chans_n0628)

        if self.do_ngc3627==True:
            self._align_cube_gal(self.cube_co10_n3627,self.cube_co21_n3627,
                self.outcube_co10_n3627,self.outcube_co21_n3627,self.basebeam_n3627,
                self.imsize_n3627,self.ra_n3627,self.dec_n3627,self.chans_n3627)

        if self.do_ngc4254==True:
            self._align_cube_gal(self.cube_co10_n4254,self.cube_co21_n4254,
                self.outcube_co10_n4254,self.outcube_co21_n4254,self.basebeam_n4254,
                self.imsize_n4254,self.ra_n4254,self.dec_n4254,self.chans_n4254)

        if self.do_ngc4321==True:
            self._align_cube_gal(self.cube_co10_n4321,self.cube_co21_n4321,
                self.outcube_co10_n4321,self.outcube_co21_n4321,self.basebeam_n4321,
                self.imsize_n4321,self.ra_n4321,self.dec_n4321,self.chans_n4321)

    ###################
    # _align_cube_gal #
    ###################

    def _align_cube_gal(
        self,
        incube1,
        incube2,
        outcube1,
        outcube2,
        beam,
        imsize,
        ra,
        dec,
        chans,
        ):
        """
        align_cubes
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(incube1,taskname)

        # staging cubes
        self._stage_cube(incube1,outcube1+"_tmp1",beam,imsize,ra,dec,115.27120)
        self._stage_cube(incube2,outcube2+"_tmp1",beam,imsize,ra,dec,230.53800)

        # align cubes
        make_gridtemplate(outcube1+"_tmp1",outcube1+"_tmp2",imsize,ra,dec,beam)
        print(outcube1+"_tmp2")
        print(glob.glob(outcube1+"_tmp2"))
        run_imregrid(outcube2+"_tmp1",outcube1+"_tmp2",outcube2+"_tmp1p5",
            axes=[0,1])

        os.system("rm -rf " + outcube1 + "_tmp1")
        os.system("rm -rf " + outcube2 + "_tmp1")
        run_imregrid(outcube2+"_tmp1p5",outcube1+"_tmp2",outcube2+"_tmp2")
        os.system("rm -rf " + outcube2 + "_tmp1p5")

        # clip edge channels
        run_immath_one(outcube1+"_tmp2",outcube1+"_tmp3","IM0",chans,delin=True)
        run_immath_one(outcube2+"_tmp2",outcube2+"_tmp3","IM0",chans,delin=True)
        run_exportfits(outcube1+"_tmp3",outcube1+"_tmp3.fits",delin=True)
        run_exportfits(outcube2+"_tmp3",outcube2+"_tmp3.fits",delin=True)
        run_importfits(outcube1+"_tmp3.fits",outcube1+"_tmp3p5",defaultaxes=True,delin=True)
        run_importfits(outcube2+"_tmp3.fits",outcube2+"_tmp3p5",defaultaxes=True,delin=True)

        # masking
        run_immath_one(outcube1+"_tmp3p5",outcube1+"_tmp4","iif(IM0>-10000000.0,1,0)", "")
        run_immath_one(outcube2+"_tmp3p5",outcube2+"_tmp4","iif(IM0>-10000000.0,1,0)", "")
        run_immath_two(outcube1+"_tmp4",outcube2+"_tmp4",outcube1+"_combined_mask",
            "IM0*IM1",delin=True)

        run_immath_two(outcube1+"_tmp3p5",outcube1+"_combined_mask",outcube1+"_tmp4","iif(IM1>0,IM0,0)")
        run_immath_two(outcube2+"_tmp3p5",outcube1+"_combined_mask",outcube2+"_tmp4","iif(IM1>0,IM0,0)",
            delin=True)
        os.system("rm -rf " + outcube1 + "_tmp3p5")
        os.system("rm -rf " + outcube2 + "_tmp3p5")

        imhead(outcube1+"_tmp4",mode="put",hdkey="beamminor",hdvalue=str(beam)+"arcsec")
        imhead(outcube1+"_tmp4",mode="put",hdkey="beammajor",hdvalue=str(beam)+"arcsec")
        imhead(outcube2+"_tmp4",mode="put",hdkey="beamminor",hdvalue=str(beam)+"arcsec")
        imhead(outcube2+"_tmp4",mode="put",hdkey="beammajor",hdvalue=str(beam)+"arcsec")

        unitconv_Jyb_K(outcube1+"_tmp4",outcube1,115.27120,unitto="Jy/beam",delin=True)
        unitconv_Jyb_K(outcube2+"_tmp4",outcube2,230.53800,unitto="Jy/beam",delin=True)

    ###############
    # _stage_cube #
    ###############

    def _stage_cube(
        self,
        incube,
        outcube,
        beam,
        imsize,
        ra,
        dec,
        restfreq=115.27120,
        ):
        """
        align_cubes - _align_cube_gal
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(incube,taskname)

        run_importfits(incube,outcube+"_tmp1")
        run_roundsmooth(outcube+"_tmp1",outcube+"_tmp2",
            beam,delin=True)
        unitconv_Jyb_K(outcube+"_tmp2",outcube+"_tmp3",restfreq,delin=True)
        self._mask_fov_edges(outcube+"_tmp3",outcube+"_fovmask")
        run_immath_two(outcube+"_tmp3",outcube+"_fovmask",outcube,
            "iif(IM1>0,IM0,0)",delin=True)
        imhead(outcube,mode="put",hdkey="beamminor",hdvalue=str(beam)+"arcsec")
        imhead(outcube,mode="put",hdkey="beammajor",hdvalue=str(beam)+"arcsec")

    ###################
    # _mask_fov_edges #
    ###################

    def _mask_fov_edges(
        self,
        imagename,
        outfile,
        delin=False,
        ):
        """
        align_cubes - _stage_cube
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(imagename,taskname)

        expr1 = "iif(IM0>=-100000000., 1, 0)"
        run_immath_one(imagename,imagename+"_mask_fov_edges_tmp1",expr1,"")
        run_roundsmooth(imagename+"_mask_fov_edges_tmp1",
            imagename+"_mask_fov_edges_tmp2",55.0,delin=True)

        maxval = imstat(imagename+"_mask_fov_edges_tmp2")["max"][0]
        expr2 = "iif(IM0>=" + str(maxval*0.6) + ", 1, 0)"
        run_immath_one(imagename+"_mask_fov_edges_tmp2",
            outfile,expr2,"",delin=True)
        #boolean_masking(imagename+"_mask_fov_edges_tmp3",outfile,delin=True)

    #

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

###################
# end of ToolsR21 #
###################