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
2022-10-17   constrcuted plot_noise, plot_recovery
Toshiki Saito@NAOJ
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

        self.dir_proj         = self._read_key("dir_proj")
        self.dir_raw          = self.dir_proj + self._read_key("dir_raw")
        self.dir_cprops       = self.dir_proj + self._read_key("dir_cprops")
        self.dir_env          = self.dir_proj + self._read_key("dir_env")
        self.dir_halpha       = self.dir_proj + self._read_key("dir_halpha")
        self.dir_wise         = self.dir_proj + self._read_key("dir_wise")
        self.dir_ready        = self.dir_proj + self._read_key("dir_ready")
        self.dir_products     = self.dir_proj + self._read_key("dir_products")
        self.dir_products_txt = self.dir_proj + self._read_key("dir_products_txt")
        self.dir_final        = self.dir_proj + self._read_key("dir_final")
        self._create_dir(self.dir_ready)
        self._create_dir(self.dir_products)
        self._create_dir(self.dir_products_txt)
        self._create_dir(self.dir_final)

    def _set_input_fits(self):
        """
        """

        self.cube_co10_n0628    = self.dir_raw + self._read_key("cube_co10_n0628")
        self.cube_co10_n3627    = self.dir_raw + self._read_key("cube_co10_n3627")
        self.cube_co10_n4254    = self.dir_raw + self._read_key("cube_co10_n4254")
        self.cube_co10_n4321    = self.dir_raw + self._read_key("cube_co10_n4321")
        self.cube_co21_n0628    = self.dir_raw + self._read_key("cube_co21_n0628")
        self.cube_co21_n3627    = self.dir_raw + self._read_key("cube_co21_n3627")
        self.cube_co21_n4254    = self.dir_raw + self._read_key("cube_co21_n4254")
        self.cube_co21_n4321    = self.dir_raw + self._read_key("cube_co21_n4321")

        self.wise1_n0628        = self.dir_wise + self._read_key("wise1_n0628")
        self.wise1_n3627        = self.dir_wise + self._read_key("wise1_n3627")
        self.wise1_n4254        = self.dir_wise + self._read_key("wise1_n4254")
        self.wise1_n4321        = self.dir_wise + self._read_key("wise1_n4321")
        self.wise2_n0628        = self.dir_wise + self._read_key("wise2_n0628")
        self.wise2_n3627        = self.dir_wise + self._read_key("wise2_n3627")
        self.wise2_n4254        = self.dir_wise + self._read_key("wise2_n4254")
        self.wise2_n4321        = self.dir_wise + self._read_key("wise2_n4321")
        self.wise3_n0628        = self.dir_wise + self._read_key("wise3_n0628")
        self.wise3_n3627        = self.dir_wise + self._read_key("wise3_n3627")
        self.wise3_n4254        = self.dir_wise + self._read_key("wise3_n4254")
        self.wise3_n4321        = self.dir_wise + self._read_key("wise3_n4321")

        self.cprops_table_n0628 = self.dir_cprops + self._read_key("cprops_n0628")
        self.cprops_table_n3627 = self.dir_cprops + self._read_key("cprops_n3627")
        self.cprops_table_n4254 = self.dir_cprops + self._read_key("cprops_n4254")
        self.cprops_table_n4321 = self.dir_cprops + self._read_key("cprops_n4321")

        self.env_bulge_n0628    = self.dir_env + self._read_key("env_bulge_n0628")
        self.env_bulge_n3627    = self.dir_env + self._read_key("env_bulge_n3627")
        self.env_bulge_n4254    = self.dir_env + self._read_key("env_bulge_n4254")
        self.env_bulge_n4321    = self.dir_env + self._read_key("env_bulge_n4321")
        self.env_arm_n0628      = self.dir_env + self._read_key("env_arm_n0628")
        self.env_arm_n3627      = self.dir_env + self._read_key("env_arm_n3627")
        self.env_arm_n4254      = self.dir_env + self._read_key("env_arm_n4254")
        self.env_arm_n4321      = self.dir_env + self._read_key("env_arm_n4321")
        self.env_bar_n0628      = self.dir_env + self._read_key("env_bar_n0628")
        self.env_bar_n3627      = self.dir_env + self._read_key("env_bar_n3627")
        self.env_bar_n4254      = self.dir_env + self._read_key("env_bar_n4254")
        self.env_bar_n4321      = self.dir_env + self._read_key("env_bar_n4321")

        self.halpha_mask_n0628  = self.dir_halpha + self._read_key("halpha_mask_n0628")
        self.halpha_mask_n3627  = self.dir_halpha + self._read_key("halpha_mask_n3627")
        self.halpha_mask_n4254  = self.dir_halpha + self._read_key("halpha_mask_n4254")
        self.halpha_mask_n4321  = self.dir_halpha + self._read_key("halpha_mask_n4321")

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

        self.pa_n0628          = float(self._read_key("pa_n0628", "gal"))
        self.pa_n3627          = float(self._read_key("pa_n3627", "gal"))
        self.pa_n4254          = float(self._read_key("pa_n4254", "gal"))
        self.pa_n4321          = float(self._read_key("pa_n4321", "gal"))

        self.incl_n0628        = float(self._read_key("incl_n0628", "gal"))
        self.incl_n3627        = float(self._read_key("incl_n3627", "gal"))
        self.incl_n4254        = float(self._read_key("incl_n4254", "gal"))
        self.incl_n4321        = float(self._read_key("incl_n4321", "gal"))

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

        self.snr_mom           = 1.5
        self.snr_gmc           = 5.0
        self.snr_ratio         = 3.0
        self.snr_showcase      = 2.5

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
        self.c_n0628                  = "tomato"
        self.c_n3627                  = "purple"
        self.c_n4254                  = "forestgreen"
        self.c_n4321                  = "deepskyblue"
        self.text_back_alpha          = 0.9

        # output txt and png
        self.outpng_noise_hist        = self.dir_products + self._read_key("outpng_noise_hist")
        self.noise_hist_xmax_snr      = 7.5
        self.noise_hist_bins          = 500
        self.noise_hist_snr4plt       = 2.5

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

        self.outpng_recovery          = self.dir_products + self._read_key("outpng_recovery")

        self.outpng_co10_n0628        = self.dir_products + self._read_key("outpng_co10_n0628")
        self.outpng_co10_n3627        = self.dir_products + self._read_key("outpng_co10_n3627")
        self.outpng_co10_n4254        = self.dir_products + self._read_key("outpng_co10_n4254")
        self.outpng_co10_n4321        = self.dir_products + self._read_key("outpng_co10_n4321")
        self.outpng_co21_n0628        = self.dir_products + self._read_key("outpng_co21_n0628")
        self.outpng_co21_n3627        = self.dir_products + self._read_key("outpng_co21_n3627")
        self.outpng_co21_n4254        = self.dir_products + self._read_key("outpng_co21_n4254")
        self.outpng_co21_n4321        = self.dir_products + self._read_key("outpng_co21_n4321")
        self.outpng_r21_n0628         = self.dir_products + self._read_key("outpng_r21_n0628")
        self.outpng_r21_n3627         = self.dir_products + self._read_key("outpng_r21_n3627")
        self.outpng_r21_n4254         = self.dir_products + self._read_key("outpng_r21_n4254")
        self.outpng_r21_n4321         = self.dir_products + self._read_key("outpng_r21_n4321")

        self.outpng_m0_vs_m8          = self.dir_products + self._read_key("outpng_m0_vs_m8")

        self.outpng_hist_550pc        = self.dir_products + self._read_key("outpng_hist_550pc")
        self.hist_550pc_cnter_radius  = 1.0 # kpc
        self.hist_550pc_bins          = 50
        self.hist_550pc_hrange        = [0.00, 1.10]

        self.outpng_violins           = self.dir_products + self._read_key("outpng_violins")

        self.outpng_r21hl_n0628       = self.dir_products + self._read_key("outpng_r21hl_n0628")
        self.outpng_cprops_n0628      = self.dir_products + self._read_key("outpng_cprops_n0628")
        self.outpng_env_n0628         = self.dir_products + self._read_key("outpng_env_n0628")
        self.outpng_halpha_n0628      = self.dir_products + self._read_key("outpng_halpha_n0628")
        self.outpng_r21hl_n3627       = self.dir_products + self._read_key("outpng_r21hl_n3627")
        self.outpng_cprops_n3627      = self.dir_products + self._read_key("outpng_cprops_n3627")
        self.outpng_env_n3627         = self.dir_products + self._read_key("outpng_env_n3627")
        self.outpng_halpha_n3627      = self.dir_products + self._read_key("outpng_halpha_n3627")
        self.outpng_r21hl_n4254       = self.dir_products + self._read_key("outpng_r21hl_n4254")
        self.outpng_cprops_n4254      = self.dir_products + self._read_key("outpng_cprops_n4254")
        self.outpng_env_n4254         = self.dir_products + self._read_key("outpng_env_n4254")
        self.outpng_halpha_n4254      = self.dir_products + self._read_key("outpng_halpha_n4254")
        self.outpng_r21hl_n4321       = self.dir_products + self._read_key("outpng_r21hl_n4321")
        self.outpng_cprops_n4321      = self.dir_products + self._read_key("outpng_cprops_n4321")
        self.outpng_env_n4321         = self.dir_products + self._read_key("outpng_env_n4321")
        self.outpng_halpha_n4321      = self.dir_products + self._read_key("outpng_halpha_n4321")

    ##################
    # run_phangs_r21 #
    ##################

    def run_phangs_r21(
        self,
        do_all          = False,
        # analysis
        do_align        = False,
        do_multismooth  = False,
        do_moments      = False,
        do_align_other  = False,
        # plot figures in paper
        plot_noise      = False,
        plot_recovery   = False,
        plot_showcase   = False,
        plot_m0_vs_m8   = False,
        plot_hist_550pc = False,
        plot_violins    = False,
        plot_masks      = False,
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
            plot_noise      = True
            plot_recovery   = True
            plot_showcase   = True
            plot_m0_vs_m8   = True

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
            self.plot_noise_hist()
            self.plot_noise_vs_beam() # ngc4254 co21 rms curve is strange.

        if plot_recovery==True:
            self.plot_recovery()

        if plot_showcase==True:
            self.plot_showcase()

        if plot_m0_vs_m8==True:
            self.plot_m0_vs_m8()

        if plot_hist_550pc==True:
            self.plot_hist_550pc()

        if plot_violins==True:
            self.plot_violins()

        if plot_masks==True:
            self.plot_masks()

    #####################
    #####################
    ### plotting part ###
    #####################
    #####################

    ##############
    # plot_masks #
    ##############

    def plot_masks(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outcube_co10_n0628,taskname)

        ###########
        # ngc0628 #
        ###########
        this_co21       = self.outmom_co21_n0628.replace("momX","mom0")
        this_eco21      = self.outmom_co21_n0628.replace("momX","emom0")
        this_co21       = self._clip_for_showcase(this_co21,this_eco21)
        this_r21hl      = self.outfits_r21hl_n0628
        this_cprops     = self.outfits_cprops_n0628
        this_env        = self.outfits_env_n0628
        this_halpha     = self.outfits_halpha_n0628
        this_out_r21hl  = self.outpng_r21hl_n0628
        this_out_cprops = self.outpng_cprops_n0628
        this_out_env    = self.outpng_env_n0628
        this_out_halpha = self.outpng_halpha_n0628
        this_imsize     = self.imsize_n0628
        this_ra         = self.ra_n0628
        this_dec        = self.dec_n0628
        this_scalebar   = 1000. / self.scale_n0628
        this_title      = "NGC 0628"
        myfig_fits2png(
            this_r21hl,
            this_out_r21hl,
            imcontour1=this_co21,
            imsize_as=this_imsize,
            ra_cnt=this_ra,
            dec_cnt=this_dec,
            levels_cont1=[0.05, 0.1, 0.2, 0.4, 0.8, 0.96],
            set_title=this_title,
            colorlog=False,
            set_cmap="rainbow",
            scalebar=this_scalebar,
            label_scalebar="1 kpc",
            comment="$R_{21}$ mask",
            clim=[0.7,3.3],
            )
        myfig_fits2png(
            this_cprops,
            this_out_cprops,
            imcontour1=this_co21,
            imsize_as=this_imsize,
            ra_cnt=this_ra,
            dec_cnt=this_dec,
            levels_cont1=[0.05, 0.1, 0.2, 0.4, 0.8, 0.96],
            set_title=this_title,
            colorlog=False,
            set_cmap="bwr",
            scalebar=this_scalebar,
            label_scalebar="1 kpc",
            comment="cloud mask",
            clim=[-0.2,1.2],
            )
        myfig_fits2png(
            this_env,
            this_out_env,
            imcontour1=this_co21,
            imsize_as=this_imsize,
            ra_cnt=this_ra,
            dec_cnt=this_dec,
            levels_cont1=[0.05, 0.1, 0.2, 0.4, 0.8, 0.96],
            set_title=this_title,
            colorlog=False,
            set_cmap="gnuplot",
            scalebar=this_scalebar,
            label_scalebar="1 kpc",
            comment="environmental mask",
            clim=[0,3.5],
            )
        myfig_fits2png(
            this_halpha,
            this_out_halpha,
            imcontour1=this_co21,
            imsize_as=this_imsize,
            ra_cnt=this_ra,
            dec_cnt=this_dec,
            levels_cont1=[0.05, 0.1, 0.2, 0.4, 0.8, 0.96],
            set_title=this_title,
            colorlog=False,
            set_cmap="PiYG_r",
            scalebar=this_scalebar,
            label_scalebar="1 kpc",
            comment="HII region mask",
            clim=[-0.2,1.2],
            )

        ###########
        # ngc3627 #
        ###########
        this_co21       = self.outmom_co21_n3627.replace("momX","mom0")
        this_eco21      = self.outmom_co21_n3627.replace("momX","emom0")
        this_co21       = self._clip_for_showcase(this_co21,this_eco21)
        this_r21hl      = self.outfits_r21hl_n3627
        this_cprops     = self.outfits_cprops_n3627
        this_env        = self.outfits_env_n3627
        this_halpha     = self.outfits_halpha_n3627
        this_out_r21hl  = self.outpng_r21hl_n3627
        this_out_cprops = self.outpng_cprops_n3627
        this_out_env    = self.outpng_env_n3627
        this_out_halpha = self.outpng_halpha_n3627
        this_imsize     = self.imsize_n3627
        this_ra         = self.ra_n3627
        this_dec        = self.dec_n3627
        this_scalebar   = 1000. / self.scale_n3627
        this_title      = "NGC 3627"
        myfig_fits2png(
            this_r21hl,
            this_out_r21hl,
            imcontour1=this_co21,
            imsize_as=this_imsize,
            ra_cnt=this_ra,
            dec_cnt=this_dec,
            levels_cont1=[0.05, 0.1, 0.2, 0.4, 0.8, 0.96],
            set_title=this_title,
            colorlog=False,
            set_cmap="rainbow",
            scalebar=this_scalebar,
            label_scalebar="1 kpc",
            comment="$R_{21}$ mask",
            clim=[0.7,3.3],
            )
        myfig_fits2png(
            this_cprops,
            this_out_cprops,
            imcontour1=this_co21,
            imsize_as=this_imsize,
            ra_cnt=this_ra,
            dec_cnt=this_dec,
            levels_cont1=[0.05, 0.1, 0.2, 0.4, 0.8, 0.96],
            set_title=this_title,
            colorlog=False,
            set_cmap="bwr",
            scalebar=this_scalebar,
            label_scalebar="1 kpc",
            comment="cloud mask",
            clim=[-0.2,1.2],
            )
        myfig_fits2png(
            this_env,
            this_out_env,
            imcontour1=this_co21,
            imsize_as=this_imsize,
            ra_cnt=this_ra,
            dec_cnt=this_dec,
            levels_cont1=[0.05, 0.1, 0.2, 0.4, 0.8, 0.96],
            set_title=this_title,
            colorlog=False,
            set_cmap="gnuplot",
            scalebar=this_scalebar,
            label_scalebar="1 kpc",
            comment="environmental mask",
            clim=[0,3.5],
            )
        myfig_fits2png(
            this_halpha,
            this_out_halpha,
            imcontour1=this_co21,
            imsize_as=this_imsize,
            ra_cnt=this_ra,
            dec_cnt=this_dec,
            levels_cont1=[0.05, 0.1, 0.2, 0.4, 0.8, 0.96],
            set_title=this_title,
            colorlog=False,
            set_cmap="PiYG_r",
            scalebar=this_scalebar,
            label_scalebar="1 kpc",
            comment="HII region mask",
            clim=[-0.2,1.2],
            )

        ###########
        # ngc4254 #
        ###########
        this_co21       = self.outmom_co21_n4254.replace("momX","mom0")
        this_eco21      = self.outmom_co21_n4254.replace("momX","emom0")
        this_co21       = self._clip_for_showcase(this_co21,this_eco21)
        this_r21hl      = self.outfits_r21hl_n4254
        this_cprops     = self.outfits_cprops_n4254
        this_env        = self.outfits_env_n4254
        this_halpha     = self.outfits_halpha_n4254
        this_out_r21hl  = self.outpng_r21hl_n4254
        this_out_cprops = self.outpng_cprops_n4254
        this_out_env    = self.outpng_env_n4254
        this_out_halpha = self.outpng_halpha_n4254
        this_imsize     = self.imsize_n4254
        this_ra         = self.ra_n4254
        this_dec        = self.dec_n4254
        this_scalebar   = 1000. / self.scale_n4254
        this_title      = "NGC 4254"
        myfig_fits2png(
            this_r21hl,
            this_out_r21hl,
            imcontour1=this_co21,
            imsize_as=this_imsize,
            ra_cnt=this_ra,
            dec_cnt=this_dec,
            levels_cont1=[0.05, 0.1, 0.2, 0.4, 0.8, 0.96],
            set_title=this_title,
            colorlog=False,
            set_cmap="rainbow",
            scalebar=this_scalebar,
            label_scalebar="1 kpc",
            comment="$R_{21}$ mask",
            clim=[0.7,3.3],
            )
        myfig_fits2png(
            this_cprops,
            this_out_cprops,
            imcontour1=this_co21,
            imsize_as=this_imsize,
            ra_cnt=this_ra,
            dec_cnt=this_dec,
            levels_cont1=[0.05, 0.1, 0.2, 0.4, 0.8, 0.96],
            set_title=this_title,
            colorlog=False,
            set_cmap="bwr",
            scalebar=this_scalebar,
            label_scalebar="1 kpc",
            comment="cloud mask",
            clim=[-0.2,1.2],
            )
        myfig_fits2png(
            this_env,
            this_out_env,
            imcontour1=this_co21,
            imsize_as=this_imsize,
            ra_cnt=this_ra,
            dec_cnt=this_dec,
            levels_cont1=[0.05, 0.1, 0.2, 0.4, 0.8, 0.96],
            set_title=this_title,
            colorlog=False,
            set_cmap="gnuplot",
            scalebar=this_scalebar,
            label_scalebar="1 kpc",
            comment="environmental mask",
            clim=[0,3.5],
            )
        myfig_fits2png(
            this_halpha,
            this_out_halpha,
            imcontour1=this_co21,
            imsize_as=this_imsize,
            ra_cnt=this_ra,
            dec_cnt=this_dec,
            levels_cont1=[0.05, 0.1, 0.2, 0.4, 0.8, 0.96],
            set_title=this_title,
            colorlog=False,
            set_cmap="PiYG_r",
            scalebar=this_scalebar,
            label_scalebar="1 kpc",
            comment="HII region mask",
            clim=[-0.2,1.2],
            )

        ###########
        # ngc4321 #
        ###########
        this_co21       = self.outmom_co21_n4321.replace("momX","mom0")
        this_eco21      = self.outmom_co21_n4321.replace("momX","emom0")
        this_co21       = self._clip_for_showcase(this_co21,this_eco21)
        this_r21hl      = self.outfits_r21hl_n4321
        this_cprops     = self.outfits_cprops_n4321
        this_env        = self.outfits_env_n4321
        this_halpha     = self.outfits_halpha_n4321
        this_out_r21hl  = self.outpng_r21hl_n4321
        this_out_cprops = self.outpng_cprops_n4321
        this_out_env    = self.outpng_env_n4321
        this_out_halpha = self.outpng_halpha_n4321
        this_imsize     = self.imsize_n4321
        this_ra         = self.ra_n4321
        this_dec        = self.dec_n4321
        this_scalebar   = 1000. / self.scale_n4321
        this_title      = "NGC 4321"
        myfig_fits2png(
            this_r21hl,
            this_out_r21hl,
            imcontour1=this_co21,
            imsize_as=this_imsize,
            ra_cnt=this_ra,
            dec_cnt=this_dec,
            levels_cont1=[0.05, 0.1, 0.2, 0.4, 0.8, 0.96],
            set_title=this_title,
            colorlog=False,
            set_cmap="rainbow",
            scalebar=this_scalebar,
            label_scalebar="1 kpc",
            comment="$R_{21}$ mask",
            clim=[0.7,3.3],
            )
        myfig_fits2png(
            this_cprops,
            this_out_cprops,
            imcontour1=this_co21,
            imsize_as=this_imsize,
            ra_cnt=this_ra,
            dec_cnt=this_dec,
            levels_cont1=[0.05, 0.1, 0.2, 0.4, 0.8, 0.96],
            set_title=this_title,
            colorlog=False,
            set_cmap="bwr",
            scalebar=this_scalebar,
            label_scalebar="1 kpc",
            comment="cloud mask",
            clim=[-0.2,1.2],
            )
        myfig_fits2png(
            this_env,
            this_out_env,
            imcontour1=this_co21,
            imsize_as=this_imsize,
            ra_cnt=this_ra,
            dec_cnt=this_dec,
            levels_cont1=[0.05, 0.1, 0.2, 0.4, 0.8, 0.96],
            set_title=this_title,
            colorlog=False,
            set_cmap="gnuplot",
            scalebar=this_scalebar,
            label_scalebar="1 kpc",
            comment="environmental mask",
            clim=[0,3.5],
            )
        myfig_fits2png(
            this_halpha,
            this_out_halpha,
            imcontour1=this_co21,
            imsize_as=this_imsize,
            ra_cnt=this_ra,
            dec_cnt=this_dec,
            levels_cont1=[0.05, 0.1, 0.2, 0.4, 0.8, 0.96],
            set_title=this_title,
            colorlog=False,
            set_cmap="PiYG_r",
            scalebar=this_scalebar,
            label_scalebar="1 kpc",
            comment="HII region mask",
            clim=[-0.2,1.2],
            )

    #

    ################
    # plot_violins #
    ################

    def plot_violins(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outcube_co10_n0628,taskname)

        ###########
        # prepare #
        ###########

        this_basebeam    = str(self.basebeam_n0628).replace(".","p").zfill(4)
        this_beams_n0628 = [s for s in self.beams_n0628 if s%4==0]
        this_r21         = self.outfits_r21_n0628.replace(this_basebeam,"????")
        this_er21        = self.outfits_er21_n0628.replace(this_basebeam,"????")
        this_co10        = self.outmom_co10_n0628.replace(this_basebeam,"????").replace("momX","mom0")
        this_co21        = self.outmom_co21_n0628.replace(this_basebeam,"????").replace("momX","mom0")
        this_ra          = float(self.ra_n0628)
        this_dec         = float(self.dec_n0628)
        this_scale       = self.scale_n0628
        this_pa          = self.pa_n0628
        this_incl        = self.incl_n0628
        co10s_n0628, co21s_n0628, r21s_n0628 = \
            self._import_violins(this_co10,this_co21,this_r21,this_er21,this_beams_n0628,this_ra,this_dec,this_scale,this_pa,this_incl)

        this_basebeam    = str(self.basebeam_n3627).replace(".","p").zfill(4)
        this_beams_n3627 = [s for s in self.beams_n3627 if s%4==0]
        this_r21         = self.outfits_r21_n3627.replace(this_basebeam,"????")
        this_er21        = self.outfits_er21_n3627.replace(this_basebeam,"????")
        this_co10        = self.outmom_co10_n3627.replace(this_basebeam,"????").replace("momX","mom0")
        this_co21        = self.outmom_co21_n3627.replace(this_basebeam,"????").replace("momX","mom0")
        this_ra          = float(self.ra_n3627)
        this_dec         = float(self.dec_n3627)
        this_scale       = self.scale_n3627
        this_pa          = self.pa_n3627
        this_incl        = self.incl_n3627
        co10s_n3627, co21s_n3627, r21s_n3627 = \
            self._import_violins(this_co10,this_co21,this_r21,this_er21,this_beams_n3627,this_ra,this_dec,this_scale,this_pa,this_incl)

        this_basebeam    = str(self.basebeam_n4254).replace(".","p").zfill(4)
        this_beams_n4254 = [s for s in self.beams_n4254 if s%4==0]
        this_r21         = self.outfits_r21_n4254.replace(this_basebeam,"????")
        this_er21        = self.outfits_er21_n4254.replace(this_basebeam,"????")
        this_co10        = self.outmom_co10_n4254.replace(this_basebeam,"????").replace("momX","mom0")
        this_co21        = self.outmom_co21_n4254.replace(this_basebeam,"????").replace("momX","mom0")
        this_ra          = float(self.ra_n4254)
        this_dec         = float(self.dec_n4254)
        this_scale       = self.scale_n4254
        this_pa          = self.pa_n4254
        this_incl        = self.incl_n4254
        co10s_n4254, co21s_n4254, r21s_n4254 = \
            self._import_violins(this_co10,this_co21,this_r21,this_er21,this_beams_n4254,this_ra,this_dec,this_scale,this_pa,this_incl)

        this_basebeam    = str(self.basebeam_n4321).replace(".","p").zfill(4)
        this_beams_n4321 = [s for s in self.beams_n4321 if s%4==0]
        this_r21         = self.outfits_r21_n4321.replace(this_basebeam,"????")
        this_er21        = self.outfits_er21_n4321.replace(this_basebeam,"????")
        this_co10        = self.outmom_co10_n4321.replace(this_basebeam,"????").replace("momX","mom0")
        this_co21        = self.outmom_co21_n4321.replace(this_basebeam,"????").replace("momX","mom0")
        this_ra          = float(self.ra_n4321)
        this_dec         = float(self.dec_n4321)
        this_scale       = self.scale_n4321
        this_pa          = self.pa_n4321
        this_incl        = self.incl_n4321
        co10s_n4321, co21s_n4321, r21s_n4321 = \
            self._import_violins(this_co10,this_co21,this_r21,this_er21,this_beams_n4321,this_ra,this_dec,this_scale,this_pa,this_incl)

        ax1_title  = "Area-weighted"
        ax2_title  = "CO(1-0)-weighted"
        ax3_title  = "CO(2-1)-weighted"
        xlim_n0628 = [np.min(this_beams_n0628)-2.0, np.max(this_beams_n0628)+2.0]
        xlim_n3627 = [np.min(this_beams_n3627)-2.0, np.max(this_beams_n3627)+2.0]
        xlim_n4254 = [np.min(this_beams_n4254)-2.0, np.max(this_beams_n4254)+2.0]
        xlim_n4321 = [np.min(this_beams_n4321)-2.0, np.max(this_beams_n4321)+2.0]
        ylim       = [0.0,1.2]
        xlabel     = "Beam size"
        ylabel     = "$R_{21}$"

        ########
        # plot #
        ########

        # set plt, ax
        plt.figure(figsize=(15,12))
        plt.subplots_adjust(bottom=0.09, left=0.07, right=0.99, top=0.95)
        gs   = gridspec.GridSpec(nrows=12, ncols=9)
        ax1  = plt.subplot(gs[0:3,0:3])
        ax2  = plt.subplot(gs[0:3,3:6])
        ax3  = plt.subplot(gs[0:3,6:9])
        ax4  = plt.subplot(gs[3:6,0:3])
        ax5  = plt.subplot(gs[3:6,3:6])
        ax6  = plt.subplot(gs[3:6,6:9])
        ax7  = plt.subplot(gs[6:9,0:3])
        ax8  = plt.subplot(gs[6:9,3:6])
        ax9  = plt.subplot(gs[6:9,6:9])
        ax10 = plt.subplot(gs[9:12,0:3])
        ax11 = plt.subplot(gs[9:12,3:6])
        ax12 = plt.subplot(gs[9:12,6:9])

        # set ax param
        factor        = 1.65
        x_beamtext    = 0.12
        font_beamtext = 15
        myax_set(ax1,  "y", xlim_n0628, ylim, ax1_title, None, ylabel)
        myax_set(ax2,  "y", xlim_n0628, ylim, ax2_title, None, None)
        myax_set(ax3,  "y", xlim_n0628, ylim, ax3_title, None, None)
        myax_set(ax4,  "y", xlim_n3627, ylim, None, None, ylabel)
        myax_set(ax5,  "y", xlim_n3627, ylim, None, None, None)
        myax_set(ax6,  "y", xlim_n3627, ylim, None, None, None)
        myax_set(ax7,  "y", xlim_n4254, ylim, None, None, ylabel)
        myax_set(ax8,  "y", xlim_n4254, ylim, None, None, None)
        myax_set(ax9,  "y", xlim_n4254, ylim, None, None, None)
        myax_set(ax10, "y", xlim_n4321, ylim, None, xlabel, ylabel)
        myax_set(ax11, "y", xlim_n4321, ylim, None, xlabel, None)
        myax_set(ax12, "y", xlim_n4321, ylim, None, xlabel, None)
        ax1.set_yticks([0.3,0.6,0.9])
        ax2.set_yticks([0.3,0.6,0.9])
        ax3.set_yticks([0.3,0.6,0.9])
        ax4.set_yticks([0.3,0.6,0.9])
        ax1.text(4.4,  x_beamtext,  "4\"", fontsize=font_beamtext)
        ax1.text(8.4,  x_beamtext,  "8\"", fontsize=font_beamtext)
        ax1.text(12.4, x_beamtext, "12\"", fontsize=font_beamtext)
        ax1.text(16.4, x_beamtext, "16\"", fontsize=font_beamtext)
        ax1.text(20.4, x_beamtext, "20\"", fontsize=font_beamtext)
        ax4.text(8.4,  x_beamtext,  "8\"", fontsize=font_beamtext)
        ax4.text(12.4, x_beamtext, "12\"", fontsize=font_beamtext)
        ax4.text(16.4, x_beamtext, "16\"", fontsize=font_beamtext)
        ax4.text(20.4, x_beamtext, "20\"", fontsize=font_beamtext)
        ax4.text(24.4, x_beamtext, "24\"", fontsize=font_beamtext)
        ax7.text(8.4,  x_beamtext,  "8\"", fontsize=font_beamtext)
        ax7.text(12.4, x_beamtext, "12\"", fontsize=font_beamtext)
        ax7.text(16.4, x_beamtext, "16\"", fontsize=font_beamtext)
        ax7.text(20.4, x_beamtext, "20\"", fontsize=font_beamtext)
        ax7.text(24.4, x_beamtext, "24\"", fontsize=font_beamtext)
        ax10.text(4.4,  x_beamtext,  "4\"", fontsize=font_beamtext)
        ax10.text(8.4,  x_beamtext,  "8\"", fontsize=font_beamtext)
        ax10.text(12.4, x_beamtext, "12\"", fontsize=font_beamtext)
        ax10.text(16.4, x_beamtext, "16\"", fontsize=font_beamtext)
        ax10.text(20.4, x_beamtext, "20\"", fontsize=font_beamtext)

        # unset xlabels
        ax1.tick_params(labelbottom=False,labelleft=True,labelright=False,labeltop=False)
        ax2.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
        ax3.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
        ax4.tick_params(labelbottom=False,labelleft=True,labelright=False,labeltop=False)
        ax5.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
        ax6.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
        ax7.tick_params(labelbottom=False,labelleft=True,labelright=False,labeltop=False)
        ax8.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
        ax9.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
        ax10.tick_params(labelbottom=True,labelleft=True,labelright=False,labeltop=False)
        ax11.tick_params(labelbottom=True,labelleft=False,labelright=False,labeltop=False)
        ax12.tick_params(labelbottom=True,labelleft=False,labelright=False,labeltop=False)
        ax10.axes.xaxis.set_ticklabels([])
        ax11.axes.xaxis.set_ticklabels([])
        ax12.axes.xaxis.set_ticklabels([])

        # plot
        self._ax_multiviolin(ax1,r21s_n0628,this_beams_n0628,ylim,self.c_n0628,0.8,weights=None)
        self._ax_multiviolin(ax2,r21s_n0628,this_beams_n0628,ylim,self.c_n0628,0.5,weights=co10s_n0628)
        self._ax_multiviolin(ax3,r21s_n0628,this_beams_n0628,ylim,self.c_n0628,0.3,weights=co21s_n0628)

        self._ax_multiviolin(ax4,r21s_n3627,this_beams_n3627,ylim,self.c_n3627,0.8,weights=None)
        self._ax_multiviolin(ax5,r21s_n3627,this_beams_n3627,ylim,self.c_n3627,0.5,weights=co10s_n3627)
        self._ax_multiviolin(ax6,r21s_n3627,this_beams_n3627,ylim,self.c_n3627,0.3,weights=co21s_n3627)

        self._ax_multiviolin(ax7,r21s_n4254,this_beams_n4254,ylim,self.c_n4254,0.8,weights=None)
        self._ax_multiviolin(ax8,r21s_n4254,this_beams_n4254,ylim,self.c_n4254,0.5,weights=co10s_n4254)
        self._ax_multiviolin(ax9,r21s_n4254,this_beams_n4254,ylim,self.c_n4254,0.3,weights=co21s_n4254)

        self._ax_multiviolin(ax10,r21s_n4321,this_beams_n4321,ylim,self.c_n4321,0.8,weights=None)
        self._ax_multiviolin(ax11,r21s_n4321,this_beams_n4321,ylim,self.c_n4321,0.5,weights=co10s_n4321)
        self._ax_multiviolin(ax12,r21s_n4321,this_beams_n4321,ylim,self.c_n4321,0.3,weights=co21s_n4321)

        # text
        t=ax1.text(0.03, 0.86, "NGC 0628", color=self.c_n0628, horizontalalignment="left", transform=ax1.transAxes, size=self.legend_fontsize-2, fontweight="bold")
        t.set_bbox(dict(facecolor="white", alpha=self.text_back_alpha, lw=0))
        t=ax4.text(0.03, 0.86, "NGC 3627", color=self.c_n3627, horizontalalignment="left", transform=ax4.transAxes, size=self.legend_fontsize-2, fontweight="bold")
        t.set_bbox(dict(facecolor="white", alpha=self.text_back_alpha, lw=0))
        t=ax7.text(0.03, 0.86, "NGC 4254", color=self.c_n4254, horizontalalignment="left", transform=ax7.transAxes, size=self.legend_fontsize-2, fontweight="bold")
        t.set_bbox(dict(facecolor="white", alpha=self.text_back_alpha, lw=0))
        t=ax10.text(0.03, 0.86, "NGC 4321", color=self.c_n4321, horizontalalignment="left", transform=ax10.transAxes, size=self.legend_fontsize-2, fontweight="bold")
        t.set_bbox(dict(facecolor="white", alpha=self.text_back_alpha, lw=0))

        plt.savefig(self.outpng_violins, dpi=self.fig_dpi)

    ###################
    # _ax_multiviolin #
    ###################

    def _ax_multiviolin(
        self,
        ax,
        r21s,
        beams,
        ylim,
        color,
        alpha,
        weights=None,
        vmin=None,
        vmax=None,
        ):
        """
        """

        # plot each violin
        list_p16, list_p50, list_p84 = [], [], []
        for i in range(len(beams)):
            this_beam    = beams[i]
            this_r21     = r21s[i]
            if weights!=None:
                this_weights = weights[i]
            else:
                this_weights = None
            p16,p50,p84  = self._ax_violin(ax,this_r21,this_beam,ylim,color,alpha,weights=this_weights)
            list_p16.append(p16)
            list_p50.append(p50)
            list_p84.append(p84)
 
        # plot running pctls
        ax.plot(beams, list_p16, "--", color="black", lw=1)
        ax.plot(beams, list_p50, "--", color="black", lw=1)
        ax.plot(beams, list_p84, "-", color="black", lw=1)

    ##############
    # _ax_violin #
    ##############

    def _ax_violin(
        self,
        ax,
        data,
        beam,
        ylim,
        color,
        alpha,
        weights=None,
        vmin=None,
        vmax=None,
        ):
        """
        """

        ygrid  = np.linspace(ylim[0], ylim[1], num=1000)

        # prepare
        if vmin==None:
            vmin = np.min(data)

        if vmax==None:
            vmax = np.max(data)

        # percentiles
        p2   = self._weighted_percentile(data,2,weights=weights)
        p16  = self._weighted_percentile(data,16,weights=weights)
        p50  = self._weighted_percentile(data,50,weights=weights)
        p84  = self._weighted_percentile(data,84,weights=weights)
        p98  = self._weighted_percentile(data,98,weights=weights)

        # kde
        h,e = np.histogram(data, bins=1000, density=True, weights=weights)
        x = np.linspace(e.min(), e.max())
        resamples = np.random.choice((e[:-1] + e[1:])/2, size=5000, p=h/h.sum())
        l = gaussian_kde(resamples)
        #l = gaussian_kde(data)
        data = np.array(l(ygrid) / np.max(l(ygrid))) / 0.55


        left  = beam-data
        right = beam+data
        cut = np.where((ygrid<vmax)&(ygrid>vmin))

        ax.plot(right[cut], ygrid[cut], lw=1, color="grey")
        ax.plot(left[cut], ygrid[cut], lw=1, color="grey")
        ax.fill_betweenx(ygrid, left, right, facecolor=color, alpha=alpha, lw=0)

        # percentiles
        ax.plot([beam,beam],[p2,p98],lw=2,color="grey")
        ax.plot([beam,beam],[p16,p84],lw=5,color="grey")
        ax.plot(beam,p50,".",color="black",markersize=8, markeredgewidth=0)

        return p16,p50,p84

    ###################
    # _import_violins #
    ###################

    def _import_violins(self,co10,co21,r21,er21,beams,ra,dec,scale,pa,incl):
        """
        plot_hist_550pc
        """

        array_co10 = []
        array_co21 = []
        array_r21 = []
        for i in range(len(beams)):
            # get names
            this_beam     = str(beams[i]).replace(".","p").zfill(4)
            this_co10     = co10.replace("????",this_beam)
            this_co21     = co21.replace("????",this_beam)
            this_r21      = r21.replace("????",this_beam)
            this_r21_err  = er21.replace("????",this_beam)

            shape         = imhead(this_co10,mode="list")["shape"]
            box           = "0,0," + str(shape[0]-1) + "," + str(shape[1]-1)
            ra_deg        = imval(this_co10,box=box)["coords"][:,:,0] * 180/np.pi
            dec_deg       = imval(this_co10,box=box)["coords"][:,:,1] * 180/np.pi
            this_co10     = imval(this_co10,box=box)["data"]
            this_co21     = imval(this_co21,box=box)["data"]
            this_r21      = imval(this_r21,box=box)["data"]
            this_r21_err  = imval(this_r21_err,box=box)["data"]

            dist_pc, _ = self._get_rel_dist_pc(ra_deg, dec_deg, ra, dec, scale, pa, incl)
            dist_kpc   = dist_pc / 1000.

            cut = np.where( (~np.isnan(this_co10)) & (~np.isinf(this_co10)) & (this_co10!=0) \
                & (~np.isnan(this_co21)) & (~np.isinf(this_co21)) & (this_co21!=0) \
                & (~np.isnan(this_r21)) & (~np.isinf(this_r21)) & (this_r21!=0) \
                & (this_r21!=this_r21_err*self.snr_ratio) & (dist_kpc > self.hist_550pc_cnter_radius) ) 

            array_co10.append(this_co10[cut].flatten())
            array_co21.append(this_co21[cut].flatten())
            array_r21.append(this_r21[cut].flatten())

        return array_co10, array_co21, array_r21

    #

    ###################
    # plot_hist_550pc #
    ###################

    def plot_hist_550pc(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outcube_co10_n0628,taskname)

        ###########
        # prepare #
        ###########

        this_basebeam = str(self.basebeam_n0628).replace(".","p").zfill(4)
        this_wisebeam = str(self.beam_wise_n0628).replace(".","p").zfill(4)
        this_r21      = self.outfits_r21_n0628.replace(this_basebeam,this_wisebeam)
        this_er21     = self.outfits_er21_n0628.replace(this_basebeam,this_wisebeam)
        this_co10     = self.outmom_co10_n0628.replace(this_basebeam,this_wisebeam).replace("momX","mom0")
        this_co21     = self.outmom_co21_n0628.replace(this_basebeam,this_wisebeam).replace("momX","mom0")
        this_ra       = float(self.ra_n0628)
        this_dec      = float(self.dec_n0628)
        this_scale    = self.scale_n0628
        this_pa       = self.pa_n0628
        this_incl     = self.incl_n0628
        hist_all_n0628, hist_inner_n0628, hist_outer_n0628, pctls_all_n0628, pctls_inner_n0628, pctls_outer_n0628 = \
            self._import_hist_550pc(this_co10,this_co21,this_r21,this_er21,this_ra,this_dec,this_scale,this_pa,this_incl)

        this_basebeam = str(self.basebeam_n3627).replace(".","p").zfill(4)
        this_wisebeam = str(self.beam_wise_n3627).replace(".","p").zfill(4)
        this_r21      = self.outfits_r21_n3627.replace(this_basebeam,this_wisebeam)
        this_er21     = self.outfits_er21_n3627.replace(this_basebeam,this_wisebeam)
        this_co10     = self.outmom_co10_n3627.replace(this_basebeam,this_wisebeam).replace("momX","mom0")
        this_co21     = self.outmom_co21_n3627.replace(this_basebeam,this_wisebeam).replace("momX","mom0")
        this_ra       = float(self.ra_n3627)
        this_dec      = float(self.dec_n3627)
        this_scale    = self.scale_n3627
        this_pa       = self.pa_n3627
        this_incl     = self.incl_n3627
        hist_all_n3627, hist_inner_n3627, hist_outer_n3627, pctls_all_n3627, pctls_inner_n3627, pctls_outer_n3627 = \
            self._import_hist_550pc(this_co10,this_co21,this_r21,this_er21,this_ra,this_dec,this_scale,this_pa,this_incl)

        this_basebeam = str(self.basebeam_n4254).replace(".","p").zfill(4)
        this_wisebeam = str(self.beam_wise_n4254).replace(".","p").zfill(4)
        this_r21      = self.outfits_r21_n4254.replace(this_basebeam,this_wisebeam)
        this_er21     = self.outfits_er21_n4254.replace(this_basebeam,this_wisebeam)
        this_co10     = self.outmom_co10_n4254.replace(this_basebeam,this_wisebeam).replace("momX","mom0")
        this_co21     = self.outmom_co21_n4254.replace(this_basebeam,this_wisebeam).replace("momX","mom0")
        this_ra       = float(self.ra_n4254)
        this_dec      = float(self.dec_n4254)
        this_scale    = self.scale_n4254
        this_pa       = self.pa_n4254
        this_incl     = self.incl_n4254
        hist_all_n4254, hist_inner_n4254, hist_outer_n4254, pctls_all_n4254, pctls_inner_n4254, pctls_outer_n4254 = \
            self._import_hist_550pc(this_co10,this_co21,this_r21,this_er21,this_ra,this_dec,this_scale,this_pa,this_incl)

        this_basebeam = str(self.basebeam_n4321).replace(".","p").zfill(4)
        this_wisebeam = str(self.beam_wise_n4321).replace(".","p").zfill(4)
        this_r21      = self.outfits_r21_n4321.replace(this_basebeam,this_wisebeam)
        this_er21     = self.outfits_er21_n4321.replace(this_basebeam,this_wisebeam)
        this_co10     = self.outmom_co10_n4321.replace(this_basebeam,this_wisebeam).replace("momX","mom0")
        this_co21     = self.outmom_co21_n4321.replace(this_basebeam,this_wisebeam).replace("momX","mom0")
        this_ra       = float(self.ra_n4321)
        this_dec      = float(self.dec_n4321)
        this_scale    = self.scale_n4321
        this_pa       = self.pa_n4321
        this_incl     = self.incl_n4321
        hist_all_n4321, hist_inner_n4321, hist_outer_n4321, pctls_all_n4321, pctls_inner_n4321, pctls_outer_n4321 = \
            self._import_hist_550pc(this_co10,this_co21,this_r21,this_er21,this_ra,this_dec,this_scale,this_pa,this_incl)


        ylim_n0628 = np.max(np.max(hist_all_n0628[2][:,1]/np.sum(hist_all_n0628[2][:,1])))
        ylim_n3627 = np.max(np.max(hist_all_n3627[2][:,1]/np.sum(hist_all_n3627[2][:,1])))
        ylim_n4254 = np.max(np.max(hist_all_n4254[2][:,1]/np.sum(hist_all_n4254[2][:,1])))
        ylim_n4321 = np.max(np.max(hist_all_n4321[2][:,1]/np.sum(hist_all_n4321[2][:,1])))
        ax1_title  = "Area-weighted"
        ax2_title  = "CO(1-0)-weighted"
        ax3_title  = "CO(2-1)-weighted"
        barwidth   = (self.hist_550pc_hrange[1] - self.hist_550pc_hrange[0]) / self.hist_550pc_bins
        xlabel     = "$R_{21}$"
        ylabel     = "Count"

        ########
        # plot #
        ########

        # set plt, ax
        plt.figure(figsize=(15,9))
        plt.subplots_adjust(bottom=0.09, left=0.07, right=0.99, top=0.95)
        gs   = gridspec.GridSpec(nrows=12, ncols=9)
        ax1  = plt.subplot(gs[0:3,0:3])
        ax2  = plt.subplot(gs[0:3,3:6])
        ax3  = plt.subplot(gs[0:3,6:9])
        ax4  = plt.subplot(gs[3:6,0:3])
        ax5  = plt.subplot(gs[3:6,3:6])
        ax6  = plt.subplot(gs[3:6,6:9])
        ax7  = plt.subplot(gs[6:9,0:3])
        ax8  = plt.subplot(gs[6:9,3:6])
        ax9  = plt.subplot(gs[6:9,6:9])
        ax10 = plt.subplot(gs[9:12,0:3])
        ax11 = plt.subplot(gs[9:12,3:6])
        ax12 = plt.subplot(gs[9:12,6:9])

        # set ax param
        factor = 1.65
        myax_set(ax1,  "x", self.hist_550pc_hrange, [0.0001,ylim_n0628*factor], ax1_title, None, ylabel)
        myax_set(ax2,  "x", self.hist_550pc_hrange, [0.0001,ylim_n0628*factor], ax2_title, None, None)
        myax_set(ax3,  "x", self.hist_550pc_hrange, [0.0001,ylim_n0628*factor], ax3_title, None, None)
        myax_set(ax4,  "x", self.hist_550pc_hrange, [0.0001,ylim_n3627*factor], None, None, None)
        myax_set(ax5,  "x", self.hist_550pc_hrange, [0.0001,ylim_n3627*factor], None, None, None)
        myax_set(ax6,  "x", self.hist_550pc_hrange, [0.0001,ylim_n3627*factor], None, None, None)
        myax_set(ax7,  "x", self.hist_550pc_hrange, [0.0001,ylim_n4254*factor], None, None, None)
        myax_set(ax8,  "x", self.hist_550pc_hrange, [0.0001,ylim_n4254*factor], None, None, None)
        myax_set(ax9,  "x", self.hist_550pc_hrange, [0.0001,ylim_n4254*factor], None, None, None)
        myax_set(ax10, "x", self.hist_550pc_hrange, [0.0001,ylim_n4321*factor], None, xlabel, None)
        myax_set(ax11, "x", self.hist_550pc_hrange, [0.0001,ylim_n4321*factor], None, xlabel, None)
        myax_set(ax12, "x", self.hist_550pc_hrange, [0.0001,ylim_n4321*factor], None, xlabel, None)

        # unset xlabels
        ax1.tick_params(labelbottom=False,labelleft=True,labelright=False,labeltop=False)
        ax2.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
        ax3.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
        ax4.tick_params(labelbottom=False,labelleft=True,labelright=False,labeltop=False)
        ax5.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
        ax6.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
        ax7.tick_params(labelbottom=False,labelleft=True,labelright=False,labeltop=False)
        ax8.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
        ax9.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
        ax10.tick_params(labelbottom=True,labelleft=True,labelright=False,labeltop=False)
        ax11.tick_params(labelbottom=True,labelleft=False,labelright=False,labeltop=False)
        ax12.tick_params(labelbottom=True,labelleft=False,labelright=False,labeltop=False)

        # plot: all
        lw_hist_all = 3
        y_ax1 = hist_all_n0628[0][:,1] / np.sum(hist_all_n0628[0][:,1])
        y_ax2 = hist_all_n0628[1][:,1] / np.sum(hist_all_n0628[1][:,1])
        y_ax3 = hist_all_n0628[2][:,1] / np.sum(hist_all_n0628[2][:,1])
        y_ax4 = hist_all_n3627[0][:,1] / np.sum(hist_all_n3627[0][:,1])
        y_ax5 = hist_all_n3627[1][:,1] / np.sum(hist_all_n3627[1][:,1])
        y_ax6 = hist_all_n3627[2][:,1] / np.sum(hist_all_n3627[2][:,1])
        y_ax7 = hist_all_n4254[0][:,1] / np.sum(hist_all_n4254[0][:,1])
        y_ax8 = hist_all_n4254[1][:,1] / np.sum(hist_all_n4254[1][:,1])
        y_ax9 = hist_all_n4254[2][:,1] / np.sum(hist_all_n4254[2][:,1])
        y_ax10 = hist_all_n4321[0][:,1] / np.sum(hist_all_n4321[0][:,1])
        y_ax11 = hist_all_n4321[1][:,1] / np.sum(hist_all_n4321[1][:,1])
        y_ax12 = hist_all_n4321[2][:,1] / np.sum(hist_all_n4321[2][:,1])

        ax1.step(hist_all_n0628[0][:,0], y_ax1, color="black", lw=lw_hist_all)
        ax2.step(hist_all_n0628[1][:,0], y_ax2, color="black", lw=lw_hist_all)
        ax3.step(hist_all_n0628[2][:,0], y_ax3, color="black", lw=lw_hist_all)
        ax4.step(hist_all_n3627[0][:,0], y_ax4, color="black", lw=lw_hist_all)
        ax5.step(hist_all_n3627[1][:,0], y_ax5, color="black", lw=lw_hist_all)
        ax6.step(hist_all_n3627[2][:,0], y_ax6, color="black", lw=lw_hist_all)
        ax7.step(hist_all_n4254[0][:,0], y_ax7, color="black", lw=lw_hist_all)
        ax8.step(hist_all_n4254[1][:,0], y_ax8, color="black", lw=lw_hist_all)
        ax9.step(hist_all_n4254[2][:,0], y_ax9, color="black", lw=lw_hist_all)
        ax10.step(hist_all_n4321[0][:,0], y_ax10, color="black", lw=lw_hist_all)
        ax11.step(hist_all_n4321[1][:,0], y_ax11, color="black", lw=lw_hist_all)
        ax12.step(hist_all_n4321[2][:,0], y_ax12, color="black", lw=lw_hist_all)

        # plot: outer
        lw_hist_outer = 2
        y_ax1 = hist_outer_n0628[0][:,1] / np.sum(hist_all_n0628[0][:,1])
        y_ax2 = hist_outer_n0628[1][:,1] / np.sum(hist_all_n0628[1][:,1])
        y_ax3 = hist_outer_n0628[2][:,1] / np.sum(hist_all_n0628[2][:,1])
        y_ax4 = hist_outer_n3627[0][:,1] / np.sum(hist_all_n3627[0][:,1])
        y_ax5 = hist_outer_n3627[1][:,1] / np.sum(hist_all_n3627[1][:,1])
        y_ax6 = hist_outer_n3627[2][:,1] / np.sum(hist_all_n3627[2][:,1])
        y_ax7 = hist_outer_n4254[0][:,1] / np.sum(hist_all_n4254[0][:,1])
        y_ax8 = hist_outer_n4254[1][:,1] / np.sum(hist_all_n4254[1][:,1])
        y_ax9 = hist_outer_n4254[2][:,1] / np.sum(hist_all_n4254[2][:,1])
        y_ax10 = hist_outer_n4321[0][:,1] / np.sum(hist_all_n4321[0][:,1])
        y_ax11 = hist_outer_n4321[1][:,1] / np.sum(hist_all_n4321[1][:,1])
        y_ax12 = hist_outer_n4321[2][:,1] / np.sum(hist_all_n4321[2][:,1])

        ax1.bar(hist_outer_n0628[0][:,0]-barwidth/2.0, y_ax1, color=self.c_n0628, lw=0, width=barwidth, align="center", alpha=0.5)
        ax2.bar(hist_outer_n0628[1][:,0]-barwidth/2.0, y_ax2, color=self.c_n0628, lw=0, width=barwidth, align="center", alpha=0.5)
        ax3.bar(hist_outer_n0628[2][:,0]-barwidth/2.0, y_ax3, color=self.c_n0628, lw=0, width=barwidth, align="center", alpha=0.5)
        ax4.bar(hist_outer_n3627[0][:,0]-barwidth/2.0, y_ax4, color=self.c_n3627, lw=0, width=barwidth, align="center", alpha=0.5)
        ax5.bar(hist_outer_n3627[1][:,0]-barwidth/2.0, y_ax5, color=self.c_n3627, lw=0, width=barwidth, align="center", alpha=0.5)
        ax6.bar(hist_outer_n3627[2][:,0]-barwidth/2.0, y_ax6, color=self.c_n3627, lw=0, width=barwidth, align="center", alpha=0.5)
        ax7.bar(hist_outer_n4254[0][:,0]-barwidth/2.0, y_ax7, color=self.c_n4254, lw=0, width=barwidth, align="center", alpha=0.5)
        ax8.bar(hist_outer_n4254[1][:,0]-barwidth/2.0, y_ax8, color=self.c_n4254, lw=0, width=barwidth, align="center", alpha=0.5)
        ax9.bar(hist_outer_n4254[2][:,0]-barwidth/2.0, y_ax9, color=self.c_n4254, lw=0, width=barwidth, align="center", alpha=0.5)
        ax10.bar(hist_outer_n4321[0][:,0]-barwidth/2.0, y_ax10, color=self.c_n4321, lw=0, width=barwidth, align="center", alpha=0.5)
        ax11.bar(hist_outer_n4321[1][:,0]-barwidth/2.0, y_ax11, color=self.c_n4321, lw=0, width=barwidth, align="center", alpha=0.5)
        ax12.bar(hist_outer_n4321[2][:,0]-barwidth/2.0, y_ax12, color=self.c_n4321, lw=0, width=barwidth, align="center", alpha=0.5)

        # inner
        lw_hist_outer = 2
        y_ax1 = hist_inner_n0628[0][:,1] / np.sum(hist_all_n0628[0][:,1])
        y_ax2 = hist_inner_n0628[1][:,1] / np.sum(hist_all_n0628[1][:,1])
        y_ax3 = hist_inner_n0628[2][:,1] / np.sum(hist_all_n0628[2][:,1])
        y_ax4 = hist_inner_n3627[0][:,1] / np.sum(hist_all_n3627[0][:,1])
        y_ax5 = hist_inner_n3627[1][:,1] / np.sum(hist_all_n3627[1][:,1])
        y_ax6 = hist_inner_n3627[2][:,1] / np.sum(hist_all_n3627[2][:,1])
        y_ax7 = hist_inner_n4254[0][:,1] / np.sum(hist_all_n4254[0][:,1])
        y_ax8 = hist_inner_n4254[1][:,1] / np.sum(hist_all_n4254[1][:,1])
        y_ax9 = hist_inner_n4254[2][:,1] / np.sum(hist_all_n4254[2][:,1])
        y_ax10 = hist_inner_n4321[0][:,1] / np.sum(hist_all_n4321[0][:,1])
        y_ax11 = hist_inner_n4321[1][:,1] / np.sum(hist_all_n4321[1][:,1])
        y_ax12 = hist_inner_n4321[2][:,1] / np.sum(hist_all_n4321[2][:,1])

        ax1.bar(hist_inner_n0628[0][:,0]-barwidth/2.0, y_ax1, color="firebrick", lw=0, width=barwidth, align="center", alpha=0.8)
        ax2.bar(hist_inner_n0628[1][:,0]-barwidth/2.0, y_ax2, color="firebrick", lw=0, width=barwidth, align="center", alpha=0.8)
        ax3.bar(hist_inner_n0628[2][:,0]-barwidth/2.0, y_ax3, color="firebrick", lw=0, width=barwidth, align="center", alpha=0.8)
        ax4.bar(hist_inner_n3627[0][:,0]-barwidth/2.0, y_ax4, color="firebrick", lw=0, width=barwidth, align="center", alpha=0.8)
        ax5.bar(hist_inner_n3627[1][:,0]-barwidth/2.0, y_ax5, color="firebrick", lw=0, width=barwidth, align="center", alpha=0.8)
        ax6.bar(hist_inner_n3627[2][:,0]-barwidth/2.0, y_ax6, color="firebrick", lw=0, width=barwidth, align="center", alpha=0.8)
        ax7.bar(hist_inner_n4254[0][:,0]-barwidth/2.0, y_ax7, color="firebrick", lw=0, width=barwidth, align="center", alpha=0.8)
        ax8.bar(hist_inner_n4254[1][:,0]-barwidth/2.0, y_ax8, color="firebrick", lw=0, width=barwidth, align="center", alpha=0.8)
        ax9.bar(hist_inner_n4254[2][:,0]-barwidth/2.0, y_ax9, color="firebrick", lw=0, width=barwidth, align="center", alpha=0.8)
        ax10.bar(hist_inner_n4321[0][:,0]-barwidth/2.0, y_ax10, color="firebrick", lw=0, width=barwidth, align="center", alpha=0.8)
        ax11.bar(hist_inner_n4321[1][:,0]-barwidth/2.0, y_ax11, color="firebrick", lw=0, width=barwidth, align="center", alpha=0.8)
        ax12.bar(hist_inner_n4321[2][:,0]-barwidth/2.0, y_ax12, color="firebrick", lw=0, width=barwidth, align="center", alpha=0.8)

        # plot: pctls
        factor = 1.5
        self._ax_pctls_bar(ax1, pctls_all_n0628[0], ylim_n0628, "all", ylim_factor=factor)
        self._ax_pctls_bar(ax2, pctls_all_n0628[1], ylim_n0628, None, ylim_factor=factor)
        self._ax_pctls_bar(ax3, pctls_all_n0628[2], ylim_n0628, None, ylim_factor=factor)
        self._ax_pctls_bar(ax4, pctls_all_n3627[0], ylim_n3627, "all", ylim_factor=factor)
        self._ax_pctls_bar(ax5, pctls_all_n3627[1], ylim_n3627, None, ylim_factor=factor)
        self._ax_pctls_bar(ax6, pctls_all_n3627[2], ylim_n3627, None, ylim_factor=factor)
        self._ax_pctls_bar(ax7, pctls_all_n4254[0], ylim_n4254, "all", ylim_factor=factor)
        self._ax_pctls_bar(ax8, pctls_all_n4254[1], ylim_n4254, None, ylim_factor=factor)
        self._ax_pctls_bar(ax9, pctls_all_n4254[2], ylim_n4254, None, ylim_factor=factor)
        self._ax_pctls_bar(ax10, pctls_all_n4321[0], ylim_n4321, "all", ylim_factor=factor)
        self._ax_pctls_bar(ax11, pctls_all_n4321[1], ylim_n4321, None, ylim_factor=factor)
        self._ax_pctls_bar(ax12, pctls_all_n4321[2], ylim_n4321, None, ylim_factor=factor)

        factor = 1.5-0.15
        self._ax_pctls_bar(ax1, pctls_outer_n0628[0], ylim_n0628, "outer", ylim_factor=factor, color=self.c_n0628)
        self._ax_pctls_bar(ax2, pctls_outer_n0628[1], ylim_n0628, None, ylim_factor=factor, color=self.c_n0628)
        self._ax_pctls_bar(ax3, pctls_outer_n0628[2], ylim_n0628, None, ylim_factor=factor, color=self.c_n0628)
        self._ax_pctls_bar(ax4, pctls_outer_n3627[0], ylim_n3627, "outer", ylim_factor=factor, color=self.c_n3627)
        self._ax_pctls_bar(ax5, pctls_outer_n3627[1], ylim_n3627, None, ylim_factor=factor, color=self.c_n3627)
        self._ax_pctls_bar(ax6, pctls_outer_n3627[2], ylim_n3627, None, ylim_factor=factor, color=self.c_n3627)
        self._ax_pctls_bar(ax7, pctls_outer_n4254[0], ylim_n4254, "outer", ylim_factor=factor, color=self.c_n4254)
        self._ax_pctls_bar(ax8, pctls_outer_n4254[1], ylim_n4254, None, ylim_factor=factor, color=self.c_n4254)
        self._ax_pctls_bar(ax9, pctls_outer_n4254[2], ylim_n4254, None, ylim_factor=factor, color=self.c_n4254)
        self._ax_pctls_bar(ax10, pctls_outer_n4321[0], ylim_n4321, "outer", ylim_factor=factor, color=self.c_n4321)
        self._ax_pctls_bar(ax11, pctls_outer_n4321[1], ylim_n4321, None, ylim_factor=factor, color=self.c_n4321)
        self._ax_pctls_bar(ax12, pctls_outer_n4321[2], ylim_n4321, None, ylim_factor=factor, color=self.c_n4321)

        factor = 1.5-0.15*2
        self._ax_pctls_bar(ax1, pctls_inner_n0628[0], ylim_n0628, "inner", ylim_factor=factor, color="firebrick")
        self._ax_pctls_bar(ax2, pctls_inner_n0628[1], ylim_n0628, None, ylim_factor=factor, color="firebrick")
        self._ax_pctls_bar(ax3, pctls_inner_n0628[2], ylim_n0628, None, ylim_factor=factor, color="firebrick")
        self._ax_pctls_bar(ax4, pctls_inner_n3627[0], ylim_n3627, "inner", ylim_factor=factor, color="firebrick", pos="left")
        self._ax_pctls_bar(ax5, pctls_inner_n3627[1], ylim_n3627, None, ylim_factor=factor, color="firebrick")
        self._ax_pctls_bar(ax6, pctls_inner_n3627[2], ylim_n3627, None, ylim_factor=factor, color="firebrick")
        self._ax_pctls_bar(ax7, pctls_inner_n4254[0], ylim_n4254, "inner", ylim_factor=factor, color="firebrick")
        self._ax_pctls_bar(ax8, pctls_inner_n4254[1], ylim_n4254, None, ylim_factor=factor, color="firebrick")
        self._ax_pctls_bar(ax9, pctls_inner_n4254[2], ylim_n4254, None, ylim_factor=factor, color="firebrick")
        self._ax_pctls_bar(ax10, pctls_inner_n4321[0], ylim_n4321, "inner", ylim_factor=factor, color="firebrick")
        self._ax_pctls_bar(ax11, pctls_inner_n4321[1], ylim_n4321, None, ylim_factor=factor, color="firebrick")
        self._ax_pctls_bar(ax12, pctls_inner_n4321[2], ylim_n4321, None, ylim_factor=factor, color="firebrick")

        # text
        xpos = 0.03
        ypos = 0.80
        yoffset = 0.15
        xoffset_n4321 = 0.58
        yoffset_n4321 = -0.33
        scale_n0628_pc = str(int(np.round(self.beam_wise_n0628*self.scale_n0628, -1))) + " pc"
        scale_n3627_pc = str(int(np.round(self.beam_wise_n3627*self.scale_n3627, -1))) + " pc"
        scale_n4254_pc = str(int(np.round(self.beam_wise_n4254*self.scale_n4254, -1))) + " pc"
        scale_n4321_pc = str(int(np.round(self.beam_wise_n4321*self.scale_n4321, -1))) + " pc"
        t=ax1.text(xpos, ypos, "NGC 0628", color=self.c_n0628, horizontalalignment="left", transform=ax1.transAxes, size=self.legend_fontsize-2, fontweight="bold")
        t.set_bbox(dict(facecolor="white", alpha=self.text_back_alpha, lw=0))
        t=ax4.text(xpos, ypos, "NGC 3627", color=self.c_n3627, horizontalalignment="left", transform=ax4.transAxes, size=self.legend_fontsize-2, fontweight="bold")
        t.set_bbox(dict(facecolor="white", alpha=self.text_back_alpha, lw=0))
        t=ax7.text(xpos, ypos, "NGC 4254", color=self.c_n4254, horizontalalignment="left", transform=ax7.transAxes, size=self.legend_fontsize-2, fontweight="bold")
        t.set_bbox(dict(facecolor="white", alpha=self.text_back_alpha, lw=0))
        t=ax10.text(xpos+xoffset_n4321, ypos+yoffset_n4321, "NGC 4321", color=self.c_n4321, horizontalalignment="left", transform=ax10.transAxes, size=self.legend_fontsize-2, fontweight="bold")
        t.set_bbox(dict(facecolor="white", alpha=self.text_back_alpha, lw=0))

        t=ax1.text(xpos, ypos-yoffset, str(self.beam_wise_n0628)+"$^{\prime}$$^{\prime}$ beam", color=self.c_n0628, horizontalalignment="left", transform=ax1.transAxes, size=self.legend_fontsize-2)
        t.set_bbox(dict(facecolor="white", alpha=self.text_back_alpha, lw=0))
        t=ax1.text(xpos, ypos-yoffset*2, scale_n0628_pc, color=self.c_n0628, horizontalalignment="left", transform=ax1.transAxes, size=self.legend_fontsize-2)
        t.set_bbox(dict(facecolor="white", alpha=self.text_back_alpha, lw=0))

        t=ax4.text(xpos, ypos-yoffset, str(self.beam_wise_n3627)+"$^{\prime}$$^{\prime}$ beam", color=self.c_n3627, horizontalalignment="left", transform=ax4.transAxes, size=self.legend_fontsize-2)
        t.set_bbox(dict(facecolor="white", alpha=self.text_back_alpha, lw=0))
        t=ax4.text(xpos, ypos-yoffset*2, scale_n3627_pc, color=self.c_n3627, horizontalalignment="left", transform=ax4.transAxes, size=self.legend_fontsize-2)
        t.set_bbox(dict(facecolor="white", alpha=self.text_back_alpha, lw=0))

        t=ax7.text(xpos, ypos-yoffset, str(self.beam_wise_n4254)+"$^{\prime}$$^{\prime}$ beam", color=self.c_n4254, horizontalalignment="left", transform=ax7.transAxes, size=self.legend_fontsize-2)
        t.set_bbox(dict(facecolor="white", alpha=self.text_back_alpha, lw=0))
        t=ax7.text(xpos, ypos-yoffset*2, scale_n4254_pc, color=self.c_n4254, horizontalalignment="left", transform=ax7.transAxes, size=self.legend_fontsize-2)
        t.set_bbox(dict(facecolor="white", alpha=self.text_back_alpha, lw=0))

        t=ax10.text(xpos+xoffset_n4321, ypos-yoffset+yoffset_n4321, str(self.beam_wise_n4321)+"$^{\prime}$$^{\prime}$ beam", color=self.c_n4321, horizontalalignment="left", transform=ax10.transAxes, size=self.legend_fontsize-2)
        t.set_bbox(dict(facecolor="white", alpha=self.text_back_alpha, lw=0))
        t=ax10.text(xpos+xoffset_n4321, ypos-yoffset*2+yoffset_n4321, scale_n4321_pc, color=self.c_n4321, horizontalalignment="left", transform=ax10.transAxes, size=self.legend_fontsize-2)
        t.set_bbox(dict(facecolor="white", alpha=self.text_back_alpha, lw=0))

        # savefig
        plt.savefig(self.outpng_hist_550pc, dpi=fig_dpi)

    #################
    # _ax_pctls_bar #
    #################

    def _ax_pctls_bar(
        self,
        ax,
        pctls,
        ylim,
        text=None,
        ylim_factor=1.35,
        color="black",
        pos="right",
        ):
        ax.plot([pctls[0], pctls[2]], [ylim*ylim_factor, ylim*ylim_factor], "-", color=color, lw=3)
        ax.plot(pctls[1], ylim*ylim_factor, "o", color=color, markersize=10, markeredgewidth=0)
        if text!=None:
            if pos=="right":
                ax.text(
                    pctls[2]+0.02,
                    ylim*ylim_factor,
                    text,
                    color=color,
                    horizontalalignment="left",
                    verticalalignment="center",
                    size=self.legend_fontsize-2)
            else:
                ax.text(
                    pctls[0]-0.02,
                    ylim*ylim_factor,
                    text,
                    color=color,
                    horizontalalignment="right",
                    verticalalignment="center",
                    size=self.legend_fontsize-2)

    ######################
    # _import_hist_550pc #
    ######################

    def _import_hist_550pc(self,co10,co21,r21,er21,ra,dec,scale,pa,incl):
        """
        plot_hist_550pc
        """

        # import
        shape           = imhead(co10,mode="list")["shape"]
        box             = "0,0," + str(shape[0]-1) + "," + str(shape[1]-1)
        ra_deg, dec_deg = imval(co10,box=box)["coords"][:,:,0] * 180/np.pi, imval(co10,box=box)["coords"][:,:,1] * 180/np.pi
        co10, co21      = imval(co10,box=box)["data"], imval(co21,box=box)["data"]
        r21, er21       = imval(r21,box=box)["data"], imval(er21,box=box)["data"]

        # trim
        cut = np.where( (~np.isnan(co10)) & (~np.isinf(co10)) & (co10!=0) & (~np.isnan(co21)) & (~np.isinf(co21)) & (co21!=0) \
            & (~np.isnan(r21)) & (~np.isinf(r21)) & (r21!=0) & (r21>=er21*self.snr_ratio) )
        ra_deg, dec_deg, co10, co21, r21, er21 = ra_deg[cut], dec_deg[cut], co10[cut], co21[cut], r21[cut], er21[cut]

        # hist
        dist_pc, _ = self._get_rel_dist_pc(ra_deg, dec_deg, ra, dec, scale, pa, incl)
        dist_kpc   = dist_pc / 1000.

        co10_inner = co10[dist_kpc <= self.hist_550pc_cnter_radius]
        co10_outer = co10[dist_kpc > self.hist_550pc_cnter_radius]
        co21_inner = co21[dist_kpc <= self.hist_550pc_cnter_radius]
        co21_outer = co21[dist_kpc > self.hist_550pc_cnter_radius]
        r21_inner  = r21[dist_kpc <= self.hist_550pc_cnter_radius]
        r21_outer  = r21[dist_kpc > self.hist_550pc_cnter_radius]

        hist_all   = self._get_weighted_hists(      co10,       co21,       r21)
        hist_inner = self._get_weighted_hists(co10_inner, co21_inner, r21_inner)
        hist_outer = self._get_weighted_hists(co10_outer, co21_outer, r21_outer)

        # pctls
        pctls_all   = self._three_three_pctls(      co10,      co21,      r21)
        pctls_inner = self._three_three_pctls(co10_inner,co21_inner,r21_inner)
        pctls_outer = self._three_three_pctls(co10_outer,co21_outer,r21_outer)

        return hist_all, hist_inner, hist_outer, pctls_all, pctls_inner, pctls_outer

    ######################
    # _three_three_pctls #
    ######################

    def _three_three_pctls(self,co10,co21,r21):
        """
        """

        weights = None
        p84 = self._weighted_percentile(r21, 84, weights=weights)
        p50 = self._weighted_percentile(r21, 50, weights=weights)
        p16 = self._weighted_percentile(r21, 16, weights=weights)
        pctl_wnone = [p16, p50, p84]

        weights = co10
        p84 = self._weighted_percentile(r21, 84, weights=weights)
        p50 = self._weighted_percentile(r21, 50, weights=weights)
        p16 = self._weighted_percentile(r21, 16, weights=weights)
        pctl_co10 = [p16, p50, p84]

        weights = co21
        p84 = self._weighted_percentile(r21, 84, weights=weights)
        p50 = self._weighted_percentile(r21, 50, weights=weights)
        p16 = self._weighted_percentile(r21, 16, weights=weights)
        pctl_co21 = [p16, p50, p84]

        return np.array([pctl_wnone, pctl_co10, pctl_co21])

    ########################
    # _weighted_percentile #
    ########################

    def _weighted_percentile(self,data,percentile,weights=None):
        """
        Args:
            data (list or numpy.array): data
            weights (list or numpy.array): weights
        """

        if weights==None:
            w_percentile = np.nanpercentile(data,percentile)
        else:
            data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
            s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
            midpoint = percentile/100. * sum(s_weights)
            if any(weights > midpoint):
                w_percentile = (data[weights == np.max(weights)])[0]
            else:
                cs_weights = np.cumsum(s_weights)
                idx = np.where(cs_weights <= midpoint)[0][-1]
                if cs_weights[idx] == midpoint:
                    w_percentile = np.mean(s_data[idx:idx+2])
                else:
                    w_percentile = s_data[idx+1]

        return w_percentile

    #######################
    # _get_weighted_hists #
    #######################

    def _get_weighted_hists(self,co10,co21,r21):
        """
        """

        hist = np.histogram(r21, bins=self.hist_550pc_bins, range=self.hist_550pc_hrange, weights=None)
        hist_wnone = np.c_[ np.delete(hist[1],-1), hist[0]]#/float(sum(hist[0])) ]

        hist = np.histogram(r21, bins=self.hist_550pc_bins, range=self.hist_550pc_hrange, weights=co10)
        hist_wco10 = np.c_[ np.delete(hist[1],-1), hist[0]]#/float(sum(hist[0])) ]

        hist = np.histogram(r21, bins=self.hist_550pc_bins, range=self.hist_550pc_hrange, weights=co21)
        hist_wco21 = np.c_[ np.delete(hist[1],-1), hist[0]]#/float(sum(hist[0])) ]

        return [hist_wnone, hist_wco10, hist_wco21]

    ####################
    # _get_rel_dist_pc #
    ####################

    def _get_rel_dist_pc(self,ra_deg,dec_deg,center_ra_deg,center_dec_deg,scale,pa,inc):
        """
        """

        tilt_cos = math.cos(math.radians(pa))
        tilt_sin = math.sin(math.radians(pa))

        ra_rel_deg  = (ra_deg - center_ra_deg)
        dec_rel_deg = (dec_deg - center_dec_deg)

        ra_rel_deproj_deg  = (ra_rel_deg*tilt_cos - dec_rel_deg*tilt_sin)
        deg_rel_deproj_deg = (ra_rel_deg*tilt_sin + dec_rel_deg*tilt_cos) / math.cos(math.radians(inc))

        distance_pc = np.sqrt(ra_rel_deproj_deg**2 + deg_rel_deproj_deg**2) * 3600 * scale
        theta_deg   = np.degrees(np.arctan2(ra_rel_deproj_deg, deg_rel_deproj_deg))

        return distance_pc, theta_deg

    #

    #################
    # plot_m0_vs_m8 #
    #################

    def plot_m0_vs_m8(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outcube_co10_n0628,taskname)

        ###########
        # prepare #
        ###########

        r21_n0628, t21_n0628, er21_n0628, et21_n0628 = \
            self._import_m0_vs_m8(self.outfits_r21_n0628, self.outfits_t21_n0628, self.outfits_er21_n0628, self.outfits_et21_n0628)
        r21_n3627, t21_n3627, er21_n3627, et21_n3627 = \
            self._import_m0_vs_m8(self.outfits_r21_n3627, self.outfits_t21_n3627, self.outfits_er21_n3627, self.outfits_et21_n3627)
        r21_n4254, t21_n4254, er21_n4254, et21_n4254 = \
            self._import_m0_vs_m8(self.outfits_r21_n4254, self.outfits_t21_n4254, self.outfits_er21_n4254, self.outfits_et21_n4254)
        r21_n4321, t21_n4321, er21_n4321, et21_n4321 = \
            self._import_m0_vs_m8(self.outfits_r21_n4321, self.outfits_t21_n4321, self.outfits_er21_n4321, self.outfits_et21_n4321)
        r21_all  = np.r_[ r21_n0628,  r21_n3627,  r21_n4254,  r21_n4321]
        t21_all  = np.r_[ t21_n0628,  t21_n3627,  t21_n4254,  t21_n4321]
        er21_all = np.r_[er21_n0628, er21_n3627, er21_n4254, er21_n4321]
        et21_all = np.r_[et21_n0628, et21_n3627, et21_n4254, et21_n4321]

        # get coreelation coeff
        cor_all   = " (r=" + str(np.round(np.corrcoef(t21_all,r21_all)[0,1], 2)).ljust(4, "0") + ")"
        cor_n0628 = " (r=" + str(np.round(np.corrcoef(t21_n0628,r21_n0628)[0,1], 2)).ljust(4, "0") + ")"
        cor_n3627 = " (r=" + str(np.round(np.corrcoef(t21_n3627,r21_n3627)[0,1], 2)).ljust(4, "0") + ")"
        cor_n4254 = " (r=" + str(np.round(np.corrcoef(t21_n4254,r21_n4254)[0,1], 2)).ljust(4, "0") + ")"
        cor_n4321 = " (r=" + str(np.round(np.corrcoef(t21_n4321,r21_n4321)[0,1], 2)).ljust(4, "0") + ")"

        xlim   = [-1.1,0.45]
        ylim   = [-1.1,0.45]
        title  = "Peak Temperature Ratio vs. Intensity Ratio"
        xlabel = "log Integrated intensity ratio"
        ylabel = "log Peak temperature ratio"

        # get contours
        contour_n0628, extent_n0628 = self._getcontour_m0_vs_m8(r21_n0628, t21_n0628, xlim, ylim)
        contour_n3627, extent_n3627 = self._getcontour_m0_vs_m8(r21_n3627, t21_n3627, xlim, ylim)
        contour_n4254, extent_n4254 = self._getcontour_m0_vs_m8(r21_n4254, t21_n4254, xlim, ylim)
        contour_n4321, extent_n4321 = self._getcontour_m0_vs_m8(r21_n4321, t21_n4321, xlim, ylim)

        # get hist
        ratio_p16 = np.percentile((t21_all-0.3) / (r21_all-0.3), 16)
        ratio_p50 = np.percentile((t21_all-0.3) / (r21_all-0.3), 50)
        ratio_p84 = np.percentile((t21_all-0.3) / (r21_all-0.3), 84)
        ylim_p16 = [ylim[0]+np.log10(ratio_p16), ylim[1]+np.log10(ratio_p16)]
        ylim_p50 = [ylim[0]+np.log10(ratio_p50), ylim[1]+np.log10(ratio_p50)]
        ylim_p84 = [ylim[0]+np.log10(ratio_p84), ylim[1]+np.log10(ratio_p84)]

        ########
        # plot #
        ########

        plt.figure(figsize=(13,10))
        gs = gridspec.GridSpec(nrows=10, ncols=10)
        ax = plt.subplot(gs[0:10,0:10])

        ad = [0.215,0.83,0.10,0.90]
        myax_set(ax, "both", xlim, ylim, title, xlabel, ylabel, adjust=ad)

        # plot
        ax.errorbar(r21_all, t21_all, xerr=er21_all, yerr=et21_all, lw=1, capsize=0, color="grey", linestyle="None")

        alpha_contourf = 0.6
        alpha_contour  = 1.0
        ax.contourf(contour_n0628, levels=[10,100], extent=extent_n0628, colors=self.c_n0628, zorder=4, linewidths=2.5, alpha=alpha_contourf)
        ax.contour(contour_n0628, levels=[10,100], extent=extent_n0628, colors=self.c_n0628, zorder=4, linewidths=2.5, alpha=alpha_contour)
        ax.contourf(contour_n3627, levels=[10,100], extent=extent_n3627, colors=self.c_n3627, zorder=5, linewidths=2.5, alpha=alpha_contourf)
        ax.contour(contour_n3627, levels=[10,100], extent=extent_n3627, colors=self.c_n3627, zorder=5, linewidths=2.5, alpha=alpha_contour)
        ax.contourf(contour_n4254, levels=[10,100], extent=extent_n4254, colors=self.c_n4254, zorder=6, linewidths=2.5, alpha=alpha_contourf)
        ax.contour(contour_n4254, levels=[10,100], extent=extent_n4254, colors=self.c_n4254, zorder=6, linewidths=2.5, alpha=alpha_contour)
        ax.contourf(contour_n4321, levels=[10,100], extent=extent_n4321, colors=self.c_n4321, zorder=7, linewidths=2.5, alpha=alpha_contourf)
        ax.contour(contour_n4321, levels=[10,100], extent=extent_n4321, colors=self.c_n4321, zorder=7, linewidths=2.5, alpha=alpha_contour)

        # ann
        ax.plot(xlim, ylim, "k--", lw=3, zorder=100000)
        ax.plot(xlim, ylim_p16, "-", color="black", lw=1, zorder=100000)
        ax.plot(xlim, ylim_p50, "-", color="black", lw=1, zorder=100000)
        ax.plot(xlim, ylim_p84, "-", color="black", lw=1, zorder=100000)

        # text
        t=ax.text(0.95, 0.25, "All"+cor_all, color="black", horizontalalignment="right", transform=ax.transAxes, size=self.legend_fontsize, fontweight="bold")
        t.set_bbox(dict(facecolor="white", alpha=self.self.text_back_alpha, lw=0))
        t=ax.text(0.95, 0.20, "NGC 0628"+cor_n0628, color=self.c_n0628, horizontalalignment="right", transform=ax.transAxes, size=self.legend_fontsize, fontweight="bold")
        t.set_bbox(dict(facecolor="white", alpha=self.self.text_back_alpha, lw=0))
        t=ax.text(0.95, 0.15, "NGC 3627"+cor_n3627, color=self.c_n3627, horizontalalignment="right", transform=ax.transAxes, size=self.legend_fontsize, fontweight="bold")
        t.set_bbox(dict(facecolor="white", alpha=self.self.text_back_alpha, lw=0))
        t=ax.text(0.95, 0.10, "NGC 4254"+cor_n4254, color=self.c_n4254, horizontalalignment="right", transform=ax.transAxes, size=self.legend_fontsize, fontweight="bold")
        t.set_bbox(dict(facecolor="white", alpha=self.self.text_back_alpha, lw=0))
        t=ax.text(0.95, 0.05, "NGC 4321"+cor_n4321, color=self.c_n4321, horizontalalignment="right", transform=ax.transAxes, size=self.legend_fontsize, fontweight="bold")
        t.set_bbox(dict(facecolor="white", alpha=self.self.text_back_alpha, lw=0))

        ax.text(0.02, 0.13, "84$^{th}$ pctl.", horizontalalignment="left", verticalalignment="bottom", rotation=45, transform=ax.transAxes, size=self.legend_fontsize)
        ax.text(0.02, 0.05, "50$^{th}$ pctl.", horizontalalignment="left", verticalalignment="bottom", rotation=45, transform=ax.transAxes, size=self.legend_fontsize)
        ax.text(0.08, 0.01, "16$^{th}$ pctl.", horizontalalignment="left", verticalalignment="bottom", rotation=45, transform=ax.transAxes, size=self.legend_fontsize)

        t=ax.text(0.02, 0.93, "84$^{th}$ percentile = " + str(np.round(ratio_p84,2)).ljust(4,"0"), horizontalalignment="left", transform=ax.transAxes, size=self.legend_fontsize)
        t.set_bbox(dict(facecolor="white", alpha=self.self.text_back_alpha, lw=0))
        t=ax.text(0.02, 0.88, "50$^{th}$ percentile = " + str(np.round(ratio_p50,2)).ljust(4,"0"), horizontalalignment="left", transform=ax.transAxes, size=self.legend_fontsize)
        t.set_bbox(dict(facecolor="white", alpha=self.self.text_back_alpha, lw=0))
        t=ax.text(0.02, 0.83, "16$^{th}$ percentile = " + str(np.round(ratio_p16,2)).ljust(4,"0"), horizontalalignment="left", transform=ax.transAxes, size=self.legend_fontsize)
        t.set_bbox(dict(facecolor="white", alpha=self.self.text_back_alpha, lw=0))

        plt.savefig(self.outpng_m0_vs_m8, dpi=self.fig_dpi)

    ########################
    # _getcontour_m0_vs_m8 #
    ########################

    def _getcontour_m0_vs_m8(
        self,
        mom0,
        mom8,
        xlim,
        ylim,
        ):
        """
        plot_m0_vs_m8
        """

        mom8_grid, mom0_grid = scipy.ndimage.zoom(mom8, 33), scipy.ndimage.zoom(mom0, 33)
        contour, xedges, yedges = np.histogram2d(mom8_grid, mom0_grid, bins=50, range=(xlim,ylim))
        this_contour = contour/contour.max() * 100
        this_extent  = [xedges[0],xedges[-1],yedges[0],yedges[-1]]

        return this_contour, this_extent

    ####################
    # _import_m0_vs_m8 #
    ####################

    def _import_m0_vs_m8(
        self,
        mom0,
        mom8,
        emom0,
        emom8,
        ):
        """
        plot_m0_vs_m8
        """

        this_r21,_  = imval_all(mom0)
        this_t21,_  = imval_all(mom8)
        this_er21,_ = imval_all(emom0)
        this_et21,_ = imval_all(emom8)
        this_r21    = this_r21["data"].flatten()
        this_t21    = this_t21["data"].flatten()
        this_er21   = this_er21["data"].flatten()
        this_et21   = this_et21["data"].flatten()

        cut = np.where( (~np.isnan(this_r21)) & (~np.isinf(this_r21)) & (this_r21!=0) \
            & (~np.isnan(this_t21)) & (~np.isinf(this_t21)) & (this_t21!=0) \
            & (this_r21>=this_er21*self.snr_ratio) & (this_t21>=this_et21*self.snr_ratio) )

        r21  = np.log10(this_r21[cut])
        t21  = np.log10(this_t21[cut])
        er21 = this_er21[cut] / (np.log(10) * this_r21[cut])
        et21 = this_et21[cut] / (np.log(10) * this_t21[cut])

        cut = np.where( (~np.isnan(r21)) & (~np.isinf(r21)) & (~np.isnan(t21)) & (~np.isinf(t21)) )

        return r21[cut], t21[cut], er21[cut], et21[cut]

    #

    #################
    # plot_showcase #
    #################

    def plot_showcase(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outcube_co10_n0628,taskname)

        ###########
        # ngc0628 #
        ###########
        this_co10      = self.outmom_co10_n0628.replace("momX","mom0")
        this_co21      = self.outmom_co21_n0628.replace("momX","mom0")
        this_eco10     = self.outmom_co10_n0628.replace("momX","emom0")
        this_eco21     = self.outmom_co21_n0628.replace("momX","emom0")
        this_r21       = self.outfits_r21_n0628
        this_out_co10  = self.outpng_co10_n0628
        this_out_co21  = self.outpng_co21_n0628
        this_out_r21   = self.outpng_r21_n0628
        this_imsize    = self.imsize_n0628
        this_ra        = self.ra_n0628
        this_dec       = self.dec_n0628
        this_scalebar  = 1000. / self.scale_n0628
        this_title     = "NGC 0628"
        this_co10      = self._clip_for_showcase(this_co10,this_eco10)
        this_co21      = self._clip_for_showcase(this_co21,this_eco21)
        myfig_fits2png(
            this_co10,
            this_out_co10,
            imcontour1=this_co10,
            imsize_as=this_imsize,
            ra_cnt=this_ra,
            dec_cnt=this_dec,
            levels_cont1=[0.05, 0.1, 0.2, 0.4, 0.8, 0.96],
            set_title=this_title,
            colorlog=True,
            set_cmap="PuBu",
            scalebar=this_scalebar,
            label_scalebar="1 kpc",
            comment="CO(1-0) integrated intensity",
            clim=None,#[10**-1,imstat(this_co10)["max"]],
            )
        myfig_fits2png(
            this_co21,
            this_out_co21,
            imcontour1=this_co21,
            imsize_as=this_imsize,
            ra_cnt=this_ra,
            dec_cnt=this_dec,
            levels_cont1=[0.05, 0.1, 0.2, 0.4, 0.8, 0.96],
            set_title=this_title,
            colorlog=True,
            set_cmap="Reds",
            scalebar=this_scalebar,
            label_scalebar="1 kpc",
            comment="CO(2-1) integrated intensity",
            clim=None,#[10**-1,imstat(this_co21)["max"]],
            )
        myfig_fits2png(
            this_r21,
            this_out_r21,
            imcontour1=this_co21,
            imsize_as=this_imsize,
            ra_cnt=this_ra,
            dec_cnt=this_dec,
            levels_cont1=[0.05, 0.1, 0.2, 0.4, 0.8, 0.96],
            set_title=this_title,
            colorlog=False,
            set_cmap="rainbow",
            scalebar=this_scalebar,
            label_scalebar="1 kpc",
            comment="CO(2-1)/CO(1-0) integrated intensity ratio",
            clim=[0.2,1.0],
            )

        ###########
        # ngc3627 #
        ###########
        this_co10      = self.outmom_co10_n3627.replace("momX","mom0")
        this_co21      = self.outmom_co21_n3627.replace("momX","mom0")
        this_eco10     = self.outmom_co10_n3627.replace("momX","emom0")
        this_eco21     = self.outmom_co21_n3627.replace("momX","emom0")
        this_r21       = self.outfits_r21_n3627
        this_out_co10  = self.outpng_co10_n3627
        this_out_co21  = self.outpng_co21_n3627
        this_out_r21   = self.outpng_r21_n3627
        this_imsize    = self.imsize_n3627
        this_ra        = self.ra_n3627
        this_dec       = self.dec_n3627
        this_scalebar  = 1000. / self.scale_n3627
        this_title     = "NGC 3627"
        this_co10      = self._clip_for_showcase(this_co10,this_eco10)
        this_co21      = self._clip_for_showcase(this_co21,this_eco21)
        myfig_fits2png(
            this_co10,
            this_out_co10,
            imcontour1=this_co10,
            imsize_as=this_imsize,
            ra_cnt=this_ra,
            dec_cnt=this_dec,
            levels_cont1=[0.02,0.04,0.08,0.16,0.32,0.64,0.96],
            set_title=this_title,
            colorlog=True,
            set_cmap="PuBu",
            scalebar=this_scalebar,
            label_scalebar="1 kpc",
            comment="CO(1-0) integrated intensity",
            clim=[10**-2,imstat(this_co10)["max"]],
            )
        myfig_fits2png(
            this_co21,
            this_out_co21,
            imcontour1=this_co21,
            imsize_as=this_imsize,
            ra_cnt=this_ra,
            dec_cnt=this_dec,
            levels_cont1=[0.02,0.04,0.08,0.16,0.32,0.64,0.96],
            set_title=this_title,
            colorlog=True,
            set_cmap="Reds",
            scalebar=this_scalebar,
            label_scalebar="1 kpc",
            comment="CO(2-1) integrated intensity",
            clim=[10**-2,imstat(this_co21)["max"]],
            )
        myfig_fits2png(
            this_r21,
            this_out_r21,
            imcontour1=this_co21,
            imsize_as=this_imsize,
            ra_cnt=this_ra,
            dec_cnt=this_dec,
            levels_cont1=[0.02,0.04,0.08,0.16,0.32,0.64,0.96],
            set_title=this_title,
            colorlog=False,
            set_cmap="rainbow",
            scalebar=this_scalebar,
            label_scalebar="1 kpc",
            comment="CO(2-1)/CO(1-0) integrated intensity ratio",
            clim=[0.2,1.0],
            )

        ###########
        # ngc4254 #
        ###########
        this_co10      = self.outmom_co10_n4254.replace("momX","mom0")
        this_co21      = self.outmom_co21_n4254.replace("momX","mom0")
        this_eco10     = self.outmom_co10_n4254.replace("momX","emom0")
        this_eco21     = self.outmom_co21_n4254.replace("momX","emom0")
        this_r21       = self.outfits_r21_n4254
        this_out_co10  = self.outpng_co10_n4254
        this_out_co21  = self.outpng_co21_n4254
        this_out_r21   = self.outpng_r21_n4254
        this_imsize    = self.imsize_n4254
        this_ra        = self.ra_n4254
        this_dec       = self.dec_n4254
        this_scalebar  = 1000. / self.scale_n4254
        this_title     = "NGC 4254"
        this_co10      = self._clip_for_showcase(this_co10,this_eco10)
        this_co21      = self._clip_for_showcase(this_co21,this_eco21)
        myfig_fits2png(
            this_co10,
            this_out_co10,
            imcontour1=this_co10,
            imsize_as=this_imsize,
            ra_cnt=this_ra,
            dec_cnt=this_dec,
            levels_cont1=[0.02,0.04,0.08,0.16,0.32,0.64,0.96],
            set_title=this_title,
            colorlog=True,
            set_cmap="PuBu",
            scalebar=this_scalebar,
            label_scalebar="1 kpc",
            comment="CO(1-0) integrated intensity",
            clim=[10**-2,imstat(this_co10)["max"]],
            )
        myfig_fits2png(
            this_co21,
            this_out_co21,
            imcontour1=this_co21,
            imsize_as=this_imsize,
            ra_cnt=this_ra,
            dec_cnt=this_dec,
            levels_cont1=[0.02,0.04,0.08,0.16,0.32,0.64,0.96],
            set_title=this_title,
            colorlog=True,
            set_cmap="Reds",
            scalebar=this_scalebar,
            label_scalebar="1 kpc",
            comment="CO(2-1) integrated intensity",
            clim=[10**-2,imstat(this_co21)["max"]],
            )
        myfig_fits2png(
            this_r21,
            this_out_r21,
            imcontour1=this_co21,
            imsize_as=this_imsize,
            ra_cnt=this_ra,
            dec_cnt=this_dec,
            levels_cont1=[0.02,0.04,0.08,0.16,0.32,0.64,0.96],
            set_title=this_title,
            colorlog=False,
            set_cmap="rainbow",
            scalebar=this_scalebar,
            label_scalebar="1 kpc",
            comment="CO(2-1)/CO(1-0) integrated intensity ratio",
            clim=[0.2,1.0],
            )

        ###########
        # ngc4321 #
        ###########
        this_co10      = self.outmom_co10_n4321.replace("momX","mom0")
        this_co21      = self.outmom_co21_n4321.replace("momX","mom0")
        this_eco10     = self.outmom_co10_n4321.replace("momX","emom0")
        this_eco21     = self.outmom_co21_n4321.replace("momX","emom0")
        this_r21       = self.outfits_r21_n4321
        this_out_co10  = self.outpng_co10_n4321
        this_out_co21  = self.outpng_co21_n4321
        this_out_r21   = self.outpng_r21_n4321
        this_imsize    = self.imsize_n4321
        this_ra        = self.ra_n4321
        this_dec       = self.dec_n4321
        this_scalebar  = 1000. / self.scale_n4321
        this_title     = "NGC 4321"
        this_co10      = self._clip_for_showcase(this_co10,this_eco10)
        this_co21      = self._clip_for_showcase(this_co21,this_eco21)
        myfig_fits2png(
            this_co10,
            this_out_co10,
            imcontour1=this_co10,
            imsize_as=this_imsize,
            ra_cnt=this_ra,
            dec_cnt=this_dec,
            levels_cont1=[0.02,0.04,0.08,0.16,0.32,0.64,0.96],
            set_title=this_title,
            colorlog=True,
            set_cmap="PuBu",
            scalebar=this_scalebar,
            label_scalebar="1 kpc",
            comment="CO(1-0) integrated intensity",
            clim=[10**-2,imstat(this_co10)["max"]],
            )
        myfig_fits2png(
            this_co21,
            this_out_co21,
            imcontour1=this_co21,
            imsize_as=this_imsize,
            ra_cnt=this_ra,
            dec_cnt=this_dec,
            levels_cont1=[0.02,0.04,0.08,0.16,0.32,0.64,0.96],
            set_title=this_title,
            colorlog=True,
            set_cmap="Reds",
            scalebar=this_scalebar,
            label_scalebar="1 kpc",
            comment="CO(2-1) integrated intensity",
            clim=[10**-2,imstat(this_co21)["max"]],
            )
        myfig_fits2png(
            this_r21,
            this_out_r21,
            imcontour1=this_co21,
            imsize_as=this_imsize,
            ra_cnt=this_ra,
            dec_cnt=this_dec,
            levels_cont1=[0.02,0.04,0.08,0.16,0.32,0.64,0.96],
            set_title=this_title,
            colorlog=False,
            set_cmap="rainbow",
            scalebar=this_scalebar,
            label_scalebar="1 kpc",
            comment="CO(2-1)/CO(1-0) integrated intensity ratio",
            clim=[0.2,1.0],
            )

    ######################
    # _clip_for_showcase #
    ######################

    def _clip_for_showcase(
        self,
        mom0,
        emom0,
        ):
        """
        plot_showcase
        """

        run_immath_two(mom0,emom0,mom0+"_clipped","iif( IM0>IM1*"+str(self.snr_showcase)+",IM0,0 )")

        return mom0+"_clipped"

    #

    #################
    # plot_recovery #
    #################

    def plot_recovery(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outcube_co10_n0628,taskname)

        ###########
        # prepare #
        ###########

        beams_new_n0628 = [s for s in self.beams_n0628[:-1] if not "11.5" in str(s)]
        list_flux_co10_n0628 = self._loop_measure_flux_norm(
            self.outmom_co10_n0628.replace(str(self.basebeam_n0628).replace(".","p").zfill(4),"????").replace(".image","_k.image").replace("momX","mom0"),
            beams_new_n0628)
        list_flux_co21_n0628 = self._loop_measure_flux_norm(
            self.outmom_co21_n0628.replace(str(self.basebeam_n0628).replace(".","p").zfill(4),"????").replace(".image","_k.image").replace("momX","mom0"),
            beams_new_n0628)

        beams_new_n3627 = [s for s in self.beams_n3627[:-2]]
        list_flux_co10_n3627 = self._loop_measure_flux_norm(
            self.outmom_co10_n3627.replace(str(self.basebeam_n3627).replace(".","p").zfill(4),"????").replace(".image","_k.image").replace("momX","mom0"),
            beams_new_n3627)
        list_flux_co21_n3627 = self._loop_measure_flux_norm(
            self.outmom_co21_n3627.replace(str(self.basebeam_n3627).replace(".","p").zfill(4),"????").replace(".image","_k.image").replace("momX","mom0"),
            beams_new_n3627)

        beams_new_n4254 = [s for s in self.beams_n4254[:-2] if not "8.7" in str(s)]
        list_flux_co10_n4254 = self._loop_measure_flux_norm(
            self.outmom_co10_n4254.replace(str(self.basebeam_n4254).replace(".","p").zfill(4),"????").replace(".image","_k.image").replace("momX","mom0"),
            beams_new_n4254)
        list_flux_co21_n4254 = self._loop_measure_flux_norm(
            self.outmom_co21_n4254.replace(str(self.basebeam_n4254).replace(".","p").zfill(4),"????").replace(".image","_k.image").replace("momX","mom0"),
            beams_new_n4254)

        beams_new_n4321 = [s for s in self.beams_n4321[:-1] if not "7.5" in str(s)]
        list_flux_co10_n4321 = self._loop_measure_flux_norm(
            self.outmom_co10_n4321.replace(str(self.basebeam_n4321).replace(".","p").zfill(4),"????").replace(".image","_k.image").replace("momX","mom0"),
            beams_new_n4321)
        list_flux_co21_n4321 = self._loop_measure_flux_norm(
            self.outmom_co21_n4321.replace(str(self.basebeam_n4321).replace(".","p").zfill(4),"????").replace(".image","_k.image").replace("momX","mom0"),
            beams_new_n4321)

        xlim   = None#[2,26]
        ylim   = [0.8,1.2]
        title  = "Total Flux vs. Beam Size"
        xlabel = "Beam size (kpc)"
        ylabel = "Total flux recovery"
        beams_new_n0628 = np.array(beams_new_n0628) * self.scale_n0628 / 1000.
        beams_new_n3627 = np.array(beams_new_n3627) * self.scale_n3627 / 1000.
        beams_new_n4254 = np.array(beams_new_n4254) * self.scale_n4254 / 1000.
        beams_new_n4321 = np.array(beams_new_n4321) * self.scale_n4321 / 1000.

        ########
        # plot #
        ########

        plt.figure(figsize=(13,10))
        gs = gridspec.GridSpec(nrows=10, ncols=10)
        ax = plt.subplot(gs[0:10,0:10])

        ad = [0.215,0.83,0.10,0.90]
        myax_set(ax, "both", xlim, ylim, title, xlabel, ylabel, adjust=ad)

        # plot co10 recovery
        ax.plot(beams_new_n0628, list_flux_co10_n0628, "o-", color=self.c_n0628, markeredgewidth=0, markersize=20, lw=3)
        ax.plot(beams_new_n3627, list_flux_co10_n3627, "o-", color=self.c_n3627, markeredgewidth=0, markersize=20, lw=3)
        ax.plot(beams_new_n4254, list_flux_co10_n4254, "o-", color=self.c_n4254, markeredgewidth=0, markersize=20, lw=3)
        ax.plot(beams_new_n4321, list_flux_co10_n4321, "o-", color=self.c_n4321, markeredgewidth=0, markersize=20, lw=3)
        # plot co21 recovery
        ax.plot(beams_new_n0628, list_flux_co21_n0628, "s--", color=self.c_n0628, markeredgewidth=0, markersize=20, lw=3)
        ax.plot(beams_new_n3627, list_flux_co21_n3627, "s--", color=self.c_n3627, markeredgewidth=0, markersize=20, lw=3)
        ax.plot(beams_new_n4254, list_flux_co21_n4254, "s--", color=self.c_n4254, markeredgewidth=0, markersize=20, lw=3)
        ax.plot(beams_new_n4321, list_flux_co21_n4321, "s--", color=self.c_n4321, markeredgewidth=0, markersize=20, lw=3)

        # text
        t=ax.text(0.95, 0.93, "NGC 0628", color=self.c_n0628, horizontalalignment="right", transform=ax.transAxes, size=self.legend_fontsize, fontweight="bold")
        t.set_bbox(dict(facecolor="white", alpha=self.self.text_back_alpha, lw=0))
        t=ax.text(0.95, 0.88, "NGC 3627", color=self.c_n3627, horizontalalignment="right", transform=ax.transAxes, size=self.legend_fontsize, fontweight="bold")
        t.set_bbox(dict(facecolor="white", alpha=self.self.text_back_alpha, lw=0))
        t=ax.text(0.95, 0.83, "NGC 4254", color=self.c_n4254, horizontalalignment="right", transform=ax.transAxes, size=self.legend_fontsize, fontweight="bold")
        t.set_bbox(dict(facecolor="white", alpha=self.self.text_back_alpha, lw=0))
        t=ax.text(0.95, 0.78, "NGC 4321", color=self.c_n4321, horizontalalignment="right", transform=ax.transAxes, size=self.legend_fontsize, fontweight="bold")
        t.set_bbox(dict(facecolor="white", alpha=self.self.text_back_alpha, lw=0))

        xtext   = 0.68
        xmarker = xtext-0.23
        ytext   = 0.93
        ymarker = ytext+0.013
        ax.text(xtext,   ytext, "CO(1-0)", color="black", horizontalalignment="right", transform=ax.transAxes, size=self.legend_fontsize, fontweight="bold")
        ax.plot(xmarker, ymarker, marker="o", color="black", markeredgewidth=0, markersize=20, transform=ax.transAxes)
        ax.plot([xmarker-0.05,xmarker+0.05], [ymarker, ymarker], "-", color="black", lw=3, transform=ax.transAxes)
        ax.text(xtext,   ytext-0.05, "CO(2-1)", color="black", horizontalalignment="right", transform=ax.transAxes, size=self.legend_fontsize, fontweight="bold")
        ax.plot(xmarker, ymarker-0.05, marker="s", color="black", markeredgewidth=0, markersize=20, transform=ax.transAxes)
        ax.plot([xmarker-0.05,xmarker+0.05], [ymarker-0.05, ymarker-0.05], "--", color="black", lw=3, transform=ax.transAxes)

        plt.savefig(self.outpng_recovery, dpi=self.fig_dpi)

    ###########################
    # _loop_measure_flux_norm #
    ###########################

    def _loop_measure_flux_norm(
        self,
        imagenames,
        beams,
        ):
        """
        plot_recovery
        """

        list_total_flux = []
        for i in range(len(beams)):
            # get names
            this_beam    = str(beams[i]).replace(".","p").zfill(4)
            this_map     = imagenames.replace("????",this_beam)

            # get data
            print("# measure total flux of " + this_map.split("/")[-1])
            this_data,_  = imval_all(this_map)
            this_data    = this_data["data"].flatten()
            this_data[np.isnan(this_data)] = 0
            this_data[np.isinf(this_data)] = 0
            this_data = this_data[this_data!=0]

            # determine number of bins
            pix       = abs(imhead(this_map,mode="list")["cdelt1"]) * 3600 * 180/np.pi
            this_flux = np.sum(this_data) * pix**2
            list_total_flux.append(this_flux)

        return list_total_flux / list_total_flux[-1]

    #

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

        beams_new_n0628 = [s for s in self.beams_n0628[:-1] if not "11.5" in str(s)]
        list_rms_co10_n0628 = self._loop_measure_log_rms(
            self.outcube_co10_n0628.replace(str(self.basebeam_n0628).replace(".","p").zfill(4),"????").replace(".image","_k.image"),
            beams_new_n0628,
            self.noise_vs_beam_co10_n0628,
            )
        list_rms_co21_n0628 = self._loop_measure_log_rms(
            self.outcube_co21_n0628.replace(str(self.basebeam_n0628).replace(".","p").zfill(4),"????").replace(".image","_k.image"),
            beams_new_n0628,
            self.noise_vs_beam_co21_n0628,
            )

        beams_new_n3627 = [s for s in self.beams_n3627[:-1]]
        list_rms_co10_n3627 = self._loop_measure_log_rms(
            self.outcube_co10_n3627.replace(str(self.basebeam_n3627).replace(".","p").zfill(4),"????").replace(".image","_k.image"),
            beams_new_n3627,
            self.noise_vs_beam_co10_n3627,
            )
        list_rms_co21_n3627 = self._loop_measure_log_rms(
            self.outcube_co21_n3627.replace(str(self.basebeam_n3627).replace(".","p").zfill(4),"????").replace(".image","_k.image"),
            beams_new_n3627,
            self.noise_vs_beam_co21_n3627,
            )

        beams_new_n4254 = [s for s in self.beams_n4254[:-1] if not "8.7" in str(s)]
        list_rms_co10_n4254 = self._loop_measure_log_rms(
            self.outcube_co10_n4254.replace(str(self.basebeam_n4254).replace(".","p").zfill(4),"????").replace(".image","_k.image"),
            beams_new_n4254,
            self.noise_vs_beam_co10_n4254,
            )
        list_rms_co21_n4254 = self._loop_measure_log_rms(
            self.outcube_co21_n4254.replace(str(self.basebeam_n4254).replace(".","p").zfill(4),"????").replace(".image","_k.image"),
            beams_new_n4254,
            self.noise_vs_beam_co21_n4254,
            )

        beams_new_n4321 = [s for s in self.beams_n4321[:-1] if not "7.5" in str(s)]
        list_rms_co10_n4321 = self._loop_measure_log_rms(
            self.outcube_co10_n4321.replace(str(self.basebeam_n4321).replace(".","p").zfill(4),"????").replace(".image","_k.image"),
            beams_new_n4321,
            self.noise_vs_beam_co10_n4321,
            )
        list_rms_co21_n4321 = self._loop_measure_log_rms(
            self.outcube_co21_n4321.replace(str(self.basebeam_n4321).replace(".","p").zfill(4),"????").replace(".image","_k.image"),
            beams_new_n4321,
            self.noise_vs_beam_co21_n4321,
            )

        xlim   = [2,28]
        ylim   = [-3.6,-0.8]
        title  = "(b) Sensitivity vs. Beam Size"
        xlabel = "Beam size (arcsec)"
        ylabel = "log rms per voxel (K)"
        index  = 1

        ########
        # plot #
        ########

        plt.figure(figsize=(13,10))
        gs = gridspec.GridSpec(nrows=10, ncols=10)
        ax = plt.subplot(gs[0:10,0:10])

        ad = [0.215,0.83,0.10,0.90]
        myax_set(ax, "both", xlim, ylim, title, xlabel, ylabel, adjust=ad)

        # plot co10 rms
        ax.plot(beams_new_n0628, list_rms_co10_n0628[:,index], "o-", color=self.c_n0628, markeredgewidth=0, markersize = 20, lw=3, label = "NGC 0628 CO(1-0)")
        ax.plot(beams_new_n3627, list_rms_co10_n3627[:,index], "o-", color=self.c_n3627, markeredgewidth=0, markersize = 20, lw=3, label = "NGC 3627 CO(1-0)")
        ax.plot(beams_new_n4254, list_rms_co10_n4254[:,index], "o-", color=self.c_n4254, markeredgewidth=0, markersize = 20, lw=3, label = "NGC 4254 CO(1-0)")
        ax.plot(beams_new_n4321, list_rms_co10_n4321[:,index], "o-", color=self.c_n4321, markeredgewidth=0, markersize = 20, lw=3, label = "NGC 4321 CO(1-0)")
        # plot co21 rms
        ax.plot(beams_new_n0628, list_rms_co21_n0628[:,index], "s--", color=self.c_n0628, markeredgewidth=0, markersize = 20, lw=3, label = "NGC 0628 CO(2-1)")
        ax.plot(beams_new_n3627, list_rms_co21_n3627[:,index], "s--", color=self.c_n3627, markeredgewidth=0, markersize = 20, lw=3, label = "NGC 3627 CO(2-1)")
        ax.plot(beams_new_n4254, list_rms_co21_n4254[:,index], "s--", color=self.c_n4254, markeredgewidth=0, markersize = 20, lw=3, label = "NGC 4254 CO(2-1)")
        ax.plot(beams_new_n4321, list_rms_co21_n4321[:,index], "s--", color=self.c_n4321, markeredgewidth=0, markersize = 20, lw=3, label = "NGC 4321 CO(2-1)")

        # text
        t=ax.text(0.95, 0.93, "NGC 0628", color=self.c_n0628, horizontalalignment="right", transform=ax.transAxes, size=self.legend_fontsize, fontweight="bold")
        t.set_bbox(dict(facecolor="white", alpha=self.self.text_back_alpha, lw=0))
        t=ax.text(0.95, 0.88, "NGC 3627", color=self.c_n3627, horizontalalignment="right", transform=ax.transAxes, size=self.legend_fontsize, fontweight="bold")
        t.set_bbox(dict(facecolor="white", alpha=self.self.text_back_alpha, lw=0))
        t=ax.text(0.95, 0.83, "NGC 4254", color=self.c_n4254, horizontalalignment="right", transform=ax.transAxes, size=self.legend_fontsize, fontweight="bold")
        t.set_bbox(dict(facecolor="white", alpha=self.self.text_back_alpha, lw=0))
        t=ax.text(0.95, 0.78, "NGC 4321", color=self.c_n4321, horizontalalignment="right", transform=ax.transAxes, size=self.legend_fontsize, fontweight="bold")
        t.set_bbox(dict(facecolor="white", alpha=self.self.text_back_alpha, lw=0))

        ax.text(0.55, 0.90, "CO(1-0) datacubes", color="black", horizontalalignment="right", transform=ax.transAxes, size=self.legend_fontsize, fontweight="bold")
        ax.text(0.48, 0.25, "CO(2-1) datacubes", color="black", horizontalalignment="right", transform=ax.transAxes, size=self.legend_fontsize, fontweight="bold")

        plt.savefig(self.outpng_noise_vs_beam, dpi=self.fig_dpi)

    #########################
    # _loop_measure_log_rms #
    #########################

    def _loop_measure_log_rms(
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

                _,_,_,_,this_rms,_,_,this_p84 = self._gaussfit_noise(this_data,this_bins)
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
        rms          = abs(np.round(popt[1], 5))
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