"""
Class to analyze Cycle 2 ALMA CO line datasets toward NGC 3110 (Kawana, Saito et al. 2021)

requirements:
CASA Version 5.4.0-70, ananlysisUtils, astropy

data:
ALMA line and cont data  2013.0.01172.S (https://almascience.nrao.edu/aq/?result_view=observation&projectCode=2013.1.01172.S)
OAO H-alpha FITS         T. Hattori et al. 2004, AJ, 127, 736
VLA FITS                 https://archive.nrao.edu/archive/archiveimage.html
VLT Ks-band FITS         Z. Randriamanakoto et al. 2013, MNRAS, 431, 554

usage:
> import os
> from scripts_n3110_co import ToolsNGC3110 as tools
> 
> tl = tools(
>     refresh     = False,
>     keyfile_gal = "/home02/saitots/myUtils/keys_n3110_co/key_ngc3110.txt",
>     keyfile_fig = "/home02/saitots/myUtils/keys_n3110_co/key_figures.txt",
>     )
> 
> tl.run_ngc3110_co(
>     do_prepare      = True, # align FITS maps (ready for CASA analysis)
>     do_lineratios   = True, # create line ratio maps
>     do_sampling     = True, # hex-sample maps (automatically skip if txt files exist)
>     plot_showcase   = True, # plot line, cont, and ratio maps with png format
>     plot_figures    = True, # plot all the other figures with png format
>     combine_figures = True, # combine png figures using image magick (ready for paper)
>     )
> 
> os.system("rm -rf *.last")

paper drafts:
Date         Filename                          To
2021-06-10   draft_v0_210610.zip               all co-Is
2021-07-02   draft_v1_210702_submitted.zip     journal
2022-01-31   draft_v2_220131.zip               all co-Is
2022-01-31   reply_v1_220131.pdf               all co-Is
2022-02-17   reply_1st_220217_v2.pdf           referee
2022-02-17   draft_v3_220217_1st_revised.zip   journal

history:
Date         Action
2016-04-01   start project with Kawana-san, Okumura-san, and Kawabe-san
2018-03-31   take over all the data and results from Kawana-san
2021-06-07   start re-analysis, write this preamble
2021-06-08   start to create paper-ready figures
2021-06-09   start up-dating the draft
2021-06-11   circulate v2 draft to the whole team
2021-06-28   move to ADC because of issues with new laptop
2021-07-02   1st submit to ApJ!
2021-08-17   receive the 1st referee report
2021-09-01   major update based on the 1st referee report
2021-10-19   bug fix in the rotation diagram part (refactor equation, Snu2)
2021-11-28   use eq1 of Nakajima et al. 2018
2022-01-31   circulate the revised draft and the reply to the whole team
2022-02-17   2nd submit to ApJ!
2022-03-19   accepted!
Toshiki Saito@Nichidai/NAOJ
"""

import os, sys, glob
import numpy as np

from mycasa_tasks import *
from mycasa_sampling import *
from mycasa_plots import *

myia = aU.createCasaTool(iatool)

###########################
### ToolsDense
###########################
class ToolsNGC3110():
    """
    Class for the NGC 3110 CO multi-line project.
    """

    ############
    # __init__ #
    ############

    def __init__(
        self,
        keyfile_fig = None,
        keyfile_gal = None,
        refresh = False,
        delete_inter = True,
        ):
        # initialize keys
        self.keyfile_gal = keyfile_gal
        self.keyfile_fig = keyfile_fig

        # intialize task
        self.refresh      = refresh
        self.delete_inter = delete_inter
        self.taskname     = None

        self.fig_dpi = 200

        # import parameters
        if keyfile_fig is not None:
            self.modname = "ToolsNGC3110."
            
            # get directories
            self.dir_proj = self._read_key("dir_proj")
            dir_raw = self.dir_proj + self._read_key("dir_raw")
            dir_other = self.dir_proj + self._read_key("dir_other")

            self.dir_ready = self.dir_proj + self._read_key("dir_ready")
            self.dir_products = self.dir_proj + self._read_key("dir_products")
            self.dir_final = self.dir_proj + self._read_key("dir_final")
            self.dir_casaregion = self.dir_proj + self._read_key("dir_casaregion")

            self._create_dir(self.dir_ready)
            self._create_dir(self.dir_products)
            self._create_dir(self.dir_final)
            self._create_dir(self.dir_casaregion)

            # input maps
            self.cube_12co10 = dir_raw + self._read_key("cube_12co10")
            self.cube_12co21 = dir_raw + self._read_key("cube_12co21")
            self.cube_13co10 = dir_raw + self._read_key("cube_13co10")
            self.cube_13co21 = dir_raw + self._read_key("cube_13co21")
            self.cube_c18o21 = dir_raw + self._read_key("cube_c18o21")

            self.pb_12co10 = dir_raw + self._read_key("pb_12co10")
            self.pb_12co21 = dir_raw + self._read_key("pb_12co21")
            self.pb_13co10 = dir_raw + self._read_key("pb_13co10")
            self.pb_13co21 = dir_raw + self._read_key("pb_13co21")
            self.pb_c18o21 = dir_raw + self._read_key("pb_c18o21")

            self.cube_12co10_uvlim = dir_raw + self._read_key("cube_12co10_uvlim")
            self.cube_12co21_uvlim = dir_raw + self._read_key("cube_12co21_uvlim")
            self.cube_13co10_uvlim = dir_raw + self._read_key("cube_13co10_uvlim")
            self.cube_13co21_uvlim = dir_raw + self._read_key("cube_13co21_uvlim")
            self.cube_c18o21_uvlim = dir_raw + self._read_key("cube_c18o21_uvlim")

            self.map_b3 = dir_raw + self._read_key("map_b3")
            self.map_b6 = dir_raw + self._read_key("map_b6")

            self.map_ssc    = dir_raw + self._read_key("map_ssc")
            self.map_halpha = dir_raw + self._read_key("map_halpha")
            self.map_vla    = dir_raw + self._read_key("map_vla")
            self.map_irac   = dir_raw + self._read_key("map_irac")

            # input files
            self.table_tkin = dir_other + self._read_key("table_tkin")
            self.table_nh2  = dir_other + self._read_key("table_nh2")

            # output maps
            self.outfits_12co10 = self.dir_ready + self._read_key("outfits_12co10")
            self.outfits_12co21 = self.dir_ready + self._read_key("outfits_12co21")
            self.outfits_13co10 = self.dir_ready + self._read_key("outfits_13co10")
            self.outfits_13co21 = self.dir_ready + self._read_key("outfits_13co21")
            self.outfits_c18o21 = self.dir_ready + self._read_key("outfits_c18o21")

            self.outfits_m0_12co10 = self.outfits_12co10.replace(".fits","_mom0.fits")
            self.outfits_m0_12co21 = self.outfits_12co21.replace(".fits","_mom0.fits")
            self.outfits_m0_13co10 = self.outfits_13co10.replace(".fits","_mom0.fits")
            self.outfits_m0_13co21 = self.outfits_13co21.replace(".fits","_mom0.fits")
            self.outfits_m0_c18o21 = self.outfits_c18o21.replace(".fits","_mom0.fits")

            self.outfits_m1_12co10 = self.outfits_12co10.replace(".fits","_mom1.fits")
            self.outfits_m1_12co21 = self.outfits_12co21.replace(".fits","_mom1.fits")
            self.outfits_m1_13co10 = self.outfits_13co10.replace(".fits","_mom1.fits")
            self.outfits_m1_13co21 = self.outfits_13co21.replace(".fits","_mom1.fits")
            self.outfits_m1_c18o21 = self.outfits_c18o21.replace(".fits","_mom1.fits")

            self.outfits_em0_12co10 = self.outfits_12co10.replace(".fits","_emom0.fits")
            self.outfits_em0_12co21 = self.outfits_12co21.replace(".fits","_emom0.fits")
            self.outfits_em0_13co10 = self.outfits_13co10.replace(".fits","_emom0.fits")
            self.outfits_em0_13co21 = self.outfits_13co21.replace(".fits","_emom0.fits")
            self.outfits_em0_c18o21 = self.outfits_c18o21.replace(".fits","_emom0.fits")

            self.outfits_b3 = self.dir_ready + self._read_key("outfits_b3")
            self.outfits_b6 = self.dir_ready + self._read_key("outfits_b6")
            self.outfits_b3_nopbcor = self.outfits_b3.replace(".fits","_nopbcor.fits")
            self.outfits_b6_nopbcor = self.outfits_b6.replace(".fits","_nopbcor.fits")

            self.outfits_ssc    = self.dir_ready + self._read_key("outfits_ssc")
            self.outfits_halpha = self.dir_ready + self._read_key("outfits_halpha")
            self.outfits_vla    = self.dir_ready + self._read_key("outfits_vla")
            self.outfits_irac   = self.dir_ready + self._read_key("outfits_irac")
            self.outfits_pb_b3  = self.dir_ready + self._read_key("outfits_pb_b3")
            self.outfits_pb_b6  = self.dir_ready + self._read_key("outfits_pb_b6")

            self.outfits_r_21    = self.dir_ready + self._read_key("outfits_r_21")
            self.outfits_r_t21   = self.dir_ready + self._read_key("outfits_r_t21")
            self.outfits_r_1213l = self.dir_ready + self._read_key("outfits_r_1213l")
            self.outfits_r_1213h = self.dir_ready + self._read_key("outfits_r_1213h")

            # ngc3110 properties
            self.z            = float(self._read_key("z", "gal"))
            self.dist         = float(self._read_key("distance", "gal"))
            self.dist_cm      = self.dist * 10**6 * 3.86*10**18

            self.ra_str       = self._read_key("ra", "gal")
            self.ra           = float(self.ra_str.replace("deg",""))
            self.dec_str      = self._read_key("dec", "gal")
            self.dec          = float(self.dec_str.replace("deg",""))

            self.ra_irac_str  = self._read_key("ra_irac", "gal")
            self.ra_irac      = float(self.ra_irac_str.replace("deg",""))
            self.dec_irac_str = self._read_key("dec_irac", "gal")
            self.dec_irac     = float(self.dec_irac_str.replace("deg",""))

            self.ra_speak     = float(self._read_key("ra_speak", "gal").replace("deg",""))
            self.dec_speak    = float(self._read_key("dec_speak", "gal").replace("deg",""))
            self.r_speak_as   = float(self._read_key("r_speak", "gal").replace("arcsec",""))

            self.scale_pc     = float(self._read_key("scale", "gal"))
            self.scale_kpc    = float(self._read_key("scale", "gal")) / 1000.

            self.pa           = np.radians(float(self._read_key("pa", "gal")))
            self.incl         = np.radians(float(self._read_key("incl", "gal")))

            # input parameters
            self.beam           = float(self._read_key("beam"))
            self.snr_mom_strong = float(self._read_key("snr_mom_strong"))
            self.snr_mom_weak   = float(self._read_key("snr_mom_weak"))
            self.imsize         = int(self._read_key("imsize"))
            self.imsize_irac    = int(self._read_key("imsize_irac"))
            self.pixelmin       = 2.0
            self.aperture_r     = float(self._read_key("aperture")) / 2.0
            self.step           = float(self._read_key("step"))
            self.ra_blc         = float(self._read_key("ra_blc"))
            self.decl_blc       = float(self._read_key("decl_blc"))
            self.num_aperture   = int(self._read_key("num_aperture"))
            self.alpha_co       = 1.7

            self.nu_12co10 = 115.27120180
            self.nu_13co10 = 110.20135430
            self.nu_12co21 = 230.53800000
            self.nu_13co21 = 220.39868420
            self.nu_b6     = 234.6075
            self.nu_b3     = 104.024625

            self.key_qrot = "/home02/saitots/myUtils/keys_n3110_co/Qrot_CDMS.txt"

            # output txt and png
            self.outpng_irac   = self.dir_products + self._read_key("outpng_irac")
            self.outpng_12co10 = self.dir_products + self._read_key("outpng_12co10")
            self.outpng_12co21 = self.dir_products + self._read_key("outpng_12co21")
            self.outpng_13co10 = self.dir_products + self._read_key("outpng_13co10")
            self.outpng_13co21 = self.dir_products + self._read_key("outpng_13co21")
            self.outpng_c18o21 = self.dir_products + self._read_key("outpng_c18o21")

            self.outpng_b3 = self.dir_products + self._read_key("outpng_b3")
            self.outpng_b6 = self.dir_products + self._read_key("outpng_b6")

            self.outpng_r_21    = self.dir_products + self._read_key("outpng_r_21")
            self.outpng_r_t21   = self.dir_products + self._read_key("outpng_r_t21")
            self.outpng_r_1213l = self.dir_products + self._read_key("outpng_r_1213l")
            self.outpng_r_1213h = self.dir_products + self._read_key("outpng_r_1213h")

            self.outtxt_hexdata = self.dir_ready + self._read_key("outtxt_hexdata")
            self.outtxt_hexphys = self.dir_ready + self._read_key("outtxt_hexphys")

            self.outpng_radial_21   = self.dir_products + self._read_key("outpng_radial_21")
            self.outpng_radial_1213 = self.dir_products + self._read_key("outpng_radial_1213")

            self.outpng_hex_index = self.dir_products + self._read_key("outpng_hex_index")
            self.outpng_hex_tkin  = self.dir_products + self._read_key("outpng_hex_tkin")
            self.outpng_hex_nh2   = self.dir_products + self._read_key("outpng_hex_nh2")
            self.outpng_hex_sfrd  = self.dir_products + self._read_key("outpng_hex_sfrd")
            self.outpng_hex_sscd  = self.dir_products + self._read_key("outpng_hex_sscd")
            self.outpng_hex_sfe   = self.dir_products + self._read_key("outpng_hex_sfe")
            self.outpng_hex_aco   = self.dir_products + self._read_key("outpng_hex_aco")

            self.outpng_aco_radial = self.dir_products + self._read_key("outpng_aco_radial")
            self.outpng_aco_hist   = self.dir_products + self._read_key("outpng_aco_hist")

            self.output_ks_fix  = self.dir_products + self._read_key("output_ks_fix")
            self.output_ks_vary = self.dir_products + self._read_key("output_ks_vary")

            self.output_index_vs_sfe_fix  = self.dir_products + self._read_key("output_index_vs_sfe_fix")
            self.output_index_vs_sfe_vary = self.dir_products + self._read_key("output_index_vs_sfe_vary")

            self.output_sfe_vs_ssc_fix  = self.dir_products + self._read_key("output_sfe_vs_ssc_fix")
            self.output_sfe_vs_ssc_vary = self.dir_products + self._read_key("output_sfe_vs_ssc_vary")

            # final product
            self.final_irac      = self.dir_final + self._read_key("final_irac")
            self.final_showline  = self.dir_final + self._read_key("final_showline")
            self.final_showcont  = self.dir_final + self._read_key("final_showcont")
            self.final_showratio = self.dir_final + self._read_key("final_showratio")
            self.final_radial    = self.dir_final + self._read_key("final_radial")
            self.final_showhex   = self.dir_final + self._read_key("final_showhex")
            self.final_aco       = self.dir_final + self._read_key("final_aco")
            self.final_scatter   = self.dir_final + self._read_key("final_scatter")
            self.final_appendix1 = self.dir_final + self._read_key("final_appendix1")

            # box
            self.box_irac     = self._read_key("box_irac")

            self.box_line_tl  = self._read_key("box_line_tl")
            self.box_line_tr  = self._read_key("box_line_tr")
            self.box_line_bl  = self._read_key("box_line_bl")
            self.box_line_br  = self._read_key("box_line_br")

            self.box_cont_b3  = self._read_key("box_cont_b3")
            self.box_cont_b6  = self._read_key("box_cont_b6")

            self.box_ratio_tl = self._read_key("box_ratio_tl")
            self.box_ratio_tr = self._read_key("box_ratio_tr")
            self.box_ratio_bl = self._read_key("box_ratio_bl")
            self.box_ratio_br = self._read_key("box_ratio_br")

            self.box_radial   = self._read_key("box_radial")

            self.box_hex1     = self._read_key("box_hex1")
            self.box_hex2     = self._read_key("box_hex2")
            self.box_hex3     = self._read_key("box_hex3")

            self.box_aco1     = self._read_key("box_aco1")
            self.box_aco2     = self._read_key("box_aco2")

            self.box_scatter1 = self._read_key("box_scatter1")
            self.box_scatter2 = self._read_key("box_scatter2")

    ##################
    # run_ngc3110_co #
    ##################

    def run_ngc3110_co(
        self,
        do_prepare      = False,
        do_lineratios   = False,
        do_sampling     = False,
        plot_showcase   = False,
        plot_figures    = False,
        combine_figures = False,
        ):
        """
        This method runs all the methods which will create figures in the paper.
        """

        if do_prepare==True:
            self.align_maps()

        if do_lineratios==True:
        	self.lineratios()

        if plot_showcase==True:
            self.showline()
            self.showcont()
            self.showratio()

        if do_sampling==True:
            done = glob.glob(self.outtxt_hexdata)
            if not done:
                self.hex_sampling_casa()
            else:
                print("# skip hex_sampling_casa()")

            done = glob.glob(self.outtxt_hexphys)
            if not done:
                self.hex_sampling_phys()
            else:
                print("# skip hex_sampling_phys()")

        if plot_figures==True:
            self.plot_radial_ratio()
            self.showhex()
            self.plot_aco()
            self.plot_scatter()

        if combine_figures==True:
            self.immagick_figures(
                do_all=True,
                do_final_showline=True,
                )

    ####################
    # immagick_figures #
    ####################

    def immagick_figures(
        self,
        do_all=False,
        do_final_irac=False,
        do_final_showline=False,
        do_final_showcont=False,
        do_final_showratio=False,
        do_final_radial=False,
        do_final_showhex=False,
        do_final_aco=False,
        do_final_scatter=False,
        do_final_appendix1=False,
        delin=False,
        ):
        """
        Re-shape and combine all the png files into the figures in the paper.
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outpng_irac,taskname)

        if do_all==True:
            do_final_irac      = True
            do_final_showline  = True
            do_final_showcont  = True
            do_final_showratio = True
            do_final_radial    = True
            do_final_showhex   = True
            do_final_aco       = True
            do_final_scatter   = True
            do_final_appendix1 = True

        # final_irac
        if do_final_irac==True:
            print("#####################")
            print("# create final_irac #")
            print("#####################")

            immagick_crop(self.outpng_irac,self.final_irac,self.box_irac)

        # final_showline
        if do_final_showline==True:
            print("")
            print("#########################")
            print("# create final_showline #")
            print("#########################")

            combine_two_png(
                self.outpng_12co10,
                self.outpng_13co10,
                self.final_showline+"_tmp1.png",
                self.box_line_tl,
                self.box_line_tr,
                )
            combine_three_png(
                self.outpng_12co21,
                self.outpng_13co21,
                self.outpng_c18o21,
                self.final_showline+"_tmp2.png",
                self.box_line_bl,
                self.box_line_br,
                self.box_line_br,
                )
            immagick_append(
                self.final_showline+"_tmp1.png",
                self.final_showline+"_tmp2.png",
                self.final_showline,
                axis="column",
                delin=True,
                )

        # final_showcont
        if do_final_showcont==True:
            print("")
            print("#########################")
            print("# create final_showcont #")
            print("#########################")

            combine_two_png(
                self.outpng_b3,
                self.outpng_b6,
                self.final_showcont,
                self.box_cont_b3,
                self.box_cont_b6,
                )

        # final_showratio
        if do_final_showratio==True:
            print("")
            print("##########################")
            print("# create final_showratio #")
            print("##########################")
        
            combine_two_png(
                self.outpng_r_21,
                self.outpng_r_t21,
                self.final_showratio+"_tmp1.png",
                self.box_ratio_tl,
                self.box_ratio_tr,
                )
            combine_two_png(
                self.outpng_r_1213l,
                self.outpng_r_1213h,
                self.final_showratio+"_tmp2.png",
                self.box_ratio_bl,
                self.box_ratio_br,
                )
            immagick_append(
                self.final_showratio+"_tmp1.png",
                self.final_showratio+"_tmp2.png",
                self.final_showratio,
                axis="column",
                delin=True,
                )

        # final_radial
        if do_final_radial==True:
            print("")
            print("#######################")
            print("# create final_radial #")
            print("#######################")
        
            combine_two_png(
                self.outpng_radial_21,
                self.outpng_radial_1213,
                self.final_radial,
                self.box_radial,
                self.box_radial,
                )

        # final_showhex
        if do_final_showhex==True:
            print("")
            print("########################")
            print("# create final_showhex #")
            print("########################")

            combine_three_png(
                self.outpng_hex_tkin,
                self.outpng_hex_nh2,
                self.outpng_hex_index,
                self.final_showhex+"_tmp1.png",
                self.box_hex1,
                self.box_hex2,
                self.box_hex2,
                )
            combine_three_png(
                self.outpng_hex_sfrd,
                self.outpng_hex_sscd,
                self.outpng_hex_sfe,
                self.final_showhex+"_tmp2.png",
                self.box_hex1,
                self.box_hex2,
                self.box_hex2,
                )
            combine_three_png(
                self.final_showhex+"_tmp1.png",
                self.final_showhex+"_tmp2.png",
                self.outpng_hex_aco,
                self.final_showhex,
                "100000x100000+0+0",
                "100000x100000+0+0",
                self.box_hex3,
                axis="column",
                )
            os.system("rm -rf " + self.final_showhex + "_tmp1.png")
            os.system("rm -rf " + self.final_showhex + "_tmp2.png")

        # final_aco
        if do_final_aco==True:
            print("")
            print("####################")
            print("# create final_aco #")
            print("####################")

            combine_two_png(
                self.outpng_aco_radial,
                self.outpng_aco_hist,
                self.final_aco,
                self.box_aco1,
                self.box_aco2,
                )

        # final_scatter
        if do_final_scatter==True:
            print("")
            print("########################")
            print("# create final_scatter #")
            print("########################")

            combine_two_png(
                self.output_ks_fix,
                self.output_index_vs_sfe_fix,
                self.final_scatter+"_tmp1.png",
                self.box_scatter1,
                self.box_scatter1,
                )
            combine_two_png(
                self.final_scatter+"_tmp1.png",
                self.output_sfe_vs_ssc_fix,
                self.final_scatter,
                "100000x100000+0+0",
                self.box_scatter2,
                axis="column",
                )
            os.system("rm -rf " + self.final_scatter + "_tmp1.png")

        # final_appendix1
        if do_final_appendix1==True:
            print("")
            print("##########################")
            print("# create final_appendix1 #")
            print("##########################")

            combine_two_png(
                self.output_ks_vary,
                self.output_index_vs_sfe_vary,
                self.final_appendix1+"_tmp1.png",
                self.box_scatter1,
                self.box_scatter1,
                )
            combine_two_png(
                self.final_appendix1+"_tmp1.png",
                self.output_sfe_vs_ssc_vary,
                self.final_appendix1,
                "100000x100000+0+0",
                self.box_scatter2,
                axis="column",
                )
            os.system("rm -rf " + self.final_appendix1 + "_tmp1.png")

    ################
    # plot_scatter #
    ################

    def plot_scatter(
        self,
        ):
        """
        Plot the KS relation, SFE vs. spectral index, and SFE vs. SSC density with two alpha_co:
        a constant alpha_co of 1.7 nad varying alpha_co (= alpha_lte(Trot=15K)).
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outtxt_hexphys,taskname)

        # import data
        data         = np.loadtxt(self.outtxt_hexphys)
        data_ra      = data[:,0]
        data_dec     = data[:,1]
        data_ra2     = ( data_ra*np.cos(self.pa) - data_dec*np.sin(self.pa) ) / np.cos(self.incl)
        data_dec2    = data_ra*np.sin(self.pa) + data_dec*np.cos(self.incl)
        dist_kpc     = np.sqrt(data_ra2**2+data_dec2**2) * 3600 * self.scale_kpc
        index        = data[:,4]
        err_index    = data[:,5]
        sfrd         = data[:,6]
        sfrd_err     = 0.3
        sscd         = data[:,7]
        dh2          = data[:,11]
        err_dh2      = data[:,12]
        sfe          = data[:,10]

        # aco
        aco_lte_trot = data[:,13]
        aco_lte_trot_err = data[:,17]
        aco_ism_trot = data[:,15]
        aco_ism_trot_err = data[:,19]
        aco_fix = self.alpha_co

        # process data
        dh2_fix      = np.log10(dh2)
        dh2_err_fix  = 1/np.log(10) * err_dh2/dh2
        dh2_vary     = np.log10(dh2 * aco_lte_trot/aco_fix)
        dh2_err_vary = 1/np.log(10) * err_dh2/dh2 * aco_lte_trot/aco_fix

        sfe_fix      = np.log10(sfe) - 9
        sfe_err_fix  = 0.3
        sfe_vary     = np.log10(sfe * aco_lte_trot/aco_fix) - 9
        sfe_err_vary = 0.3

        index        = np.log10(index)
        index_err    = 1/np.log(10) * err_index/10**index

        sscd         = np.log10(sscd)
        sfrd         = np.log10(sfrd)

        # data of the peak S
        ra_speak  = self.ra_speak - self.ra
        dec_speak = self.dec_speak - self.dec
        data_ra_from_speak  = data_ra - ra_speak
        data_dec_from_speak = data_dec - dec_speak
        data_r_from_speak   = np.sqrt(data_ra_from_speak**2 + data_dec_from_speak**2) * 3600

        dh2_fix_speak       = dh2_fix[data_r_from_speak<self.r_speak_as]
        dh2_err_fix_speak   = dh2_err_fix[data_r_from_speak<self.r_speak_as]
        dh2_vary_speak      = dh2_vary[data_r_from_speak<self.r_speak_as]
        dh2_err_vary_speak  = dh2_err_vary[data_r_from_speak<self.r_speak_as]
        sfe_fix_speak       = sfe_fix[data_r_from_speak<self.r_speak_as]
        sfe_vary_speak      = sfe_vary[data_r_from_speak<self.r_speak_as]
        sfrd_speak          = sfrd[data_r_from_speak<self.r_speak_as]
        dist_kpc_speak      = dist_kpc[data_r_from_speak<self.r_speak_as]
        index_speak         = index[data_r_from_speak<self.r_speak_as]
        index_err_speak     = index_err[data_r_from_speak<self.r_speak_as]
        sscd_speak          = sscd[data_r_from_speak<self.r_speak_as]

        ### ks relation
        xlim = [0.3,3.3]
        ylim = [-2.8,0.2]
        clim = 10.0

        # plot ks with a fixed aco
        plt.figure()
        plt.rcParams["font.size"] = 16
        plt.subplots_adjust(bottom = 0.15)
        gs = gridspec.GridSpec(nrows=30, ncols=30)
        ax = plt.subplot(gs[0:30,0:30])
        ax.grid(which="both")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"log $\Sigma_{H_2}$ ($M_{\odot}$ pc$^{-2}$)")
        ax.set_ylabel(r"log $\Sigma_{SFR}$ ($M_{\odot}$ kpc$^{-2}$ yr$^{-1}$)")
        ax.set_aspect('equal', adjustable='box')

        cax = ax.scatter(dh2_fix, sfrd, s=100, c=dist_kpc, cmap="rainbow_r", linewidths=0, alpha=0.7,zorder=-1e9)
        for i in range(len(dh2_fix)):
            x    = dh2_fix[i]
            y    = sfrd[i]
            xerr = dh2_err_fix[i]
            yerr = sfrd_err
            c    = cm.rainbow_r( dist_kpc[i] / clim )

            _, _, bars = ax.errorbar(x,y,xerr=xerr,yerr=yerr,fmt="o",c=c,capsize=5,markeredgewidth=0,markersize=0,lw=2,zorder=-1e9)
            [bar.set_alpha(0.7) for bar in bars]

        cbar = plt.colorbar(cax)
        cbar.set_label("Deprojected Distance (kpc)")
        cbar.set_clim([0,clim])
        cbar.outline.set_linewidth(1.0)

        for i in range(len(dh2_fix_speak)):
            x    = dh2_fix_speak[i]
            xerr = dh2_err_fix_speak[i]
            y    = sfrd_speak[i]
            yerr = sfrd_err
            c    = "black"
            ax.scatter(x, y, s=100, c=c, linewidth=2.0, zorder=1e9)
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, capsize=0, c=c, linewidth=2.0, zorder=1e9)

        ax.plot(xlim,[xlim[0]-3.0,xlim[1]-3.0], "k--")
        ax.plot(xlim,[xlim[0]-2.0,xlim[1]-2.0], "k--")

        ax.set_title(r"KS Relation ($\alpha_{CO}$ = "+str(aco_fix)+")")
        ax.text(0.5, -2.5+0.4, "SFE = 10$^{-9}$ yr$^{-1}$", rotation=45, fontsize=12, weight="bold", zorder=1e10)
        ax.text(0.5, -1.5+0.8, "SFE = 10$^{-8}$ yr$^{-1}$", rotation=45, fontsize=12, weight="bold", zorder=1e10)

        os.system("rm -rf " + self.output_ks_fix)
        plt.savefig(self.output_ks_fix, dpi=300)


        # plot ks with varying aco
        plt.figure()
        plt.rcParams["font.size"] = 16
        plt.subplots_adjust(bottom = 0.15)
        gs = gridspec.GridSpec(nrows=30, ncols=30)
        ax = plt.subplot(gs[0:30,0:30])
        ax.grid(which="both")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"log $\Sigma_{H_2}$ ($M_{\odot}$ pc$^{-2}$)")
        ax.set_ylabel(r"log $\Sigma_{SFR}$ ($M_{\odot}$ kpc$^{-2}$ yr$^{-1}$)")
        ax.set_aspect('equal', adjustable='box')

        cax = ax.scatter(dh2_vary, sfrd, s=100, c=dist_kpc, cmap="rainbow_r", linewidths=0, alpha=0.7,zorder=-1e9)
        for i in range(len(dh2_fix)):
            x    = dh2_vary[i]
            y    = sfrd[i]
            xerr = dh2_err_vary[i]
            yerr = sfrd_err
            c    = cm.rainbow_r( dist_kpc[i] / clim )

            _, _, bars = ax.errorbar(x,y,xerr=xerr,yerr=yerr,fmt="o",c=c,capsize=5,markeredgewidth=0,markersize=0,lw=2,zorder=-1e9)
            [bar.set_alpha(0.7) for bar in bars]

        for i in range(len(dh2_vary_speak)):
            x    = dh2_vary_speak[i]
            xerr = dh2_err_vary_speak[i]
            y    = sfrd_speak[i]
            yerr = sfrd_err
            c    = "black"
            ax.scatter(x, y, s=100, c=c, linewidth=2.0, zorder=1e9)
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, capsize=0, c=c, linewidth=2.0, zorder=1e9)

        cbar = plt.colorbar(cax)
        cbar.set_label("Deprojected Distance (kpc)")
        cbar.set_clim([0,clim])
        cbar.outline.set_linewidth(1.0)

        ax.plot(xlim,[xlim[0]-3.0,xlim[1]-3.0], "k--")
        ax.plot(xlim,[xlim[0]-2.0,xlim[1]-2.0], "k--")

        ax.set_title(r"KS Relation ($\alpha_{CO}$ = $\alpha_{LTE}$)")
        ax.text(0.5, -2.5+0.4, "SFE = 10$^{-9}$ yr$^{-1}$", rotation=45, fontsize=12, weight="bold", zorder=1e10)
        ax.text(0.5, -1.5+0.8, "SFE = 10$^{-8}$ yr$^{-1}$", rotation=45, fontsize=12, weight="bold", zorder=1e10)

        os.system("rm -rf " + self.output_ks_vary)
        plt.savefig(self.output_ks_vary, dpi=300)


        # sfe vs iindex
        xlim = [-1.0+0.2,0.5+0.2]
        ylim = [-9.5-0.2,-8.0-0.2]

        # plot sfe vs index with a fixed aco
        plt.figure()
        plt.rcParams["font.size"] = 16
        plt.subplots_adjust(bottom = 0.15)
        gs = gridspec.GridSpec(nrows=30, ncols=30)
        ax = plt.subplot(gs[0:30,0:30])
        ax.grid(which="both")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"log Spectral Index")
        ax.set_ylabel(r"log SFE (yr$^{-1}$)")
        ax.set_aspect('equal', adjustable='box')

        cax = ax.scatter(index, sfe_fix, s=100, c=dist_kpc, cmap="rainbow_r", linewidths=0, alpha=0.7,zorder=-1e9)
        for i in range(len(dh2_fix)):
            x    = index[i]
            y    = sfe_fix[i]
            xerr = index_err[i]
            yerr = sfe_err_fix
            c    = cm.rainbow_r( dist_kpc[i] / clim )

            _, _, bars = ax.errorbar(x,y,xerr=xerr,yerr=yerr,fmt="o",c=c,capsize=5,markeredgewidth=0,markersize=0,lw=2,zorder=-1e9)
            [bar.set_alpha(0.7) for bar in bars]

        for i in range(len(index_speak)):
            x    = index_speak[i]
            y    = sfe_fix_speak[i]
            xerr = index_err_speak[i]
            yerr = sfe_err_fix
            c    = "black"
            ax.scatter(x, y, s=100, c=c, linewidth=2.0, zorder=1e9)
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, capsize=0, c=c, linewidth=2.0, zorder=1e9)

        cbar = plt.colorbar(cax)
        cbar.set_label("Deprojected Distance (kpc)")
        cbar.set_clim([0,clim])
        cbar.outline.set_linewidth(1.0)

        ax.set_title(r"log SFE vs. log Index ($\alpha_{CO}$ = "+str(aco_fix)+")")

        os.system("rm -rf " + self.output_index_vs_sfe_fix)
        plt.savefig(self.output_index_vs_sfe_fix, dpi=300)


        # plot sfe vs index with varying aco
        plt.figure()
        plt.rcParams["font.size"] = 16
        plt.subplots_adjust(bottom = 0.15)
        gs = gridspec.GridSpec(nrows=30, ncols=30)
        ax = plt.subplot(gs[0:30,0:30])
        ax.grid(which="both")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"log Spectral Index")
        ax.set_ylabel(r"log SFE (yr$^{-1}$)")
        ax.set_aspect('equal', adjustable='box')

        cax = ax.scatter(index, sfe_vary, s=100, c=dist_kpc, cmap="rainbow_r", linewidths=0, alpha=0.7,zorder=-1e9)
        for i in range(len(dh2_vary)):
            x    = index[i]
            y    = sfe_vary[i]
            xerr = index_err[i]
            yerr = sfe_err_vary
            c    = cm.rainbow_r( dist_kpc[i] / clim )

            _, _, bars = ax.errorbar(x,y,xerr=xerr,yerr=yerr,fmt="o",c=c,capsize=5,markeredgewidth=0,markersize=0,lw=2,zorder=-1e9)
            [bar.set_alpha(0.7) for bar in bars]

        for i in range(len(index_speak)):
            x    = index_speak[i]
            y    = sfe_vary_speak[i]
            c    = "black"
            xerr = index_err_speak[i]
            yerr = sfe_err_vary
            ax.scatter(x, y, s=100, c=c, linewidth=2.0, zorder=1e9)
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, capsize=0, c=c, linewidth=2.0, zorder=1e9)

        cbar = plt.colorbar(cax)
        cbar.set_label("Deprojected Distance (kpc)")
        cbar.set_clim([0,clim])
        cbar.outline.set_linewidth(1.0)

        ax.set_title(r"log SFE vs. log Index ($\alpha_{CO}$ = $\alpha_{LTE}$)")

        os.system("rm -rf " + self.output_index_vs_sfe_vary)
        plt.savefig(self.output_index_vs_sfe_vary, dpi=300)


        # sfe vs ssc
        xlim = [-2.0+0.2,1.0+0.2]
        ylim = [-10.0-0.2,-7.0-0.2]

        # plot sfe vs ssc with a fixed aco
        plt.figure()
        plt.rcParams["font.size"] = 16
        plt.subplots_adjust(bottom = 0.15)
        gs = gridspec.GridSpec(nrows=30, ncols=30)
        ax = plt.subplot(gs[0:30,0:30])
        ax.grid(which="both")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"log $\Sigma_{SSC}$ (kpc$^{-2}$)")
        ax.set_ylabel(r"log SFE (yr$^{-1}$)")
        ax.set_aspect('equal', adjustable='box')

        cax = ax.scatter(sscd, sfe_fix, s=100, c=dist_kpc, cmap="rainbow_r", linewidths=0, alpha=0.7,zorder=-1e9)
        for i in range(len(sfe_fix)):
            x    = sscd[i]
            y    = sfe_fix[i]
            yerr = sfe_err_fix
            c    = cm.rainbow_r( dist_kpc[i] / clim )

            _, _, bars = ax.errorbar(x,y,yerr=yerr,fmt="o",c=c,capsize=5,markeredgewidth=0,markersize=0,lw=2,zorder=-1e9)
            [bar.set_alpha(0.7) for bar in bars]

        for i in range(len(sscd_speak)):
            x    = sscd_speak[i]
            y    = sfe_fix_speak[i]
            yerr = sfe_err_fix
            c    = "black"
            ax.scatter(x, y, s=100, c=c, linewidth=2.0, zorder=1e9)
            ax.errorbar(x, y, yerr=yerr, capsize=0, c=c, linewidth=2.0, zorder=1e9)

        cbar = plt.colorbar(cax)
        cbar.set_label("Deprojected Distance (kpc)")
        cbar.set_clim([0,clim])
        cbar.outline.set_linewidth(1.0)

        ax.set_title(r"log SFE vs. log $\Sigma_{SSC}$ ($\alpha_{CO}$ = "+str(aco_fix)+")")

        os.system("rm -rf " + self.output_sfe_vs_ssc_fix)
        plt.savefig(self.output_sfe_vs_ssc_fix, dpi=300)

        # plot sfe vs ssc with varying aco
        plt.figure()
        plt.rcParams["font.size"] = 16
        plt.subplots_adjust(bottom = 0.15)
        gs = gridspec.GridSpec(nrows=30, ncols=30)
        ax = plt.subplot(gs[0:30,0:30])
        ax.grid(which="both")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"log $\Sigma_{SSC}$ (kpc$^{-2}$)")
        ax.set_ylabel(r"log SFE (yr$^{-1}$)")
        ax.set_aspect('equal', adjustable='box')

        cax = ax.scatter(sscd, sfe_vary, s=100, c=dist_kpc, cmap="rainbow_r", linewidths=0, alpha=0.7,zorder=-1e9)
        for i in range(len(sfe_vary)):
            x    = sscd[i]
            y    = sfe_vary[i]
            yerr = sfe_err_vary
            c    = cm.rainbow_r( dist_kpc[i] / clim )

            _, _, bars = ax.errorbar(x,y,yerr=yerr,fmt="o",c=c,capsize=5,markeredgewidth=0,markersize=0,lw=2,zorder=-1e9)
            [bar.set_alpha(0.7) for bar in bars]

        for i in range(len(sscd_speak)):
            x    = sscd_speak[i]
            y    = sfe_vary_speak[i]
            yerr = sfe_err_fix
            c    = "black"
            ax.scatter(x, y, s=100, c=c, linewidth=2.0, zorder=1e9)
            ax.errorbar(x, y, yerr=yerr, capsize=0, c=c, linewidth=2.0, zorder=1e9)

        cbar = plt.colorbar(cax)
        cbar.set_label("Deprojected Distance (kpc)")
        cbar.set_clim([0,clim])
        cbar.outline.set_linewidth(1.0)

        ax.set_title(r"log SFE vs. log $\Sigma_{SSC}$ ($\alpha_{CO}$ = $\alpha_{LTE}$)")

        os.system("rm -rf " + self.output_sfe_vs_ssc_vary)
        plt.savefig(self.output_sfe_vs_ssc_vary, dpi=300)

    ############
    # plot_aco #
    ############

    def plot_aco(
        self,
        ):
        """
        Plot alpha_lte(Trot) and alpha_ism(Trot). Flag data points when alpha_co(Tkin) < 
        alpha_co(Trot) (i.e., non-thermal equilibrium).
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outtxt_hexphys,taskname)

        # import data
        data         = np.loadtxt(self.outtxt_hexphys)
        data_ra      = data[:,0]
        data_dec     = data[:,1]
        data_ra2     = ( data_ra*np.cos(self.pa) - data_dec*np.sin(self.pa) ) / np.cos(self.incl)
        data_dec2    = data_ra*np.sin(self.pa) + data_dec*np.cos(self.incl)
        aco_lte_trot = data[:,13]
        aco_lte_tkin = data[:,14]
        aco_ism_trot = data[:,15]
        aco_ism_tkin = data[:,16]
        aco_lte_trot_err = data[:,17]
        aco_lte_tkin_err = data[:,18]
        aco_ism_trot_err = data[:,19]
        aco_ism_tkin_err = data[:,20]

        # process data
        dist_kpc  = np.sqrt(data_ra2**2+data_dec2**2) * 3600 * self.scale_kpc

        cut_lte = np.where((aco_lte_trot>0) & (aco_lte_tkin>0) & (aco_lte_trot<aco_lte_tkin))
        dist_lte_trot = dist_kpc[cut_lte]
        aco_lte_trot_err = 1/np.log(10) * aco_lte_trot_err[cut_lte]/aco_lte_trot[cut_lte]
        aco_lte_trot  = np.log10(aco_lte_trot[cut_lte])

        cut_ism = np.where((aco_ism_trot>0) & (aco_ism_tkin>0) & (aco_ism_trot<aco_ism_tkin))
        dist_ism_trot = dist_kpc[cut_ism] # np.log10(dist[cut_ism])
        aco_ism_trot_err = 1/np.log(10) * aco_ism_trot_err[cut_ism]/aco_ism_trot[cut_ism]
        aco_ism_trot  = np.log10(aco_ism_trot[cut_ism])

        # plot radial dist
        plt.figure()
        plt.rcParams["font.size"] = 16
        plt.subplots_adjust(left=0.23,right=0.78,bottom=0.15)
        gs = gridspec.GridSpec(nrows=30, ncols=30)
        ax = plt.subplot(gs[0:30,0:30])
        ax.grid(which="both")

        xlim = [0.0, 10**1.0] # [-1.0+0.1,1.0+0.1]
        ax.set_xlim(xlim)
        ax.set_ylim([-1.0+0.4,1.0+0.4])
        ax.set_xlabel("Deprojected Distance (kpc)")
        ax.set_ylabel(r"log $\alpha_{CO}$ ($M_{\odot}$ (K km s$^{-1}$ pc$^2$)$^{-1}$)")

        _, _, bars = ax.errorbar(dist_lte_trot,aco_lte_trot,yerr=aco_lte_trot_err,fmt="o",c="tomato",capsize=0,alpha=0.5,markeredgewidth=0)
        [bar.set_alpha(0.5) for bar in bars]
        _, _, bars = ax.errorbar(dist_ism_trot,aco_ism_trot,yerr=aco_ism_trot_err,fmt="o",c="deepskyblue",capsize=0,alpha=0.5,markeredgewidth=0)
        [bar.set_alpha(0.5) for bar in bars]

        ax.plot(xlim, [np.log10(0.8),np.log10(0.8)], "k-", lw=3)
        ax.plot(xlim, [np.log10(4.3),np.log10(4.3)], "k-", lw=3)

        aco_lte_trot = 10**aco_lte_trot
        lte_p50 = np.percentile(aco_lte_trot,50)
        lte_p16 = str(np.round(lte_p50 - np.percentile(aco_lte_trot,16), 2))
        lte_p84 = str(np.round(np.percentile(aco_lte_trot,84) - lte_p50, 2))
        lte_p50 = str(np.round(lte_p50, 2))
        value = "$" + lte_p50 + "_{-"+lte_p16+"}^{+"+lte_p84+"}$"
        t=ax.text(0.05,0.90,r"$\alpha_{LTE}$ = "+value,color="tomato",transform=ax.transAxes)
        t.set_bbox(dict(facecolor="white", alpha=0.8, lw=0))

        aco_ism_trot = 10**aco_ism_trot
        lte_p50 = np.percentile(aco_ism_trot,50)
        lte_p16 = str(np.round(lte_p50 - np.percentile(aco_ism_trot,16), 2))
        lte_p84 = str(np.round(np.percentile(aco_ism_trot,84) - lte_p50, 2))
        lte_p50 = str(np.round(lte_p50, 2))
        value = "$" + lte_p50 + "_{-"+lte_p16+"}^{+"+lte_p84+"}$"
        t=ax.text(0.05,0.82,r"$\alpha_{ISM}$ = "+value,color="deepskyblue",transform=ax.transAxes)
        t.set_bbox(dict(facecolor="white", alpha=0.8, lw=0))

        ax.text(10*0.95,np.log10(4.3)+0.03,"MW value",color="black",ha="right",va="bottom")
        ax.text(10*0.95,np.log10(0.8)-0.03,"ULIRG value",color="black",ha="right",va="top")

        os.system("rm -rf " + self.outpng_aco_radial)
        plt.savefig(self.outpng_aco_radial, dpi=300)

        # plot hist
        histdata = np.histogram(aco_lte_trot, bins=25, range=[0.5,3.5])
        x1, y1 = histdata[1][:-1], histdata[0]/float(np.sum(histdata[0]))
        histdata = np.histogram(aco_ism_trot, bins=25, range=[0.5,3.5])
        x2, y2 = histdata[1][:-1], histdata[0]/float(np.sum(histdata[0]))

        plt.figure()
        plt.rcParams["font.size"] = 16
        plt.subplots_adjust(bottom = 0.15)
        gs = gridspec.GridSpec(nrows=30, ncols=30)
        ax = plt.subplot(gs[0:30,0:30])
        ax.grid(which="x")
        ax.set_xlim([0.1,3.3])
        ax.set_ylim([0,0.3])
        ax.set_xlabel(r"$\alpha_{CO}$ ($M_{\odot}$ (K km s$^{-1}$ pc$^2$)$^{-1}$)")

        #
        ax.bar(x1, y1+y2, lw=0, color="black", width=x1[1]-x1[0], alpha=0.5)
        
        popt, pcov = curve_fit(self._func, x1, y1+y2, p0=[0.3,1.5,0.5])
        best_fit   = self._func(x2, popt[0],popt[1],popt[2])
        ax.plot(x2, best_fit, "tomato", lw=5, alpha=0.5)
        ax.text(0.03, 0.92,
            r"best-fit $\mu$ = " + str(np.round(popt[1],2)) + r", $\sigma$ = " + str(np.round(popt[2],2)),
            color="black", transform=ax.transAxes)
        
        os.system("rm -rf " + self.outpng_aco_hist)
        plt.savefig(self.outpng_aco_hist, dpi=300)

    ###########
    # showhex #
    ###########

    def showhex(
        self,
        ):
        """
        Plot hex-sampled physical parameter maps.
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outtxt_hexphys,taskname)

        # import data
        data        = np.loadtxt(self.outtxt_hexphys)
        data_ra     = data[:,0] * 3600
        data_dec    = data[:,1] * 3600
        tkin        = data[:,2]
        nh2         = data[:,3]
        index       = data[:,4]
        err_index   = data[:,5]
        sfrd        = data[:,6] # err = 0.3dex
        sscd        = data[:,7]
        co10        = data[:,8]
        err_co10    = data[:,9]
        sfe         = data[:,10]
        aco         = data[:,13]

        # co10
        cut  = np.where(co10>0)
        X,Y,C = data_ra[cut],data_dec[cut],co10[cut]

        # spectral index
        cut = np.where((co10>0) & (index!=0))
        x,y,c,title = data_ra[cut],data_dec[cut],index[cut],"Spectral Index"
        self._plot_hexmap(self.outpng_hex_index,x,y,c,X,Y,C,title,title)

        # tkin
        cut = np.where((co10>0) & (tkin>0))
        x,y,c,title = data_ra[cut],data_dec[cut],tkin[cut],"Kinetic Temperature ($T_{kin}$)"
        self._plot_hexmap(self.outpng_hex_tkin,x,y,c,X,Y,C,title,"(K)")

        # nh2
        cut = np.where((co10>0) & (nh2>0))
        x,y,c,title = data_ra[cut],data_dec[cut],np.log10(nh2[cut]),"log H$_2$ Volume Density ($n_{H_2}$)"
        self._plot_hexmap(self.outpng_hex_nh2,x,y,c,X,Y,C,title,"log (cm$^{-3}$)",True)

        # sfr density
        cut = np.where((co10>0) & (sfrd>0))
        x,y,c,title = data_ra[cut],data_dec[cut],np.log10(sfrd[cut]),"log Extinction-corrected SFR Density ($\Sigma_{SFR}$)"
        self._plot_hexmap(self.outpng_hex_sfrd,x,y,c,X,Y,C,title,"log ($M_{\odot}$ kpc$^{-2}$ yr$^{-1}$)")

        # ssc density
        cut = np.where((co10>0) & (sscd>0.0))
        x,y,c,title = data_ra[cut],data_dec[cut],sscd[cut],"SSC Density ($\Sigma_{SSC}$)"
        # clip lowest sscd in order to change color range of the hex map.
        # this should not affect any other analysis.
        c[c<0.5] = 0.5
        self._plot_hexmap(self.outpng_hex_sscd,x,y,c,X,Y,C,title,"(kpc$^{-2}$)")

        # sfe
        cut = np.where((sfe>0) & (co10>np.percentile(co10,67)))
        x,y,c,title = data_ra[cut],data_dec[cut],np.log10(sfe[cut]/1e9),"log SFE"
        self._plot_hexmap(self.outpng_hex_sfe,x,y,c,X,Y,C,title,"log (yr$^{-1}$)")

        # aco
        cut = np.where((co10>0) & (aco>0))
        x,y,c,title = data_ra[cut],data_dec[cut],aco[cut],r"CO-to-H$_2$ Conversion Factor ($\alpha_{LTE}$)"
        # clip highest aco in order to change color range of the hex map.
        # this should not affect any other analysis.
        c[c>2.8] = 2.8
        self._plot_hexmap(self.outpng_hex_aco,x,y,c,X,Y,C,title,"($M_{\odot}$ (K km s$^{-1}$ pc$^2$)$^{-1}$)")

    ###############
    # plot_radial #
    ###############

    def plot_radial_ratio(
        self,
        ):
        """
        Plot radial CO line ratio distributions and their histograms with some stats.
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outtxt_hexdata,taskname)

        # import data
        data        = np.loadtxt(self.outtxt_hexdata)
        data_ra     = data[:,0] - self.ra
        data_dec    = data[:,1] - self.dec
        data_ra2    = ( data_ra*np.cos(self.pa) - data_dec*np.sin(self.pa) ) / np.cos(self.incl)
        data_dec2   = data_ra*np.sin(self.pa) + data_dec*np.cos(self.incl)
        data_12co10 = data[:,2]
        err_12co10  = data[:,3]
        data_12co21 = data[:,4]
        err_12co21  = data[:,5]
        data_13co10 = data[:,6]
        err_13co10  = data[:,7]
        data_13co21 = data[:,8]
        err_13co21  = data[:,9]

        # process data
        dist_kpc  = np.sqrt(data_ra2**2+data_dec2**2) * 3600 * self.scale_kpc
        
        # r21
        cut  = np.where((data_12co10!=0) & (data_12co21!=0))
        x    = data_12co21[cut]
        y    = data_12co10[cut]
        errx = err_12co21[cut]
        erry = err_12co10[cut]

        dist_r21 = dist_kpc[cut]
        data_r21 = x / y / (self.nu_12co21/self.nu_12co10)**2
        err_r21  = data_r21 * np.sqrt((errx/x)**2 + (erry/y)**2)

        # rt21
        cut  = np.where((data_13co10!=0) & (data_13co21!=0))
        x    = data_13co21[cut]
        y    = data_13co10[cut]
        errx = err_13co21[cut]
        erry = err_13co10[cut]
        
        dist_rt21 = dist_kpc[cut]
        data_rt21 = x / y / (self.nu_13co21/self.nu_13co10)**2
        err_rt21  = data_rt21 * np.sqrt((errx/x)**2 + (erry/y)**2)

        # r1213l
        cut  = np.where((data_12co10!=0) & (data_13co10!=0))
        x    = data_12co10[cut]
        y    = data_13co10[cut]
        errx = err_12co10[cut]
        erry = err_13co10[cut]
        
        dist_r1213l = dist_kpc[cut]
        data_r1213l = x / y / (self.nu_12co10/self.nu_13co10)**2
        err_r1213l  = data_r1213l * np.sqrt((errx/x)**2 + (erry/y)**2)

        # r1213h
        cut  = np.where((data_12co21!=0) & (data_13co21!=0))
        x    = data_12co21[cut]
        y    = data_13co21[cut]
        errx = err_12co21[cut]
        erry = err_13co21[cut]
        
        dist_r1213h = dist_kpc[cut]
        data_r1213h = x / y / (self.nu_12co21/self.nu_13co21)**2
        err_r1213h  = data_r1213h * np.sqrt((errx/x)**2 + (erry/y)**2)

        # plot
        self._plot_radial(
            self.outpng_radial_21,dist_r21,dist_rt21,data_r21,data_rt21,err_r21,err_rt21,
            comment1="$^{12}R_{21/10}$",comment2="$^{13}R_{21/10}$")

        self._plot_radial(
            self.outpng_radial_1213,dist_r1213l,dist_r1213h,data_r1213l,data_r1213h,err_r1213l,err_r1213h,
            ylim=[-0.5, 2.2],histrange=[0,50],comment1="$^{12/13}R_{10}$",comment2="$^{12/13}R_{21}$",
            xticks=[10,20,30,40,50],num_round=0)

    #####################
    # hex_sampling_phys #
    #####################

    def hex_sampling_phys(
        self,
        ):
        """
        Derive physical paramters using the hex-sampled catalog. See self.outtxt_hexdata.
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outtxt_hexdata,taskname)

        # import observed data
        data = np.loadtxt(self.outtxt_hexdata)
        data[:,0] = data[:,0] - self.ra
        data[:,1] = data[:,1] - self.dec
        data_tkin = np.loadtxt(self.table_tkin)
        data_nh2  = np.loadtxt(self.table_nh2)

        # physical paramters
        area_kpc = np.pi * (3./2. * self.scale_kpc)**2
        area_pc  = np.pi * (3./2. * self.scale_pc)**2
        tkin     = data_tkin[:,5]
        nh2      = data_nh2[:,5]

        # spectral index
        index     = np.log10(data[:,16]/data[:,14]) / np.log10(self.nu_b6/self.nu_b3)
        err_index = np.log(10) / np.log10(self.nu_b6/self.nu_b3)
        err_index = err_index * np.sqrt( (data[:,15]/data[:,14])**2 + (data[:,17]/data[:,16])**2 )

        # ssc density
        sscd     = data[:,19] / area_kpc / (1.99**2*np.pi/np.sqrt(4*np.log(2))/0.25**2) # divided by beam area

        # extinction-corrected sfr and sfr density
        l_halpha = data[:,18] * (36.5*4.*np.pi) * self.dist_cm**2 * 0.4**2
        beamcorr = 26.7658
        l_vla    = data[:,12] / beamcorr * 0.32**2 * 1.e-23 * (4.*np.pi*self.dist_cm**2)
        sfr      = (l_halpha + 0.39e+13 * l_vla) / 10. ** 41.27
        sfrd     = sfr / area_kpc

        # err_sfr = 0.3 dex of sfr

        # co luminosity
        beamarea = beam_area(self.outfits_m0_12co10)
        data_12co10_Jykms = data[:,2] / beamarea
        nu_obs_12co10 = self.nu_12co10 / (1+self.z)
        lumi_co10 = 3.25e+7 * data_12co10_Jykms / nu_obs_12co10**2 * \
            self.dist**2 / (1+self.z)**2

        err_12co10_Jykms = data[:,3] / beamarea
        err_lumi_co10 = 3.25e+7 * err_12co10_Jykms / nu_obs_12co10**2 * \
            self.dist**2 / (1+self.z)**2

        # sfe
        gmass = lumi_co10 * self.alpha_co
        sfe   = sfr / gmass * 10**9
        dh2   = gmass / area_pc

        err_gmass = err_lumi_co10 * self.alpha_co
        err_dh2   = err_gmass / area_pc

        # alpha_LTE
        nu_obs_13co21 = self.nu_13co21 / (1+self.z)
        kelvin_13co21 = data[:,8] * (1.222e6 / 2.0**2 / nu_obs_13co21**2)
        kelvin_13co21_err = data[:,9] * (1.222e6 / 2.0**2 / nu_obs_13co21**2)
        kelvin_12co10 = data[:,2] * (1.222e6 / 2.0**2 / nu_obs_12co10**2)
        kelvin_12co10_err = data[:,3] * (1.222e6 / 2.0**2 / nu_obs_12co10**2)

        list_alpha_lte_trot = []
        list_alpha_lte_tkin = []
        list_alpha_lte_trot_err = []
        list_alpha_lte_tkin_err = []
        Xco    = 3e-4
        Rcotco = 70
        X13co   = Xco / Rcotco

        for i in range(len(kelvin_13co21)):
            this_k_13co21 = kelvin_13co21[i]
            err_k_13co21 = kelvin_13co21_err[i]
            this_k_12co10 = kelvin_12co10[i]
            err_k_12co10 = kelvin_12co10_err[i]

            # a_lte(trot)
            logN_rot, Qrot = self._trot_from_rotation_diagram_13co(
                15.0, this_k_13co21, txtdata = self.key_qrot)
            N_tot = 10**logN_rot
            Xco = N_tot / X13co / this_k_12co10
            a_lte_trot = 4.3 * Xco / 2e+20
            list_alpha_lte_trot.append(a_lte_trot)

            print(np.log10(N_tot / X13co), a_lte_trot)

            # a_lte(trot) error
            N_tot_err = N_tot * err_k_13co21 / this_k_13co21
            Xco_err = Xco * np.sqrt((err_k_12co10/this_k_12co10)**2 + (N_tot_err/N_tot)**2)
            a_lte_trot_err = 4.3 * Xco_err / 2e+20
            list_alpha_lte_trot_err.append(a_lte_trot_err)

            # a_lte(tkin)
            logN_rot, Qrot = self._trot_from_rotation_diagram_13co(
                tkin[i], this_k_13co21, txtdata = self.key_qrot)
            N_tot = 10**logN_rot
            Xco = N_tot / X13co / this_k_12co10
            a_lte_tkin = 4.3 * Xco / 2e+20
            list_alpha_lte_tkin.append(a_lte_tkin)

            # a_lte(tkin) error
            N_tot_err = N_tot * err_k_13co21 / this_k_13co21
            Xco_err = Xco * np.sqrt((err_k_12co10/this_k_12co10)**2 + (N_tot_err/N_tot)**2)
            a_lte_tkin_err = 4.3 * Xco_err / 2e+20
            list_alpha_lte_tkin_err.append(a_lte_tkin_err)

        list_alpha_lte_trot = np.array(list_alpha_lte_trot)
        print(np.mean(list_alpha_lte_trot[list_alpha_lte_trot>0]))
        # alpha_ISM
        list_alpha_ism_trot = []
        list_alpha_ism_tkin = []
        list_alpha_ism_trot_err = []
        list_alpha_ism_tkin_err = []
        beamarea = beam_area(self.outfits_b6)

        for i in range(len(data[:,16])):
        	# a_ims(trot)
            factor     = self._factor_contin_to_ism_mass(15., self.dist, self.z)
            ism_mass   = data[:,16][i]/beamarea * factor
            a_ism_trot = ism_mass / lumi_co10[i]
            list_alpha_ism_trot.append(a_ism_trot)

            # a_ism(trot) err
            ism_mass_err = ism_mass * data[:,17][i] / data[:,16][i]
            a_ism_trot_err = a_ism_trot * np.sqrt((ism_mass_err/ism_mass)**2 + (err_lumi_co10[i]/lumi_co10[i])**2)
            list_alpha_ism_trot_err.append(a_ism_trot_err)

            # a_ism(tkin)
            factor     = self._factor_contin_to_ism_mass(tkin[i], self.dist, self.z)
            ism_mass   = data[:,16][i]/beamarea * factor
            a_ism_tkin = ism_mass / lumi_co10[i]
            list_alpha_ism_tkin.append(a_ism_tkin)

            # a_ism(tkin) err
            ism_mass_err = ism_mass * data[:,17][i] / data[:,16][i]
            a_ism_tkin_err = a_ism_trot * np.sqrt((ism_mass_err/ism_mass)**2 + (err_lumi_co10[i]/lumi_co10[i])**2)
            list_alpha_ism_tkin_err.append(a_ism_tkin_err)

        # combine
        data_science_ready = np.c_[
            data[:,0], # err = n/a
            data[:,1], # err = n/a
            tkin,
            nh2,
            index,
            err_index,
            sfrd, # err = 0.3 dex
            sscd, # err = n/a
            lumi_co10,
            err_lumi_co10,
            sfe, # err = 0.3 dex
            dh2,
            err_dh2,
            list_alpha_lte_trot,
            list_alpha_lte_tkin,
            list_alpha_ism_trot,
            list_alpha_ism_tkin,
            list_alpha_lte_trot_err,
            list_alpha_lte_tkin_err,
            list_alpha_ism_trot_err,
            list_alpha_ism_tkin_err,
            ]
        data_science_ready[np.isnan(data_science_ready)] = 0
        data_science_ready[np.isinf(data_science_ready)] = 0

        os.system("rm -rf " + self.outtxt_hexphys)
        fmt    = "%12.9f %10.7f %2.0f %7.0f %.2f %.2f %.4f %5.2f %9.0f %9.0f %7.4f %8.4f %8.4f %.2f %5.2f %5.2f %5.2f %.2f %5.2f %5.2f %5.2f"
        header = \
            "ra dec Tkin nH2 Index err Sig_SFR Sig_SSC Lco10 err SFE Sig_H2 err a_LTE_Trot a_LTE_Tkin a_ISM_Trot a_ISM_Tkin"
        np.savetxt(self.outtxt_hexphys, data_science_ready, fmt=fmt, header=header)

    #####################
    # hex_sampling_casa #
    #####################

    def hex_sampling_casa(
        self,
        ):
        """
        Hex-sample ALMA, OAO, VLA, and VLA SSC maps.
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outfits_m0_12co10,taskname)

        # create apertures with CASA region format
        self._create_casa_apertures(
            self.ra_blc,
            self.decl_blc,
            self.num_aperture,
            self.num_aperture,
            self.aperture_r,
            self.step,
            )
        casa_apertures = glob.glob(self.dir_casaregion + "*.region")
        casa_apertures.sort()

        total = str(self.num_aperture * self.num_aperture * 2)

        # measure rms
        rms_vla = measure_rms(self.outfits_vla)
        rms_b3 = measure_rms(self.outfits_b3_nopbcor)
        rms_b6 = measure_rms(self.outfits_b6_nopbcor)
        rms_halpha = 6.0e-19

        # sampling using CASA apertures
        os.system("rm -rf " + self.outtxt_hexdata)

        f = open(self.outtxt_hexdata, "a")
        f.write("#x y co10 err co21 err 13co10 err 13co21 err c18o21 err 1.45GHz err b3 err b6 err halpha(log err=0.3) ssc_density\n")
        f.close()

        for i, this_aperture in enumerate(casa_apertures):
            print("# get values at " + str(i+1) + "th aperture (total=" + total + ")")
            # measure fluxes and positions
            data_ra, data_dec = self._casa2radec(this_aperture)

            data_12co10 = self._eazy_imval(self.outfits_m0_12co10,this_aperture)
            err_12co10  = self._eazy_imval(self.outfits_em0_12co10,this_aperture)
            err_12co10  = np.sqrt(err_12co10**2 + (data_12co10*0.05)**2)

            data_12co21 = self._eazy_imval(self.outfits_m0_12co21,this_aperture)
            err_12co21  = self._eazy_imval(self.outfits_em0_12co21,this_aperture)
            err_12co21  = np.sqrt(err_12co21**2 + (data_12co21*0.10)**2)

            data_13co10 = self._eazy_imval(self.outfits_m0_13co10,this_aperture)
            err_13co10  = self._eazy_imval(self.outfits_em0_13co10,this_aperture)
            err_13co10  = np.sqrt(err_13co10**2 + (data_13co10*0.05)**2)

            data_13co21 = self._eazy_imval(self.outfits_m0_13co21,this_aperture)
            err_13co21  = self._eazy_imval(self.outfits_em0_13co21,this_aperture)
            err_13co21  = np.sqrt(err_13co21**2 + (data_13co21*0.10)**2)

            data_c18o21 = self._eazy_imval(self.outfits_m0_c18o21,this_aperture)
            err_c18o21  = self._eazy_imval(self.outfits_em0_c18o21,this_aperture)
            err_c18o21  = np.sqrt(err_c18o21**2 + (data_c18o21*0.10)**2)

            data_vla    = self._eazy_imval(self.outfits_vla,this_aperture,rms=rms_vla,roundval=5)
            err_vla     = np.sqrt(rms_vla**2 + (data_vla*0.03)**2)

            data_b3     = self._eazy_imval(self.outfits_b3,this_aperture,rms=rms_b3,snr=1,roundval=6)
            err_b3      = np.sqrt(rms_b3**2 + (data_b3*0.05)**2)

            data_b6     = self._eazy_imval(self.outfits_b6,this_aperture,rms=rms_b6,snr=1,roundval=6)
            err_b6      = np.sqrt(rms_b6**2 + (data_b6*0.05)**2)

            data_halpha = self._eazy_imval(self.outfits_halpha,this_aperture,rms=rms_halpha,roundval=20)

            data_ssc    = self._eazy_imval(self.outfits_ssc,this_aperture)

            # export to txt file
            f = open(self.outtxt_hexdata, "a")
            data = \
                str(data_ra).rjust(9) + " " + \
                str(data_dec).rjust(10) + " " + \
                str(data_12co10).rjust(6) + " " + \
                str(err_12co10).rjust(6) + " " + \
                str(data_12co21).rjust(6) + " " + \
                str(err_12co21).rjust(6) + " " + \
                str(data_13co10).rjust(5) + " " + \
                str(err_13co10).rjust(5) + " " + \
                str(data_13co21).rjust(6) + " " + \
                str(err_13co21).rjust(6) + " " + \
                str(data_c18o21).rjust(5) + " " + \
                str(err_c18o21).rjust(5) + " " + \
                str(data_vla).rjust(7) + " " + \
                str(err_vla).rjust(7) + " " + \
                str(data_b3).rjust(8) + " " + \
                str(err_b3).rjust(8) + " " + \
                str(data_b6).rjust(8) + " " + \
                str(err_b6).rjust(8) + " " + \
                str(data_halpha).rjust(9) + " " + \
                str(data_ssc).rjust(5)
            f.write(data + "\n")
            f.close()

        os.system("rm -rf " + self.dir_casaregion)

    ############
    # showcont #
    ############

    def showratio(
        self,
        ):
        """
        Plot line ratio maps. 
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outfits_irac,taskname)

        scalebar = 2000. / self.scale_pc
        label_scalebar = "2 kpc"

        # 12co21 12co10 ratio
        myfig_fits2png(
            imcolor=self.outfits_r_21,
            outfile=self.outpng_r_21,
            imcontour1=self.outfits_m0_12co10,
            imsize_as=self.imsize,
            ra_cnt=self.ra_str,
            dec_cnt=self.dec_str,
            levels_cont1=[0.01,0.02,0.04,0.08,0.16,0.32,0.64,0.96],
            width_cont1=[1.5],
            color_cont1="black",
            set_title="$^{12}$CO(2-1)/$^{12}$CO(1-0) Ratio ($^{12}R_{21/10}$)",
            colorlog=False,
            scalebar=scalebar,
            label_scalebar=label_scalebar,
            color_scalebar="black",
            set_cbar=True,
            label_cbar="Ratio",
            clim=[0.1,1.5],
            #numann="n3110_co_moms",
            )

        # 12co21 12co10 ratio
        myfig_fits2png(
            imcolor=self.outfits_r_t21,
            outfile=self.outpng_r_t21,
            imcontour1=self.outfits_m0_12co10,
            imsize_as=self.imsize,
            ra_cnt=self.ra_str,
            dec_cnt=self.dec_str,
            levels_cont1=[0.01,0.02,0.04,0.08,0.16,0.32,0.64,0.96],
            width_cont1=[1.5],
            color_cont1="black",
            set_title="$^{13}$CO(2-1)/$^{13}$CO(1-0) Ratio ($^{13}R_{21/10}$)",
            colorlog=False,
            scalebar=scalebar,
            label_scalebar=label_scalebar,
            color_scalebar="black",
            set_cbar=True,
            label_cbar="Ratio",
            clim=[0.1,1.5],
            #numann="n3110_co_moms",
            )

        # 12co10 13co10 ratio
        myfig_fits2png(
            imcolor=self.outfits_r_1213l,
            outfile=self.outpng_r_1213l,
            imcontour1=self.outfits_m0_12co10,
            imsize_as=self.imsize,
            ra_cnt=self.ra_str,
            dec_cnt=self.dec_str,
            levels_cont1=[0.01,0.02,0.04,0.08,0.16,0.32,0.64,0.96],
            width_cont1=[1.5],
            color_cont1="black",
            set_title="$^{12}$CO(1-0)/$^{13}$CO(1-0) Ratio ($^{12/13}R_{10}$)",
            colorlog=False,
            scalebar=scalebar,
            label_scalebar=label_scalebar,
            color_scalebar="black",
            set_cbar=True,
            label_cbar="Ratio",
            clim=[7.5,30.0],
            #numann="n3110_co_moms",
            )

        # 12co21 13co21 ratio
        myfig_fits2png(
            imcolor=self.outfits_r_1213h,
            outfile=self.outpng_r_1213h,
            imcontour1=self.outfits_m0_12co10,
            imsize_as=self.imsize,
            ra_cnt=self.ra_str,
            dec_cnt=self.dec_str,
            levels_cont1=[0.01,0.02,0.04,0.08,0.16,0.32,0.64,0.96],
            width_cont1=[1.5],
            color_cont1="black",
            set_title="$^{12}$CO(2-1)/$^{13}$CO(2-1) Ratio ($^{12/13}R_{21}$)",
            colorlog=False,
            scalebar=scalebar,
            label_scalebar=label_scalebar,
            color_scalebar="black",
            set_cbar=True,
            label_cbar="Ratio",
            clim=[7.5,30.0],
            #numann="n3110_co_moms",
            )

    ############
    # showcont #
    ############

    def showcont(
        self,
        ):
        """
        Plot ALMA continuum maps.
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outfits_irac,taskname)

        scalebar = 2000. / self.scale_pc
        label_scalebar = "2 kpc"

        # b3
        myfig_fits2png(
            imcolor=self.outfits_b3_nopbcor,
            outfile=self.outpng_b3,
            imcontour1=self.outfits_b3_nopbcor,
            imsize_as=self.imsize,
            ra_cnt=self.ra_str,
            dec_cnt=self.dec_str,
            unit_cont1=0.0418, # 1sigma level in Jy/beam
            levels_cont1=[-2,2,4,6,8,10],
            width_cont1=[1.5],
            color_cont1="black",
            set_title="2.9 mm Continuum",
            set_cmap="PuBu",
            colorlog=False,
            scalebar=scalebar,
            label_scalebar=label_scalebar,
            color_scalebar="black",
            set_cbar=True,
            label_cbar="mJy beam$^{-1}$",
            set_bg_color=cm.PuBu(0),
            clim=[0,0.50],
            #numann="n3110_co_moms",
            )

        # b6
        myfig_fits2png(
            imcolor=self.outfits_b6_nopbcor,
            outfile=self.outpng_b6,
            imcontour1=self.outfits_b6_nopbcor,
            imsize_as=self.imsize,
            ra_cnt=self.ra_str,
            dec_cnt=self.dec_str,
            unit_cont1=0.0627, # 1sigma level in Jy/beam
            levels_cont1=[-2.5,2.5,5.0,7.5,10.0,15.0,20.0],
            width_cont1=[1.5],
            color_cont1="black",
            set_title="1.3 mm Continuum",
            set_cmap="PuBu",
            colorlog=False,
            scalebar=scalebar,
            label_scalebar=label_scalebar,
            color_scalebar="black",
            set_cbar=True,
            label_cbar="mJy beam$^{-1}$",
            set_bg_color=cm.PuBu(0),
            clim=[0,1.70],
            #numann="n3110_co_moms",
            )

    ############
    # showline #
    ############

    def showline(
        self,
        ):
        """
        Plot ALMA line maps.
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outfits_irac,taskname)

        scalebar = 2000. / self.scale_pc
        label_scalebar = "2 kpc"
        scalebar_irac = 5000. / self.scale_pc
        label_scalebar_irac = "5 kpc"

        # irac and alma b3/b6 fov
        myfig_fits2png(
            imcolor=self.outfits_irac,
            outfile=self.outpng_irac,
            imcontour1=self.outfits_irac,
            imcontour2=self.outfits_pb_b3,
            imcontour3=self.outfits_pb_b6,
            imsize_as=self.imsize_irac,
            ra_cnt=self.ra_irac_str,
            dec_cnt=self.dec_irac_str,
            levels_cont1=[0.02,0.04,0.08,0.16,0.32,0.64,0.96],
            levels_cont2=[0.5],
            levels_cont3=[0.5],
            width_cont1=[1.0],
            color_cont1="black",
            width_cont2=[2.0],
            color_cont2="white",
            width_cont3=[2.0],
            color_cont3="white",
            set_title="IRAC 3.6 um",
            colorlog=False,
            scalebar=scalebar_irac,
            label_scalebar=label_scalebar_irac,
            color_scalebar="white",
            set_cbar=True,
            label_cbar="MJy sr$^{-1}$",
            clim=[0,30],
            numann="n3110_irac",
            textann=True,
            showbeam=False,
            )

        # 12co10
        myfig_fits2png(
            imcolor=self.outfits_m1_12co10,
            outfile=self.outpng_12co10,
            imcontour1=self.outfits_m0_12co10,
            imsize_as=self.imsize,
            ra_cnt=self.ra_str,
            dec_cnt=self.dec_str,
            levels_cont1=[0.01,0.02,0.04,0.08,0.16,0.32,0.64,0.96],
            width_cont1=[1.5],
            color_cont1="black",
            set_title="$^{12}$CO(1-0)",
            colorlog=False,
            scalebar=scalebar,
            label_scalebar=label_scalebar,
            color_scalebar="black",
            set_cbar=True,
            label_cbar="km s$^{-1}$",
            clim=[-245,245],
            numann="n3110_co_moms",
            textann=True,
            )

        # 12co21
        myfig_fits2png(
            imcolor=self.outfits_m1_12co21,
            outfile=self.outpng_12co21,
            imcontour1=self.outfits_m0_12co21,
            imsize_as=self.imsize,
            ra_cnt=self.ra_str,
            dec_cnt=self.dec_str,
            levels_cont1=[0.01,0.02,0.04,0.08,0.16,0.32,0.64,0.96],
            width_cont1=[1.5],
            color_cont1="black",
            set_title="$^{12}$CO(2-1)",
            colorlog=False,
            scalebar=scalebar,
            label_scalebar=label_scalebar,
            color_scalebar="black",
            set_cbar=True,
            label_cbar="km s$^{-1}$",
            clim=[-245,245],
            numann="n3110_co_moms",
            textann=False,
            )

        # 13co10
        myfig_fits2png(
            imcolor=self.outfits_m1_13co10,
            outfile=self.outpng_13co10,
            imcontour1=self.outfits_m0_13co10,
            imsize_as=self.imsize,
            ra_cnt=self.ra_str,
            dec_cnt=self.dec_str,
            levels_cont1=[0.02,0.04,0.08,0.16,0.32,0.64,0.96],
            width_cont1=[1.5],
            color_cont1="black",
            set_title="$^{13}$CO(1-0)",
            colorlog=False,
            scalebar=scalebar,
            label_scalebar=label_scalebar,
            color_scalebar="black",
            set_cbar=True,
            label_cbar="km s$^{-1}$",
            clim=[-245,245],
            numann="n3110_co_moms",
            textann=False,
            )

        # 13co21
        myfig_fits2png(
            imcolor=self.outfits_m1_13co21,
            outfile=self.outpng_13co21,
            imcontour1=self.outfits_m0_13co21,
            imsize_as=self.imsize,
            ra_cnt=self.ra_str,
            dec_cnt=self.dec_str,
            levels_cont1=[0.02,0.04,0.08,0.16,0.32,0.64,0.96],
            width_cont1=[1.5],
            color_cont1="black",
            set_title="$^{13}$CO(2-1)",
            colorlog=False,
            scalebar=scalebar,
            label_scalebar=label_scalebar,
            color_scalebar="black",
            set_cbar=True,
            label_cbar="km s$^{-1}$",
            clim=[-245,245],
            numann="n3110_co_moms",
            textann=False,
            )

        # c18o21
        myfig_fits2png(
            imcolor=self.outfits_m1_c18o21,
            outfile=self.outpng_c18o21,
            imcontour1=self.outfits_m0_c18o21,
            imsize_as=self.imsize,
            ra_cnt=self.ra_str,
            dec_cnt=self.dec_str,
            levels_cont1=[0.02,0.04,0.08,0.16,0.32,0.64,0.96],
            width_cont1=[1.5],
            color_cont1="black",
            set_title="C$^{18}$O(2-1)",
            colorlog=False,
            scalebar=scalebar,
            label_scalebar=label_scalebar,
            color_scalebar="black",
            set_cbar=True,
            label_cbar="km s$^{-1}$",
            clim=[-245,245],
            numann="n3110_co_moms",
            textann=False,
            )

    ##############
    # lineratios #
    ##############

    def lineratios(
        self,
        ):
        """
        create line ratio FITS maps.
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outfits_m0_12co10,taskname)

        # 12co21 12co10 ratio
        self._create_ratios(
            self.outfits_m0_12co21,
            self.outfits_m0_12co10,
            self.outfits_em0_12co21,
            self.outfits_em0_12co10,
            self.nu_12co21,
            self.nu_12co10,
            self.outfits_r_21,
            self.outfits_r_21.replace(".fits","_error.fits")
            )

        # 13co21 13co10 ratio
        self._create_ratios(
            self.outfits_m0_13co21,
            self.outfits_m0_13co10,
            self.outfits_em0_13co21,
            self.outfits_em0_13co10,
            self.nu_13co21,
            self.nu_13co10,
            self.outfits_r_t21,
            self.outfits_r_t21.replace(".fits","_error.fits")
            )

        # 12co21 13co21 ratio
        self._create_ratios(
            self.outfits_m0_12co21,
            self.outfits_m0_13co21,
            self.outfits_em0_12co21,
            self.outfits_em0_13co21,
            self.nu_12co21,
            self.nu_13co21,
            self.outfits_r_1213h,
            self.outfits_r_1213h.replace(".fits","_error.fits")
            )

        # 12co10 13co10 ratio
        self._create_ratios(
            self.outfits_m0_12co10,
            self.outfits_m0_13co10,
            self.outfits_em0_12co10,
            self.outfits_em0_13co10,
            self.nu_12co10,
            self.nu_13co10,
            self.outfits_r_1213l,
            self.outfits_r_1213l.replace(".fits","_error.fits")
            )

    ##############
    # align_maps #
    ##############

    def align_maps(
        self,
        ):
        """
        Covolve all the ALMA line cubes and cont maps to 2.0 arcsec resolution.
        Regrid all the datacubes to the grid of the 12CO(1-0) map (pixel=0.25 arcsec).
        Primary beam correction.
        Create moment maps after masking datacubes by 12CO(1-0)-SNR-based mask.
        Leave other facility data as is (i.e., different grid and resolution).
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.cube_12co10,taskname)

        #######################
        # moment map creation #
        #######################

        # import to casa
        run_importfits(self.map_halpha,self.map_halpha.replace(".fits",".image"))
        run_importfits(self.map_vla,self.map_vla.replace(".fits",".image"))

        # create 12co10-based cube mask
        run_roundsmooth(self.cube_12co10,self.cube_12co10+"_mask1",targetbeam=3.0)
        run_roundsmooth(self.cube_12co10,self.cube_12co10+"_mask2",targetbeam=5.5)
        signal_masking(self.cube_12co10,self.cube_12co10+"_mask3",0.0035)
        signal_masking(self.cube_12co10+"_mask1",self.cube_12co10+"_mask4",0.0045,delin=True)
        signal_masking(self.cube_12co10+"_mask2",self.cube_12co10+"_mask5",0.0105,delin=True)
        expr = "iif( IM0+IM1+IM2>=2.0, 1.0, 0.0 )"
        run_immath_three(
            self.cube_12co10+"_mask3",self.cube_12co10+"_mask4",self.cube_12co10+"_mask5",
            self.cube_12co10+"_mask",expr=expr,delin=True)

        remove_small_masks(self.cube_12co10+"_mask",None,self.cube_12co10,self.pixelmin)
        run_exportfits(self.cube_12co10+"_mask",self.cube_12co10+"_mask2",False,False,True)
        run_importfits(self.cube_12co10+"_mask2",self.cube_12co10+"_mask",True,True,["RA","Dec","1GHz","Stokes"])

        # create 13co21-based cube mask
        run_roundsmooth(self.cube_13co21,self.cube_13co21+"_mask1",targetbeam=3.0)
        run_roundsmooth(self.cube_13co21,self.cube_13co21+"_mask2",targetbeam=5.5)
        signal_masking(self.cube_13co21,self.cube_13co21+"_mask3",0.0035)
        signal_masking(self.cube_13co21+"_mask1",self.cube_13co21+"_mask4",0.0045,delin=True)
        signal_masking(self.cube_13co21+"_mask2",self.cube_13co21+"_mask5",0.0105,delin=True)
        expr = "iif( IM0+IM1+IM2>=2.0, 1.0, 0.0 )"
        run_immath_three(
            self.cube_13co21+"_mask3",self.cube_13co21+"_mask4",self.cube_13co21+"_mask5",
            self.cube_13co21+"_mask6",expr=expr,delin=True)
        expr = "iif( IM1>0, IM0, 0 )"
        run_immath_two(self.cube_13co21+"_mask6",self.cube_12co10+"_mask",
            self.cube_13co21+"_mask",expr)
        os.system("rm -rf " + self.cube_13co21+"_mask6")

        remove_small_masks(self.cube_13co21+"_mask",None,self.cube_13co21,self.pixelmin)
        run_exportfits(self.cube_13co21+"_mask",self.cube_13co21+"_mask2",False,False,True)
        run_importfits(self.cube_13co21+"_mask2",self.cube_13co21+"_mask",True,True,["RA","Dec","1GHz","Stokes"])

        # beam to 2.0 arcsec
        run_roundsmooth(self.cube_12co10,self.outfits_12co10+"_tmp1",targetbeam=self.beam)
        run_roundsmooth(self.cube_12co21,self.outfits_12co21+"_tmp1",targetbeam=self.beam)
        run_roundsmooth(self.cube_13co10,self.outfits_13co10+"_tmp1",targetbeam=self.beam)
        run_roundsmooth(self.cube_13co21,self.outfits_13co21+"_tmp1",targetbeam=self.beam)
        run_roundsmooth(self.cube_c18o21,self.outfits_c18o21+"_tmp1",targetbeam=self.beam)
        run_roundsmooth(self.cube_c18o21,self.outfits_c18o21+"_tmp1",targetbeam=self.beam)

        run_roundsmooth(self.map_b3,self.outfits_b3+"_tmp1",targetbeam=self.beam)
        run_roundsmooth(self.map_b6,self.outfits_b6+"_tmp1",targetbeam=self.beam)
        run_roundsmooth(self.map_ssc,self.outfits_ssc+"_tmp1",targetbeam=self.beam)

        # get rms
        rms_12co10 = measure_rms(self.outfits_12co10+"_tmp1")
        rms_12co21 = measure_rms(self.outfits_12co21+"_tmp1")
        rms_13co10 = measure_rms(self.outfits_13co10+"_tmp1")
        rms_13co21 = measure_rms(self.outfits_13co21+"_tmp1")
        rms_c18o21 = measure_rms(self.outfits_c18o21+"_tmp1")

        # pbcorr
        run_impbcor(self.outfits_12co10+"_tmp1",self.pb_12co10,self.outfits_12co10+"_tmp2",delin=True)
        run_impbcor(self.outfits_12co21+"_tmp1",self.pb_12co21,self.outfits_12co21+"_tmp2",delin=True)
        run_impbcor(self.outfits_13co10+"_tmp1",self.pb_13co10,self.outfits_13co10+"_tmp2",delin=True)
        run_impbcor(self.outfits_13co21+"_tmp1",self.pb_13co21,self.outfits_13co21+"_tmp2",delin=True)
        run_impbcor(self.outfits_c18o21+"_tmp1",self.pb_c18o21,self.outfits_c18o21+"_tmp2",delin=True)

        run_immath_one(self.pb_12co10,self.pb_12co10+"_tmp1_b3","IM0","10")
        run_immath_one(self.pb_12co21,self.pb_12co21+"_tmp1_b6","IM0","10")
        run_imregrid(self.pb_12co10+"_tmp1_b3",self.outfits_b3+"_tmp1",self.pb_12co10+"_tmp2_b3",delin=True)
        run_imregrid(self.pb_12co21+"_tmp1_b6",self.outfits_b6+"_tmp1",self.pb_12co21+"_tmp2_b6",delin=True)
        run_immath_one(self.outfits_b3+"_tmp1",self.outfits_b3+"_tmp2","IM0*1000.",delin=True)
        run_immath_one(self.outfits_b6+"_tmp1",self.outfits_b6+"_tmp2","IM0*1000.",delin=True)
        run_impbcor(self.outfits_b3+"_tmp2",self.pb_12co10+"_tmp2_b3",self.outfits_b3+"_tmp3",delin=False)
        run_impbcor(self.outfits_b6+"_tmp2",self.pb_12co21+"_tmp2_b6",self.outfits_b6+"_tmp3",delin=False)
        run_exportfits(self.outfits_b3+"_tmp2",self.outfits_b3_nopbcor,True,True,True)
        run_exportfits(self.outfits_b6+"_tmp2",self.outfits_b6_nopbcor,True,True,True)

        # casa to fits: co lines
        run_exportfits(self.outfits_12co10+"_tmp2",self.outfits_12co10,True,True,False)
        run_exportfits(self.outfits_12co21+"_tmp2",self.outfits_12co21,True,True,True)
        run_exportfits(self.outfits_13co10+"_tmp2",self.outfits_13co10,True,True,True)
        run_exportfits(self.outfits_13co21+"_tmp2",self.outfits_13co21,True,True,True)
        run_exportfits(self.outfits_c18o21+"_tmp2",self.outfits_c18o21,True,True,False)

        # casa to fits: alma continuum
        run_exportfits(self.outfits_b3+"_tmp3",self.outfits_b3,True,True,True)
        run_exportfits(self.outfits_b6+"_tmp3",self.outfits_b6,True,True,True)

        # casa to fits: vlt/naco k-band ssc catalogue
        self._align_one_map(self.outfits_ssc+"_tmp1",self.outfits_12co10+"_tmp2",
        	self.outfits_ssc+"_tmp2",axes=[0,1])
        run_exportfits(self.outfits_ssc+"_tmp2",self.outfits_ssc,True,True,True)
        os.system("rm -rf " + self.outfits_12co10+"_tmp1")

        # casa to fits: oao/halpha, vla/1.45, irac
        os.system("cp -r " + self.map_halpha + " " + self.outfits_halpha)
        os.system("cp -r " + self.map_vla + " " + self.outfits_vla)
        run_exportfits(self.map_irac,self.outfits_irac,True,True,False)

        # moment map (skip imrebin with factor=[2,2,1,1])
        self._create_moments(
            self.outfits_12co10,self.cube_12co10+"_mask",rms_12co10,
            self.outfits_m0_12co10,self.outfits_em0_12co10,self.outfits_m1_12co10, self.snr_mom_strong)

        self._create_moments(
            self.outfits_12co21,self.cube_12co10+"_mask",rms_12co21,
            self.outfits_m0_12co21,self.outfits_em0_12co21,self.outfits_m1_12co21, self.snr_mom_strong)

        self._create_moments(
            self.outfits_13co10,self.cube_13co21+"_mask",rms_13co10,
            self.outfits_m0_13co10,self.outfits_em0_13co10,self.outfits_m1_13co10, self.snr_mom_weak)

        self._create_moments(
            self.outfits_13co21,self.cube_13co21+"_mask",rms_13co21,
            self.outfits_m0_13co21,self.outfits_em0_13co21,self.outfits_m1_13co21, self.snr_mom_weak)

        self._align_one_map(self.cube_13co21+"_mask",self.outfits_c18o21+"_tmp2",
            self.cube_13co21+"_mask2",axes=-1)
        self._create_moments(
            self.outfits_c18o21,self.cube_13co21+"_mask2",rms_c18o21,
            self.outfits_m0_c18o21,self.outfits_em0_c18o21,self.outfits_m1_c18o21, self.snr_mom_weak)

        os.system("rm -rf " + self.cube_12co10 + "_mask")
        os.system("rm -rf " + self.cube_13co21 + "_mask")
        os.system("rm -rf " + self.cube_13co21 + "_mask2")
        os.system("rm -rf " + self.outfits_12co10 + "_tmp2")
        os.system("rm -rf " + self.outfits_c18o21 + "_tmp2")
        os.system("rm -rf " + self.dir_ready + "*_tmp1")

        # pb for figure 1 (irac map)
        run_imregrid(self.pb_12co10+"_tmp2_b3",self.map_irac,self.pb_12co10+"_tmp3_b3",delin=True)
        run_imregrid(self.pb_12co21+"_tmp2_b6",self.map_irac,self.pb_12co21+"_tmp3_b6",delin=True)
        run_exportfits(self.pb_12co10+"_tmp3_b3",self.outfits_pb_b3,True,True,True)
        run_exportfits(self.pb_12co21+"_tmp3_b6",self.outfits_pb_b6,True,True,True)

    #########
    # _func #
    #########

    def _func(self, x, a, mu, sigma):
        return a*np.exp(-(x-mu)**2/(2*sigma**2))

    ################
    # _plot_hexmap #
    ################

    def _plot_hexmap(
        self,
        outpng,
        x,y,c,
        X,Y,C,
        title,
        title_cbar,
        nH2=False,
        ):
        """
        """

        # set plt, ax
        plt.figure(figsize=(13,10))
        plt.rcParams["font.size"] = 16
        plt.subplots_adjust(bottom=0.10, left=0.19, right=0.99, top=0.90)
        gs = gridspec.GridSpec(nrows=10, ncols=10)
        ax = plt.subplot(gs[0:10,0:10])

        # set ax parameter
        myax_set(
        ax,
        grid="both",
        xlim=[32.5, -32.5],
        ylim=[-32.5, 32.5],
        title=title,
        xlabel="R.A. offset (arcsec)",
        ylabel="Decl. offset (arcsec)",
        )

        contour_levels = map(lambda x: x * np.max(C), [0.02,0.04,0.08,0.16,0.32,0.64,0.96])
        ax.tricontour(X, Y, C, colors=["black"], levels=contour_levels)
        cax = ax.scatter(x, y, s=270, c=c, cmap="rainbow", marker="h", linewidths=0)
        ax.set_aspect('equal', adjustable='box')

        # cbar
        cbar = plt.colorbar(cax)
        cbar.set_label(title_cbar)
        if nH2==True:
            cbar.set_ticks([2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5])
        cbar.outline.set_linewidth(2.5)

        # circle
        circ = patches.Ellipse(xy=(5.579989200043656, -22.764995999999726), width=self.r_speak_as*2, height=self.r_speak_as*2, angle=0, fill=False, edgecolor="tomato", alpha=1.0, lw=4)
        ax.add_patch(circ)

        os.system("rm -rf " + outpng)
        plt.savefig(outpng, dpi=300)

    ################
    # _plot_radial #
    ################

    def _plot_radial(
        self,
        outpng,
        dist1,
        dist2,
        value1,
        value2,
        err1,
        err2,
        xlim=[-1.3,1.4],
        ylim=[-2.0,0.7],
        histrange=[0,1.6],
        xticks=[0.4,0.8,1.2,1.6],
        num_round=2,
        comment1=None,
        comment2=None,
        ):
        """
        """

        histdata  = np.histogram(value1, bins=25, range=histrange)
        histx1, histy1 = histdata[1][:-1], histdata[0]/float(np.sum(histdata[0]))

        histdata = np.histogram(value2, bins=25, range=histrange)
        histx2, histy2 = histdata[1][:-1], histdata[0]/float(np.sum(histdata[0]))

        value1_p50 = np.percentile(value1,50)
        value2_p50 = np.percentile(value2,50)
        value1_p16 = value1_p50 - np.percentile(value1,16)
        value2_p16 = value2_p50 - np.percentile(value2,16)
        value1_p84 = np.percentile(value1,84) - value1_p50
        value2_p84 = np.percentile(value2,84) - value2_p50

        # plot
        plt.figure()
        plt.rcParams["font.size"] = 16
        plt.subplots_adjust(bottom = 0.15)
        gs  = gridspec.GridSpec(nrows=30, ncols=30)
        ax1 = plt.subplot(gs[0:30,0:30])
        ax1.set_aspect('equal', adjustable='box')
        ax2 = plt.axes([0,0,1,1])
        ip = InsetPosition(ax1, [0.15,0.13,0.5,0.33])
        ax2.set_axes_locator(ip)

        _, _, bars = ax1.errorbar(np.log10(dist1), np.log10(value1), yerr=1/np.log(10)*err1/value1, fmt="o", capsize=0, markersize=5, markeredgewidth=0, c="tomato", linewidth=1, alpha=0.5)
        [bar.set_alpha(0.5) for bar in bars]
        _, _, bars = ax1.errorbar(np.log10(dist2), np.log10(value2), yerr=1/np.log(10)*err2/value2, fmt="o", capsize=0, markersize=5, markeredgewidth=0, c="deepskyblue", linewidth=1, alpha=0.5)
        [bar.set_alpha(0.5) for bar in bars]

        ax2.bar(histx1, histy1, lw=0, color="tomato", width=histx1[1]-histx1[0], alpha=0.5)
        ax2.bar(histx2, histy2, lw=0, color="deepskyblue", width=histx2[1]-histx2[0], alpha=0.5)

        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        ax1.grid(which="both")
        ax1.set_xlabel("log Deprojected Distance (kpc)")
        ax1.set_ylabel("log Ratio")

        ax2.set_xlim(histrange)
        ax2.set_xticks(xticks)
        ax2.set_xlabel("Ratio")

        t=ax1.text(0.05,0.90,comment1,size=16,weight="bold",color="tomato", transform=ax1.transAxes)
        t.set_bbox(dict(facecolor="white", alpha=0.8, lw=0))
        t=ax1.text(0.05,0.82,comment2,size=16,weight="bold",color="deepskyblue", transform=ax1.transAxes)
        t.set_bbox(dict(facecolor="white", alpha=0.8, lw=0))

        if num_round!=0:
            text = "$" + str(np.round(value1_p50,num_round)).ljust(4,"0") + \
                "_{-" + str(np.round(value1_p16,num_round)).ljust(4,"0") + "}" + \
                "^{+" + str(np.round(value1_p84,num_round)).ljust(4,"0") + "}$"
        elif num_round==0:
            text = "$" + str(int(value1_p50)) + \
                "_{-" + str(int(value1_p16)) + "}" + \
                "^{+" + str(int(value1_p84)) + "}$"

        ax2.text(0.9,0.80,text,size=14,color="tomato",transform=ax2.transAxes,horizontalalignment='right')

        if num_round!=0:
            text = "$" + str(np.round(value2_p50,num_round)).ljust(4,"0") + \
                "_{-" + str(np.round(value2_p16,num_round)).ljust(4,"0") + "}" + \
                "^{+" + str(np.round(value2_p84,num_round)).ljust(4,"0") + "}$"
        elif num_round==0:
            text = "$" + str(int(value2_p50)) + \
                "_{-" + str(int(value2_p16)) + "}" + \
                "^{+" + str(int(value2_p84)) + "}$"

        ax2.text(0.9,0.60,text,size=14,color="deepskyblue",transform=ax2.transAxes,horizontalalignment='right')

        os.system("rm -rf " + outpng)
        plt.savefig(outpng, dpi=300)

    ##############################
    # _factor_contin_to_ism_mass #
    ##############################

    def _factor_contin_to_ism_mass(
    	self,
        Td,
        D_L, # Mpc
        z,
        ):
        """
        """

        h = 6.626e-27 # erg.s
        k = 1.38e-16 # erg/K
        nu_obs = 234.6075e+9 / (1 + z) #GHz
        nu_0   = 352.6970094e+9
        a850   = 6.7e+19

        factor = h * nu_obs * (1+z) / (k * Td)
        factor_0 = h * 352.6970094e+9 * (1+0) / (k * Td)
        gamma_rj = factor / (np.exp(factor) - 1)
        gamma_0 = factor_0 / (np.exp(factor_0) - 1)
        fac = 1.78 * (1+z)**-4.8 * (nu_obs/nu_0)**-3.8 * (6.7e+19/a850)
        fac = fac * gamma_0/gamma_rj * (D_L/1000.)**2 * 1e+10

        return fac

    ###################
    # _partition_func #
    ###################

    def _partition_func(self, Trot, datacol, txtdata = "Qrot_CDMS.txt"):
        """
        Derive partition funcition of a molecule at a given temperature
        using the CDMS table under LTE.  Interpolating 2 nearest values.
        (http://www.astro.uni-koeln.de/site/vorhersagen/catalog/
        partition_function.html)

        Parameters
        ----------
        Trot (K): int, float
            rotation temperature your molecule under LTE
        datacol: int
            column of "../Qrot_CDMS.txt" which contains descrete partition
            functions of your molecule
        data: str
            txt file containing partition function, otherwise use
            "../Qrot_CDMS.txt"

        Returns
        ----------
        Qrot: float
            derived partition function

        reference
        ----------
        Mueller, H. S. P. et al. 2001, A&A, 370, L49
        Mueller, H. S. P. et al. 2005, JMoSt, 742, 215
        """

        table = np.loadtxt(txtdata, usecols = (0,datacol))
        row = np.sum(table[:,0] < Trot) - 1
        t1 = table[:,0][row]
        t2 = table[:,0][row + 1]
        logQ1 = table[:,1][row]
        logQ2 = table[:,1][row + 1]
        a = (logQ1 - logQ2) / (t1 - t2)
        b = (t2*logQ1 - t1*logQ2) / (t2 - t1)
        Qrot = np.exp(a*Trot + b)

        return Qrot

    ####################################
    # _trot_from_rotation_diagram_13co #
    ####################################

    def _trot_from_rotation_diagram_13co(
        self,
        Trot,
        flux_hj,
        txtdata,
        ):
        """
        use equation 1 of Nakajima et al. 2018
        """

        # prepare
        k_B       = 1.38064852e-16 # erg.K^-1
        v_13co10  = 110.20135e9 # s^-1
        v_13co21  = 220.39868e9 # s^-1
        h_p       = 6.6260755e-27 # erg.s
        clight    = 3e5 # km.s^-1
        A_13co10  = 10**-7.198 # s^-1
        A_13co21  = 10**-6.216 # s^-1
        gu_13co10 = 3
        gu_13co21 = 5
        Eu_13co10 = 5.28880
        Eu_13co21 = 15.86618

        # item 1
        numer = 8 * np.pi * k_B * v_13co21**2 * flux_hj
        denom = h_p * clight**3 * A_13co21 * gu_13co21
        log_item1 = np.log10(numer/denom)

        # item 2
        log_item2 = Eu_13co21 * np.log10(np.e) / Trot

        # item 3
        Qrot      = self._partition_func(Trot, datacol=1, txtdata=txtdata)
        log_item3 = np.log10(Qrot)

        log_Ntot  = log_item1 + log_item2 + log_item3

        """
        # script before 1st referee comment
        k_B = 1.38064852e-16 # erg/K
        h_p = 6.6260755e-27 # erg.s
        Tbg = 2.73 # K
        Eu = {
            1: 5.28880,
            2: 15.86618,
            3: 31.73179,
            4: 52.88517,
            5: 79.32525,
            6: 111.05126,
            } # Eu = Eu/k [K]
        Snu2 = {
            1: 0.01220,
            2: 0.02436,
            3: 0.04869,
            4: 0.07297,
            5: 0.09717,
            6: 0.12124,
            7: 0.14518,
            } # Debye^2
        gl = 3 # http://akrmys.com/PhCh2011/docs/note/note03.pdf
        gu = 5

        clight = 2.99792e10 # cm/s

        Al = 6.294e-08
        Au = 6.038e-07

        #y_hj = 3 * k_B * flux_hj / (8 * np.pi * Snu2[hj_upp] * 110.20135 * hj_upp) * 1e32 # cm^2
        #b = np.log(y_hj) + Eu[hj_upp] / Trot
        Qrot = self._partition_func(Trot, datacol=1, txtdata=txtdata)
        exp_rot = np.exp(h_p * 110.20135e+9 * hj_upp / k_B / Trot) - 1.
        exp_bg = np.exp(h_p * 110.20135e+9 * hj_upp / k_B / Tbg) - 1.

        #log_Ntot = (b + Qrot - np.log(1 - (exp_rot / exp_bg))) / np.log(10)
        #Ntot = y_hj * Qrot / (1 - (exp_rot / exp_bg)) / np.exp(Eu[hj_upp]/Trot)
        #factor = (8 * np.pi**3 * Snu2[hj_upp] * 110.20135 * hj_upp) / (3 * k_B * Qrot) * 1e-32
        #factor = factor * (1 - (exp_rot / exp_bg)) * np.exp(-1 * Eu[hj_upp]/Trot)
        #Ntot = flux_hj / factor
        #log_Ntot = np.log10(Ntot)

        # gammaWg = gamma * W / gu (eq.23 of Goldsmith & Langer 1999)
        gammaWg  = flux_hj * 8 * np.pi * k_B * (110.20135 * hj_upp)**2
        gammaWg  = gammaWg / (gu * h_p * (clight)**3 * Au) * 1e23 # cm^-2
        log_Ntot = np.log10( gammaWg * Qrot * np.exp(Eu[hj_upp]/Trot) )
        """

        return round(log_Ntot, 2), round(Qrot, 2)

    ###############
    # _eazy_imval #
    ###############

    def _eazy_imval(
        self,
        imagename,
        casa_aperture,
        rms=0,
        snr=3,
        roundval=3,
        ):

        value        = imval(imagename=imagename,region=casa_aperture)
        value_masked = value["data"] * value["mask"]
        value_masked[np.isnan(value_masked)] = 0
        value_masked[np.isinf(value_masked)] = 0

        data         = value_masked.sum(axis = (0, 1))
        data_1d      = value_masked.flatten()
        num_all      = float(len(data_1d))
        num_detect   = len(data_1d[data_1d>rms*snr])

        if num_detect/num_all < 0.5:
            data = 0.0

        return np.round(data,roundval)

    ###############
    # _casa2radec #
    ###############

    def _casa2radec(self,casa_aperture):

        # import ra and dec
        f = open(casa_aperture)
        lines = f.readlines()
        f.close()
        str_xy = lines[3].replace("circle[[","").replace("deg","").replace("]","")
        data_ra = str_xy.split(",")[0]
        data_dec = str_xy.split(",")[1].replace(" ", "")

        return data_ra, data_dec

    ##########################
    # _create_casa_apertures #
    ##########################

    def _create_casa_apertures(
        self,
        ra_blc,
        decl_blc,
        numx,
        numy,
        aperture_r,
        step,
        ):
        """
        """

        step_ra   = step / 3600.
        step_decl = step / 3600. * np.sqrt(3)
        ra_deg    = ra_blc + step_ra
        decl_deg  = decl_blc - step_decl

        # replace with itertools.product(numx, numy)
        for i in range(numx):
            ra_deg   = ra_deg - step_ra
            ra_deg2  = ra_deg - step_ra / 2.
            decl_deg = decl_blc

            for j in range(numy):
                decl_deg = decl_deg + step_decl
                region_name = "_"+str(i).zfill(2)+"_"+str(j).zfill(2)+".region"

                # region A
                region_file = self.dir_casaregion + "A" + region_name
                f = open(region_file, "w")
                f.write("#CRTFv0\n")
                f.write("global coord=J2000\n")
                f.write("\n")
                f.write("circle[[" + str(round(ra_deg, 5)) + "deg, " + \
                    str(round(decl_deg,7)) + "deg], " + str(aperture_r) +"arcsec]")
                f.write("")
                f.close()

                # region B
                region_file2 = self.dir_casaregion + "B" + region_name
                decl_deg2 = decl_deg + step_decl / 2.
                f = open(region_file2, "w")
                f.write("#CRTFv0\n")
                f.write("global coord=J2000\n")
                f.write("\n")
                f.write("circle[[" + str(round(ra_deg2, 5)) + "deg, " + \
                    str(round(decl_deg2,7)) + "deg], " + str(aperture_r) +"arcsec]")
                f.write("")
                f.close()

    ##################
    # _create_ratios #
    ##################

    def _create_ratios(
        self,
        umap,
        lmap,
        uemap,
        lemap,
        ufreq,
        lfreq,
        outmap,
        outemap,
        snr=4.0,
        ):
        """
        """

        # ratio map
        expr = "IM0/IM1/" + str(ufreq)+"/"+str(ufreq)+"*"+str(lfreq)+"*"+str(lfreq)
        run_immath_two(umap,uemap,umap+"_tmp_masked","iif(IM0>IM1*"+str(snr)+",IM0,0)")
        run_immath_two(lmap,lemap,lmap+"_tmp_masked","iif(IM0>IM1*"+str(snr)+",IM0,0)")
        run_immath_two(umap+"_tmp_masked",lmap+"_tmp_masked",outmap+"_tmp1",expr,delin=True)
        run_immath_one(outmap+"_tmp1",outmap+"_tmp2","iif(IM0<100000,IM0,0)",delin=True)
        run_exportfits(outmap+"_tmp2",outmap,True,True,True)

        # ratio error map (only statistical error)
        expr = "IM0*sqrt(IM1+IM2)"
        run_immath_two(uemap,umap,"error1.map","iif(IM1>IM0*"+str(snr)+",IM0*IM0/IM1/IM1,0)")
        run_immath_two(lemap,lmap,"error2.map","iif(IM1>IM0*"+str(snr)+",IM0*IM0/IM1/IM1,0)")
        run_immath_three(outmap,"error1.map","error2.map",outemap+"_tmp1",expr)
        run_exportfits(outemap+"_tmp1",outemap,True,True,True)
        os.system("rm -rf error1.map error2.map")

    ###################
    # _create_moments #
    ###################

    def _create_moments(self, imagename, mask, rms, outmom0, outemom0, outmom1, snr_mom):
        """
        """

        print("# run _create_moments")
        expr = "iif( IM1>0, IM0, 0 )"

        run_exportfits(imagename,imagename+"_tmp1",False,False,False)
        run_importfits(imagename+"_tmp1",imagename+"_tmp2",True,True,["RA","Dec","1GHz","Stokes"])

        # mom0
        run_immoments(imagename+"_tmp2",mask,outmom0+"_tmp1",0,rms,snr_mom,outemom0+"_tmp1",vdim=3)
        run_exportfits(outmom0+"_tmp1",outmom0+"_tmp2",False,False,True)
        run_importfits(outmom0+"_tmp2",outmom0+"_tmp1",True,True,["RA","Dec","1GHz","Stokes"])
        run_exportfits(outemom0+"_tmp1",outemom0+"_tmp2",False,False,True)
        run_importfits(outemom0+"_tmp2",outemom0+"_tmp1",True,True,["RA","Dec","1GHz","Stokes"])

        signal_masking(outmom0+"_tmp1",outmom0+"_tmp2",0,delin=False)

        remove_small_masks(outmom0+"_tmp2",None,outmom0+"_tmp1",self.pixelmin)

        os.system("rm -rf this_mask.image")
        os.system("cp -r " + outmom0 + "_tmp2 this_mask.image")
        os.system("rm -rf " + outmom0 + "_tmp2")

        os.system("rm -rf " + outmom0 + "_tmp3")
        immath(
            imagename = [outmom0+"_tmp1","this_mask.image"],
            expr      = "IM0*IM1",
            outfile   = outmom0 + "_tmp3",
            mask      = 'mask("this_mask.image")',
            )
        os.system("rm -rf " + outemom0 + "_tmp3")
        immath(
            imagename = [outemom0+"_tmp1","this_mask.image"],
            expr      = "IM0*IM1",
            outfile   = outemom0 + "_tmp3",
            mask      = 'mask("this_mask.image")',
            )
        os.system("rm -rf " + outmom0 + "_tmp1")
        os.system("rm -rf " + outemom0 + "_tmp1")

        run_exportfits(outmom0+"_tmp3",outmom0,True,True,True)
        run_exportfits(outemom0+"_tmp3",outemom0,True,True,True)

        # mom1
        run_immoments(imagename+"_tmp2",mask,outmom1+"_tmp1",1,rms,snr_mom,vdim=3)
        os.system("rm -rf " + outmom1 + "_tmp3")
        immath(
            imagename = [outmom1+"_tmp1","this_mask.image"],
            expr      = "IM0*IM1",
            outfile   = outmom1 + "_tmp3",
            mask      = 'mask("this_mask.image")',
            )
        run_exportfits(outmom1+"_tmp3",outmom1,True,True,True)

        os.system("rm -rf " + imagename + "_tmp2")
        os.system("rm -rf this_mask.image")

    ##################
    # _align_one_map #
    ##################

    def _align_one_map(self, imagename, template, outfits, axes=-1):
        """
        """

        delim  = False
        deltmp = False

        # make sure casa image
        if imagename[-5:]==".fits":
            run_importfits(
                fitsimage   = imagename,
                imagename   = imagename.replace(".fits",".image"),
                defaultaxes = True,
                defaultaxesvalues = ["RA","Dec","1GHz","Stokes"],
                )
            imagename = imagename.replace(".fits",".image")
            delim     = True

        # make sure casa image
        if template[-5:]==".fits":
            run_importfits(
                fitsimage   = template,
                imagename   = template.replace(".fits",".image"),
                defaultaxes = True,
                defaultaxesvalues = ["RA","Dec","1GHz","Stokes"],
                )
            template = template.replace(".fits",".image")
            deltmp   = True

        # regrid
        run_imregrid(
            imagename = imagename,
            template  = template,
            outfile   = imagename + ".regrid",
            axes      = axes,
            )

        # exportfits
        run_exportfits(
            imagename = imagename + ".regrid",
            fitsimage = outfits,
            dropdeg    = True,
            dropstokes = True,
            delin      = True,
            )

        # delete
        os.system("rm -rf " + imagename + ".regrid")
        os.system("rm -rf " + imagename)

        if delim==True:
            os.system("rm -rf " + imagename)

        if deltmp==True:
            os.system("rm -rf " + template)

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

