"""
Class to analyze Cycle 2 ALMA CO line datasets toward NGC 3110

history:
2016-04-01   start project with Kawana-san, Okumura-san, and Kawabe-san
2021-06-07   start re-analysis, write this README
2021-06-08   start to create paper-ready figures
2021-06-09   start up-dating the draft
2021-06-11   circulate v2 draft to the whole team
2021-06-28   move to ADC because of issues with new laptop
2021-07-02   submit to ApJ!
2021-09-01   major update in order to revise the draft
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

            self.map_ssc = dir_raw + self._read_key("map_ssc")
            self.map_halpha = dir_raw + self._read_key("map_halpha")
            self.map_vla = dir_raw + self._read_key("map_vla")
            self.map_irac = dir_raw + self._read_key("map_irac")

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

            self.outfits_ssc = self.dir_ready + self._read_key("outfits_ssc")
            self.outfits_halpha = self.dir_ready + self._read_key("outfits_halpha")
            self.outfits_vla = self.dir_ready + self._read_key("outfits_vla")
            self.outfits_irac = self.dir_ready + self._read_key("outfits_irac")
            self.outfits_pb_b3 = self.dir_ready + self._read_key("outfits_pb_b3")
            self.outfits_pb_b6 = self.dir_ready + self._read_key("outfits_pb_b6")

            self.outfits_r_21 = self.dir_ready + self._read_key("outfits_r_21")
            self.outfits_r_t21 = self.dir_ready + self._read_key("outfits_r_t21")
            self.outfits_r_1213l = self.dir_ready + self._read_key("outfits_r_1213l")
            self.outfits_r_1213h = self.dir_ready + self._read_key("outfits_r_1213h")

            # ngc3110 properties
            self.ra_str = self._read_key("ra", "gal")
            self.ra = float(self.ra_str.replace("deg",""))
            self.dec_str = self._read_key("dec", "gal")
            self.dec = float(self.dec_str.replace("deg",""))

            self.ra_irac_str = self._read_key("ra_irac", "gal")
            self.ra_irac = float(self.ra_irac_str.replace("deg",""))
            self.dec_irac_str = self._read_key("dec_irac", "gal")
            self.dec_irac = float(self.dec_irac_str.replace("deg",""))

            self.scale_pc = float(self._read_key("scale", "gal"))
            self.scale_kpc = float(self._read_key("scale", "gal")) / 1000.

            # input parameters
            self.beam = float(self._read_key("beam"))
            self.snr_mom = float(self._read_key("snr_mom"))
            self.imsize = int(self._read_key("imsize"))
            self.imsize_irac = int(self._read_key("imsize_irac"))
            self.pixelmin = 2.0

            # output txt and png
            self.outpng_irac = self.dir_products + self._read_key("outpng_irac")
            self.outpng_12co10 = self.dir_products + self._read_key("outpng_12co10")
            self.outpng_12co21 = self.dir_products + self._read_key("outpng_12co21")
            self.outpng_13co10 = self.dir_products + self._read_key("outpng_13co10")
            self.outpng_13co21 = self.dir_products + self._read_key("outpng_13co21")
            self.outpng_c18o21 = self.dir_products + self._read_key("outpng_c18o21")

            self.outpng_b3 = self.dir_products + self._read_key("outpng_b3")
            self.outpng_b6 = self.dir_products + self._read_key("outpng_b6")

    ##################
    # run_ngc3110_co #
    ##################

    def run_ngc3110_co(
        self,
        do_prepare    = False,
        do_lineratios = False,
        plot_showcase = False,
        ):

        if do_prepare==True:
            self.align_maps()

        if do_lineratios==True:
        	self.lineratios()

        if plot_showcase==True:
            self.showline()
            self.showcont()
            #self.showratio()

    ##############
    # lineratios #
    ##############

    def lineratios(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outfits_m0_12co10,taskname)

        nu_12co10 = 115.27120180
        nu_13co10 = 110.20135430
        nu_12co21 = 230.53800000
        nu_13co21 = 220.39868420

        # 12co21 12co10 ratio
        self._create_ratios(
            self.outfits_m0_12co21,
            self.outfits_m0_12co10,
            self.outfits_em0_12co21,
            self.outfits_em0_12co10,
            nu_12co21,
            nu_12co10,
            self.outfits_r_21,
            self.outfits_r_21.replace(".fits","_error.fits")
            )

        # 13co21 13co10 ratio
        self._create_ratios(
            self.outfits_m0_13co21,
            self.outfits_m0_13co10,
            self.outfits_em0_13co21,
            self.outfits_em0_13co10,
            nu_13co21,
            nu_13co10,
            self.outfits_r_t21,
            self.outfits_r_t21.replace(".fits","_error.fits")
            )

        # 12co21 13co21 ratio
        self._create_ratios(
            self.outfits_m0_12co21,
            self.outfits_m0_13co21,
            self.outfits_em0_12co21,
            self.outfits_em0_13co21,
            nu_12co21,
            nu_13co21,
            self.outfits_r_1213h,
            self.outfits_r_1213h.replace(".fits","_error.fits")
            )

        # 12co10 13co10 ratio
        self._create_ratios(
            self.outfits_m0_12co10,
            self.outfits_m0_13co10,
            self.outfits_em0_12co10,
            self.outfits_em0_13co10,
            nu_12co10,
            nu_13co10,
            self.outfits_r_1213l,
            self.outfits_r_1213l.replace(".fits","_error.fits")
            )

    ############
    # showcont #
    ############

    def showcont(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outfits_irac,taskname)

        scalebar = 2000. / self.scale_pc
        label_scalebar = "2 kpc"

        # b3
        myfig_fits2png(
            imcolor=self.outfits_b3,
            outfile=self.outpng_b3,
            imcontour1=self.outfits_b3,
            imsize_as=self.imsize,
            ra_cnt=self.ra_str,
            dec_cnt=self.dec_str,
            unit_cont1=0.0418, # 1sigma level in Jy/beam
            levels_cont1=[-2.5,2.5,5.0,7.5,10.0],
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
            )

        # b6
        myfig_fits2png(
            imcolor=self.outfits_b6,
            outfile=self.outpng_b6,
            imcontour1=self.outfits_b6,
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
            )

    ############
    # showline #
    ############

    def showline(
        self,
        ):
        """
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
            numann=2,
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
            levels_cont1=[0.02,0.04,0.08,0.16,0.32,0.64,0.96],
            width_cont1=[1.5],
            color_cont1="black",
            set_title="$^{12}$CO(1-0)",
            colorlog=False,
            scalebar=scalebar,
            label_scalebar=label_scalebar,
            color_scalebar="black",
            set_cbar=True,
            label_cbar="km s$^{-1}$",
            clim=[-250,250],
            )

        # 12co21
        myfig_fits2png(
            imcolor=self.outfits_m1_12co21,
            outfile=self.outpng_12co21,
            imcontour1=self.outfits_m0_12co21,
            imsize_as=self.imsize,
            ra_cnt=self.ra_str,
            dec_cnt=self.dec_str,
            levels_cont1=[0.02,0.04,0.08,0.16,0.32,0.64,0.96],
            width_cont1=[1.5],
            color_cont1="black",
            set_title="$^{12}$CO(2-1)",
            colorlog=False,
            scalebar=scalebar,
            label_scalebar=label_scalebar,
            color_scalebar="black",
            set_cbar=True,
            label_cbar="km s$^{-1}$",
            clim=[-250,250],
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
            clim=[-250,250],
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
            clim=[-250,250],
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
            clim=[-250,250],
            )

    ##############
    # align_maps #
    ##############

    def align_maps(
        self,
        ):
        """
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
        run_impbcor(self.outfits_b3+"_tmp1",self.pb_12co10+"_tmp2_b3",self.outfits_b3+"_tmp2",delin=True)
        run_impbcor(self.outfits_b6+"_tmp1",self.pb_12co21+"_tmp2_b6",self.outfits_b6+"_tmp2",delin=True)
        run_immath_one(self.outfits_b3+"_tmp2",self.outfits_b3+"_tmp3","IM0/1000.",delin=True)
        run_immath_one(self.outfits_b6+"_tmp2",self.outfits_b6+"_tmp3","IM0/1000.",delin=True)

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
            self.outfits_m0_12co10,self.outfits_em0_12co10,self.outfits_m1_12co10)

        self._create_moments(
            self.outfits_12co21,self.cube_12co10+"_mask",rms_12co21,
            self.outfits_m0_12co21,self.outfits_em0_12co21,self.outfits_m1_12co21)

        self._create_moments(
            self.outfits_13co10,self.cube_13co21+"_mask",rms_13co10,
            self.outfits_m0_13co10,self.outfits_em0_13co10,self.outfits_m1_13co10)

        self._create_moments(
            self.outfits_13co21,self.cube_13co21+"_mask",rms_13co21,
            self.outfits_m0_13co21,self.outfits_em0_13co21,self.outfits_m1_13co21)

        self._align_one_map(self.cube_13co21+"_mask",self.outfits_c18o21+"_tmp2",
            self.cube_13co21+"_mask2",axes=-1)
        self._create_moments(
            self.outfits_c18o21,self.cube_13co21+"_mask2",rms_c18o21,
            self.outfits_m0_c18o21,self.outfits_em0_c18o21,self.outfits_m1_c18o21)

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

    def _create_moments(self, imagename, mask, rms, outmom0, outemom0, outmom1):
        """
        """
        print("# run _create_moments")
        expr = "iif( IM1>0, IM0, 0 )"

        run_exportfits(imagename,imagename+"_tmp1",False,False,False)
        run_importfits(imagename+"_tmp1",imagename+"_tmp2",True,True,["RA","Dec","1GHz","Stokes"])

        # mom0
        run_immoments(imagename+"_tmp2",mask,outmom0+"_tmp1",0,rms,self.snr_mom,outemom0+"_tmp1",vdim=3)
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
        run_immoments(imagename+"_tmp2",mask,outmom1+"_tmp1",1,rms,self.snr_mom,vdim=3)
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
