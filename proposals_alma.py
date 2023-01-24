"""
Class for ALMA proposals.

requirements:
CASA Version 5.4.0-70, ananlysisUtils, astropy

usage:
see "def _set_cycle_8p5a_specscan" for example

ALMA timeline:
cycle    release date         deadline
C8supp   2021-08-08 15:00UT   2021-10-06 15:00UT
outcome  2021-12-16 15:59UT
C10main  2023-05-?? 15:00UT   2023-??-?? 15:00UT

Submitted project:
cycle   project code    outcome  rank
c8supp  2021.2.00049.S  grade C  1,1,1,1,2,2,3,4,4,6
c10     N/A             N/A      N/A

history:
2021-09-21   start Cycle 8.5
2021-10-05   submit the 8p5a proposal (see run_cycle_8p5a_specscan)
2023-01-23   start Cycle 10
Toshiki Saito@Nichidai/NAOJ
"""

import os, sys, glob, csv
import numpy as np

from mycasa_sampling import *
from mycasa_tasks import *
from mycasa_plots import *

myia = aU.createCasaTool(iatool)

#################
# ProposalsALMA #
#################
class ProposalsALMA():
    """
    Class for ALMA proposals.
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
        self.keyfile_gal  = keyfile_gal
        self.keyfile_fig  = keyfile_fig

        # intialize task
        self.refresh      = refresh
        self.delete_inter = delete_inter
        self.taskname     = None
        self.fig_dpi      = 500

        # import parameters
        if keyfile_fig is not None:
            self.modname = "ProposalsALMA."

            # get alma cycle
            self.cycle = self._read_key("this_cycle")
            
            # set directories
            self._set_dir()

            # cycle 10
            if self.cycle=="cycle10":
                self._set_cycle_10()

            # cycle 9
            if self.cycle=="cycle09":
                self._set_cycle_9()

            # cycle 8p5
            if self.cycle=="cycle08p5":
                self._set_cycle_8p5a_specscan()
                self._set_cycle_8p5b_catom21()

    ##############################################################################################
    ##############################################################################################
    ##################################                         ###################################
    ################################## ALMA cycle 10 main call ###################################
    ##################################                         ###################################
    ##############################################################################################
    ##############################################################################################

    #################
    # _set_cycle_10 #
    #################

    def _set_cycle_10(
        self,
        ):
        """
        import os
        from proposals_alma import ProposalsALMA as tools
        
        dir_proj = /home02/saitots/myUtils/keys_alma_proposal/
        
        ### Cycle 10 main call
        # 2022.1.?????.S
        # [explain]
        tl = tools(
            refresh     = False,
            keyfile_fig = dir_proj + "key_cyle09.txt",
            )
        tl.run_cycle_10(
            ??? = True,
            )
        
        os.system("rm -rf *.last")
        """

        # input data
        self.image_12co10       = self.dir_raw + self._read_key("image_12co10")
        self.image_13co10       = self.dir_raw + self._read_key("image_13co10")
        self.image_h13cn        = self.dir_raw + self._read_key("image_h13cn")
        self.image_oiiioii      = self.dir_raw + self._read_key("image_oiiioii")
        self.image_13co10_150pc = self.dir_raw + self._read_key("image_13co10_150pc")
        self.image_ch3oh_150pc  = self.dir_raw + self._read_key("image_ch3oh_150pc")

        # ngc1068
        self.z             = float(self._read_key("z"))
        self.scale         = float(self._read_key("scale"))
        self.ra            = self._read_key("ra")
        self.dec           = self._read_key("dec")
        self.ra_agn        = self._read_key("ra_agn")
        self.dec_agn       = self._read_key("dec_agn")

        # output fits
        self.outfits_mask  = self.dir_ready + self._read_key("outfits_mask")

        # output png
        self.png_mask_map  = self.dir_products + self._read_key("png_mask_map")
        self.imsize_as     = float(self._read_key("imsize_as"))

        # final products
        self.final_mask    = self.dir_final + self._read_key("final_mask")

    ################
    # run_cycle_10 #
    ################

    def run_cycle_10_band1(
        self,
        plot_mask_map = False,
        ):
        """
        Content: Band 1 spectral scans with 12m+7m toward NGC 1068.
        Status:  preparing
        Code:    N/A
        Outcome: N/A
        Rank:    N/A
        Note:    N/A
        """

        if plot_mask_map==True:
            self.c10_plot_mask_map()

    #####################
    # c10_plot_mask_map #
    #####################

    def c10_plot_mask_map(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.image_12co10,taskname)

        # param
        scalebar = 500 / self.scale

        # regrid
        os.system("rm -rf template.image")
        template = "template.image" 
        run_importfits(self.image_12co10,template)
        run_importfits(self.image_oiiioii,self.image_oiiioii+"_tmp1")
        run_exportfits(self.image_oiiioii+"_tmp1",self.image_oiiioii+"_tmp2.fits",True,True,True)
        run_imregrid(self.image_13co10,               template, self.image_13co10+"_regrid",       delin=False)
        run_imregrid(self.image_13co10_150pc,         template, self.image_13co10_150pc+"_regrid", delin=False)
        run_imregrid(self.image_ch3oh_150pc,          template, self.image_ch3oh_150pc+"_regrid",  delin=False)
        run_imregrid(self.image_h13cn,                template, self.image_h13cn+"_regrid",        delin=False)
        run_imregrid(self.image_oiiioii+"_tmp2.fits", template, self.image_oiiioii+"_regrid",      delin=True)

        # CH3OH/13CO line ratio
        run_immath_two(self.image_ch3oh_150pc+"_regrid",self.image_13co10_150pc+"_regrid",self.outfits_mask+"_ch3oh_13co","iif(IM1>0.2,IM0/IM1,0)",delin=False)
        os.system("rm -rf " + self.image_ch3oh_150pc + "_regrid")

        # smooth OIII/OII ratio
        run_immath_one(self.image_oiiioii+"_regrid",self.image_oiiioii+"_regrid2","iif(IM0>=2.2,1,0)",delin=True)
        run_roundsmooth(self.image_oiiioii+"_regrid2",self.image_oiiioii+"_regrid3",2.4,inputbeam=0.8,delin=True)
        run_immath_one(self.image_oiiioii+"_regrid3",self.image_oiiioii+"_regrid4","iif(IM0>=0.3,1,0)",delin=True)

        # smooth 12CO
        run_immath_one(template,self.outfits_mask+"_tmp1","iif(IM0>=1,1,0)",delin=False)
        os.system("rm -rf template.image")
        run_roundsmooth(self.outfits_mask+"_tmp1",self.outfits_mask+"_tmp1b",2.4,inputbeam=0.8,delin=True)
        run_immath_one(self.outfits_mask+"_tmp1b",self.outfits_mask+"_tmp1c","iif(IM0>=0.5,1,0)",delin=True)

        # masking
        run_immath_two(self.image_13co10+"_regrid",self.outfits_mask+"_tmp1c",self.outfits_mask+"_tmp2","iif(IM0>=20,2,IM1)",delin=True)
        run_immath_two(self.outfits_mask+"_ch3oh_13co",self.outfits_mask+"_tmp2",self.outfits_mask+"_tmp3","iif(IM0>=0.15,3,IM1)",delin=False)
        run_immath_two(self.image_oiiioii+"_regrid4",self.outfits_mask+"_tmp3",self.outfits_mask+"_tmp4","iif(IM0==1,4,IM1)",delin=True)
        run_immath_two(self.image_h13cn+"_regrid",self.outfits_mask+"_tmp4",self.outfits_mask+"_tmp5","iif(IM0>=11,5,IM1)",delin=True)
        run_exportfits(self.outfits_mask+"_tmp5",self.outfits_mask,True,True,True)

        # plot
        myfig_fits2png(
            # general
            self.outfits_mask,
            self.png_mask_map,
            imsize_as=self.imsize_as,
            ra_cnt=self.ra_agn,
            dec_cnt=self.dec_agn,
            # imshow
            fig_dpi=self.fig_dpi,
            set_grid=None,
            set_title="NGC 1068 environments",
            clim=[1,5],
            set_cmap="Set1_r",
            showzero=False,
            showbeam=False,
            scalebar=scalebar,
            label_scalebar="0.5 kpc",
            color_scalebar="black",
            # annotation
            #numann=4,
            #textann=False,
            )

    ############################################################################################
    ############################################################################################
    ##############################                                ##############################
    ############################## ALMA cycle 8 supplemental call ##############################
    ##############################                                ##############################
    ############################################################################################
    ############################################################################################

    ##################
    # _set_cycle_8p5 #
    ##################

    def _set_cycle_8p5a_specscan(
        self,
        ):
        """
        import os
        from proposals_alma import ProposalsALMA as tools
        
        dir_proj = /home02/saitots/myUtils/keys_alma_proposal/
        
        ### Cycle 8 supplemental call
        # 2021.2.00049.S
        # Band 3 and 6 spectral scans with 7m+TP toward NGC 1068.
        tl = tools(
            refresh           = False,
            keyfile_fig       = dir_proj + "key_cyle08p5.txt",
            )
        tl.run_cycle_8p5a_specscan(
            plot_spw_setup    = True,
            plot_missingflux  = True,
            plot_proposed_fov = True,
            combine_figures   = True,
            )
        
        os.system("rm -rf *.last")
        """

        # input data
        self.image_co10_12m7m    = dir_raw + self._read_key("image_co10_12m7m")
        self.image_co10_12m      = dir_raw + self._read_key("image_co10_12m")
        self.image_cs21          = dir_raw + self._read_key("image_cs21")
        self.archive_csv         = dir_raw + self._read_key("archive_csv")
        self.txt_fov_b3          = dir_raw + self._read_key("fov_b3")
        self.txt_fov_b6          = dir_raw + self._read_key("fov_b6")

        # spectral scan setup
        self.line_key            = self.dir_proj + "scripts/keys/key_lines.txt"
        l                        = self._read_key("b3_spw_setup")
        self.b3_spw_setup        = [float(s) for s in l.split(",")]
        l                        = self._read_key("b6_spw_setup")
        self.b6_spw_setup        = [float(s) for s in l.split(",")]

        # ngc1068
        self.z                   = float(self._read_key("z"))
        self.scale               = float(self._read_key("scale"))
        self.ra                  = self._read_key("ra")
        self.dec                 = self._read_key("dec")
        self.ra_agn              = self._read_key("ra_agn")
        self.dec_agn             = self._read_key("dec_agn")

        # output fits
        self.outfits_missingflux = self.dir_ready + self._read_key("outfits_missingflux")
        self.outfits_co10        = self.dir_ready + self._read_key("outfits_co10")

        # output png
        self.png_specscan_b3     = self.dir_products + self._read_key("png_specscan_b3")
        self.png_specscan_b6     = self.dir_products + self._read_key("png_specscan_b6")

        self.png_missingflux     = self.dir_products + self._read_key("png_missingflux")
        self.imsize_as           = float(self._read_key("imsize_as"))

        self.png_histogram       = self.dir_products + self._read_key("png_histogram")

        self.png_fov_map         = self.dir_products + self._read_key("png_fov_map")
        self.imsize_fov_as       = float(self._read_key("imsize_fov_as"))

        # final products
        self.final_specscan      = self.dir_final + self._read_key("final_specscan")
        self.box_specscan        = self._read_key("box_specscan")

        self.final_missingflux   = self.dir_final + self._read_key("final_missingflux")
        self.box_missingflux     = self._read_key("box_missingflux")
        self.box_missingflux2    = self._read_key("box_missingflux2")

        self.final_fov           = self.dir_final + self._read_key("final_fov")
        self.box_fov_map         = self._read_key("box_fov_map")

    def _set_cycle_8p5b_catom21(
        self,
        ):
        """
        import os
        from proposals_alma import ProposalsALMA as tools
        
        dir_proj = /home02/saitots/myUtils/keys_alma_proposal/
        
        ### Cycle 8 supplemental call
        # not submitted (no Band 10 mosaic in cycle 8)
        # Band 10 7m-only mosaic toward NGC 1068
        tl.run_cycle_8p5b_catom21(
            plot_ci_co_ratio      = True,
            plot_expected_catom21 = True,
            combine_figures       = True,
            )
        
        os.system("rm -rf *.last")
        """

        # input data
        self.tpeak_ci10_1p64      = dir_raw + self._read_key("tpeak_ci10_1p64")
        self.image_ci10_1p64      = dir_raw + self._read_key("image_ci10_1p64")
        self.image_co10_1p64      = dir_raw + self._read_key("image_co10_1p64")

        self.txt_fov_b10_fov1     = dir_raw + self._read_key("fov_b10_fov1")
        self.txt_fov_b10_fov2     = dir_raw + self._read_key("fov_b10_fov2")
        self.imsize_catom_as      = float(self._read_key("imsize_catom_as"))

        # output fits
        self.outfits_expected_ci21_tpeak \
                                  = self.dir_ready + self._read_key("outfits_expected_ci21_tpeak")
        self.outfits_ci_co_ratio  = self.dir_ready + self._read_key("outfits_ci_co_ratio")

        # output png
        self.png_expected_catom21 = self.dir_products + self._read_key("png_expected_catom21")
        self.box_expected_catom21 = self._read_key("box_expected_catom21")

        self.png_ci10_1p64        = self.dir_products + self._read_key("png_ci10_1p64")
        self.box_ci10_1p64        = self._read_key("box_ci10_1p64")

        self.png_ci_co_ratio_1p64 = self.dir_products + self._read_key("png_ci_co_ratio_1p64")
        self.box_ci_co_ratio_1p64 = self._read_key("box_ci_co_ratio_1p64")

        # final products
        self.final_catom10        = self.dir_final + self._read_key("final_catom10")
        self.final_catom21        = self.dir_final + self._read_key("final_catom21")

    #################
    # run_cycle_8p5 #
    #################

    def run_cycle_8p5a_specscan(
        self,
        plot_spw_setup    = False,
        plot_missingflux  = False,
        plot_proposed_fov = False,
        combine_figures   = False,
        ):
        """
        Content: Band 3 and 6 spectral scans with 7m+TP toward NGC 1068.
        Status:  submitted
        Code:    2021.2.00049.S
        Outcome: accepted (grade C)
        Rank:    1,1,1,1,2,2,3,4,4,6
        Note:    N/A
        """

        if plot_spw_setup==True:
            self.c8p5_plot_spw_setup_b3()
            self.c8p5_plot_spw_setup_b6()

        if plot_missingflux==True:
            self.c8p5_plot_missingflux()

        if plot_proposed_fov==True:
            self.c8p5_fov_with_map()

        if combine_figures==True:
            self.c8p5_create_figure_spws()
            self.c8p5_create_figure_missingflux()
            self.c8p5_create_figure_fov()

    def run_cycle_8p5b_catom21(
        self,
        plot_ci_co_ratio      = False,
        plot_expected_catom21 = False,
        combine_figures       = False,
        ):
        """
        Content: CI 2-1 mapping of NGC 1068
        Status:  Not submitted
        Code:    N/A
        Outcome: N/A
        Rank:    N/A
        Note:
        this proposal (7m-only Band 10 mosaic toward NGC 1068) was not submitted in Cycle8.5,
        because Cycle 8 did not offer Band 10 mosaic. This proposal will be modified and then
        submitted in the future main call (e.g., 12m+7m array).
        """

        if plot_ci_co_ratio==True:
            self.c8p5b_plot_ci_co_ratio()

        if plot_expected_catom21==True:
            self.c8p5b_plot_expected_catom21()

        if combine_figures==True:
            self.c8p5b_create_figure_catom21()
            self.c8p5b_create_figure_catom10()

    #####################
    # c8p5_fov_with_map #
    #####################

    def c8p5_fov_with_map(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.image_cs21,taskname)

        # map
        scalebar = 500 / self.scale

        myfig_fits2png(
            # general
            self.image_cs21,
            self.png_fov_map,
            imsize_as=self.imsize_fov_as,
            ra_cnt=self.ra_agn,
            dec_cnt=self.dec_agn,
            # imshow
            fig_dpi=self.fig_dpi,
            set_grid=None,
            set_title="12m-only CS(2-1) intensity map",
            colorlog=True,
            clim=[10**0.1,10**2],
            set_cmap="PuBu",
            showzero=False,
            showbeam=False,
            color_beam="black",
            scalebar=scalebar,
            label_scalebar="0.5 kpc",
            color_scalebar="black",
            # annotation
            numann=4,
            textann=True,
            txtfiles=[self.txt_fov_b3,self.txt_fov_b6],
            )

    ##########################
    # c8p5_create_figure_fov #
    ##########################

    def c8p5_create_figure_fov(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.png_missingflux,taskname)

        immagick_crop(self.png_fov_map,self.final_fov,self.box_fov_map,True)

    ##################################
    # c8p5_create_figure_missingflux #
    ##################################

    def c8p5_create_figure_missingflux(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.png_missingflux,taskname)

        combine_two_png(
        self.png_missingflux,
        self.png_histogram,
        self.final_missingflux,
        self.box_missingflux,
        self.box_missingflux2,
        )

    #########################
    # c8p5_plot_missingflux #
    #########################

    def c8p5_plot_missingflux(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.image_co10_12m7m,taskname)

        run_importfits(self.image_co10_12m7m,self.image_co10_12m7m+"_tmp1",defaultaxes=True,
            defaultaxesvalues=["RA","Dec","1GHz","Stokes"])
        run_importfits(self.image_co10_12m,self.image_co10_12m+"_tmp1",defaultaxes=True,
            defaultaxesvalues=["RA","Dec","1GHz","Stokes"])
        run_exportfits(self.image_co10_12m+"_tmp1",self.outfits_co10,True,True,False)

        run_imregrid(self.image_co10_12m7m+"_tmp1",self.image_co10_12m+"_tmp1",
            self.image_co10_12m7m+"_tmp2",delin=True)

        expr = "iif(IM1>0,(IM0-IM1)/IM0*100,0)"
        run_immath_two(self.image_co10_12m7m+"_tmp2",self.image_co10_12m+"_tmp1",
            self.outfits_missingflux+"_tmp1",expr)

        signal_masking(self.outfits_missingflux+"_tmp1",self.outfits_missingflux+"_tmp2",0)

        expr = "iif(IM1>0,IM0,0)"
        run_immath_two(self.outfits_missingflux+"_tmp1",self.outfits_missingflux+"_tmp2",
            self.outfits_missingflux+"_tmp3",expr,delin=True)

        signal_masking(self.image_co10_12m7m+"_tmp2",self.image_co10_12m7m+"_tmp3",0,True)
        signal_masking(self.image_co10_12m+"_tmp1",self.image_co10_12m+"_tmp2",0,True)
        expr = "iif(IM0>0,IM0-IM1,0)"
        run_immath_two(self.image_co10_12m7m+"_tmp3",self.image_co10_12m+"_tmp2",
            self.outfits_missingflux+"_tmp4",expr,delin=True)

        expr = "iif(IM1>0,100,IM0)"
        run_immath_two(self.outfits_missingflux+"_tmp3",self.outfits_missingflux+"_tmp4",
            self.outfits_missingflux+"_tmp5",expr,delin=True)

        imhead(self.outfits_missingflux+"_tmp5",mode="put",hdkey="beammajor",hdvalue="0.8arcsec")
        imhead(self.outfits_missingflux+"_tmp5",mode="put",hdkey="beamminor",hdvalue="0.8arcsec")

        run_exportfits(self.outfits_missingflux+"_tmp5",self.outfits_missingflux,True,True,True)

        # plot map
        scalebar = 500 / self.scale

        myfig_fits2png(
            # general
            self.outfits_missingflux,
            self.png_missingflux,
            imcontour1=self.outfits_co10,
            imsize_as=self.imsize_as,
            ra_cnt=self.ra_agn,
            dec_cnt=self.dec_agn,
            # contour 1
            unit_cont1=None,
            levels_cont1=[0.01,0.02,0.04,0.08,0.16,0.32,0.64,0.96],
            width_cont1=[1.0],
            color_cont1="black",
            # imshow
            fig_dpi=self.fig_dpi,
            set_grid=None,
            set_title="CO(1-0) missing flux map",
            showzero=False,
            showbeam=True,
            color_beam="black",
            scalebar=scalebar,
            label_scalebar="0.5 kpc",
            color_scalebar="black",
            # imshow colorbar
            set_cbar=True,
            clim=[0,100],
            label_cbar="missing flux (%)",
            # annotation
            numann=3,
            textann=True,
            )

        # plt histogram
        data,box = imval_all(self.outfits_missingflux)
        data     = data["data"] * data["mask"]
        data     = data.flatten()
        data     = data[np.where((data>0) & (data<100))]

        p16 = np.percentile(data,16)
        p50 = np.percentile(data,50)
        p84 = np.percentile(data,84)

        data_hist = np.histogram(data, bins=100, range=[0,100], weights=None)
        x = np.delete(data_hist[1],-1)
        y = data_hist[0] / float(np.sum(data_hist[0]))

        # plot
        plt.figure(figsize=(13,10))
        gs = gridspec.GridSpec(nrows=10, ncols=10)
        ax = plt.subplot(gs[0:10,0:10])
        ad = [0.215,0.83,0.10,0.90]
        myax_set(ax,grid=None,xlim=[0,100],ylim=[0,0.03],title="CO(1-0) missing flux histogram",
            xlabel="missing flux (%)",ylabel="count density",adjust=ad)

        width = abs(x[1] - x[0])
        for i in range(100):
            ax.bar(x[np.where((x>=i) & (x<i+1))]+width, y[np.where((x>=i) & (x<i+1))], lw=0,
                color=cm.rainbow(i/100.), alpha=1.0, width=width, align="edge")       

        ax.plot([p50,p50],[0.027,0.027],"o",color="black",lw=0,markersize=15)
        ax.plot([p16,p84],[0.027,0.027],"-",color="black",lw=2)

        ax.text(p50,0.0265,"median = "+str(int(p50))+"%",fontsize=22,ha="center",va="top")

        ax.text(p16,0.0275,"16$^{th}$ pctl. = "+str(int(p16))+"%",
            fontsize=22,ha="center",va="bottom")
        ax.text(p84,0.0275,"84$^{th}$ pctl. = "+str(int(p84))+"%",
            fontsize=22,ha="center",va="bottom")

        plt.savefig(self.png_histogram, dpi=self.fig_dpi)

    ###########################
    # c8p5_create_figure_spws #
    ###########################

    def c8p5_create_figure_spws(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.png_specscan_b6,taskname)

        combine_two_png(
        self.png_specscan_b3,
        self.png_specscan_b6,
        self.final_specscan,
        self.box_specscan,
        self.box_specscan,
        )

    ##########################
    # c8p5_plot_spw_setup_b6 #
    ##########################

    def c8p5_plot_spw_setup_b6(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.archive_csv,taskname)

        # read csv
        with open(self.archive_csv) as f:
            reader = csv.reader(f)
            l = [row for row in reader]

        l = np.array(l)

        ### get info
        header    = l[0]
        project   = l[:,0]
        galname   = l[:,1]
        band      = l[:,4]
        freq_info = l[:,6]
        ang_res   = l[:,9]
        pi_name   = l[:,23]
        array     = l[:,11]

        # exclude ToO projects
        mask = []
        for i in range(len(project)):
            this_project = project[i]
            if ".S" in this_project:
                mask.append(i)

        project   = project[mask]
        galname   = galname[mask]
        band      = band[mask]
        freq_info = freq_info[mask]
        ang_res   = ang_res[mask]
        pi_name   = np.array([s.split(",")[0] for s in pi_name[mask]])
        array     = array[mask]

        # get data
        sci_band = []
        for i in range(len(band)):
            this_band = band[i]
            if this_band=="6":
                sci_band.append(i)

        data = np.c_[project[sci_band], freq_info[sci_band], ang_res[sci_band], pi_name[sci_band], galname[sci_band], array[sci_band]]

        # prepare for plot: sci_band spws
        list_spw,list_color,list_lw = [],[],[]
        for i in range(len(data[:,1])):
            this_freq  = data[i,1]
            this_array = data[i,5]
            this_spws  = this_freq.split(" U ")

            if this_array=="12m":
                this_color = "black"
                this_lw = 1.0
            else:
                this_color = "tomato"
                this_lw = 2.0

            for j in range(len(this_spws)):
                this_spw = this_spws[j].split(",")[0].lstrip("[")
                this_spw = this_spw.rstrip("GHz").split("..")
                this_spw = [float(this_spw[0]),float(this_spw[1])]
                list_spw.append(this_spw)
                list_color.append(this_color)
                list_lw.append(this_lw)

        list_data = np.c_[list_spw,list_color]
        list_data = list_data[np.argsort(list_data[:, 0].astype(np.float64))]

        # prepare for plot: lines
        list_line = np.loadtxt(self.line_key,dtype="str")
        list_linefreq = [float(s[2]) for s in list_line if "b6" in s[0]]
        list_linename = [s[0].split("line_b6_")[1] for s in list_line if "b6" in s[0]]
        list_lineoffset = [float(s[1]) for s in list_line if "b6" in s[0]]

        # plot
        plt.figure(figsize=(10,8))
        gs = gridspec.GridSpec(nrows=5, ncols=30)
        ax1 = plt.subplot(gs[0:3,1:30])
        ax2 = plt.subplot(gs[3:5,1:30])

        # ax setup
        myax_set(ax1,xlim=[209,277],ylim=[0.0,10.0],labelbottom=False,labelleft=False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.spines["bottom"].set_visible(False)
        ax1.spines["left"].set_visible(False)
        ax1.tick_params("x", length=0, which="major")
        ax1.tick_params("y", length=0, which="major")

        myax_set(ax2,xlim=[209,277],ylim=[-5,100],labelbottom=False,labelleft=False,lw_outline=1.5)
        ax2.tick_params("x", length=0, which="major")
        ax2.tick_params("y", length=0, which="major")

        # ax1
        list_text = []
        for i in range(len(list_linename)):
        	this_name   = list_linename[i]
        	this_freq   = list_linefreq[i] / (1+self.z)
        	this_offset = list_lineoffset[i]
        	ax1.plot([this_freq,this_freq],[1.0,6.0],color="green",lw=2)
        	this_text = ax1.text(this_freq+this_offset,6.0,this_name,
        	    rotation=60,fontsize=15,ha="left",va="bottom")

        # ax2: arcival spw
        for i in range(len(list_data)):
            x = [float(list_data[i][0]),float(list_data[i][1])]
            ax2.plot(x,[i+1,i+1],"-",color=list_data[i][2],lw=list_lw[i])

        # ax2: proposed spw
        for j in range(len(self.b6_spw_setup)):
            x = [self.b6_spw_setup[j]-1.875/2.0, self.b6_spw_setup[j]+1.875/2.0]
            y = [j*2*4/5.+65,j*2*4/5.+65]
            if j>19:
                y = [j*2*4/5.+65-16*2,j*2*4/5.+65-16*2]

            ax2.plot(x,y,color="blue",lw=5)

        # text
        ax1.text(0.50,0.90,"ALMA Band 6 Coverage",
            color="black",weight="bold",transform=ax1.transAxes,fontsize=20,ha="center")
        ax2.text(0.95,0.05,"proposed 7m+TP SPWs",
            color="blue",weight="bold",transform=ax2.transAxes,fontsize=17,ha="right")
        ax2.text(0.95,0.25,"archival 12m SPWs",
            color="black",transform=ax2.transAxes,fontsize=17,ha="right")
        ax2.text(0.95,0.16,"archival 7m SPWs",
            color="tomato",transform=ax2.transAxes,fontsize=17,ha="right")

        # ax1 grid
        width = 0.4
        ax1.plot([211,275],[1,1],lw=2,color="black",zorder=1e9)
        ax1.plot([220,220],[1.0-width,1.0+width],lw=2,color="black",zorder=1e9)
        ax1.plot([230,230],[1.0-width,1.0+width],lw=2,color="black",zorder=1e9)
        ax1.plot([240,240],[1.0-width,1.0+width],lw=2,color="black",zorder=1e9)
        ax1.plot([250,250],[1.0-width,1.0+width],lw=2,color="black",zorder=1e9)
        ax1.plot([260,260],[1.0-width,1.0+width],lw=2,color="black",zorder=1e9)
        ax1.plot([270,270],[1.0-width,1.0+width],lw=2,color="black",zorder=1e9)
        ax1.text(220,1.0-width-0.1,"220",ha="center",va="top",fontsize=15)
        ax1.text(230,1.0-width-0.1,"230",ha="center",va="top",fontsize=15)
        ax1.text(240,1.0-width-0.1,"240",ha="center",va="top",fontsize=15)
        ax1.text(250,1.0-width-0.1,"250",ha="center",va="top",fontsize=15)
        ax1.text(260,1.0-width-0.1,"260",ha="center",va="top",fontsize=15)
        ax1.text(270,1.0-width-0.1,"270",ha="center",va="top",fontsize=15)

        plt.savefig(self.png_specscan_b6, dpi=self.fig_dpi)

    ##########################
    # c8p5_plot_spw_setup_b3 #
    ##########################

    def c8p5_plot_spw_setup_b3(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.archive_csv,taskname)

        # read csv
        with open(self.archive_csv) as f:
            reader = csv.reader(f)
            l = [row for row in reader]

        l = np.array(l)

        ### get info
        header    = l[0]
        project   = l[:,0]
        galname   = l[:,1]
        band      = l[:,4]
        freq_info = l[:,6]
        ang_res   = l[:,9]
        pi_name   = l[:,23]
        array     = l[:,11]

        # exclude ToO projects
        mask = []
        for i in range(len(project)):
            this_project = project[i]
            if ".S" in this_project:
                mask.append(i)

        project   = project[mask]
        galname   = galname[mask]
        band      = band[mask]
        freq_info = freq_info[mask]
        ang_res   = ang_res[mask]
        pi_name   = np.array([s.split(",")[0] for s in pi_name[mask]])
        array     = array[mask]

        # get data
        sci_band = []
        for i in range(len(band)):
            this_band = band[i]
            if this_band=="3":
                sci_band.append(i)

        data = np.c_[project[sci_band], freq_info[sci_band], ang_res[sci_band], pi_name[sci_band], galname[sci_band], array[sci_band]]

        # prepare for plot: sci_band spws
        list_spw,list_color,list_lw = [],[],[]
        for i in range(len(data[:,1])):
            this_freq  = data[i,1]
            this_array = data[i,5]
            this_spws  = this_freq.split(" U ")

            if this_array=="12m":
                this_color = "black"
                this_lw = 1.0
            else:
                this_color = "tomato"
                this_lw = 2.0

            for j in range(len(this_spws)):
                this_spw = this_spws[j].split(",")[0].lstrip("[")
                this_spw = this_spw.rstrip("GHz").split("..")
                this_spw = [float(this_spw[0]),float(this_spw[1])]
                list_spw.append(this_spw)
                list_color.append(this_color)
                list_lw.append(this_lw)

        list_data = np.c_[list_spw,list_color]
        list_data = list_data[np.argsort(list_data[:, 0].astype(np.float64))]

        # prepare for plot: lines
        list_line = np.loadtxt(self.line_key,dtype="str")
        list_linefreq = [float(s[2]) for s in list_line if "b3" in s[0]]
        list_linename = [s[0].split("line_b3_")[1] for s in list_line if "b3" in s[0]]
        list_lineoffset = [float(s[1]) for s in list_line if "b3" in s[0]]

        # plot
        plt.figure(figsize=(10,8))
        gs = gridspec.GridSpec(nrows=5, ncols=30)
        ax1 = plt.subplot(gs[0:3,1:30])
        ax2 = plt.subplot(gs[3:5,1:30])

        # ax setup
        myax_set(ax1,xlim=[83,117],ylim=[0.0,10.0],labelbottom=False,labelleft=False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.spines["bottom"].set_visible(False)
        ax1.spines["left"].set_visible(False)
        ax1.tick_params("x", length=0, which="major")
        ax1.tick_params("y", length=0, which="major")

        myax_set(ax2,xlim=[83,117],ylim=[-5,100],labelbottom=False,labelleft=False,lw_outline=1.5)
        ax2.tick_params("x", length=0, which="major")
        ax2.tick_params("y", length=0, which="major")

        # ax1
        list_text = []
        for i in range(len(list_linename)):
        	this_name   = list_linename[i]
        	this_freq   = list_linefreq[i] / (1+self.z)
        	this_offset = list_lineoffset[i]
        	ax1.plot([this_freq,this_freq],[1.0,6.0],color="green",lw=2)
        	this_text = ax1.text(this_freq+this_offset,6.0,this_name,
        	    rotation=60,fontsize=15,ha="left",va="bottom")

        # ax2: arcival spw
        for i in range(len(list_data)):
            x = [float(list_data[i][0]),float(list_data[i][1])]
            ax2.plot(x,[i+1,i+1],"-",color=list_data[i][2],lw=list_lw[i])

        # ax2: proposed spw
        for j in range(len(self.b3_spw_setup)):
            x = [self.b3_spw_setup[j]-1.875/2.0, self.b3_spw_setup[j]+1.875/2.0]
            y = [j*2+65,j*2+65]
            ax2.plot(x,y,color="blue",lw=5)

        # text
        ax1.text(0.50,0.90,"ALMA Band 3 Coverage",
            color="black",weight="bold",transform=ax1.transAxes,fontsize=20,ha="center")
        ax2.text(0.95,0.05,"proposed 7m+TP SPWs",
            color="blue",weight="bold",transform=ax2.transAxes,fontsize=17,ha="right")
        ax2.text(0.95,0.25,"archival 12m SPWs",
            color="black",transform=ax2.transAxes,fontsize=17,ha="right")
        ax2.text(0.95,0.16,"archival 7m SPWs",
            color="tomato",transform=ax2.transAxes,fontsize=17,ha="right")

        # ax1 grid
        width = 0.4
        ax1.plot([84,116],[1,1],lw=2,color="black",zorder=1e9)
        ax1.plot([85,85],[1.0-width,1.0+width],lw=2,color="black",zorder=1e9)
        ax1.plot([95,95],[1.0-width,1.0+width],lw=2,color="black",zorder=1e9)
        ax1.plot([105,105],[1.0-width,1.0+width],lw=2,color="black",zorder=1e9)
        ax1.plot([115,115],[1.0-width,1.0+width],lw=2,color="black",zorder=1e9)
        ax1.plot([115,115],[1.0-width,1.0+width],lw=2,color="black",zorder=1e9)
        ax1.text(85,1.0-width-0.1,"85",ha="center",va="top",fontsize=15)
        ax1.text(95,1.0-width-0.1,"95",ha="center",va="top",fontsize=15)
        ax1.text(105,1.0-width-0.1,"105",ha="center",va="top",fontsize=15)
        ax1.text(115,1.0-width-0.1,"115",ha="center",va="top",fontsize=15)
        ax1.text(115,1.0-width-0.1,"115",ha="center",va="top",fontsize=15)

        plt.savefig(self.png_specscan_b3, dpi=self.fig_dpi)

    ##########################
    # c8p5b_plot_ci_co_ratio #
    ##########################

    def c8p5b_plot_ci_co_ratio(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.image_ci10_1p64,taskname)

        #
        run_importfits(self.image_ci10_1p64,"template.image")
        run_imregrid(self.image_co10_1p64,"template.image",self.image_co10_1p64+"_tmp1")
        os.system("rm -rf template.image")
        expr = "iif( IM1>0, IM1/IM0, 0 )"
        run_immath_two(self.image_co10_1p64+"_tmp1",self.image_ci10_1p64,self.outfits_ci_co_ratio+"_tmp1",expr)
        os.system("rm -rf " + self.image_co10_1p64 + "_tmp1")
        run_exportfits(self.outfits_ci_co_ratio+"_tmp1",self.outfits_ci_co_ratio,True,True,True)

        # plot
        scalebar = 500 / self.scale

        myfig_fits2png(
            # general
            self.image_ci10_1p64,
            self.png_ci10_1p64,
            self.image_ci10_1p64,
            imsize_as=self.imsize_catom_as,
            ra_cnt=self.ra,
            dec_cnt=self.dec,
            # contour 1
            levels_cont1=[0.02,0.04,0.08,0.16,0.32,0.64,0.96],
            width_cont1=[2.5],
            color_cont1="black",
            # imshow
            fig_dpi=self.fig_dpi,
            set_grid=None,
            set_title="Observed [CI](1-0) map",
            colorlog=True,
            set_bg_color=cm.rainbow(0),
            set_cmap="rainbow",
            showzero=False,
            showbeam=True,
            color_beam="black",
            scalebar=scalebar,
            label_scalebar="0.5 kpc",
            color_scalebar="black",
            # annotation
            #numann=5,
            #textann=True,
            #txtfiles=[self.txt_fov_b10_fov1,self.txt_fov_b10_fov2],
            )

        myfig_fits2png(
            # general
            self.outfits_ci_co_ratio,
            self.png_ci_co_ratio_1p64,
            self.image_ci10_1p64,
            imsize_as=self.imsize_catom_as,
            ra_cnt=self.ra,
            dec_cnt=self.dec,
            # contour 1
            levels_cont1=[0.02,0.04,0.08,0.16,0.32,0.64,0.96],
            width_cont1=[2.5],
            color_cont1="black",
            # imshow
            fig_dpi=self.fig_dpi,
            set_grid=None,
            set_title="Observed [CI](1-0)/CO(1-0) ratio",
            colorlog=False,
            #set_bg_color=cm.rainbow(0),
            set_cmap="rainbow",
            clim=[0,1],
            showzero=False,
            showbeam=True,
            color_beam="black",
            scalebar=scalebar,
            label_scalebar="0.5 kpc",
            color_scalebar="black",
            # annotation
            #numann=5,
            #textann=True,
            #txtfiles=[self.txt_fov_b10_fov1,self.txt_fov_b10_fov2],
            )

    ###############################
    # c8p5b_create_figure_catom10 #
    ###############################

    def c8p5b_create_figure_catom10(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.png_ci_co_ratio_1p64,taskname)

        combine_two_png(self.png_ci10_1p64,self.png_ci_co_ratio_1p64,
            self.final_catom10,self.box_ci10_1p64,self.box_ci_co_ratio_1p64,delin=True)

    ###############################
    # c8p5b_create_figure_catom21 #
    ###############################

    def c8p5b_create_figure_catom21(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.png_expected_catom21,taskname)

        immagick_crop(self.png_expected_catom21,self.final_catom21,self.box_expected_catom21,True)

    ###############################
    # c8p5b_plot_expected_catom21 #
    ###############################

    def c8p5b_plot_expected_catom21(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.image_cs21,taskname)

        #
        run_immath_one(self.tpeak_ci10,self.outfits_expected_ci21_tpeak+"_tmp1","IM0*0.7")
        run_exportfits(self.outfits_expected_ci21_tpeak+"_tmp1",self.outfits_expected_ci21_tpeak,True,True,True)

        # map
        scalebar = 500 / self.scale

        myfig_fits2png(
            # general
            self.outfits_expected_ci21_tpeak,
            self.png_expected_catom21,
            self.outfits_expected_ci21_tpeak,
            imsize_as=self.imsize_catom_as,
            ra_cnt=self.ra,
            dec_cnt=self.dec,
            # contour 1
            unit_cont1=0.1, # required rms = 0.1 K
            levels_cont1=[3,5],
            width_cont1=[2.5],
            color_cont1="black",
            # imshow
            fig_dpi=self.fig_dpi,
            set_grid=None,
            set_title="Expected [CI](2-1) peak temperature",
            colorlog=True,
            set_bg_color=cm.rainbow(0),
            set_cmap="rainbow",
            showzero=False,
            showbeam=True,
            color_beam="white",
            scalebar=scalebar,
            label_scalebar="0.5 kpc",
            color_scalebar="white",
            # annotation
            numann=5,
            textann=True,
            txtfiles=[self.txt_fov_b10_fov1,self.txt_fov_b10_fov2],
            )

    ###########################################################################################
    ###########################################################################################
    #######################################             #######################################
    ####################################### common part #######################################
    #######################################             #######################################
    ###########################################################################################
    ###########################################################################################

    ############
    # _set_dir #
    ############

    def _set_dir(self):
        """
        """

        self.dir_proj     = self._read_key("dir_proj")
        dir_raw           = self.dir_proj + self._read_key("dir_raw") + self.cycle + "/"
        self.dir_raw      = self.dir_proj + self._read_key("dir_raw") + self.cycle + "/"
        self.dir_ready    = self.dir_proj + self._read_key("dir_ready") + self.cycle + "/"
        self.dir_products = self.dir_proj + self._read_key("dir_products") + self.cycle + "/"
        self.dir_final    = self.dir_proj + self._read_key("dir_final") + self.cycle + "/"
        self._create_dir(self.dir_ready)
        self._create_dir(self.dir_products)
        self._create_dir(self.dir_final)

    ###############
    # _create_dir #
    ###############

    def _create_dir(self, this_dir):

        if self.refresh==True:
            print("## refresh " + this_dir)
            os.system("rm -rf " + this_dir)

        if not glob.glob(this_dir):
            print("## create " + this_dir)
            os.makedirs(this_dir)

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


########################
# end of ProposalsALMA #
########################