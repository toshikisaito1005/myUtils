"""
Python class for the NGC 1068 PCA project

requirements:
CASA Version 5.4.0-70, ananlysisUtils, astropy

data:

usage:
> import os
> from scripts_sim_lst_wp import ToolsLSTSim as tools
>
> # key
> tl = tools(s
>     refresh     = False,
>     keyfile_gal = "/home02/saitots/myUtils/keys_sim_lst_wp/key_ngc1068.txt",
>     keyfile_fig = "/home02/saitots/myUtils/keys_sim_lst_wp/key_figures.txt",
>     )
>
> # main
> tl.run_sim_lst_alma(
>     # analysis
>     do_prepare             = True,
>     )
>
> os.system("rm -rf *.last")

white paper drafts:
Date         Filename                To
2022-??-??

history:
2022-03-08   created
Toshiki Saito@Nichidai/NAOJ
"""

import os, sys, glob
import numpy as np

from mycasa_tasks import *
from mycasa_plots import *
from mycasa_simobs import *

###############
# ToolsLSTSim #
###############
class ToolsLSTSim():
    """
    Class for the LST white paper 2022 project.
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
        self.dir_keyfile = "/".join(self.keyfile_gal.split("/")[:-1]) + "/"

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
            self.modname = "ToolsPCA."
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

        # simobserve
        self.project_n1068 = self._read_key("project_n1068")
        self.project_n1097 = self._read_key("project_n1097")
        self.config_c1     = self.dir_keyfile + self._read_key("config_c1")
        self.config_c10    = self.dir_keyfile + self._read_key("config_c10")
        self.config_7m     = self.dir_keyfile + self._read_key("config_7m")

        # phangs-alma pipeline
        self.dir_pipeline = self._read_key("dir_pipeline")

    def _set_input_fits(self):
        """
        """

        self.n1068_template_file = self._read_key("n1068_template_file")
        self.n1068_template_mask = self._read_key("n1068_template_mask")

        self.n1097_template_file = self._read_key("n1097_template_file")
        self.n1097_template_mask = self._read_key("n1097_template_mask")

    def _set_output_fits(self):
        """
        """

        self.n1068_template_in_jypix        = self.project_n1068 + "_template_jypixel.image"
        self.n1068_template_clipped         = self.project_n1068 + "_template_clipped.image"
        self.n1068_template_mask_imported   = self.project_n1068 + "_template_mask.image"
        self.n1068_template_rotated         = self.project_n1068 + "_template_rotated.image"
        self.n1068_template_shrunk          = self.project_n1068 + "_template_shrunk.image"
        self.n1068_template_fullspec        = self.project_n1068 + "_template_fullspec.image"
        self.n1068_template_fullspec_div3   = self.project_n1068 + "_template_fullspec_div3.image"
        self.n1068_template_fullspec_div5   = self.project_n1068 + "_template_fullspec_div5.image"
        self.n1068_template_fullspec_div10  = self.project_n1068 + "_template_fullspec_div10.image"
        self.n1068_template_fullspec_div30  = self.project_n1068 + "_template_fullspec_div30.image"
        self.n1068_template_fullspec_div100 = self.project_n1068 + "_template_fullspec_div100.image"
        self.n1068_template_withcont        = self.project_n1068 + "_template_withcont.image"
        self.n1068_template_withcont_div3   = self.project_n1068 + "_template_withcont_div3.image"
        self.n1068_template_withcont_div5   = self.project_n1068 + "_template_withcont_div5.image"
        self.n1068_template_withcont_div10  = self.project_n1068 + "_template_withcont_div10.image"
        self.n1068_template_withcont_div30  = self.project_n1068 + "_template_withcont_div30.image"
        self.n1068_template_withcont_div100 = self.project_n1068 + "_template_withcont_div100.image"
        self.n1068_sdnoise_image            = self.project_n1068 + "_aca_tp_noise_nocont.image"
        self.n1068_sdimage_fullspec         = self.project_n1068 + "_aca_tp_nocont.image"
        self.n1068_lstnoise_image            =self.project_n1068 + "_lst_noise_nocont.image"
        self.n1068_lstimage_fullspec         =self.project_n1068 + "_lst_nocont.image"

        self.n1097_template_in_jypix        = self.project_n1097 + "_template_jypixel.image"
        self.n1097_template_clipped         = self.project_n1097 + "_template_clipped.image"
        self.n1097_template_mask_imported   = self.project_n1097 + "_template_mask.image"
        self.n1097_template_rotated         = self.project_n1097 + "_template_rotated.image"
        self.n1097_template_shrunk          = self.project_n1097 + "_template_shrunk.image"
        self.n1097_template_fullspec        = self.project_n1097 + "_template_fullspec.image"
        self.n1097_template_fullspec_div3   = self.project_n1097 + "_template_fullspec_div3.image"
        self.n1097_template_fullspec_div5   = self.project_n1097 + "_template_fullspec_div5.image"
        self.n1097_template_fullspec_div10  = self.project_n1097 + "_template_fullspec_div10.image"
        self.n1097_template_fullspec_div30  = self.project_n1097 + "_template_fullspec_div30.image"
        self.n1097_template_fullspec_div100 = self.project_n1097 + "_template_fullspec_div100.image"
        self.n1097_template_withcont        = self.project_n1097 + "_template_withcont.image"
        self.n1097_template_withcont_div3   = self.project_n1097 + "_template_withcont_div3.image"
        self.n1097_template_withcont_div5   = self.project_n1097 + "_template_withcont_div5.image"
        self.n1097_template_withcont_div10  = self.project_n1097 + "_template_withcont_div10.image"
        self.n1097_template_withcont_div30  = self.project_n1097 + "_template_withcont_div30.image"
        self.n1097_template_withcont_div100 = self.project_n1097 + "_template_withcont_div100.image"
        self.n1097_sdnoise_image            = self.project_n1097 + "_aca_tp_noise_nocont.image"
        self.n1097_sdimage_fullspec         = self.project_n1097 + "_aca_tp_nocont.image"
        self.n1097_lstnoise_image           = self.project_n1097 + "_lst_noise_nocont.image"
        self.n1097_lstimage_fullspec        = self.project_n1097 + "_lst_nocont.image"

    def _set_input_param(self):
        """
        """

        # sim properties
        self.singledish_noise = 0.102 # Jy/beam at final res
        self.singledish_res   = "28.37arcsec" # resolution
        self.image_roration   = "23deg"

        # ngc1068 properties
        self.ra_agn    = float(self._read_key("ra_agn", "gal").split("deg")[0])
        self.dec_agn   = float(self._read_key("dec_agn", "gal").split("deg")[0])
        self.scale_pc  = float(self._read_key("scale", "gal"))
        self.scale_kpc = self.scale_pc / 1000.

        self.beam      = 2.14859173174056 # 150pc in arcsec
        self.snr_mom   = 4.0
        self.r_cnd     = 3.0 * self.scale_pc / 1000. # kpc
        self.r_cnd_as  = 3.0
        self.r_sbr     = 10.0 * self.scale_pc / 1000. # kpc
        self.r_sbr_as  = 10.0

    def _set_output_txt_png(self):
        """
        """

        self.outpng_config_12m   = self.dir_products + self._read_key("outpng_config_12m")
        self.outpng_uv_alma_lst1 = self.dir_products + self._read_key("outpng_uv_alma_lst1")

    ####################
    # run_sim_lst_alma #
    ####################

    def run_sim_lst_alma(
        self,
        # ngc1097sim
        tinteg_n1097sim      = 48, # 7m total observing time
        observed_freq        = 492.16065100, # GHz, determine LST and TP beam sizes
        do_template_n1097sim = False, # create "wide" template cube for mapping simobserve
        do_simint_n1097im    = False, # sim ACA band 8 for big ngc1097sim
        do_imaging_n1097sim  = False, # imaging sim ms
        dryrun_simSD         = False, # just output SD mapping parameters
        do_simTP_n1097im     = False, # sim ACA TP alone
        do_simLST_n1097im    = False, # sim LST alone
        # ngc1068sim
        tinteg_n1068sim      = 2,
        do_template_n1068sim = False, # create "compact" template cube for long-baseline simobserve
        do_simint_n1068im    = False, # sim C-10 band 8 for small ngc1068sim
        do_imaging_n1068sim  = False, # imaging sim ms
        # plot
        plot_config          = False,
        # calc
        calc_collectingarea  = False,
        ):
        """
        This method runs all the methods which will create figures in the white paper.
        """

        # set observe frequency
        self.observed_freq = observed_freq
        self.incenter      = observed_freq

        # n1097sim_7m from tinteg_n1097sim
        totaltime_n1097sim_7m = str(float(tinteg_n1097sim))+"h"
        totaltimetint_n1097sim_7m = totaltime_n1097sim_7m.replace(".","p")

        # determine LST and TP beam sizes
        lst_beam_n1097sim       = str(12.979 * 115.27120 / self.observed_freq)+"arcsec"
        lst_beam_n1097sim_float = 12.979 * 115.27120 / self.observed_freq
        tp_beam_n1097sim        = str(50.6   * 115.27120 / self.observed_freq)+"arcsec"
        tp_beam_n1097sim_float  = 50.6 * 115.27120 / self.observed_freq

        # n1097sim_aca_tp from tinteg_n1097sim
        # TP integration time = 7m time * 1.7 (Table 7.4 of ALMA Technical Handbook 9.1.1)
        totaltime_n1097sim_tp  = str(np.round(tinteg_n1097sim * 1.7 * (tp_beam_n1097sim_float/7.)**2, 1))+"h"
        totaltime_n1097sim_lst = str(np.round(tinteg_n1097sim * 1.7 * (lst_beam_n1097sim_float/7.)**2, 1))+"h"
        totaltimetint_n1097sim_tp  = (str(np.round(tinteg_n1097sim, 1))+"h7m_"+totaltime_n1097sim_tp+"tp").replace(".","p")
        totaltimetint_n1097sim_lst = (str(np.round(tinteg_n1097sim, 1))+"h7m_"+totaltime_n1097sim_lst+"lst").replace(".","p")

        # ngc1097sim
        if do_template_n1097sim==True:
            self.prepare_template_n1097sim()

        if do_simint_n1097im==True:
            self.simaca_n1097sim(
                totaltime=totaltime_n1097sim_7m,
                totaltimetint=totaltimetint_n1097sim_7m,
                )

        if do_imaging_n1097sim==True:
            self.phangs_pipeline_imaging(
                self.project_n1097,
                "7m",
                self.project_n1097+"_"+totaltimetint_n1097sim_7m,
                )

        if do_simTP_n1097im==True:
            self.simtp_n1097sim(
                singledish_res=tp_beam_n1097sim,
                totaltime=totaltime_n1097sim_tp,
                totaltimetint=totaltimetint_n1097sim_tp,
                dryrun=dryrun_simSD,
                )

        if do_simLST_n1097im==True:
            self.simlst_n1097sim(
                singledish_res=lst_beam_n1097sim,
                totaltime=totaltime_n1097sim_lst,
                totaltimetint=totaltimetint_n1097sim_lst,
                dryrun=dryrun_simSD,
                )

        # n1097sim_7m from tinteg_n1097sim
        totaltime_n1068sim_12m = str(float(tinteg_n1068sim))+"h"
        totaltimetint_n1068sim_12m = totaltime_n1068sim_12m.replace(".","p")

        # ngc1068sim
        if do_template_n1068sim==True:
            self.prepare_template_n1068sim()

        if do_simint_n1068im==True:
            self.simaca_n1068sim(
            totaltime=totaltime_n1068sim_12m,
            totaltimetint=totaltimetint_n1068sim_12m,
            ) # imsize too large! manually change it!

        if do_imaging_n1068sim==True:
            self.phangs_pipeline_imaging(
                self.project_n1068,
                "12m",
                self.project_n1068+"_"+totaltimetint_n1068sim_12m,
                )

        # plot
        if plot_config==True:
            self.plot_config()

        # calc
        if calc_collectingarea==True:
            self.calc_collectingarea()

    #######################
    # calc_collectingarea #
    #######################

    def calc_collectingarea(self):
        """
        """

        # alma
        area_12m_array = 50 * (12/2.)**2 * np.pi
        area_7m_array  = 12 * (7/2.)**2 * np.pi
        area_tp_array  = 4 * (12/2.)**2 * np.pi
        area_alma      = 50 * (12/2.)**2 * np.pi + 12 * (7/2.)**2 * np.pi + 4 * (12/2.)**2 * np.pi
        area_aca       = 12 * (7/2.)**2 * np.pi + 4 * (12/2.)**2 * np.pi

        # LST
        area_lst       = (50/2.)**2 * np.pi

        # ratio to LST and str
        print_12m  = str(int(np.round(area_12m_array))) + " m^2 (this/lst = " + str(np.round(area_12m_array/area_lst,2)) + ")"
        print_7m   = str(int(np.round(area_7m_array))) + " m^2 (this/lst = " + str(np.round(area_7m_array/area_lst,2)) + ")"
        print_tp   = str(int(np.round(area_tp_array))) + " m^2 (this/lst = " + str(np.round(area_tp_array/area_lst,2)) + ")"
        print_alma = str(int(np.round(area_alma))) + " m^2 (this/lst = " + str(np.round(area_alma/area_lst,2)) + ")"
        print_aca  = str(int(np.round(area_aca))) + " m^2 (this/lst = " + str(np.round(area_aca/area_lst,2)) + ")"

        print_lst  = str(int(np.round(area_lst))) + " m^2"
        print_lst_12m  = str(int(np.round(area_lst+area_12m_array))) + " m^2 (this/12m = " + str(np.round((area_lst+area_12m_array)/area_12m_array,2)) + ")"
        print_lst_alma = str(int(np.round(area_lst+area_alma))) + " m^2 (this/alma = " + str(np.round((area_lst+area_alma)/area_alma,2)) + ")"
        print_lst_aca  = str(int(np.round(area_lst+area_aca))) + \
            " m^2 (this/aca = " + str(np.round((area_lst+area_aca)/area_aca,2)) + \
            ", this/lst = " + str(np.round((area_lst+area_aca)/area_lst,2)) + ")"

        print("#################################################")
        print("### Array collecting area and ratio")
        print("# ALMA 12m-array     = " + print_12m)
        print("# ALMA  7m-array     =  " + print_7m)
        print("# ALMA  TP-array     =  " + print_tp)
        print("# ")
        print("# ALMA 12m+7m+TP     = " + print_alma)
        print("# ACA 7m+TP          =  " + print_aca)
        print("#")
        print("# LST                = " + print_lst)
        print("#")
        print("# LST+ALMA 12m       = " + print_lst_12m)
        print("# LST+ALMA 12m+7m+TP = " + print_lst_alma)
        print("# LST+ACA 7m+TP      = " + print_lst_aca)
        print("##################################################")

        #################################################
        ### Array collecting area and ratio
        # ALMA 12m-array     = 5655 m^2 (this/lst = 2.88)
        # ALMA  7m-array     =  462 m^2 (this/lst = 0.24)
        # ALMA  TP-array     =  452 m^2 (this/lst = 0.23)
        # 
        # ALMA 12m+7m+TP     = 6569 m^2 (this/lst = 3.35)
        # ACA 7m+TP          =  914 m^2 (this/lst = 0.47)
        #
        # LST                = 1963 m^2
        #
        # LST+ALMA 12m       = 7618 m^2 (this/12m = 1.35)
        # LST+ALMA 12m+7m+TP = 8533 m^2 (this/alma = 1.3)
        # LST+ACA 7m+TP      = 2878 m^2 (this/aca = 3.15, this/lst = 1.47)
        ##################################################

    ###############
    # plot_config #
    ###############

    def plot_config(self):
        """
        Reference:
        http://math_research.uct.ac.za/~siphelo/admin/interferometry/4_Visibility_Space/4_4_1_UV_Coverage_UV_Tracks.html
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.config_c10,taskname)

        decl = -37.755 # 0=celestial equator, 90=north pole, -90=south pole
        tinteg = 1
        #lst_position = np.array([0,0,0]) # km/s
        lst_position = np.array([6.452141+0.1, 7.886675+0.1, -0.245131]) # km/s

        # get data
        data  = np.loadtxt(self.config_c10,"str")
        x_12m = data[:,0].astype(np.float32) / 1000.
        y_12m = data[:,1].astype(np.float32) / 1000.
        z_12m = data[:,2].astype(np.float32) / 1000.

        data  = np.loadtxt(self.config_7m,"str")
        x_7m  = data[:,0].astype(np.float32) / 1000.
        y_7m  = data[:,1].astype(np.float32) / 1000.
        z_7m  = data[:,2].astype(np.float32) / 1000.

        # get dist and angle: alma-alma baselines
        this_data = np.c_[x_12m.flatten(),y_12m.flatten(),z_12m.flatten()]
        #this_data = np.c_[x_12m.flatten()+[lst_position[0]],y_12m.flatten()+[lst_position[1]],z_12m.flatten()+[lst_position[2]]]
        u_alma, v_alma = self._get_baselines(this_data,this_data,decl=decl,tinteg=tinteg)
        u1_lst_center, v1_lst_center = self._get_baselines([lst_position],this_data,decl=decl,tinteg=tinteg)
        u2_lst_center, v2_lst_center = self._get_baselines(this_data,[lst_position],decl=decl,tinteg=tinteg)

        ##########################
        # plot: antenna position #
        ##########################
        ad    = [0.215,0.83,0.10,0.90]
        xlim  = [-10,10]
        ylim  = [-10,10]
        title = "Antenna positions"
        xlabel = "East-West (km)"
        ylabel = "North-South (km)"

        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        plt.subplots_adjust(left=ad[0], right=ad[1], bottom=ad[2], top=ad[3])
        myax_set(ax1, None, xlim, ylim, title, xlabel, ylabel, adjust=ad)

        ax1.scatter(6.452141+0.1, 7.886675+0.1, color="tomato", marker="*", lw=0, s=900)
        ax1.scatter(8, -8, color="tomato", marker="*", lw=0, s=900)
        ax1.scatter(-8, 0, color="tomato", marker="*", lw=0, s=900)
        ax1.scatter(0, 0, color="tomato", marker="*", lw=0, s=900)
        ax1.scatter(x_12m, y_12m, color="grey", lw=0, s=100)
        ax1.scatter(x_7m, y_7m, color="deepskyblue", lw=0, s=100)

        # text
        ax1.text(0.05,0.92, "ALMA 12-m array", color="grey", weight="bold", transform=ax1.transAxes)
        ax1.text(0.05,0.87, "ACA 7-m array", color="deepskyblue", weight="bold", transform=ax1.transAxes)
        ax1.text(0.05,0.82, "LSTsim 50-m", color="tomato", weight="bold", transform=ax1.transAxes)

        # save
        plt.subplots_adjust(hspace=.0)
        os.system("rm -rf " + self.outpng_config_12m)
        plt.savefig(self.outpng_config_12m, dpi=self.fig_dpi)

        ############
        # plot: uv #
        ############
        ad    = [0.215,0.83,0.10,0.90]
        xlim  = [-20,20]
        ylim  = [-20,20]
        title = "$u-v$ coverage"
        xlabel = "East-West (km)"
        ylabel = "North-South (km)"

        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        plt.subplots_adjust(left=ad[0], right=ad[1], bottom=ad[2], top=ad[3])
        myax_set(ax1, "both", xlim, ylim, title, xlabel, ylabel, adjust=ad)

        ax1.scatter(u_alma, v_alma, color="grey", lw=0, s=5, alpha=0.5)
        ax1.scatter(u1_lst_center, v1_lst_center, color="tomato", lw=0, s=5, alpha=0.5)
        ax1.scatter(u2_lst_center, v2_lst_center, color="tomato", lw=0, s=5, alpha=0.5)

        # text
        ax1.text(0.05,0.92, "Baselines: ALMA - ALMA", color="grey", weight="bold", transform=ax1.transAxes)
        ax1.text(0.05,0.87, "Baselines: ALMA - LSTsim", color="tomato", weight="bold", transform=ax1.transAxes)

        # save
        plt.subplots_adjust(hspace=.0)
        os.system("rm -rf " + self.outpng_uv_alma_lst1)
        plt.savefig(self.outpng_uv_alma_lst1, dpi=self.fig_dpi)

    ###########################
    # phangs_pipeline_imaging #
    ###########################

    def phangs_pipeline_imaging(self,this_proj,this_array,this_target):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        #check_first(self.template_fullspec,taskname)

        # ms
        sim_12m_ms_orig = self.dir_ready + "ms/" + this_proj + "." + self.config_c1 + ".noisy.ms"
        sim_7m_ms_orig  = self.dir_ready + "ms/" + this_proj + "." + self.config_7m + ".noisy.ms"

        # prepare dir_cleanmask
        dir_cleanmask = self.dir_ready + "outputs/cleanmasks/"
        if not glob.glob(dir_cleanmask):
            os.mkdir(dir_cleanmask)

        # set piepline
        master_key = self.dir_pipeline + "master_key.txt"

        pipepath = os.environ.get('PHANGSPIPE')
        if pipepath is not None:
            sys.path.append(os.environ.get('PHANGSPIPE'))
        else:
            sys.path.append(os.getcwd())

        # Check casa environment by importing CASA-only packages
        try:
            import taskinit
        except ImportError:
            print('Please run this script inside CASA!')
            sys.exit()

        # Set the logging
        from phangsPipeline import phangsLogger as pl
        reload(pl)
        pl.setup_logger(level='DEBUG', logfile=None)
        # Imports

        from phangsPipeline import handlerKeys as kh
        from phangsPipeline import handlerVis as uvh
        from phangsPipeline import handlerImaging as imh
        from phangsPipeline import handlerPostprocess as pph

        # Reloads for debugging
        reload(kh)
        reload(uvh)
        reload(imh)
        reload(pph)

        # Initialize key handler
        this_kh  = kh.KeyHandler(master_key = master_key)
        this_uvh = uvh.VisHandler(key_handler = this_kh)
        this_imh = imh.ImagingHandler(key_handler = this_kh)
        this_pph = pph.PostProcessHandler(key_handler= this_kh)
        dry_run_key = False
        this_uvh.set_dry_run(dry_run_key)
        this_imh.set_dry_run(dry_run_key)
        this_pph.set_dry_run(dry_run_key)

        # set handlers
        for this_hander in [this_uvh,this_imh,this_pph]:
            this_hander.set_targets(only=[this_target])
            this_hander.set_line_products(only=["ci10"])
            this_hander.set_no_cont_products(True)
            this_hander.set_no_line_products(False)
            this_hander.set_interf_configs(only=[this_array])

        # run piepline
        this_uvh.loop_stage_uvdata(\
                do_copy           = True,
                do_remove_staging = True,
                do_contsub        = False,
                do_extract_line   = True,
                do_extract_cont   = False,
                overwrite         = False,
                )
        this_imh.loop_imaging(\
                do_dirty_image          = True,
                do_revert_to_dirty      = False,
                do_read_clean_mask      = False,
                do_multiscale_clean     = False,
                do_revert_to_multiscale = False,
                do_singlescale_mask     = False,
                do_singlescale_clean    = True,
                do_export_to_fits       = False,
                extra_ext_in            = '',
                extra_ext_out           = '',
                )
        this_pph.loop_postprocess(\
                do_prep               = True,
                do_feather            = False,
                do_mosaic             = True,
                do_cleanup            = True,
                do_summarize          = True,
                # feather_apod          = True,
                feather_noapod        = True,
                # feather_before_mosaic = False,
                # feather_after_mosaic  = False,
                )

    ###################
    # simlst_n1097sim #
    ###################

    def simlst_n1097sim(self,singledish_res="3.04arcsec",totaltime="2.0h",totaltimetint="2p0tph",dryrun=True):
        """
        The totaltime is the ACA TP integration time to calculate TP achievable sensitivity.
        This module will calculate and map the same sensitivity (in K, not Jy/beam) using LST.
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.dir_ready+"inputs/"+self.n1097_template_fullspec,taskname)

        # ACA LST sim at 492.16065100 GHz
        # 3.04 arcsec resolution
        # "TP" sensitivity at 492.16065100 GHz based on ASC => same noise in K units (scaled by dish size when Jy/beam units)
        singledish_noise = 3.033450239598523 / 1000. / np.sqrt(float(totaltime.replace("h",""))) * (50.**2 / 12.**2)

        # calc pointing number
        header       = imhead(self.dir_ready+"inputs/"+self.n1097_template_fullspec,mode="list")
        area_in_as   = (header["shape"][0]*header["cdelt2"]*3600*180/np.pi) * (header["shape"][1]*header["cdelt2"]*3600*180/np.pi)
        one_hex_as   = (float(singledish_res.replace("arcsec",""))/2.0)**2 * 6/np.sqrt(3) # hex with half-beam length
        num_pointing = int(np.ceil(area_in_as / one_hex_as))

        # calc sensitivity per pointing
        singledish_noise_per_pointing = singledish_noise * np.sqrt(num_pointing)
        singledish_noise_per_pointing_K = 1.222e6 * float(singledish_res.replace("arcsec",""))**-2 * self.observed_freq**-2 * singledish_noise_per_pointing

        print("### LST observations with Tinteg     = use LST sensitivity calculator")
        print("# outputname = " + self.n1097_lstimage_fullspec.replace(".image","_"+totaltimetint+".image"))
        print("# sensitivity per pointing (Jy/beam) = " + str(np.round(singledish_noise_per_pointing,5)))
        print("# sensitivity per pointing (K)       = " + str(np.round(singledish_noise_per_pointing_K,5)))
        print("# beam size (arcsec)                 = " + str(np.round(float(singledish_res.replace("arcsec","")),2)))
        print("# number of pointing                 = " + str(num_pointing))
        print("# survey area (arcsec^2)             = " + str(int(area_in_as)))
        print("#")

        # run
        if dryrun==False:
            simtp(
                working_dir=self.dir_ready,
                template_fullspec=self.n1097_template_fullspec,
                sdimage_fullspec=self.n1097_lstimage_fullspec.replace(".image","_"+totaltimetint+".image"),
                sdnoise_image=self.n1097_lstnoise_image.replace(".image","_"+totaltimetint+".image"),
                singledish_res=singledish_res,
                singledish_noise=singledish_noise_per_pointing, # Jy/beam at final res
                )
        else:
            print("# skipped simtp as dryrun==True")
            print("#")

    ##################
    # simtp_n1097sim #
    ##################

    def simtp_n1097sim(self,singledish_res="11.8arcsec",totaltime="2.0h",totaltimetint="2p0h",dryrun=True):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.dir_ready+"inputs/"+self.n1097_template_fullspec,taskname)

        # ACA TP sim at 492.16065100 GHz
        # 11.8 arcsec resolution
        # sensitivity at 492.16065100 GHz based on ASC (1hr = 3.033450239598523 Jy/beam)
        singledish_noise = 3.033450239598523 / 1000. / np.sqrt(float(totaltime.replace("h","")))

        # calc pointing number
        header       = imhead(self.dir_ready+"inputs/"+self.n1097_template_fullspec,mode="list")
        area_in_as   = (header["shape"][0]*header["cdelt2"]*3600*180/np.pi) * (header["shape"][1]*header["cdelt2"]*3600*180/np.pi)
        one_hex_as   = (float(singledish_res.replace("arcsec",""))/4.0)**2 * 6/np.sqrt(3) # hex with 1/4-beam length
        num_pointing = int(np.ceil(area_in_as / one_hex_as))

        # calc sensitivity per pointing
        singledish_noise_per_pointing = singledish_noise * np.sqrt(num_pointing)
        singledish_noise_per_pointing_K = 1.222e6 * float(singledish_res.replace("arcsec",""))**-2 * self.observed_freq**-2 * singledish_noise_per_pointing

        print("### ACA TP observations with Tinteg  = " + totaltime)
        print("# outputname = " + self.n1097_sdimage_fullspec.replace(".image","_"+totaltimetint+".image"))
        print("# sensitivity per pointing (Jy/beam) = " + str(np.round(singledish_noise_per_pointing,5)))
        print("# sensitivity per pointing (K)       = " + str(np.round(singledish_noise_per_pointing_K,5)))
        print("# beam size (arcsec)                 = " + str(np.round(float(singledish_res.replace("arcsec","")),2)))
        print("# number of pointing                 = " + str(num_pointing))
        print("# survey area (arcsec^2)             = " + str(int(area_in_as)))
        print("#")

        # run
        if dryrun==False:
            simtp(
                working_dir=self.dir_ready,
                template_fullspec=self.n1097_template_fullspec,
                sdimage_fullspec=self.n1097_sdimage_fullspec.replace(".image","_"+totaltimetint+".image"),
                sdnoise_image=self.n1097_sdnoise_image.replace(".image","_"+totaltimetint+".image"),
                singledish_res=singledish_res,
                singledish_noise=singledish_noise_per_pointing, # Jy/beam at final res
                )
        else:
            print("# skipped simtp as dryrun==True")
            print("#")

    ###################
    # simaca_n1097sim #
    ###################

    def simaca_n1097sim(self,totaltime="2.0h",totaltimetint="2p0h"):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.n1097_template_fullspec,taskname)

        run_simobserve(
            working_dir=self.dir_ready,
            template=self.n1097_template_fullspec,
            antennalist=self.config_7m,
            project=self.project_n1097+"_7m_"+totaltimetint,
            totaltime=totaltime,
            incenter=self.incenter,
            )

    ###################
    # simaca_n1068sim #
    ###################

    def simaca_n1068sim(self,totaltime="2.0h",totaltimetint="2p0h"):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.n1068_template_fullspec,taskname)

        run_simobserve(
            working_dir=self.dir_ready,
            template=self.n1068_template_fullspec,
            antennalist=self.config_c10,
            project=self.project_n1068+"_7m_"+totaltimetint,
            totaltime=totaltime,
            incenter=self.incenter,
            )

    #############################
    # prepare_template_n1097sim #
    #############################

    def prepare_template_n1097sim(self):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.n1097_template_file,taskname)

        gen_cube(
            template_dir=self.dir_raw,
            template_file=self.n1097_template_file,
            template_mask=self.n1097_template_mask,
            working_dir=self.dir_ready,
            template_in_jypix=self.n1097_template_in_jypix,
            template_clipped=self.n1097_template_clipped,
            template_mask_imported=self.n1097_template_mask_imported,
            template_rotated=self.n1097_template_rotated,
            template_shrunk=self.n1097_template_shrunk,
            template_fullspec=self.n1097_template_fullspec,
            template_fullspec_div3=self.n1097_template_fullspec_div3,
            template_fullspec_div5=self.n1097_template_fullspec_div5,
            template_fullspec_div10=self.n1097_template_fullspec_div10,
            template_fullspec_div30=self.n1097_template_fullspec_div30,
            template_fullspec_div100=self.n1097_template_fullspec_div100,
            template_withcont=self.n1097_template_withcont,
            template_withcont_div3=self.n1097_template_withcont_div3,
            template_withcont_div5=self.n1097_template_withcont_div5,
            template_withcont_div10=self.n1097_template_withcont_div10,
            template_withcont_div30=self.n1097_template_withcont_div30,
            template_withcont_div100=self.n1097_template_withcont_div100,
            pa=self.image_roration, # rotation angle
            )

    #############################
    # prepare_template_n1068sim #
    #############################

    def prepare_template_n1068sim(self):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.n1068_template_file,taskname)

        gen_cube(
            template_dir=self.dir_raw,
            template_file=self.n1068_template_file,
            template_mask=self.n1068_template_mask,
            working_dir=self.dir_ready,
            template_in_jypix=self.n1068_template_in_jypix,
            template_clipped=self.n1068_template_clipped,
            template_mask_imported=self.n1068_template_mask_imported,
            template_rotated=self.n1068_template_rotated,
            template_shrunk=self.n1068_template_shrunk,
            template_fullspec=self.n1068_template_fullspec,
            template_fullspec_div3=self.n1068_template_fullspec_div3,
            template_fullspec_div5=self.n1068_template_fullspec_div5,
            template_fullspec_div10=self.n1068_template_fullspec_div10,
            template_fullspec_div30=self.n1068_template_fullspec_div30,
            template_fullspec_div100=self.n1068_template_fullspec_div100,
            template_withcont=self.n1068_template_withcont,
            template_withcont_div3=self.n1068_template_withcont_div3,
            template_withcont_div5=self.n1068_template_withcont_div5,
            template_withcont_div10=self.n1068_template_withcont_div10,
            template_withcont_div30=self.n1068_template_withcont_div30,
            template_withcont_div100=self.n1068_template_withcont_div100,
            pa=self.image_roration, # rotation angle
            shrink=0.1,
            )

    ##################
    # _get_baselines #
    ##################

    def _get_baselines(self,x,y,decl=60,tinteg=0):
        """
        """
        latitude = np.radians(-67.755) # degree, alma site

        list_dist  = []
        list_theta = []
        list_phi   = []
        combinations = itertools.product(x,y)
        for comb in combinations:
            this_vec = comb[0] - comb[1]

            this_d = np.linalg.norm(this_vec)
            list_dist.append(this_d)

            this_a = np.degrees(np.arcsin(this_vec[2] / this_d))
            list_theta.append(this_a)

            this_h = np.degrees(np.arctan2(this_vec[0], this_vec[1]))
            list_phi.append(this_h)

        l = np.array(list_dist)
        t = np.radians( np.array(list_theta) )
        p = np.radians( np.array(list_phi) )
        dec = np.radians(decl)

        X = l*(np.cos(latitude)*np.sin(t) - np.sin(latitude)*np.cos(t)*np.cos(p)) # l*np.sin(t)*np.cos(p)
        Y = l*np.cos(t)*np.sin(p)
        Z = l*(np.sin(latitude)*np.sin(t) + np.cos(latitude)*np.cos(t)*np.cos(p)) # l*np.cos(t)

        # output
        list_u = []
        list_v = []
        trange = np.r_[np.arange(-tinteg/24.*360/2.0, tinteg/24.*360/2.0, 0.1), tinteg/24.*360/2.0]
        for this_t in trange:
            H = np.radians(this_t)
 
            this_u = X*np.sin(H) + Y*np.cos(H)
            this_v = -X*np.sin(dec)*np.cos(H) + Y*np.sin(dec)*np.sin(H) + Z*np.cos(dec)

            # output
            list_u = np.r_[list_u, this_u]
            list_v = np.r_[list_v, this_v]

        return list_u, list_v

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

######################
# end of ToolsLSTSim #
######################