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
> tl = tools(
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
2022-03-19   CASA tools sm and vp for heterogenous array sim
Toshiki Saito@Nichidai/NAOJ
"""

import os, sys, glob
import numpy as np
from scipy.stats import gaussian_kde

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
            self.modname = "ToolsLSTSim."
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
        self.project_torus = self._read_key("project_torus")
        self.project_n1097 = self._read_key("project_n1097")
        self.project_check = "checksim"
        self.config_c1     = self.dir_keyfile + self._read_key("config_c1")
        self.config_c9     = self.dir_keyfile + self._read_key("config_c9")
        self.config_c9_lst = self.dir_keyfile + self._read_key("config_c9_lst")
        self.config_c10    = self.dir_keyfile + self._read_key("config_c10")
        self.config_7m     = self.dir_keyfile + self._read_key("config_7m")
        self.config_lst    = self.dir_keyfile + self._read_key("config_lst")

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

        self.n1097_feather_tp_7m            = self.dir_ready + "outputs/" + self._read_key("n1097_feather_tp_7m")
        self.n1097_feather_lst_7m           = self.dir_ready + "outputs/" + self._read_key("n1097_feather_lst_7m")

        self.torus_template_file            = self._read_key("torus_template_file")
        self.check_template_file            = self._read_key("check_template_file")

        self.mom0_input                     = self.dir_ready + "outputs/" + "ngc1097sim_mom0_input.fits"
        self.mom0_tp                        = self.dir_ready + "outputs/" + "ngc1097sim_mom0_TP12m.fits"
        self.mom0_7m_tp                     = self.dir_ready + "outputs/" + "ngc1097sim_mom0_7m+TP.fits"
        self.mom0_lst                       = self.dir_ready + "outputs/" + "ngc1097sim_mom0_LST50m.fits"
        self.mom0_7m_lst                    = self.dir_ready + "outputs/" + "ngc1097sim_mom0_7m+LST.fits"

        self.dust_input                     = self.dir_ready + "outputs/" + "torussim_dust_input.fits"
        self.dust_c9                        = self.dir_ready + "outputs/" + "torussim_dust_c9.fits"
        self.dust_c9_lst                    = self.dir_ready + "outputs/" + "torussim_dust_c9_lst.fits"

    def _set_input_param(self):
        """
        """

        # sim properties
        self.singledish_noise = 0.102 # Jy/beam at final res
        self.singledish_res   = "28.37arcsec" # resolution
        self.image_rot_n1068sim = "0deg"
        self.image_rot_n1097sim = "35deg"

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

        self.outpng_config_12m    = self.dir_products + self._read_key("outpng_config_12m")
        self.outpng_config_7m     = self.dir_products + self._read_key("outpng_config_7m")
        self.outpng_uv_alma_lst1  = self.dir_products + self._read_key("outpng_uv_alma_lst1")
        self.outpng_mosaic_7m     = self.dir_products + self._read_key("outpng_mosaic_7m")
        self.outpng_mosaic_c1     = self.dir_products + self._read_key("outpng_mosaic_c1")
        self.outpng_uv_aca        = self.dir_products + self._read_key("outpng_uv_aca")

        self.outpng_mom0_input     = self.dir_products + self._read_key("outpng_mom0_input")
        self.outpng_mom0_tp        = self.dir_products + self._read_key("outpng_mom0_tp")
        self.outpng_mom0_lst50m    = self.dir_products + self._read_key("outpng_mom0_lst50m")
        self.outpng_mom0_tp_7m     = self.dir_products + self._read_key("outpng_mom0_tp_7m")
        self.outpng_mom0_7m_lst50m = self.dir_products + self._read_key("outpng_mom0_7m_lst50m")
        self.outpng_scatter_n1097sim = self.dir_products + self._read_key("outpng_scatter_n1097sim")
        self.outpng_hist_n1097sim  = self.dir_products + self._read_key("outpng_hist_n1097sim")

        self.outpng_dust_input    = self.dir_products + self._read_key("outpng_dust_input")
        self.outpng_dust_input_original = self.dir_products + self._read_key("outpng_dust_input_original")
        self.outpng_dust_c9_lst   = self.dir_products + self._read_key("outpng_dust_c9_lst")
        self.outpng_dust_c9       = self.dir_products + self._read_key("outpng_dust_c9")

    ####################
    # run_sim_lst_alma #
    ####################

    def run_sim_lst_alma(
        self,
        ##############
        # ngc1097sim #
        ##############
        # prepare
        tinteg_n1097sim        = 48,    # 7m total observing time
        observed_freq          = 492.16065100, # GHz, determine LST and TP beam sizes
        do_template_n1097sim   = False, # create "wide" template cube for mapping simobserve
        # ACA-alone
        do_simACA_n1097sim     = False, # sim ACA band 8 for big ngc1097sim
        do_imaging_n1097sim    = False, # imaging sim ms
        # SD-alone
        dryrun_simSD           = False, # just output SD mapping parameters
        do_simTP_n1097sim      = False, # sim ACA TP alone; after do_imaging_n1097sim
        do_simLST_n1097sim     = False, # sim LST alone; after do_imaging_n1097sim
        do_feather             = False,
        do_tp2vis              = False, # not implemented yet
        # postprocess
        do_mom0_n1097sim       = False,
        # LST-connected 7m array (decomissioned)
        # unrearistic observing time due to too small FoV of 50m!
        #do_simACA_LST_n1097sim = False,
        #do_imaging_simACA_LST  = False,
        #
        ############
        # torussim #
        ############
        # prepare
        tinteg_torussim        = 24,
        do_template_torussim   = False, # create "compact" template cube for long-baseline simobserve
        do_simint_torussim     = False, # sim C-10 band 9
        do_imaging_torussim    = False, # imaging sim ms
        do_process_torussim    = False,
        #
        ########
        # plot #
        ########
        # plot
        plot_config            = False,
        plot_mosaic            = False,
        plot_mom0_n1097sim     = False,
        plot_scatter_n1097sim  = False,
        plot_mom0_torussim     = False,
        # calc
        calc_collectingarea    = False,
        #
        ############
        # checksim #
        ############
        tinteg_checksim        = 1,
        do_template_checksim   = False,
        do_simint_checksim     = False,
        do_imaging_checksim    = False,
        ):
        """
        This method runs all the methods which will create figures in the white paper.
        """

        # keep these toggles on (for code-readability purpose)
        toggle_ngc1097sim = True
        toggle_torussim   = True
        toggle_plot       = True
        toggle_checksim   = True

        ##############
        # ngc1097sim #
        ##############
        if toggle_ngc1097sim==True:
            #############################
            # set ngc1097sim parameters #
            #############################
            # observed frequency
            self.observed_freq = observed_freq
            self.incenter      = str(observed_freq)+"GHz"

            # n1097sim_7m from tinteg_n1097sim
            tinteg_7m    = str(float(tinteg_n1097sim))+"h"
            tintegstr_7m = tinteg_7m.replace(".","p")
            this_target  = self.project_n1097+"_"+tintegstr_7m
            this_target_connected = self.project_n1097+"xLST_"+tintegstr_7m

            # determine LST and TP beam sizes
            lst_beam    = str(12.979 * 115.27120 / self.observed_freq)+"arcsec"
            tp_beam     = str(50.6   * 115.27120 / self.observed_freq)+"arcsec"
            lst30m_beam = str(21.631 * 115.27120 / self.observed_freq)+"arcsec"

            # define products
            cube_tp  = self.dir_ready+"outputs/"+self.n1097_sdimage_fullspec.replace(".image","_"+tintegstr_7m+"7m.image")
            cube_lst = self.dir_ready+"outputs/"+self.n1097_lstimage_fullspec.replace(".image","_"+tintegstr_7m+"7m.image")
            cube_7m  = self.dir_ready+"outputs/postprocess/"+this_target+"/"+this_target+"_7m_ci10_pbcorr_trimmed.image"

            ##################
            # run ngc1097sim #
            ##################
            if do_template_n1097sim==True:
                self.prepare_template_n1097sim()

            if do_simACA_n1097sim==True:
                self.simaca_n1097sim(tinteg_7m,tintegstr_7m)

            if do_imaging_n1097sim==True:
                self.phangs_pipeline_imaging(
                    this_proj=self.project_n1097,
                    this_array="7m",
                    this_target=this_target,
                    )

            if do_simTP_n1097sim==True:
                self.simtp_n1097sim(tp_beam,tintegstr_7m,dryrun_simSD)

            if do_simLST_n1097sim==True:
                self.simlst_n1097sim(lst_beam,tp_beam,tintegstr_7m,dryrun_simSD)
                self.simlst_n1097sim(lst30m_beam,tp_beam,tintegstr_7m,True)

            if do_feather==True:
                self.do_feather(cube_7m,cube_tp,self.n1097_feather_tp_7m,-1)
                self.do_feather(cube_7m,cube_lst,self.n1097_feather_lst_7m,-1)

            if do_mom0_n1097sim==True:
                self.create_mom0(tintegstr_7m)

            #if do_simACA_LST_n1097sim==True:
            #    self.do_simaca_lst_n1097sim(tinteg_7m,tintegstr_7m)
            #
            #if do_imaging_simACA_LST==True:
            #    self.phangs_pipeline_imaging(
            #        this_proj=self.project_n1097+"_LSTconnected_7m_"+tintegstr_7m,
            #        this_array="7m",
            #        this_target=this_target_connected,
            #        )

        ############
        # torussim #
        ############
        if toggle_torussim==True:
            ###########################
            # set torussim parameters #
            ###########################
            tinteg_12m      = str(float(tinteg_torussim))+"h"
            tintegstr_12m   = tinteg_12m.replace(".","p")
            this_target     = self.project_torus+"_"+tintegstr_12m
            this_target_lst = self.project_torus+"_lst_"+tintegstr_12m

            ################
            # run torussim #
            ################
            if do_template_torussim==True:
                self.prepare_template_torussim()

            if do_simint_torussim==True:
                self.sim12m_torussim(tinteg_12m,tintegstr_12m)

            if do_imaging_torussim==True:
                #############
                # config_c9 #
                #############
                # stage instead of pipeline
                msname  = self.project_torus + "_12m_" + tintegstr_12m + "."+self.config_c9.split("/")[-1].split(".cfg")[0]+".noisy.ms"
                ms_from = self.dir_ready + "ms/" + self.project_torus + "_12m_" + tintegstr_12m + "/" + msname
                dir_to  = self.dir_ready + "outputs/imaging/" + this_target + "/"
                ms_to   = dir_to + this_target + "_12m_cont.ms"
                os.system("rm -rf " + ms_to)
                os.system("rm -rf " + dir_to)
                os.makedirs(dir_to)
                os.system("cp -r " + ms_from + " " + ms_to)

                # run
                self.phangs_pipeline_imaging(
                    this_proj=self.project_torus,
                    this_array="12m",
                    this_target=this_target,
                    do_cont=True,
                    only_dirty=False,
                    )

                #################
                # config_c9_lst #
                #################
                # stage instead of pipeline
                msname  = self.project_torus + "_12m_lst_" + tintegstr_12m + "."+self.config_c9_lst.split("/")[-1].split(".cfg")[0]+".noisy.ms"
                ms_from = self.dir_ready + "ms/" + self.project_torus + "_12m_lst_" + tintegstr_12m + "/" + msname
                dir_to  = self.dir_ready + "outputs/imaging/" + this_target_lst + "/"
                ms_to   = dir_to + this_target_lst + "_12m_cont.ms"
                os.system("rm -rf " + ms_to)
                os.system("rm -rf " + dir_to)
                os.makedirs(dir_to)
                os.system("cp -r " + ms_from + " " + ms_to)

                # run
                self.phangs_pipeline_imaging(
                    this_proj=self.project_torus,
                    this_array="12m",
                    this_target=this_target_lst,
                    do_cont=True,
                    only_dirty=False,
                    )

            if do_process_torussim==True:
                self.create_map_torussim()

        ########
        # plot #
        ########
        if toggle_plot==True:
            # plot
            if plot_config==True:
                self.plot_config()

                this_target = self.project_n1097+"_"+tintegstr_7m
                dir_ms  = self.dir_ready + "outputs/imaging/" + this_target + "/"
                this_ms = dir_ms + this_target + "_7m_ci10.ms"
                self.plot_uv(this_ms,self.outpng_uv_aca,[-50,50,-50,50])

                this_target = self.project_n1097+"_"+tintegstr_12m
                dir_ms  = self.dir_ready + "outputs/imaging/" + this_target + "/"
                this_ms = dir_ms + this_target + "_12m_cont.ms"
                self.plot_uv(this_ms,self.outpng_uv_aca,[-20,20,-20,20],unit="km")

            if plot_mosaic==True:
                self.plot_mosaic_7m(tintegstr_7m,self.observed_freq)
                self.plot_mosaic_C1(tintegstr_ch,693.9640232)

            if plot_mom0_n1097sim==True:
                self.plot_mom0()

            if plot_scatter_n1097sim==True:
                self.plot_scatter_n1097sim()

            if plot_mom0_torussim==True:
                self.plot_mom0_torussim()

            # calc
            if calc_collectingarea==True:
                self.calc_collectingarea()

        ############
        # checksim #
        ############
        if toggle_checksim==True:
            ###########################
            # set checksim parameters #
            ###########################
            tinteg_ch    = str(float(tinteg_checksim))+"h"
            tintegstr_ch = tinteg_ch.replace(".","p")
            this_target  = self.project_check+"_"+tintegstr_ch
            lst_beam     = str(12.979 * 115.27120 / 693.9640232)+"arcsec"
            lst_nyquist  = str(12.979/2.0 * 115.27120 / 693.9640232)+"arcsec"

            ################
            # run checksim #
            ################
            if do_template_checksim==True:
                self.prepare_template_checksim()

            if do_simint_checksim==True:
                self.sim12m_checksim(tinteg_ch,tintegstr_ch,lst_nyquist)

            if do_imaging_checksim==True:
                # stage instead of pipeline
                msname  = self.project_check + "_12m_" + tintegstr_ch + "."+self.config_c1.split("/")[-1].split(".cfg")[0]+".noisy.ms"
                ms_from = self.dir_ready + "ms/" + self.project_check + "_12m_" + tintegstr_ch + "/" + msname
                dir_to  = self.dir_ready + "outputs/imaging/" + this_target + "/"
                ms_to   = dir_to + this_target + "_12m_cont.ms"
                os.system("rm -rf " + ms_to)
                os.system("rm -rf " + dir_to)
                os.makedirs(dir_to)
                os.system("cp -r " + ms_from + " " + ms_to)

                # run
                self.phangs_pipeline_imaging(
                    this_proj=self.project_check,
                    this_array="12m",
                    this_target=this_target,
                    do_cont=True,
                    )

    #########################
    # plot_scatter_n1097sim #
    #########################

    def plot_scatter_n1097sim(
        self,
        snr=5.0,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.mom0_input,taskname)

        # import mom0
        l,_ = imval_all(self.mom0_input)
        l = l["data"]
        data_input = np.array(l.flatten())

        l,_ = imval_all(self.mom0_7m_tp)
        l = l["data"]
        data_7m_tp = np.array(l.flatten())

        l,_ = imval_all(self.mom0_lst)
        l = l["data"]
        data_lst = np.array(l.flatten())

        # import emom0
        l,_ = imval_all(self.mom0_7m_tp.replace(".fits","_err.fits"))
        l = l["data"]
        err_7m_tp = np.array(l.flatten())

        l,_ = imval_all(self.mom0_lst.replace(".fits","_err.fits"))
        l = l["data"]
        err_lst = np.array(l.flatten())

        # process data
        cut        = np.where((data_7m_tp>abs(err_7m_tp)*snr)&(data_input>0))
        x_7m_tp    = np.log10(data_input[cut])
        y_7m_tp    = np.log10(data_7m_tp[cut])
        yerr_7m_tp = abs(err_7m_tp[cut]) / data_7m_tp[cut]

        cut        = np.where((data_lst>abs(err_lst)*snr)&(data_input>0))
        x_lst      = np.log10(data_input[cut])
        y_lst      = np.log10(data_lst[cut])
        yerr_lst   = abs(err_lst[cut]) / data_lst[cut]

        # histogram
        kdelim = [-1.0,2.0]
        ygrid  = np.linspace(kdelim[0], kdelim[1], num=1000)
        l = gaussian_kde(y_7m_tp - x_7m_tp)
        kde_7m_tp = np.array(l(ygrid))
        p16_7m_tp = str(np.round(np.percentile(y_7m_tp - x_7m_tp, 16),2))
        p50_7m_tp = str(np.round(np.percentile(y_7m_tp - x_7m_tp, 50),2))
        p84_7m_tp = str(np.round(np.percentile(y_7m_tp - x_7m_tp, 84),2))

        l = gaussian_kde(y_lst - x_lst)
        kde_lst = np.array(l(ygrid))
        p16_lst = str(np.round(np.percentile(y_lst - x_lst, 16),2))
        p50_lst = str(np.round(np.percentile(y_lst - x_lst, 50),2))
        p84_lst = str(np.round(np.percentile(y_lst - x_lst, 84),2))

        ########
        # plot #
        ########
        ad   = [0.215,0.83,0.10,0.90]
        xlim = [-1.2,2.2]
        ylim = [-1.2,2.2]

        # prepare
        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        plt.subplots_adjust(left=ad[0], right=ad[1], bottom=ad[2], top=ad[3])
        myax_set(ax1, "both", xlim, ylim, "pixel-by-pixel scatter", "Convolved input (K km s$^{^1}$)", "Convolved output (K km s$^{^1}$)", adjust=ad)

        # plot
        ax1.scatter(x_7m_tp, y_7m_tp, c="deepskyblue", lw=0, s=20, zorder=1e9)
        ax1.errorbar(x_7m_tp, y_7m_tp, yerr=yerr_7m_tp, lw=0.5, capsize=0, color="grey", linestyle="None")

        ax1.scatter(x_lst, y_lst, c="tomato", lw=0, s=20, zorder=1e9)
        ax1.errorbar(x_lst, y_lst, yerr=yerr_lst, lw=0.5, capsize=0, color="grey", linestyle="None")

        # text
        ax1.text(0.05,0.92, "7m+TP map", color="deepskyblue", weight="bold", transform=ax1.transAxes)
        ax1.text(0.05,0.87, "LST map", color="tomato", weight="bold", transform=ax1.transAxes)

        # ann
        ax1.plot(xlim, ylim, "--", color="black", lw=2,zorder=1e10)
        ax1.plot(xlim, [ylim[0]+1.0,ylim[1]+1.0], "--", color="black", lw=2,zorder=1e10)
        ax1.plot(xlim, [ylim[0]-1.0,ylim[1]-1.0], "--", color="black", lw=2,zorder=1e10)

        # save
        plt.subplots_adjust(hspace=.0)
        os.system("rm -rf " + self.outpng_scatter_n1097sim)
        plt.savefig(self.outpng_scatter_n1097sim, dpi=self.fig_dpi)

        ##################
        # plot histogram #
        ##################
        ad    = [0.215,0.83,0.10,0.90]
        xlim  = kdelim
        ylim  = [0,np.max(kde_7m_tp)*1.1]
        ylim2 = [0,np.max(kde_lst)*1.1]

        # prepare
        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:5,0:10])
        ax2 = plt.subplot(gs[5:10,0:10])
        plt.subplots_adjust(left=ad[0], right=ad[1], bottom=ad[2], top=ad[3])
        myax_set(ax1, "x", xlim, ylim, "Output/input KDE", None, None, adjust=ad)
        myax_set(ax2, "x", xlim, ylim2, None, "log Ratio", "Density", adjust=ad)
        ax1.tick_params(labelbottom=False)

        # plot
        ax1.plot(ygrid, kde_7m_tp, lw=2, color="grey")
        ax1.fill_between(ygrid, 0, kde_7m_tp, facecolor="deepskyblue", alpha=0.5, lw=0)

        ax2.plot(ygrid, kde_lst, lw=2, color="grey")
        ax2.fill_between(ygrid, 0, kde_lst, facecolor="tomato", alpha=0.5, lw=0)

        # text
        ax1.text(0.95,0.87, "7m+TP/input ratio", color="deepskyblue", weight="bold", ha="right", transform=ax1.transAxes)
        ax2.text(0.95,0.87, "LST/input ratio", color="tomato", weight="bold", ha="right", transform=ax2.transAxes)

        # ann
        ax1.text(0.75,0.77, "16$^{th}$ = " + p16_7m_tp, ha="left", transform=ax1.transAxes)
        ax1.text(0.75,0.67, "50$^{th}$ = " + p50_7m_tp, ha="left", transform=ax1.transAxes)
        ax1.text(0.75,0.57, "84$^{th}$ = " + p84_7m_tp, ha="left", transform=ax1.transAxes)

        ax2.text(0.75,0.77, "16$^{th}$ = " + p16_lst, ha="left", transform=ax2.transAxes)
        ax2.text(0.75,0.67, "50$^{th}$ = " + p50_lst, ha="left", transform=ax2.transAxes)
        ax2.text(0.75,0.57, "84$^{th}$ = " + p84_lst, ha="left", transform=ax2.transAxes)

        # save
        plt.subplots_adjust(hspace=.0)
        os.system("rm -rf " + self.outpng_hist_n1097sim)
        plt.savefig(self.outpng_hist_n1097sim, dpi=self.fig_dpi)

    ###############
    # create_mom0 #
    ###############

    def create_mom0(
        self,
        totaltimetint,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.dir_ready+"inputs/"+self.n1097_template_fullspec,taskname)

        ################
        # define input #
        ################
        cube_input  = self.dir_ready+"inputs/"+self.n1097_template_fullspec
        cube_tp     = self.dir_ready+"outputs/"+self.n1097_sdimage_fullspec.replace(".image","_"+totaltimetint+"7m.image")
        cube_7m_tp  = self.n1097_feather_tp_7m
        cube_7m_lst = self.n1097_feather_lst_7m
        cube_lst    = self.dir_ready+"outputs/"+self.n1097_lstimage_fullspec.replace(".image","_"+totaltimetint+"7m.image")

        ##############
        # importfits #
        ##############
        run_importfits(cube_7m_tp,cube_7m_tp.replace(".fits",".image"))
        cube_7m_tp = cube_7m_tp.replace(".fits",".image")

        run_importfits(cube_7m_lst,cube_7m_lst.replace(".fits",".image"))
        cube_7m_lst = cube_7m_lst.replace(".fits",".image")

        ################################################
        # convolve beam and regrid velocity resolution #
        ################################################
        imhead(cube_input,mode="del",hdkey="beammajor")
        run_roundsmooth(cube_input,cube_input+"_tmp2",3.5,0.001,targetres=False)
        imhead(cube_input+"_tmp2",mode="put",hdkey="restfreq",hdvalue=str(self.observed_freq*1e9))
        imhead(cube_input+"_tmp2",mode="put",hdkey="crval3",hdvalue=str(self.observed_freq*1e9))

        ############################
        # regrid to common xy grid #
        ############################
        os.system("cp -r " + cube_input + " xytemplate.image")
        imhead("xytemplate.image",mode="del",hdkey="beamminor")
        imhead("xytemplate.image",mode="del",hdkey="beammajor")

        run_imregrid(cube_tp,"xytemplate.image",cube_tp+"_tmp0",axes=[0,1])
        run_imregrid(cube_7m_tp,"xytemplate.image",cube_7m_tp+"_tmp0",axes=[0,1])
        run_imregrid(cube_7m_lst,"xytemplate.image",cube_7m_lst+"_tmp0",axes=[0,1])
        run_imregrid(cube_lst,"xytemplate.image",cube_lst+"_tmp0",axes=[0,1])

        run_roundsmooth(cube_tp+"_tmp0",cube_tp+"_tmp1",3.5,delin=True)
        run_roundsmooth(cube_7m_tp+"_tmp0",cube_7m_tp+"_tmp1",3.5,delin=True)
        run_roundsmooth(cube_7m_lst+"_tmp0",cube_7m_lst+"_tmp1",3.5,delin=True)
        run_roundsmooth(cube_lst+"_tmp0",cube_lst+"_tmp1",3.5,delin=True)

        ###########
        # masking #
        ###########
        run_immath_one(cube_tp+"_tmp1","mask.image","iif(IM0>=0.3,1,0)")

        bmaj = imhead(cube_tp+"_tmp1",mode="get",hdkey="beammajor")["value"]
        bmin = imhead(cube_tp+"_tmp1",mode="get",hdkey="beamminor")["value"]
        bpa  = imhead(cube_tp+"_tmp1",mode="get",hdkey="beampa")["value"]
        os.system("rm -rf this_temp.image")
        os.system("cp -r " + cube_tp+"_tmp1 this_temp.image")
        run_imregrid("mask.image","this_temp.image","mask.image2",axes=[2])
        run_immath_two(cube_tp+"_tmp1","mask.image2",cube_tp+"_tmp2","iif(IM1>0,IM0,0)",delin=True)
        imhead(cube_tp+"_tmp2",mode="del",hdkey="beammajor")
        imhead(cube_tp+"_tmp2",mode="put",hdkey="beammajor",hdvalue=str(bmaj)+"arcsec")
        imhead(cube_tp+"_tmp2",mode="put",hdkey="beamminor",hdvalue=str(bmin)+"arcsec")
        imhead(cube_tp+"_tmp2",mode="put",hdkey="beampa",hdvalue=str(bpa)+"deg")

        bmaj = imhead(cube_7m_tp+"_tmp1",mode="get",hdkey="beammajor")["value"]
        bmin = imhead(cube_7m_tp+"_tmp1",mode="get",hdkey="beamminor")["value"]
        bpa  = imhead(cube_7m_tp+"_tmp1",mode="get",hdkey="beampa")["value"]
        os.system("rm -rf this_temp.image")
        os.system("cp -r " + cube_7m_tp+"_tmp1 this_temp.image")
        run_imregrid("mask.image","this_temp.image","mask.image2",axes=[2])
        run_immath_two(cube_7m_tp+"_tmp1","mask.image2",cube_7m_tp+"_tmp2","iif(IM1>0,IM0,0)",delin=True)
        imhead(cube_7m_tp+"_tmp2",mode="del",hdkey="beammajor")
        imhead(cube_7m_tp+"_tmp2",mode="put",hdkey="beammajor",hdvalue=str(bmaj)+"arcsec")
        imhead(cube_7m_tp+"_tmp2",mode="put",hdkey="beamminor",hdvalue=str(bmin)+"arcsec")
        imhead(cube_7m_tp+"_tmp2",mode="put",hdkey="beampa",hdvalue=str(bpa)+"deg")

        bmaj = imhead(cube_7m_lst+"_tmp1",mode="get",hdkey="beammajor")["value"]
        bmin = imhead(cube_7m_lst+"_tmp1",mode="get",hdkey="beamminor")["value"]
        bpa  = imhead(cube_7m_lst+"_tmp1",mode="get",hdkey="beampa")["value"]
        os.system("rm -rf this_temp.image")
        os.system("cp -r " + cube_7m_lst+"_tmp1 this_temp.image")
        run_imregrid("mask.image","this_temp.image","mask.image2",axes=[2])
        run_immath_two(cube_7m_lst+"_tmp1","mask.image2",cube_7m_lst+"_tmp2","iif(IM1>0,IM0,0)",delin=True)
        imhead(cube_7m_lst+"_tmp2",mode="del",hdkey="beammajor")
        imhead(cube_7m_lst+"_tmp2",mode="put",hdkey="beammajor",hdvalue=str(bmaj)+"arcsec")
        imhead(cube_7m_lst+"_tmp2",mode="put",hdkey="beamminor",hdvalue=str(bmin)+"arcsec")
        imhead(cube_7m_lst+"_tmp2",mode="put",hdkey="beampa",hdvalue=str(bpa)+"deg")

        bmaj = imhead(cube_lst+"_tmp1",mode="get",hdkey="beammajor")["value"]
        bmin = imhead(cube_lst+"_tmp1",mode="get",hdkey="beamminor")["value"]
        bpa  = imhead(cube_lst+"_tmp1",mode="get",hdkey="beampa")["value"]
        os.system("rm -rf this_temp.image")
        os.system("cp -r " + cube_lst+"_tmp1 this_temp.image")
        run_imregrid("mask.image","this_temp.image","mask.image2",axes=[2])
        run_immath_two(cube_lst+"_tmp1","mask.image2",cube_lst+"_tmp2","iif(IM1>0,IM0,0)",delin=True)
        imhead(cube_lst+"_tmp2",mode="del",hdkey="beammajor")
        imhead(cube_lst+"_tmp2",mode="put",hdkey="beammajor",hdvalue=str(bmaj)+"arcsec")
        imhead(cube_lst+"_tmp2",mode="put",hdkey="beamminor",hdvalue=str(bmin)+"arcsec")
        imhead(cube_lst+"_tmp2",mode="put",hdkey="beampa",hdvalue=str(bpa)+"deg")

        os.system("rm -rf mask.image mask.image2 this_temp.image xytemplate.image")

        #####################
        # convert to Kelvin #
        #####################
        bmaj = imhead(imagename=cube_input+"_tmp2",mode="get",hdkey="beammajor")["value"]
        bmin = imhead(imagename=cube_input+"_tmp2",mode="get",hdkey="beamminor")["value"]
        expr = "IM0*"+str(1.222e6/bmaj/bmin/self.observed_freq**2)
        run_immath_one(cube_input+"_tmp2",cube_input+"_tmp3",expr,delin=True)
        imhead(cube_input+"_tmp3",mode="del",hdkey="bunit")
        imhead(cube_input+"_tmp3",mode="put",hdkey="bunit",hdvalue="K")

        bmaj = imhead(imagename=cube_tp+"_tmp2",mode="get",hdkey="beammajor")["value"]
        bmin = imhead(imagename=cube_tp+"_tmp2",mode="get",hdkey="beamminor")["value"]
        expr = "IM0*"+str(1.222e6/bmaj/bmin/self.observed_freq**2)
        run_immath_one(cube_tp+"_tmp2",cube_tp+"_tmp3",expr,delin=True)
        imhead(cube_tp+"_tmp3",mode="del",hdkey="bunit")
        imhead(cube_tp+"_tmp3",mode="put",hdkey="bunit",hdvalue="K")

        bmaj = imhead(imagename=cube_7m_tp+"_tmp2",mode="get",hdkey="beammajor")["value"]
        bmin = imhead(imagename=cube_7m_tp+"_tmp2",mode="get",hdkey="beamminor")["value"]
        expr = "IM0*"+str(1.222e6/bmaj/bmin/self.observed_freq**2)
        run_immath_one(cube_7m_tp+"_tmp2",cube_7m_tp+"_tmp3",expr,delin=True)
        imhead(cube_7m_tp+"_tmp3",mode="del",hdkey="bunit")
        imhead(cube_7m_tp+"_tmp3",mode="put",hdkey="bunit",hdvalue="K")

        bmaj = imhead(imagename=cube_7m_lst+"_tmp2",mode="get",hdkey="beammajor")["value"]
        bmin = imhead(imagename=cube_7m_lst+"_tmp2",mode="get",hdkey="beamminor")["value"]
        expr = "IM0*"+str(1.222e6/bmaj/bmin/self.observed_freq**2)
        run_immath_one(cube_7m_lst+"_tmp2",cube_7m_lst+"_tmp3",expr,delin=True)
        imhead(cube_7m_lst+"_tmp3",mode="del",hdkey="bunit")
        imhead(cube_7m_lst+"_tmp3",mode="put",hdkey="bunit",hdvalue="K")

        bmaj = imhead(imagename=cube_lst+"_tmp2",mode="get",hdkey="beammajor")["value"]
        bmin = imhead(imagename=cube_lst+"_tmp2",mode="get",hdkey="beamminor")["value"]
        expr = "IM0*"+str(1.222e6/bmaj/bmin/self.observed_freq**2)
        run_immath_one(cube_lst+"_tmp2",cube_lst+"_tmp3",expr,delin=True)
        imhead(cube_lst+"_tmp3",mode="del",hdkey="bunit")
        imhead(cube_lst+"_tmp3",mode="put",hdkey="bunit",hdvalue="K")

        #################
        # mom0 creation #
        #################
        os.system("rm -rf " + cube_input+"_tmp4")
        immoments(imagename=cube_input+"_tmp3",includepix=[0,100000],outfile=cube_input+"_tmp4")
        #os.system("rm -rf " + cube_input+"_tmp3")
        run_exportfits(cube_input+"_tmp4",self.mom0_input,True,True,True)

        os.system("rm -rf " + cube_tp+"_tmp4")
        immoments(imagename=cube_tp+"_tmp3",includepix=[0,100000],outfile=cube_tp+"_tmp4")
        rms=abs(measure_rms(cube_tp+"_tmp3",snr=3.0))
        self._immoments_err(cube_tp+"_tmp3",rms,self.mom0_tp.replace(".fits","_err.fits"))
        #os.system("rm -rf " + cube_tp+"_tmp3")
        run_exportfits(cube_tp+"_tmp4",self.mom0_tp,True,True,True)

        os.system("rm -rf " + cube_7m_tp+"_tmp4")
        immoments(imagename=cube_7m_tp+"_tmp3",includepix=[0,100000],outfile=cube_7m_tp+"_tmp4")
        rms=abs(measure_rms(cube_7m_tp+"_tmp3",snr=3.0))
        self._immoments_err(cube_7m_tp+"_tmp3",rms,self.mom0_7m_tp.replace(".fits","_err.fits"))
        #os.system("rm -rf " + cube_7m_tp+"_tmp3")
        run_exportfits(cube_7m_tp+"_tmp4",self.mom0_7m_tp,True,True,True)

        os.system("rm -rf " + cube_7m_lst+"_tmp4")
        immoments(imagename=cube_7m_lst+"_tmp3",includepix=[0,100000],outfile=cube_7m_lst+"_tmp4")
        rms=abs(measure_rms(cube_7m_lst+"_tmp3",snr=3.0))
        self._immoments_err(cube_7m_lst+"_tmp3",rms,self.mom0_7m_lst.replace(".fits","_err.fits"))
        #os.system("rm -rf " + cube_7m_lst+"_tmp3")
        run_exportfits(cube_7m_lst+"_tmp4",self.mom0_7m_lst,True,True,True)

        os.system("rm -rf " + cube_lst+"_tmp4")
        immoments(imagename=cube_lst+"_tmp3",includepix=[0,100000],outfile=cube_lst+"_tmp4")
        rms=abs(measure_rms(cube_lst+"_tmp3",snr=3.0))
        self._immoments_err(cube_lst+"_tmp3",rms,self.mom0_lst.replace(".fits","_err.fits"))
        #os.system("rm -rf " + cube_lst+"_tmp3")
        run_exportfits(cube_lst+"_tmp4",self.mom0_lst,True,True,True)

    ##################
    # _immoments_err #
    ##################

    def _immoments_err(
        self,
        imagename,
        rms,
        outfile,
        ):
        """
        """

        run_immath_one(imagename,imagename+"_tmp1","iif(IM0==0,0,1)",delin=False)
        immoments(imagename=imagename+"_tmp1",moments=[0],outfile=imagename+"_tmp2")
        data = imval(imagename+"_tmp1")["coords"][:,2]
        restfreq = imhead(imagename+"_tmp1",mode="list")["restfreq"][0]
        chanwidth = str(np.round(abs(data[1]-data[0])/restfreq * 299792.458, 2))
        os.system("rm -rf " + imagename + "_tmp1")

        run_immath_one(imagename+"_tmp2",imagename+"_tmp3",str(rms)+"*"+chanwidth+"*sqrt(IM0/"+chanwidth+")",delin=True)
        run_exportfits(imagename+"_tmp3",outfile,True,True,True)

    #######################
    # create_map_torussim #
    #######################

    def create_map_torussim(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.dir_ready+"inputs/"+self.torus_template_file,taskname)

        ################
        # define input #
        ################
        map_input  = self.dir_ready+"inputs/"+self.torus_template_file
        map_c9     = self.dir_ready+"outputs/imaging/torussim_12p0h/torussim_12p0h_12m_cont.image.tt0"
        map_c9_lst = self.dir_ready+"outputs/imaging/torussim_lst_12p0h/torussim_lst_12p0h_12m_cont.image.tt0"

        ##############
        # importfits #
        ##############
        run_importfits(map_input,map_input.replace(".fits",".image"))
        map_input = map_input.replace(".fits",".image")

        #################
        # convolve beam #
        #################
        imhead(map_input,mode="del",hdkey="beammajor")
        run_roundsmooth(map_input,map_input+"_tmp1",0.015,0.000001,targetres=False)
        run_immath_one(map_input+"_tmp1",map_input+"_tmp2","iif(IM0>0,IM0*1000.,0)",chans="0",delin=True)
        imrebin(map_input+"_tmp2",map_input+"_tmp3",factor=[8,8,1,1])
        os.system("rm -rf "+map_input+"_tmp2 template.image")
        os.system("cp -r " + map_input+"_tmp3 template.image")

        run_roundsmooth(map_c9,map_c9+"_tmp1",0.015)
        run_imregrid(map_c9+"_tmp1","template.image",map_c9+"_tmp2",delin=True)
        run_immath_one(map_c9+"_tmp2",map_c9+"_tmp3","IM0*1000.",chans="0",delin=True)

        run_roundsmooth(map_c9_lst,map_c9_lst+"_tmp1",0.015)
        run_imregrid(map_c9_lst+"_tmp1","template.image",map_c9_lst+"_tmp2",delin=True)
        run_immath_one(map_c9_lst+"_tmp2",map_c9_lst+"_tmp3","IM0*1000.",chans="0",delin=True)

        ##############
        # exportfits #
        ##############
        run_exportfits(map_input+"_tmp3",self.dust_input,True,True,True)
        run_exportfits(map_c9+"_tmp3",self.dust_c9,True,True,True)
        run_exportfits(map_c9_lst+"_tmp3",self.dust_c9_lst,True,True,True)

    ######################
    # plot_mom0_torussim #
    ######################

    def plot_mom0_torussim(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.dust_input,taskname)

        ra_cnt       = "40.66962133deg"
        dec_cnt      = "-0.01331803deg"
        levels_cont1 = [0.04,0.08,0.16,0.32,0.64,0.96]

        ##############
        # plot input #
        ##############
        run_importfits(
            self.dir_ready+"inputs/"+self.torus_template_file,
            "original_tmp1",
            )
        run_immath_one("original_tmp1","original_tmp2","iif(IM0>0,IM0*1000.,0)",chans="0",delin=True)
        run_exportfits("original_tmp2","original.fits",True,True,True)
        myfig_fits2png(
            # general
            "original.fits",
            self.outpng_dust_input_original,
            imcontour1="original.fits",
            imsize_as=0.20,
            ra_cnt=ra_cnt,
            dec_cnt=dec_cnt,
            # contour 1
            unit_cont1=None,
            levels_cont1=levels_cont1,
            width_cont1=[1.0],
            color_cont1="white",
            # imshow
            set_title="Input: torussim continuum",
            colorlog=False,
            set_cmap="rainbow",
            set_bg_color=cm.rainbow(0),
            showbeam=True,
            color_beam="black",
            scalebar=None,
            label_scalebar=None,
            comment=None,
            # imshow colorbar
            clim=None,#[0,120],
            label_cbar="(mJy pixel$^{-1}$)",
            # annotation
            numann=None,
            textann=True,
            )
        os.system("rm -rf original.fits")

        myfig_fits2png(
            # general
            self.dust_input,
            self.outpng_dust_input,
            imcontour1=self.dust_input,
            imsize_as=0.20,
            ra_cnt=ra_cnt,
            dec_cnt=dec_cnt,
            # contour 1
            unit_cont1=None,
            levels_cont1=levels_cont1,
            width_cont1=[1.0],
            color_cont1="white",
            # imshow
            set_title="Input (convolved): torussim continuum",
            colorlog=False,
            set_cmap="rainbow",
            set_bg_color=cm.rainbow(0),
            showbeam=True,
            color_beam="black",
            scalebar=None,
            label_scalebar=None,
            comment=None,
            # imshow colorbar
            clim=[0,1.25],
            label_cbar="(mJy beam$^{-1}$)",
            # annotation
            numann=None,
            textann=True,
            )

        myfig_fits2png(
            # general
            self.dust_c9,
            self.outpng_dust_c9,
            imcontour1=self.dust_c9,
            imsize_as=0.20,
            ra_cnt=ra_cnt,
            dec_cnt=dec_cnt,
            # contour 1
            unit_cont1=None,
            levels_cont1=levels_cont1,
            width_cont1=[1.0],
            color_cont1="white",
            # imshow
            set_title="C-9 (convolved): torussim continuum",
            colorlog=False,
            set_cmap="rainbow",
            set_bg_color=cm.rainbow(0),
            showbeam=True,
            color_beam="black",
            scalebar=None,
            label_scalebar=None,
            comment=None,
            # imshow colorbar
            clim=[0,1.25],
            label_cbar="(mJy beam$^{-1}$)",
            # annotation
            numann=None,
            textann=True,
            )

        myfig_fits2png(
            # general
            self.dust_c9_lst,
            self.outpng_dust_c9_lst,
            imcontour1=self.dust_c9_lst,
            imsize_as=0.20,
            ra_cnt=ra_cnt,
            dec_cnt=dec_cnt,
            # contour 1
            unit_cont1=None,
            levels_cont1=levels_cont1,
            width_cont1=[1.0],
            color_cont1="white",
            # imshow
            set_title="C-9+LST (convolved): torussim continuum",
            colorlog=False,
            set_cmap="rainbow",
            set_bg_color=cm.rainbow(0),
            showbeam=True,
            color_beam="black",
            scalebar=None,
            label_scalebar=None,
            comment=None,
            # imshow colorbar
            clim=[0,1.25],
            label_cbar="(mJy beam$^{-1}$)",
            # annotation
            numann=None,
            textann=True,
            )

    #############
    # plot_mom0 #
    #############

    def plot_mom0(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.n1097_template_fullspec,taskname)

        levels_cont1 = [0.01,0.02,0.04,0.08,0.16,0.32,0.64,0.96]
        levels_cont2 = [0.02,0.04,0.08,0.16,0.32,0.64,0.96]

        ##############
        # plot input #
        ##############
        myfig_fits2png(
            # general
            self.mom0_input,
            self.outpng_mom0_input,
            imcontour1=self.mom0_input,
            imsize_as=150,
            ra_cnt="41.5763deg",
            dec_cnt="-30.2771deg",
            # contour 1
            unit_cont1=None,
            levels_cont1=levels_cont1,
            width_cont1=[1.0],
            color_cont1="white",
            # imshow
            set_title="Input (convolved): n1097sim [CI] mom0",
            colorlog=False,
            set_cmap="rainbow",
            set_bg_color=cm.rainbow(0),
            showbeam=True,
            color_beam="black",
            scalebar=None,
            label_scalebar=None,
            comment=None,
            # imshow colorbar
            clim=[0,120],
            label_cbar="(K km s$^{-1}$)",
            # annotation
            numann="lst_n1097sim",
            textann=True,
            )

        ###########
        # plot TP #
        ###########
        myfig_fits2png(
            # general
            self.mom0_tp,
            self.outpng_mom0_tp,
            imcontour1=self.mom0_tp,
            imsize_as=150,
            ra_cnt="41.5763deg",
            dec_cnt="-30.2771deg",
            # contour 1
            unit_cont1=None,
            levels_cont1=levels_cont1,
            width_cont1=[1.0],
            color_cont1="white",
            # imshow
            set_title="TP 12m: n1097sim [CI] mom0",
            colorlog=False,
            set_cmap="rainbow",
            set_bg_color=cm.rainbow(0),
            showbeam=True,
            color_beam="black",
            scalebar=None,
            label_scalebar=None,
            comment=None,
            # imshow colorbar
            clim=None,
            label_cbar="(K km s$^{-1}$)",
            # annotation
            numann="lst_n1097sim",
            textann=True,
            )

        ##############
        # plot 7m+TP #
        ##############
        myfig_fits2png(
            # general
            self.mom0_7m_tp,
            self.outpng_mom0_tp_7m,
            imcontour1=self.mom0_7m_tp,
            imsize_as=150,
            ra_cnt="41.5763deg",
            dec_cnt="-30.2771deg",
            # contour 1
            unit_cont1=None,
            levels_cont1=levels_cont2,
            width_cont1=[1.0],
            color_cont1="white",
            # imshow
            set_title="7m+TP: n1097sim [CI] mom0",
            colorlog=False,
            set_cmap="rainbow",
            set_bg_color=cm.rainbow(0),
            showbeam=True,
            color_beam="black",
            scalebar=None,
            label_scalebar=None,
            comment=None,
            # imshow colorbar
            clim=[0,120],
            label_cbar="(K km s$^{-1}$)",
            # annotation
            numann="lst_n1097sim",
            textann=True,
            )

        ################
        # plot LST 50m #
        ################
        myfig_fits2png(
            # general
            self.mom0_lst,
            self.outpng_mom0_lst50m,
            imcontour1=self.mom0_lst,
            imsize_as=150,
            ra_cnt="41.5763deg",
            dec_cnt="-30.2771deg",
            # contour 1
            unit_cont1=None,
            levels_cont1=levels_cont1,
            width_cont1=[1.0],
            color_cont1="white",
            # imshow
            set_title="LST 50m: n1097sim [CI] mom0",
            colorlog=False,
            set_cmap="rainbow",
            set_bg_color=cm.rainbow(0),
            showbeam=True,
            color_beam="black",
            scalebar=None,
            label_scalebar=None,
            comment=None,
            # imshow colorbar
            clim=[0,120],
            label_cbar="(K km s$^{-1}$)",
            # annotation
            numann="lst_n1097sim",
            textann=True,
            )

        ###################
        # plot 7m+LST 50m #
        ###################
        myfig_fits2png(
            # general
            self.mom0_7m_lst,
            self.outpng_mom0_7m_lst50m,
            imcontour1=self.mom0_7m_lst,
            imsize_as=150,
            ra_cnt="41.5763deg",
            dec_cnt="-30.2771deg",
            # contour 1
            unit_cont1=None,
            levels_cont1=levels_cont1,
            width_cont1=[1.0],
            color_cont1="white",
            # imshow
            set_title="7m+LST 50m: n1097sim [CI] mom0",
            colorlog=False,
            set_cmap="rainbow",
            set_bg_color=cm.rainbow(0),
            showbeam=True,
            color_beam="black",
            scalebar=None,
            label_scalebar=None,
            comment=None,
            # imshow colorbar
            clim=[0,120],
            label_cbar="(K km s$^{-1}$)",
            # annotation
            numann="lst_n1097sim",
            textann=True,
            )

    ###################
    # sim12m_checksim #
    ###################

    def sim12m_checksim(
        self,
        totaltime="2.0h",
        totaltimetint="2p0h",
        pointingspacing="",
        ):
        """
        References:
        https://casa.nrao.edu/casadocs/casa-5.4.0/global-tool-list/tool_simulator/methods
        http://ska-sdp.org/sites/default/files/attachments/ska-tel-sdp-img-memo-001.pdf (P.28~)
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.torus_template_file,taskname)

        """
        u = simutil()
        direction  = "J2000 02h42m40.70912s -00d00m47.9449s"
        direction2 = "J2000 02h42m40.70912s -00d00m57.9449s"
        os.system("rm -rf ALMAxLST.vptable")
        os.system("rm -rf test.ms")

        # importfits
        run_importfits(
            fitsimage=self.dir_ready+"inputs/"+self.check_template_file,
            imagename="image.image",
            )

        # set LST voltage pattern as "ACA"
        mysm.open("test.ms")

        myvp.setpbgauss(
            telescope="ALMA",
            halfwidth='63.0arcsec',
            maxrad='999.0arcsec',
            reffreq='100.0GHz',
            dopb=True,
            )
        myid1 = myvp.getuserdefault("ALMA")
        myvp.setuserdefault(myid1, "ALMA", "DV")
        myvp.setuserdefault(myid1, "ALMA", "DA")
        myvp.setuserdefault(myid1, "ALMA", "PM")
        myvp.setuserdefault(myid1, "ALMA", "CM")

        myvp.setpbgauss(
            telescope="ALMA",
            halfwidth='15.0arcsec',
            maxrad='999.0arcsec',
            reffreq='100.0GHz',
            dopb=True,
            )
        myid2 = myvp.getuserdefault("ALMA")
        myvp.setuserdefault(myid2, "ALMA", "")

        myvp.summarizevps()
        myvp.saveastable("ALMAxLST.vptable")
        mysm.setvp(dovp=True,usedefaultvp=False)#,vptable="ALMAxLST.vptable")

        mysm.setoptions(
            cache=10000000,
            tile=32,
            gridfunction="BOX",
            location=myme.observatory("ALMA"),
            )

        x,y,z,d,padnames,telescope,posobs = u.readantenna(self.config_c1)
        x2,y2,z2,d2,padnames2,telescope2,posobs2 = u.readantenna(self.config_lst)

        mysm.setconfig(
            telescopename="ALMA",
            x=np.append(x,x2),
            y=np.append(y,y2),
            z=np.append(z,z2),
            dishdiameter=np.append(d,d2),
            mount=['alt-az'],
            antname=np.append(padnames,padnames2).tolist(),
            coordsystem='global',
            referencelocation=posobs,
            )
        mysm.setoptions(ftmachine="mosaic")

        mysm.setspwindow(spwname="spw1", freq="690.4640232GHz",
          deltafreq="1.0GHz",freqresolution="1GHz",nchannels=1,stokes='XX YY')
        mysm.setspwindow(spwname="spw2", freq="691.4640232GHz",
          deltafreq="1.0GHz",freqresolution="1GHz",nchannels=1,stokes='XX YY')
        mysm.setspwindow(spwname="spw3", freq="692.4640232GHz",
          deltafreq="1.0GHz",freqresolution="1GHz",nchannels=1,stokes='XX YY')
        mysm.setspwindow(spwname="spw4", freq="693.4640232GHz",
          deltafreq="1.0GHz",freqresolution="1GHz",nchannels=1,stokes='XX YY')
        mysm.setspwindow(spwname="spw5", freq="694.4640232GHz",
          deltafreq="1.0GHz",freqresolution="1GHz",nchannels=1,stokes='XX YY')
        mysm.setspwindow(spwname="spw6", freq="695.4640232GHz",
          deltafreq="1.0GHz",freqresolution="1GHz",nchannels=1,stokes='XX YY')
        mysm.setspwindow(spwname="spw7", freq="696.4640232GHz",
          deltafreq="1.0GHz",freqresolution="1GHz",nchannels=1,stokes='XX YY')

        mysm.setfeed(mode='perfect X Y',pol=[''])
        mysm.setlimits(shadowlimit=0.01, elevationlimit='10deg')
        mysm.setauto(0.0)

        mysm.setfield(sourcename="src1", 
          sourcedirection=direction,
          calcode="OBJ", distance='0m')

        mysm.settimes(integrationtime="10s", usehourangle=True, 
          referencetime=myme.epoch('TAI', "2012/01/01/00:00:00"))

        etime="600s"
        mysm.observe(sourcename="src1", spwname="spw1",
          starttime=myqa.mul(-1,qa.quantity(etime)),
          stoptime=myqa.quantity(0,"s"))
        mysm.observe(sourcename="src1", spwname="spw2",
          starttime=myqa.mul(-1,qa.quantity(etime)),
          stoptime=myqa.quantity(0,"s"))
        mysm.observe(sourcename="src1", spwname="spw3",
          starttime=myqa.mul(-1,qa.quantity(etime)),
          stoptime=myqa.quantity(0,"s"))
        mysm.observe(sourcename="src1", spwname="spw4",
          starttime=myqa.mul(-1,qa.quantity(etime)),
          stoptime=myqa.quantity(0,"s"))
        mysm.observe(sourcename="src1", spwname="spw5",
          starttime=myqa.mul(-1,qa.quantity(etime)),
          stoptime=myqa.quantity(0,"s"))

        mysm.setoptions(ftmachine="mosaic")
        mysm.predict(imagename="image.image")

        mysm.summary()
        mysm.done()
        mysm.close()

        os.system("rm -rf test_all.* test_onlyLST.* test_noLST.*")
        tclean(vis="test.ms",imagename="test_all",gridder="mosaic",cell="0.1arcsec",imsize=[256,256])
        tclean(vis="test.ms",imagename="test_onlyLST",gridder="mosaic",cell="0.1arcsec",imsize=[256,256],antenna="test")
        tclean(vis="test.ms",imagename="test_noLST",gridder="mosaic",cell="0.1arcsec",imsize=[256,256],antenna="!LST")
        """

        #"""
        run_simobserve(
            working_dir=self.dir_ready,
            template=self.check_template_file,
            antennalist=self.config_c1,
            project=self.project_check+"_12m_"+totaltimetint,
            totaltime=totaltime,
            incenter="693.9640232GHz",
            pointingspacing=pointingspacing,
            )
        #"""

    #############################
    # prepare_template_checksim #
    #############################

    def prepare_template_checksim(self):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        #check_first(self.n1097_template_fullspec,taskname)

        # cleanup directories
        input_dir  = self.dir_ready + "inputs/"
        output_dir = self.dir_ready + "outputs/"
        ms_dir     = self.dir_ready + "ms/"
        #os.system("rm -rf " + input_dir)
        #os.system("rm -rf " + output_dir)
        #os.system("rm -rf " + ms_dir)
        if not glob.glob(input_dir):
            os.mkdir(input_dir)
        if not glob.glob(output_dir):
            os.mkdir(output_dir)
        if not glob.glob(ms_dir):
            os.mkdir(ms_dir)

        # assume ngc1068 torus
        rmaj_out     = "4arcsec" # arcsec, 10pc at ngc1068, Gamez-Rosas et al. 2022 Nature
        rmin_out     = "1arcsec" # arcsec, 10pc at ngc1068, Gamez-Rosas et al. 2022 Nature
        pa           = '-50.0deg' # Gamez-Rosas et al. 2022 Nature
        totalflux    = 500 # continuum flux (mJy) at 432um (693.9640232 GHz), Garcia-Burillo et al. 2017

        direction = "J2000 02h42m40.70912s -00d00m47.9449s" # ngc1068 decl = -00d00m47.859690204s
        mycl.done()
        mycl.addcomponent(dir=direction, flux=totalflux, fluxunit='Jy', freq='693.9640232GHz', shape="disk", 
                        majoraxis=rmaj_out, minoraxis=rmin_out, positionangle=pa)
        #
        myia.fromshape("torus.im",[100,100,1,7],overwrite=True)
        cs=myia.coordsys()
        cs.setunits(['rad','rad','','Hz'])
        cell_rad=myqa.convert(myqa.quantity("0.05arcsec"),"rad")['value']
        cs.setincrement([-cell_rad,cell_rad],'direction')
        cs.setreferencevalue([myqa.convert("2.7113080889h",'rad')['value'],myqa.convert("-0.01331803deg",'rad')['value']],type="direction")
        cs.setreferencevalue("693.9640232GHz",'spectral')
        cs.setincrement('1.0GHz','spectral')
        myia.setcoordsys(cs.torecord())
        myia.setbrightnessunit("Jy/pixel")
        myia.modify(mycl.torecord(),subtract=False)
        exportfits(imagename='torus.im',fitsimage=input_dir+self.check_template_file,overwrite=True)

        myia.close()
        mycl.close()
        os.system("rm -rf torus.im")

    ##################
    # plot_mosaic_C1 #
    ##################

    def plot_mosaic_C1(self,tintegstr,freq):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        #check_first(self.torus_template_file,taskname)

        # LSTx12m ms
        target = self.project_check + "_12m_" + tintegstr
        ms_7m  = self.dir_ready + "ms/" + target + "/" + target + "." + self.config_c1.split("/")[-1].split(".cfg")[0] + ".noisy.ms"
        mymsmd.open(ms_7m)
        pointings_7m = mymsmd.sourcedirs()
        mymsmd.done()
        fov_7m = 21 * 300 / freq * (12./50.) / 3600.

        ra_7m  = []
        dec_7m = []
        for this in pointings_7m.keys():
            ra_7m.append(pointings_7m[this]["m0"]["value"]*180/np.pi)
            dec_7m.append(pointings_7m[this]["m1"]["value"]*180/np.pi)

        ra_7m  = np.array(ra_7m) - np.mean(ra_7m)
        dec_7m = np.array(dec_7m) - np.mean(dec_7m)

        ###################
        # plot LSTx12m ms #
        ###################
        ad    = [0.215,0.83,0.10,0.90]
        offset = 0.002
        xlim  = [-offset,offset]
        ylim  = [-offset,offset]
        title = "LSTx12-m mosaic (checksim)"
        xlabel = "R.A (degree)"
        ylabel = "Decl. (degree)"

        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        plt.subplots_adjust(left=ad[0], right=ad[1], bottom=ad[2], top=ad[3])
        myax_set(ax1, "both", xlim, ylim, title, xlabel, ylabel, adjust=ad)

        for i in range(len(ra_7m)):
            this_x = ra_7m[i]
            this_y = dec_7m[i]
            fov    = patches.Ellipse(xy=(this_x,this_y), width=fov_7m,
                height=fov_7m, angle=0, fill=False, color="black", edgecolor="black",
                alpha=1.0, lw=1)
            ax1.add_patch(fov)

        # text
        #ax1.text(0.05,0.92, "ALMA 12-m array", color="forestgreen", weight="bold", transform=ax1.transAxes)

        # save
        plt.subplots_adjust(hspace=.0)
        os.system("rm -rf " + self.outpng_mosaic_c1)
        plt.savefig(self.outpng_mosaic_c1, dpi=self.fig_dpi)

    ##################
    # plot_mosaic_7m #
    ##################

    def plot_mosaic_7m(self,tintegstr,freq):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        #check_first(self.torus_template_file,taskname)

        # 7m-only ms
        target = self.project_n1097 + "_7m_" + tintegstr
        ms_7m  = self.dir_ready + "ms/" + target + "/" + target + "." + self.config_7m.split("/")[-1].split(".cfg")[0] + ".noisy.ms"
        mymsmd.open(ms_7m)
        pointings_7m = mymsmd.sourcedirs()
        mymsmd.done()
        fov_7m = 21 * 300 / freq * (12./7.) / 3600.

        ra_7m  = []
        dec_7m = []
        for this in pointings_7m.keys():
            ra_7m.append(pointings_7m[this]["m0"]["value"]*180/np.pi)
            dec_7m.append(pointings_7m[this]["m1"]["value"]*180/np.pi)

        ra_7m  = np.array(ra_7m) - np.mean(ra_7m)
        dec_7m = np.array(dec_7m) - np.mean(dec_7m)

        ###################
        # plot 7m-only ms #
        ###################
        ad    = [0.215,0.83,0.10,0.90]
        xlim  = [-0.022,0.022]
        ylim  = [-0.022,0.022]
        title = "7-m mosaic"
        xlabel = "R.A (degree)"
        ylabel = "Decl. (degree)"

        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        plt.subplots_adjust(left=ad[0], right=ad[1], bottom=ad[2], top=ad[3])
        myax_set(ax1, "both", xlim, ylim, title, xlabel, ylabel, adjust=ad)

        for i in range(len(ra_7m)):
            this_x = ra_7m[i]
            this_y = dec_7m[i]
            fov    = patches.Ellipse(xy=(this_x,this_y), width=fov_7m,
                height=fov_7m, angle=0, fill=False, color="black", edgecolor="black",
                alpha=1.0, lw=1)
            ax1.add_patch(fov)

        # text
        #ax1.text(0.05,0.92, "ALMA 12-m array", color="forestgreen", weight="bold", transform=ax1.transAxes)

        # save
        plt.subplots_adjust(hspace=.0)
        os.system("rm -rf " + self.outpng_mosaic_7m)
        plt.savefig(self.outpng_mosaic_7m, dpi=self.fig_dpi)

    ###################
    # sim12m_torussim #
    ###################

    def sim12m_torussim(self,totaltime="2.0h",totaltimetint="2p0h"):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.torus_template_file,taskname)

        run_simobserve(
            working_dir=self.dir_ready,
            template=self.torus_template_file,
            antennalist=self.config_c9_lst,
            project=self.project_torus+"_12m_lst_"+totaltimetint,
            totaltime=totaltime,
            incenter="693.9640232GHz",
            )

        """
        run_simobserve(
            working_dir=self.dir_ready,
            template=self.torus_template_file,
            antennalist=self.config_c9,
            project=self.project_torus+"_12m_"+totaltimetint,
            totaltime=totaltime,
            incenter="693.9640232GHz",
            )
        """

    #############################
    # prepare_template_torussim #
    #############################

    def prepare_template_torussim(self):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        #check_first(self.n1097_template_fullspec,taskname)

        # cleanup directories
        input_dir  = self.dir_ready + "inputs/"
        output_dir = self.dir_ready + "outputs/"
        ms_dir     = self.dir_ready + "ms/"
        #os.system("rm -rf " + input_dir)
        #os.system("rm -rf " + output_dir)
        #os.system("rm -rf " + ms_dir)
        if not glob.glob(input_dir):
            os.mkdir(input_dir)
        if not glob.glob(output_dir):
            os.mkdir(output_dir)
        if not glob.glob(ms_dir):
            os.mkdir(ms_dir)

        # assume ngc1068 torus
        rmaj_out     = str(10.  / 72. / 2.)+"arcsec" # arcsec, 10pc at ngc1068, Gamez-Rosas et al. 2022 Nature
        rmin_out     = str(1.74 / 72. / 2.)+"arcsec" # arcsec, 10pc at ngc1068, Gamez-Rosas et al. 2022 Nature
        pa           = '-50.0deg' # Gamez-Rosas et al. 2022 Nature
        #totalflux    = 13.8 / 1000. * 345**3.8 / 693.9640232**3.8 * 2 # continuum flux (mJy) at 432um (693.9640232 GHz), Garcia-Burillo et al. 2017
        totalflux    = 13.8 / 1000.

        rmaj_in      = str(10.  / 5. / 72.)+"arcsec"
        rmin_in      = str(1.74 / 5. / 72.)+"arcsec"
        totalflux_in = -1 * totalflux / (2.5*2.5)

        scale = float(totalflux) / float(totalflux-totalflux_in)
        totalflux = totalflux * scale
        totalflux_in = totalflux_in * scale

        direction = "J2000 02h42m40.70912s -00d00m47.9449s" # ngc1068 decl = -00d00m47.859690204s
        mycl.done()
        mycl.addcomponent(dir=direction, flux=totalflux, fluxunit='Jy', freq='693.9640232GHz', shape="Gauss", 
                        majoraxis=rmaj_out, minoraxis=rmin_out, positionangle=pa)
        mycl.addcomponent(dir=direction, flux=totalflux_in, fluxunit='Jy', freq='693.9640232GHz', shape="Gauss", 
                        majoraxis=rmaj_in, minoraxis=rmin_in, positionangle=pa)
        #
        myia.fromshape("torus.im",[512,512,1,7],overwrite=True)
        cs=myia.coordsys()
        cs.setunits(['rad','rad','','Hz'])
        cell_rad=myqa.convert(myqa.quantity("0.0005arcsec"),"rad")['value']
        cs.setincrement([-cell_rad,cell_rad],'direction')
        cs.setreferencevalue([myqa.convert("2.7113080889h",'rad')['value'],myqa.convert("-0.01331803deg",'rad')['value']],type="direction")
        cs.setreferencevalue("693.9640232GHz",'spectral')
        cs.setincrement('1.0GHz','spectral')
        myia.setcoordsys(cs.torecord())
        myia.setbrightnessunit("Jy/pixel")
        myia.modify(mycl.torecord(),subtract=False)
        immath(imagename='torus.im',outfile="torus.im2",expr="iif(IM0>0,IM0,0)")
        exportfits(imagename='torus.im2',fitsimage=input_dir+self.torus_template_file,overwrite=True)

        myia.close()
        mycl.close()
        os.system("rm -rf torus.im torus.im2")

    ##########################
    # do_simaca_lst_n1097sim #
    ##########################

    def do_simaca_lst_n1097sim(self,totaltime="2.0h",totaltimetint="2p0h"):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.n1097_template_fullspec,taskname)

        run_simobserve(
            working_dir=self.dir_ready,
            template=self.n1097_template_fullspec,
            antennalist=self.config_7m_lst,
            project=self.project_n1097+"_LSTconnected_7m_"+totaltimetint,
            totaltime=totaltime,
            incenter=self.incenter,
            )

    #############
    # do_tp2vis #
    #############

    def do_tp2vis(self,sdimage,vis,outfile):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(lowres,taskname)

        # calculate hex ptg table (e.g., J2000 05h39m45.660s -70d07m57.524s)
        # use matplotlib.hexbin?
        print("# TBE.")

        # run tp2vis
        execfile("tp2vis.py")
        #tp2vis('tp.im','tp.ms','12m.ptg',rms=0.67)

    ##############
    # do_feather #
    ##############

    def do_feather(self,highres,lowres,outfile,effdishdiam):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(lowres,taskname)
        check_first(highres)

        os.system("rm -rf " + outfile+"_tmp1")
        feather(
            imagename = outfile+"_tmp1",
            highres = highres,
            lowres = lowres,
            effdishdiam = effdishdiam,
            )
        run_exportfits(
            imagename = outfile+"_tmp1",
            fitsimage = outfile,
            delin = True,
            dropdeg = True,
            dropstokes = True,
            )

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

    ###########
    # plot_uv #
    ###########

    def plot_uv(self,vis,outpng,plotrange,unit="m"):
        """
        Reference:
        https://safe.nrao.edu/wiki/bin/view/ALMA/Uvplt
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(vis,taskname)

        aU.uvplot(vis, field='0', plotrange=plotrange, figfile=outpng, markersize=10, density=self.fig_dpi, units=unit, mirrorPoints=True)

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

        # define center as ACA
        x_cnt = -50.06162725-0.9
        y_cnt = -568.9553881-0.4
        decl  = -37.755 # 0=celestial equator, 90=north pole, -90=south pole
        tinteg = 5.7
        #lst_position = np.array([0,0,0]) # km/s
        lst_position = np.array([6.452141+0.1, 7.886675+0.1, -0.245131]) # km/s

        # get data
        data  = np.loadtxt(self.config_c9,"str")
        x_12m = data[:,0].astype(np.float32) / 1000. - x_cnt / 1000.
        y_12m = data[:,1].astype(np.float32) / 1000. - y_cnt / 1000.
        z_12m = data[:,2].astype(np.float32) / 1000.

        data  = np.loadtxt(self.config_7m,"str")
        x_7m  = data[:,0].astype(np.float32) / 1000. - x_cnt / 1000.
        y_7m  = data[:,1].astype(np.float32) / 1000. - y_cnt / 1000.
        z_7m  = data[:,2].astype(np.float32) / 1000.

        #############################
        # plot: 7m antenna position #
        #############################
        ad    = [0.215,0.83,0.10,0.90]
        dev   = 65
        xlim  = [-dev,dev]
        ylim  = [-dev,dev]
        title = "Morita Array (7m+TP) antenna positions"
        xlabel = "East-West (m)"
        ylabel = "North-South (m)"

        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        plt.rcParams["font.family"] = "Calibri"
        plt.subplots_adjust(left=ad[0], right=ad[1], bottom=ad[2], top=ad[3])
        myax_set(ax1, None, xlim, ylim, title, xlabel, ylabel, adjust=ad)

        for i in range(len(x_7m)):
            this_x = x_7m[i] * 1000.
            this_y = y_7m[i] * 1000.
            antenna = patches.Ellipse(xy=(this_x,this_y), width=7.0,
                height=7.0, angle=0, fill=True, color="deepskyblue", edgecolor="deepskyblue",
                alpha=1.0, lw=0)
            ax1.add_patch(antenna)

        # TPx4
        antenna = patches.Ellipse(xy=(-36.00,27.50), width=12.0,
            height=12.0, angle=0, fill=True, color="grey", edgecolor="grey",
            alpha=1.0, lw=0)
        ax1.add_patch(antenna)
        antenna = patches.Ellipse(xy=(-31.00,-32.00), width=12.0,
            height=12.0, angle=0, fill=True, color="grey", edgecolor="grey",
            alpha=1.0, lw=0)
        ax1.add_patch(antenna)
        antenna = patches.Ellipse(xy=(35.50,-26.00), width=12.0,
            height=12.0, angle=0, fill=True, color="grey", edgecolor="grey",
            alpha=1.0, lw=0)
        ax1.add_patch(antenna)
        antenna = patches.Ellipse(xy=(36.00,29.50), width=12.0,
            height=12.0, angle=0, fill=True, color="grey", edgecolor="grey",
            alpha=1.0, lw=0)
        ax1.add_patch(antenna)

        # LST at the center
        antenna = patches.Ellipse(xy=(0,0), width=50.0,
            height=50.0, angle=0, fill=False, color="tomato", edgecolor="tomato",
            alpha=0.7, lw=3, ls="dashed")
        ax1.add_patch(antenna)

        # ann
        # text
        ax1.text(0.05,0.92, "ACA 7-m array", color="deepskyblue", weight="bold", transform=ax1.transAxes)
        ax1.text(0.05,0.87, "ACA TP array", color="grey", weight="bold", transform=ax1.transAxes)
        ax1.text(0.05,0.82, "LST 50m", color="tomato", weight="bold", transform=ax1.transAxes)
        # save
        plt.subplots_adjust(hspace=.0)
        os.system("rm -rf " + self.outpng_config_7m)
        plt.savefig(self.outpng_config_7m, dpi=self.fig_dpi)

        ##############################
        # plot: C-9 antenna position #
        ##############################
        ad    = [0.215,0.83,0.10,0.90]
        xlim  = [-8,8]
        ylim  = [-8,8]
        title = "ALMA 12-m array and LST$_{\mathrm{sim,50m}}$ positions"
        xlabel = "East-West (km)"
        ylabel = "North-South (km)"

        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        plt.subplots_adjust(left=ad[0], right=ad[1], bottom=ad[2], top=ad[3])
        myax_set(ax1, None, xlim, ylim, title, xlabel, ylabel, adjust=ad)

        # LST 0
        antenna = patches.Ellipse(xy=(-50.06162725/1000.-x_cnt/1000.,-457.2313425/1000.-y_cnt/1000.), width=0.8,
            height=0.8, angle=0, fill=True, color="tomato", edgecolor="tomato",
            alpha=0.7, lw=0)
        ax1.add_patch(antenna)

        # LST 1
        antenna = patches.Ellipse(xy=(5.0-x_cnt/1000.,5.0-y_cnt/1000.+0.1), width=0.8,
            height=0.8, angle=0, fill=True, color="tomato", edgecolor="tomato",
            alpha=0.7, lw=0)
        ax1.add_patch(antenna)

        for i in range(len(x_12m)):
            this_x = x_12m[i]
            this_y = y_12m[i]
            antenna = patches.Ellipse(xy=(this_x,this_y), width=0.3,
                height=0.3, angle=0, fill=True, color="deepskyblue", edgecolor="deepskyblue",
                alpha=1.0, lw=0)
            ax1.add_patch(antenna)

        # text
        ax1.text(0.05,0.92, "ALMA 12-m array", color="deepskyblue", weight="bold", transform=ax1.transAxes)
        ax1.text(0.05,0.87, "two LST$_{\mathrm{sim,50m}}$ positions", color="tomato", weight="bold", transform=ax1.transAxes)

        # save
        plt.subplots_adjust(hspace=.0)
        os.system("rm -rf " + self.outpng_config_12m)
        plt.savefig(self.outpng_config_12m, dpi=self.fig_dpi)

    ###########################
    # phangs_pipeline_imaging #
    ###########################

    def phangs_pipeline_imaging(self,this_proj,this_array,this_target,do_cont=False,only_dirty=False):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        #check_first(self.template_fullspec,taskname)

        # prepare dir_cleanmask = dir_singledish
        dir_cleanmask = self.dir_ready + "outputs/"
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

        if do_cont==False:
            set_no_cont_products = True
        else:
            set_no_cont_products = False

        # set handlers
        for this_hander in [this_uvh,this_imh,this_pph]:
            this_hander.set_targets(only=[this_target])
            this_hander.set_line_products(only=["ci10"])
            this_hander.set_no_cont_products(set_no_cont_products)
            this_hander.set_no_line_products(False)
            this_hander.set_interf_configs(only=[this_array])

        # run piepline
        if do_cont==False:
            this_uvh.loop_stage_uvdata(\
                    do_copy           = True,
                    do_remove_staging = True,
                    do_contsub        = False,
                    do_extract_line   = True,
                    do_extract_cont   = False,
                    overwrite         = False,
                    )

        if only_dirty==True:
            do_singlescale_clean = False
        else:
            do_singlescale_clean = True

        this_imh.loop_imaging(\
                do_dirty_image          = True,
                do_revert_to_dirty      = False,
                do_read_clean_mask      = False,
                do_multiscale_clean     = False,
                do_revert_to_multiscale = False,
                do_singlescale_mask     = False,
                do_singlescale_clean    = do_singlescale_clean,
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

    def simlst_n1097sim(self,lst_res="3.04arcsec",tp_res="11.8arcsec",totaltimetint="2p0h",dryrun=True):
        """
        1. measure rms_7m (= rms level of the 7m-only cleaned image)
        2. measure tp-tinteg (= 7m-tinteg * 1.7)
        3. provide tp-tinteg and 7m-tinteg to ASC to calculate achievable sensitivity and TP/7m sensitivity ratio (= 2.326550129182734/1.4789569480812979)
        4. scale rms_7m to get rms_tp
        5. rms_tp must be also scaled by LST/TP beam area ratio (to match the sensitivity in K units, not Jy/beam)
        6. then, scale to match sensitivity "after" convolution a common TP beam.
        """

        inputcube = self.dir_ready+"inputs/"+self.n1097_template_fullspec.replace(".image",".fits")

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(inputcube,taskname)

        image_7m = self.dir_ready + "outputs/imaging/"+self.project_n1097+"_"+totaltimetint + "/"+self.project_n1097+"_"+totaltimetint+"_7m_ci10.image"
        rms_7m = measure_rms(image_7m, snr=3.0,rms_or_p84 = "p84")
        rms_lst = rms_7m * 2.326550129182734/1.4789569480812979 * float(lst_res.replace("arcsec",""))**2 / float(tp_res.replace("arcsec",""))**2
        rms_lst = rms_lst * float(tp_res.replace("arcsec","")) / float(lst_res.replace("arcsec",""))
        rms_lst_K = 1.222e6 * float(lst_res.replace("arcsec",""))**-2 * self.observed_freq**-2 * rms_lst

        # calc pointing number
        header       = imhead(inputcube,mode="list")
        area_in_as   = (header["shape"][0]*header["cdelt2"]*3600*180/np.pi) * (header["shape"][1]*header["cdelt2"]*3600*180/np.pi)
        one_hex_as   = (float(lst_res.replace("arcsec",""))/4.0)**2 * 6/np.sqrt(3) # hex with 1/4-beam length
        num_pointing = int(np.ceil(area_in_as / one_hex_as))

        # ACA TP sim at 492.16065100 GHz
        print("### LST observations")
        print("# achieved 7m sensitivity (Jy/b)  = " + str(np.round(rms_7m,5)))
        print("# required LST sensntivity (Jy/b) = " + str(np.round(rms_lst,5)))
        print("# sensitivity per pointing (K)    = " + str(np.round(rms_lst_K,5)))
        print("# beam size (arcsec)              = " + str(np.round(float(lst_res.replace("arcsec","")),2)))
        print("# number of pointing              = " + str(num_pointing))
        print("# survey area (degree^2)          = " + str(np.round(area_in_as/3600.**2,8)))
        print("#")
        print("# 7m input   = " + image_7m.split("/")[-1])
        print("# LST output = " + self.n1097_lstimage_fullspec.replace(".image","_"+totaltimetint+"7m.image"))
        print("#")

        # run
        if dryrun==False:
            simtp(
                working_dir=self.dir_ready,
                template_fullspec=self.n1097_template_fullspec.replace(".image",".fits"),
                sdimage_fullspec=self.n1097_lstimage_fullspec.replace(".image","_"+totaltimetint+"7m.image"),
                sdnoise_image=self.n1097_lstnoise_image.replace(".image","_"+totaltimetint+"7m.image"),
                singledish_res=lst_res,
                singledish_noise=rms_lst, # Jy/beam at final res
                observed_freq=self.observed_freq*1e9,
                )
        else:
            print("# skipped simtp as dryrun==True")
            print("#")

    ##################
    # simtp_n1097sim #
    ##################

    def simtp_n1097sim(self,singledish_res="11.8arcsec",totaltimetint="2p0h",dryrun=True):
        """
        1. measure rms_7m (= rms level of the 7m-only cleaned image)
        2. measure tp-tinteg (= 7m-tinteg * 1.7)
        3. provide tp-tinteg and 7m-tinteg to ASC to calculate achievable sensitivity and TP/7m sensitivity ratio (= 2.326550129182734/1.4789569480812979)
        4. scale rms_7m to get rms_tp
        """

        inputcube = self.dir_ready+"inputs/"+self.n1097_template_fullspec.replace(".image",".fits")

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(inputcube,taskname)

        image_7m = self.dir_ready + "outputs/imaging/"+self.project_n1097+"_"+totaltimetint + "/"+self.project_n1097+"_"+totaltimetint+"_7m_ci10.image"
        rms_7m = measure_rms(image_7m, snr=3.0,rms_or_p84="rms")
        rms_tp = rms_7m * 2.326550129182734/1.4789569480812979 # ASC TPrms/7mrms ratio
        rms_tp_K = 1.222e6 * float(singledish_res.replace("arcsec",""))**-2 * self.observed_freq**-2 * rms_tp

        # calc pointing number
        header       = imhead(inputcube,mode="list")
        area_in_as   = (header["shape"][0]*header["cdelt2"]*3600*180/np.pi) * (header["shape"][1]*header["cdelt2"]*3600*180/np.pi)
        one_hex_as   = (float(singledish_res.replace("arcsec",""))/4.0)**2 * 6/np.sqrt(3) # hex with 1/4-beam length
        num_pointing = int(np.ceil(area_in_as / one_hex_as))

        """ failed to calculate SD sensitivity based on 7m observing time...
        # tinteg per pointing
        # sensitivity at 492.16065100 GHz based on ASC (1hr = 3.033450239598523 Jy/beam)
        tinteg_per_pointing = float(totaltime.replace("h","")) / num_pointing
        singledish_noise_per_pointing = 3.033450239598523 / 1000. / tinteg_per_pointing**2
        singledish_noise_per_pointing_K = 1.222e6 * float(singledish_res.replace("arcsec",""))**-2 * self.observed_freq**-2 * singledish_noise_per_pointing
        """

        # ACA TP sim at 492.16065100 GHz
        print("### ACA TP observations")
        print("# achieved 7m sensitivity (Jy/b) = " + str(np.round(rms_7m,5)))
        print("# required TP sensntivity (Jy/b) = " + str(np.round(rms_tp,5)))
        print("# sensitivity per pointing (K)   = " + str(np.round(rms_tp_K,5)))
        print("# beam size (arcsec)             = " + str(np.round(float(singledish_res.replace("arcsec","")),2)))
        print("# number of pointing             = " + str(num_pointing))
        print("# survey area (degree^2)         = " + str(np.round(area_in_as/3600.**2,8)))
        print("#")
        print("# 7m input  = " + image_7m.split("/")[-1])
        print("# TP output = " + self.n1097_sdimage_fullspec.replace(".image","_"+totaltimetint+"7m.image"))
        print("#")

        # run
        if dryrun==False:
            simtp(
                working_dir=self.dir_ready,
                template_fullspec=self.n1097_template_fullspec.replace(".image",".fits"),
                sdimage_fullspec=self.n1097_sdimage_fullspec.replace(".image","_"+totaltimetint+"7m.image"),
                sdnoise_image=self.n1097_sdnoise_image.replace(".image","_"+totaltimetint+"7m.image"),
                singledish_res=singledish_res,
                singledish_noise=rms_tp, # Jy/beam at final res
                observed_freq=self.observed_freq*1e9,
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
            pa=self.image_rot_n1097sim, # rotation angle
            )

    ##################
    # _get_baselines #
    ##################

    def _get_baselines(self,x,y,decl=60,tinteg=0):
        """
        some calculation (both u and v) is wrong apparently!
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