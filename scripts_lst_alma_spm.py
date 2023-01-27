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
2023-01-25   created
2023-01-26   simulations and imaging with GM Aur
Toshiki Saito@NAOJ
"""

import os, sys, glob
import numpy as np
from scipy.stats import gaussian_kde

from mycasa_tasks import *
from mycasa_plots import *
from mycasa_simobs import *

##################
# ToolsLSTSpMSim #
##################
class ToolsLSTSpMSim():
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
        done!
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

        # phangs-alma pipeline
        self.dir_pipeline  = self._read_key("dir_pipeline")
        self.project_gmaur = self._read_key("project_gmaur")

    def _set_input_fits(self):
        """
        """

        self.gmaur_template_file = self._read_key("gmaur_template_file")
        self.gmaur_template_mask = self._read_key("gmaur_template_mask")

    def _set_output_fits(self):
        """
        """

        self.gmaur_template_noshrunk = self.project_gmaur + "_template_shrunk.image"
        self.gmaur_template_shurnk2  = self.project_gmaur + "gmaur_template_shurnk2.image"
        self.gmaur_template_shurnk4  = self.project_gmaur + "gmaur_template_shurnk4.image"

    def _set_input_param(self):
        """
        """

        # simobserve
        self.config_c1        = self.dir_keyfile + self._read_key("config_c1")
        self.config_c9        = self.dir_keyfile + self._read_key("config_c9")
        self.config_c10       = self.dir_keyfile + self._read_key("config_c10")
        self.config_c10_lstI  = self.dir_keyfile + self._read_key("config_c10_lstI")
        self.config_c10_lstII = self.dir_keyfile + self._read_key("config_c10_lstII")
        self.config_7m        = self.dir_keyfile + self._read_key("config_7m")
        self.config_lstI      = self.dir_keyfile + self._read_key("config_lstI")
        self.config_lstII     = self.dir_keyfile + self._read_key("config_lstII")

    def _set_output_txt_png(self):
        """
        """

        self.outpng_config_12m = self.dir_products + self._read_key("outpng_config_12m")

    ####################
    # run_sim_lst_alma #
    ####################

    def run_sim_lst_alma(
        self,
        ############
        # GMAursim #
        ############
        # prepare
        tinteg_GMaursim      = 24,    # 12m total observing time
        observed_freq        = 230.0, # GHz, determine LST and TP beam sizes
        do_template_GMaursim = False, # create template simobserve
        do_simint_GMaursim   = False, # sim C-10 at observed_freq
        do_imaging_GMaursim  = False,
        plot_config          = False,
        ):
        """
        This method runs all the methods which will create figures in the white paper.
        """

        ###########################
        # set GMAursim parameters #
        ###########################
        # observed frequency
        self.observed_freq = observed_freq
        self.incenter      = str(observed_freq)+"GHz"

        # GMaursim_12m from tinteg_GMaursim
        tinteg_12m      = str(float(tinteg_GMaursim))+"h"
        tintegstr_12m   = tinteg_12m.replace(".","p")
        this_target     = self.project_gmaur+"_"+tintegstr_12m
        this_target_lst = self.project_gmaur+"_lstI_"+tintegstr_12m

        if do_template_GMaursim==True:
            self.prepare_template_gmaursim()

        if do_simint_GMaursim==True:
            self.simobs_gmaursim(tinteg_12m,tintegstr_12m)

        if do_imaging_GMaursim==True:
            #############
            # config 10 #
            #############
            # stage instead of pipeline
            msname  = self.project_gmaur + "_12m_" + tintegstr_12m + "."+self.config_c10.split("/")[-1].split(".cfg")[0]+".noisy.ms"
            ms_from = self.dir_ready + "ms/" + self.project_gmaur + "_12m_" + tintegstr_12m + "/" + msname
            dir_to  = self.dir_ready + "outputs/imaging/" + this_target + "/"
            ms_to   = dir_to + this_target + "_12m_cont.ms"
            os.system("rm -rf " + ms_to)
            os.system("rm -rf " + dir_to)
            os.makedirs(dir_to)
            os.system("cp -r " + ms_from + " " + ms_to)

            # imaging
            self.phangs_pipeline_imaging(
                this_proj=self.project_gmaur,
                this_array="12m",
                this_target=this_target,
                only_dirty=False,
                )

            #######################
            # config 10 + LST 50m #
            #######################
            # stage instead of pipeline
            msname  = self.project_gmaur + "_12m_lstI_" + tintegstr_12m + "."+self.config_c10_lstI.split("/")[-1].split(".cfg")[0]+".noisy.ms"
            ms_from = self.dir_ready + "ms/" + self.project_gmaur + "_12m_lstI_" + tintegstr_12m + "/" + msname
            dir_to  = self.dir_ready + "outputs/imaging/" + this_target_lst + "/"
            ms_to   = dir_to + this_target_lst + "_12m_cont.ms"
            os.system("rm -rf " + ms_to)
            os.system("rm -rf " + dir_to)
            os.makedirs(dir_to)
            os.system("cp -r " + ms_from + " " + ms_to)

            # imaging
            self.phangs_pipeline_imaging(
                this_proj=self.project_gmaur,
                this_array="12m",
                this_target=this_target_lst,
                only_dirty=False,
                )

        if plot_config:
            self.plot_config()

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

        # get data
        data    = np.loadtxt(self.config_c10,"str")
        x_12m   = data[:,0].astype(np.float32) / 1000.
        y_12m   = data[:,1].astype(np.float32) / 1000. 
        z_12m   = data[:,2].astype(np.float32) / 1000.

        data    = np.loadtxt(self.config_lstI,"str")
        x_lstI  = data[0].astype(np.float32) / 1000.
        y_lstI  = data[1].astype(np.float32) / 1000. 
        z_lstI  = data[2].astype(np.float32) / 1000.

        data    = np.loadtxt(self.config_lstII,"str")
        x_lstII = data[0].astype(np.float32) / 1000.
        y_lstII = data[1].astype(np.float32) / 1000. 
        z_lstII = data[2].astype(np.float32) / 1000.

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

        # LST I
        antenna = patches.Ellipse(xy=(x_lstI, y_lstI, z_lstI), width=0.8,
            height=0.8, angle=0, fill=True, color="tomato", edgecolor="tomato",
            alpha=0.7, lw=0)
        ax1.add_patch(antenna)

        # LST II
        antenna = patches.Ellipse(xy=(x_lstII, y_lstII, z_lstII), width=0.8,
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

    def phangs_pipeline_imaging(
        self,
        this_proj,
        this_array,
        this_target,
        only_dirty=False,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name

        # prepare dir_cleanmask = dir_singledish
        dir_cleanmask = self.dir_ready + "outputs/"
        if not glob.glob(dir_cleanmask):
            os.mkdir(dir_cleanmask)

        # set piepline
        master_key = self.dir_pipeline + "master_key_lst_alma_spm.txt"

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
            this_hander.set_no_cont_products(False)
            this_hander.set_no_line_products(True)
            this_hander.set_interf_configs(only=[this_array])

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
                do_singlescale_mask     = True,
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
    # simobs_gmaursim #
    ###################

    def simobs_gmaursim(self,totaltime="2.0h",totaltimetint="2p0h"):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.gmaur_template_noshrunk,taskname)

        run_simobserve(
            working_dir = self.dir_ready,
            template    = self.gmaur_template_noshrunk,
            antennalist = self.config_c10,
            project     = self.project_gmaur+"_12m_"+totaltimetint,
            totaltime   = totaltime,
            incenter    = self.incenter,
            pointingspacing = "3arcsec",
            )

        run_simobserve(
            working_dir = self.dir_ready,
            template    = self.gmaur_template_noshrunk,
            antennalist = self.config_c10_lstI,
            project     = self.project_gmaur+"_12m_lstI_"+totaltimetint,
            totaltime   = totaltime,
            incenter    = self.incenter,
            pointingspacing = "3arcsec",
            )

    #############################
    # prepare_template_gmaursim #
    #############################

    def prepare_template_gmaursim(self):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name

        gen_cont(
            self.dir_raw,
            self.gmaur_template_file,
            self.gmaur_template_mask,
            self.dir_ready,
            self.gmaur_template_noshrunk,
            self.gmaur_template_shurnk2,
            self.gmaur_template_shurnk4,
            )

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

#########################
# end of ToolsLSTSpMSim #
#########################